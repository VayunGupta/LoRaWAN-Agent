from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.react_agent import (
    AttackScenario,
    GatewayCoordinatorServer,
    OllamaAdjudicator,
    OllamaConfig,
    ReActGatewayAgent,
    ServerConfig,
)

ROLE_CHOICES = [
    "gateway_local",
    "temporal_consistency",
    "physical_consistency",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the server-side ReAct LoRaWAN attack detection agent."
    )
    parser.add_argument(
        "--attack-type",
        default="none",
        choices=[
            "none",
            "sensor_foil",
            "gateway_bias",
            "random_noise",
            "packet_drop",
            "replay_attack",
            "delayed_replay",
            "gateway_fabrication",
            "selective_suppression",
            "counter_corruption",
        ],
    )
    parser.add_argument("--sensor", default=None, help="Target sensor for sensor-side attacks.")
    parser.add_argument("--gateway", default=None, help="Target gateway for gateway-side attacks.")
    parser.add_argument("--rssi-shift-db", type=float, default=0.0)
    parser.add_argument("--noise-sigma-db", type=float, default=0.0)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument("--replay-delay-s", type=float, default=0.0)
    parser.add_argument("--fabricate-fraction", type=float, default=0.0)
    parser.add_argument("--fabricate-shift-db", type=float, default=8.0)
    parser.add_argument("--counter-shift", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="Optional path for the full JSON report.")
    parser.add_argument(
        "--llm-mode",
        default="adjudicate",
        choices=["off", "explain", "adjudicate"],
        help="Use the LLM only for explanations or bounded adjudication on ambiguous cases.",
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b", help="Ollama model name.")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama HTTP endpoint for non-streaming generation.",
    )
    parser.add_argument(
        "--use-environment-context",
        action="store_true",
        help="Use cached satellite/SAM path context, falling back to OpenStreetMap context, to regularize weak anomalies.",
    )
    parser.add_argument(
        "--architecture",
        default="energy_graph",
        choices=["loramas", "centralized_trust", "consistency_graph", "energy_graph", "localization_only"],
        help="Choose LoRaMAS, the centralized trust baseline, the consistency-graph verifier, the energy-style verifier, or an RF/localization-only baseline.",
    )
    parser.add_argument(
        "--role-reasoning",
        default="rules",
        choices=["rules", "llm"],
        help="Use deterministic specialist-role rules or LLM-rewritten specialist claims.",
    )
    parser.add_argument(
        "--disable-role",
        nargs="*",
        default=[],
        choices=ROLE_CHOICES,
        help="Disable one or more LoRaMAS agent roles for ablation runs.",
    )
    parser.add_argument(
        "--supervisor-max-rounds",
        type=int,
        default=2,
        help="Maximum number of supervisor rounds, including rebuttal rounds.",
    )
    return parser


def format_trace(report: dict) -> str:
    verdict_label = "Supervisor verdict"
    prediction_label = "Supervisor prediction"
    confidence_label = "Supervisor confidence"
    if report.get("architecture") in {"centralized_trust", "localization_only"}:
        verdict_label = "Centralized verdict"
        prediction_label = "Centralized prediction"
        confidence_label = "Centralized confidence"
    lines = [
        f"Predicted attack: {report['predicted_attack_type']}",
        f"Attack detected: {report['attack_detected']}",
        f"Confidence: {report['confidence']:.2f}",
        f"Suspicious sensor: {report['suspicious_sensor']}",
        f"Suspicious gateway: {report['suspicious_gateway']}",
        f"Architecture: {report.get('architecture', 'loramas')}",
        f"Role reasoning: {report.get('role_backend', 'rules')}",
        f"{verdict_label}: {report['heuristic']['verdict']}",
        f"{prediction_label}: {report['heuristic']['predicted_attack_type']}",
        f"{confidence_label}: {report['heuristic']['confidence']:.2f}",
        f"LLM invoked: {report['llm']['invoked']}",
        f"LLM mode: {report['llm']['mode']}",
        "Evidence:",
    ]
    for key, value in report["evidence"].items():
        lines.append(f"  - {key}: {value}")
    if report["heuristic"]["trigger_reasons"]:
        lines.append("Heuristic trigger reasons:")
        for reason in report["heuristic"]["trigger_reasons"]:
            lines.append(f"  - {reason}")
    if report.get("agent_claims"):
        lines.append("LoRaMAS agent claims:")
        for claim in report["agent_claims"]:
            lines.append(
                f"  - {claim['agent_name']} ({claim['role']}, round={claim.get('round_index', 1)}): label={claim['label']} confidence={claim['confidence']}"
            )
            lines.append(f"    rationale: {claim['rationale']}")
            if claim["evidence_keys"]:
                lines.append(f"    evidence_keys: {claim['evidence_keys']}")
    if report["llm"]["invoked"]:
        lines.append("LLM adjudication:")
        lines.append(f"  - available: {report['llm']['available']}")
        lines.append(f"  - model: {report['llm']['model']}")
        lines.append(f"  - final_label: {report['llm']['final_label']}")
        lines.append(f"  - confidence: {report['llm']['confidence']}")
        lines.append(f"  - faithful: {report['llm']['faithful']}")
        lines.append(f"  - request_more_evidence: {report['llm']['request_more_evidence']}")
        lines.append(f"  - next_tool: {report['llm']['next_tool']}")
        lines.append(f"  - evidence_used: {report['llm']['evidence_used']}")
        lines.append(f"  - fallback_reason: {report['llm']['fallback_reason']}")
        if report["llm"]["rationale"]:
            lines.append(f"  - rationale: {report['llm']['rationale']}")
    lines.append("Trace:")
    for idx, step in enumerate(report["trace"], start=1):
        lines.append(f"  {idx}. Thought: {step['thought']}")
        lines.append(f"     Action: {step['action']}")
        lines.append(f"     Observation: {step['observation']}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    scenario = AttackScenario(
        attack_type=args.attack_type,
        sensor=args.sensor,
        gateway=args.gateway,
        rssi_shift_db=args.rssi_shift_db,
        noise_sigma_db=args.noise_sigma_db,
        drop_prob=args.drop_prob,
        replay_fraction=args.replay_fraction,
        replay_delay_s=args.replay_delay_s,
        fabricate_fraction=args.fabricate_fraction,
        fabricate_shift_db=args.fabricate_shift_db,
        counter_shift=args.counter_shift,
        seed=args.seed,
    )

    server = GatewayCoordinatorServer(
        ServerConfig(use_environment_context=args.use_environment_context)
    )
    server.ensure_metadata()
    llm_client = None
    if args.llm_mode != "off" or args.role_reasoning == "llm":
        llm_client = OllamaAdjudicator(
            OllamaConfig(model=args.llm_model, endpoint=args.ollama_endpoint)
        )
    enabled_roles = set(ROLE_CHOICES) - set(args.disable_role)
    agent = ReActGatewayAgent(
        server,
        llm_client=llm_client,
        llm_mode=args.llm_mode,
        architecture=args.architecture,
        enabled_roles=enabled_roles,
        supervisor_max_rounds=args.supervisor_max_rounds,
        role_backend=args.role_reasoning,
    )
    report = agent.investigate(scenario).to_dict()

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(format_trace(report))


if __name__ == "__main__":
    main()
