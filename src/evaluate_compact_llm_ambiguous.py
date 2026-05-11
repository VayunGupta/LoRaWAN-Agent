from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.build_results_bundle import build_scenarios
from src.react_agent import (
    GatewayCoordinatorServer,
    OllamaAdjudicator,
    OllamaConfig,
    ReActGatewayAgent,
    ServerConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate compact-payload LLM adjudication on the ambiguous slice only."
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
    )
    parser.add_argument("--out-dir", default="outputs/compact_llm_ambiguous")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of ambiguous scenarios to evaluate.",
    )
    return parser


def select_ambiguous_scenarios(limit: int | None = None) -> list[dict]:
    base_server = GatewayCoordinatorServer(ServerConfig(use_environment_context=False))
    base_server.ensure_metadata()
    scenarios = build_scenarios(base_server)
    agent = ReActGatewayAgent(base_server, llm_mode="off")

    selected: list[dict] = []
    for item in scenarios:
        report = agent.investigate(item["scenario"]).to_dict()
        if report["heuristic"]["should_invoke_llm"]:
            selected.append(item)

    if limit is not None:
        return selected[:limit]
    return selected


def evaluate_condition(
    *,
    scenarios: list[dict],
    use_environment_context: bool,
    llm_model: str,
    ollama_endpoint: str,
) -> tuple[pd.DataFrame, dict]:
    server = GatewayCoordinatorServer(ServerConfig(use_environment_context=use_environment_context))
    server.ensure_metadata()
    llm_client = OllamaAdjudicator(
        OllamaConfig(model=llm_model, endpoint=ollama_endpoint, timeout_s=45.0)
    )
    agent = ReActGatewayAgent(server, llm_client=llm_client, llm_mode="adjudicate")

    rows: list[dict] = []
    for item in scenarios:
        report = agent.investigate(item["scenario"]).to_dict()
        rows.append(
            {
                "condition": "compact_llm_osm" if use_environment_context else "compact_llm_base",
                "scenario_name": item["scenario_name"],
                "attack_family": item["attack_family"],
                "severity": item["severity"],
                "expected_label": item["expected_label"],
                "predicted_label": report["predicted_attack_type"],
                "correct": report["predicted_attack_type"] == item["expected_label"],
                "attack_detected": report["attack_detected"],
                "expected_attack_detected": item["expected_label"] != "none",
                "heuristic_prediction": report["heuristic"]["predicted_attack_type"],
                "heuristic_confidence": report["heuristic"]["confidence"],
                "heuristic_reasons": "|".join(report["heuristic"]["trigger_reasons"]),
                "llm_invoked": report["llm"]["invoked"],
                "llm_available": report["llm"]["available"],
                "llm_faithful": report["llm"]["faithful"],
                "llm_final_label": report["llm"]["final_label"],
                "llm_confidence": report["llm"]["confidence"],
                "llm_request_more_evidence": report["llm"]["request_more_evidence"],
                "llm_next_tool": report["llm"]["next_tool"],
                "llm_fallback_reason": report["llm"]["fallback_reason"],
                "sensor_environment_plausibility": report["evidence"].get("sensor_environment_plausibility", 0.0),
                "gateway_environment_plausibility": report["evidence"].get("gateway_environment_plausibility", 0.0),
                "sensor_expected_extra_attenuation_db": report["evidence"].get("sensor_expected_extra_attenuation_db", 0.0),
                "gateway_expected_extra_attenuation_db": report["evidence"].get("gateway_expected_extra_attenuation_db", 0.0),
                "top_sensor_score_raw": report["evidence"].get("top_sensor_score_raw", report["evidence"]["top_sensor_score"]),
                "top_sensor_score": report["evidence"]["top_sensor_score"],
                "top_gateway_score_raw": report["evidence"].get("top_gateway_score_raw", report["evidence"]["top_gateway_score"]),
                "top_gateway_score": report["evidence"]["top_gateway_score"],
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "condition": "compact_llm_osm" if use_environment_context else "compact_llm_base",
        "scenario_count": int(df.shape[0]),
        "overall_accuracy": float(df["correct"].mean()),
        "attack_detection_accuracy": float(
            (df["attack_detected"] == df["expected_attack_detected"]).mean()
        ),
        "llm_available_rate": float(df["llm_available"].mean()),
        "llm_invocation_rate": float(df["llm_invoked"].mean()),
        "llm_faithfulness_rate": float(df["llm_faithful"].mean()) if not df.empty else 0.0,
        "request_more_evidence_rate": float(df["llm_request_more_evidence"].mean()) if not df.empty else 0.0,
    }
    return df, summary


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = select_ambiguous_scenarios(limit=args.limit)
    base_rows, base_summary = evaluate_condition(
        scenarios=scenarios,
        use_environment_context=False,
        llm_model=args.llm_model,
        ollama_endpoint=args.ollama_endpoint,
    )
    osm_rows, osm_summary = evaluate_condition(
        scenarios=scenarios,
        use_environment_context=True,
        llm_model=args.llm_model,
        ollama_endpoint=args.ollama_endpoint,
    )

    all_rows = pd.concat([base_rows, osm_rows], ignore_index=True)
    summary = {
        "ambiguous_scenario_count": len(scenarios),
        "conditions": [base_summary, osm_summary],
    }

    all_rows.to_csv(out_dir / "compact_llm_ambiguous_rows.csv", index=False)
    (out_dir / "compact_llm_ambiguous_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
