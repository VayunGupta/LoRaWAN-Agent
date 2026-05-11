from __future__ import annotations

import argparse
import csv
import itertools
import re
import json
import random
import time
from pathlib import Path
from typing import Any

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

ARCHITECTURE_PRESETS = {
    "localization_only": {
        "architecture": "localization_only",
        "enabled_roles": set(),
        "supervisor_max_rounds": 1,
    },
    "centralized_trust": {
        "architecture": "centralized_trust",
        "enabled_roles": set(),
        "supervisor_max_rounds": 1,
    },
    "loramas": {
        "architecture": "energy_graph",
        "enabled_roles": set(ROLE_CHOICES),
        "supervisor_max_rounds": 2,
    },
    "consistency_graph": {
        "architecture": "consistency_graph",
        "enabled_roles": set(ROLE_CHOICES),
        "supervisor_max_rounds": 2,
    },
    "energy_graph": {
        "architecture": "energy_graph",
        "enabled_roles": set(ROLE_CHOICES),
        "supervisor_max_rounds": 2,
    },
    "loramas_no_gateway": {
        "architecture": "energy_graph",
        "enabled_roles": set(ROLE_CHOICES) - {"gateway_local"},
        "supervisor_max_rounds": 2,
    },
    "loramas_no_temporal": {
        "architecture": "energy_graph",
        "enabled_roles": set(ROLE_CHOICES) - {"temporal_consistency"},
        "supervisor_max_rounds": 2,
    },
    "loramas_no_physical": {
        "architecture": "energy_graph",
        "enabled_roles": set(ROLE_CHOICES) - {"physical_consistency"},
        "supervisor_max_rounds": 2,
    },
}

METHOD_LABELS = {
    "localization_only": "RF/Trilateration",
    "centralized_trust": "Trust Baseline",
    "consistency_graph": "LoRaMAS-Graph",
    "energy_graph": "LoRaMAS-Rules",
    "loramas": "LoRaMAS-Rules",
    "loramas_no_gateway": "LoRaMAS-NoGateway",
    "loramas_no_temporal": "LoRaMAS-NoTemporal",
    "loramas_no_physical": "LoRaMAS-NoPhysical",
}


def paper_method_name(architecture_name: str, role_reasoning: str) -> str:
    if architecture_name in {"energy_graph", "loramas"} and role_reasoning == "llm":
        return "LoRaMAS-LLM"
    if architecture_name in {"energy_graph", "loramas"}:
        return "LoRaMAS-Rules"
    return METHOD_LABELS.get(architecture_name, architecture_name)


def architecture_uses_energy(architecture_name: str) -> bool:
    return str(ARCHITECTURE_PRESETS[architecture_name]["architecture"]) == "energy_graph"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate centralized trust-consensus and LoRaMAS detectors with paper-ready summaries."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off"],
        choices=["off", "explain", "adjudicate"],
        help="LLM modes to evaluate for each architecture.",
    )
    parser.add_argument(
        "--role-reasoning",
        nargs="+",
        default=["rules"],
        choices=["rules", "llm"],
        help="Reasoning backend for specialist roles. `rules` is deterministic; `llm` asks the configured LLM to rewrite each role claim.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=[
            "localization_only",
            "centralized_trust",
            "loramas",
            "loramas_no_temporal",
            "loramas_no_physical",
        ],
        choices=sorted(ARCHITECTURE_PRESETS.keys()),
        help="Architectures and LoRaMAS ablations to compare.",
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON path for the evaluation summary. Defaults to <out-dir>/react_eval_summary.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/paper_eval",
        help="Directory where JSON and CSV summaries will be written.",
    )
    parser.add_argument(
        "--scenario-set",
        default="full",
        choices=["full", "quick", "benchmark"],
        help="Use the legacy full/quick suite or the richer benchmark suite with severity and attribution metadata.",
    )
    parser.add_argument(
        "--use-environment-context",
        action="store_true",
        help="Use cached/fetched OpenStreetMap path context in the heuristic and evaluator.",
    )
    parser.add_argument(
        "--include-gateway-fabrication",
        action="store_true",
        help="Include gateway fabrication scenarios. Leave off to match the paper's current scope.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Log progress every N scenarios within each run.",
    )
    parser.add_argument(
        "--benchmark-split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Filter benchmark scenarios by split when --scenario-set benchmark is used.",
    )
    parser.add_argument(
        "--energy-calibration-limit",
        type=int,
        default=0,
        help="Optional cap on the number of train scenarios used to fit the energy verifier. Use 0 for all.",
    )
    parser.add_argument(
        "--energy-fixed-params-json",
        default=None,
        help="Optional JSON file containing frozen energy_graph tunable params to apply after fitting.",
    )
    return parser


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def scenario_family(item: dict) -> str:
    if item["expected_label"] == "none":
        return item["group"]
    return item["expected_label"]


def benchmark_split_for_name(name: str) -> str:
    bucket = sum(ord(ch) for ch in name) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "val"
    return "test"


def scenario_difficulty(
    *,
    expected_label: str,
    severity: float,
    corrupted_observer_count: int,
    environment_plausibility: str = "neutral",
) -> str:
    if expected_label == "none":
        return "easy" if severity <= 1.5 else "medium"
    score = severity
    if corrupted_observer_count > 1:
        score += 1.0
    if environment_plausibility == "high":
        score -= 0.5
    if score <= 3.0:
        return "hard"
    if score <= 7.0:
        return "medium"
    return "easy"


def build_scenario_record(
    *,
    name: str,
    expected_label: str,
    group: str,
    scenario: AttackScenario,
    severity: float = 0.0,
    severity_bucket: str = "none",
    benchmark_track: str = "legacy",
    corruption_scope: str = "none",
    attribution_target_type: str = "none",
    attribution_target_id: str | None = None,
    corrupted_observer_count: int = 0,
    environment_plausibility: str = "neutral",
    observer_regime: str = "single",
    attack_surface: str = "rf",
) -> dict[str, Any]:
    return {
        "name": name,
        "expected_label": expected_label,
        "group": group,
        "scenario": scenario,
        "severity": float(severity),
        "severity_bucket": severity_bucket,
        "benchmark_track": benchmark_track,
        "corruption_scope": corruption_scope,
        "attribution_target_type": attribution_target_type,
        "attribution_target_id": attribution_target_id,
        "corrupted_observer_count": corrupted_observer_count,
        "environment_plausibility": environment_plausibility,
        "observer_regime": observer_regime,
        "attack_surface": attack_surface,
        "difficulty": scenario_difficulty(
            expected_label=expected_label,
            severity=float(severity),
            corrupted_observer_count=corrupted_observer_count,
            environment_plausibility=environment_plausibility,
        ),
        "split": benchmark_split_for_name(name),
    }


def make_scenarios(
    server: GatewayCoordinatorServer,
    *,
    include_gateway_fabrication: bool,
) -> list[dict]:
    sensors = sorted(server.metadata["SENSORS_LATLON"].keys())
    gateways = sorted(server.metadata["GATEWAYS_LATLON"].keys())
    scenarios: list[dict] = [
        build_scenario_record(
            name="baseline_clean",
            expected_label="none",
            group="clean",
            scenario=AttackScenario(attack_type="none"),
        )
    ]
    for sensor in sensors[:4]:
        scenarios.append(
            build_scenario_record(
                name=f"sensor_foil_{sensor}",
                expected_label="sensor_foil",
                group="attack",
                scenario=AttackScenario(
                    attack_type="sensor_foil",
                    sensor=sensor,
                    rssi_shift_db=-12.0,
                ),
                severity=12.0,
                severity_bucket="strong",
                corruption_scope="sensor",
                attribution_target_type="sensor",
                attribution_target_id=sensor,
            )
        )
    for gateway in gateways:
        scenarios.append(
            build_scenario_record(
                name=f"gateway_bias_{gateway}",
                expected_label="gateway_bias",
                group="attack",
                scenario=AttackScenario(
                    attack_type="gateway_bias",
                    gateway=gateway,
                    rssi_shift_db=-10.0,
                ),
                severity=10.0,
                severity_bucket="strong",
                corruption_scope="gateway",
                attribution_target_type="gateway",
                attribution_target_id=gateway,
                corrupted_observer_count=1,
            )
        )
    for sensor in sensors[4:7]:
        scenarios.append(
            build_scenario_record(
                name=f"random_noise_{sensor}",
                expected_label="random_noise",
                group="attack",
                scenario=AttackScenario(
                    attack_type="random_noise",
                    sensor=sensor,
                    noise_sigma_db=6.0,
                    seed=7,
                ),
                severity=6.0,
                severity_bucket="strong",
                corruption_scope="sensor",
                attribution_target_type="sensor",
                attribution_target_id=sensor,
            )
        )
    for gateway in gateways:
        scenarios.append(
            build_scenario_record(
                name=f"packet_drop_{gateway}",
                expected_label="packet_drop",
                group="attack",
                scenario=AttackScenario(
                    attack_type="packet_drop",
                    gateway=gateway,
                    drop_prob=0.7,
                    seed=7,
                ),
                severity=0.7,
                severity_bucket="strong",
                benchmark_track="legacy",
                corruption_scope="gateway",
                attribution_target_type="gateway",
                attribution_target_id=gateway,
                corrupted_observer_count=1,
            )
        )
    for gateway in gateways:
        scenarios.append(
            build_scenario_record(
                name=f"replay_attack_{gateway}",
                expected_label="replay_attack",
                group="attack",
                scenario=AttackScenario(
                    attack_type="replay_attack",
                    gateway=gateway,
                    replay_fraction=0.10,
                    replay_delay_s=60.0,
                    seed=13,
                ),
                severity=6.0,
                severity_bucket="medium",
                corruption_scope="gateway",
                attribution_target_type="gateway",
                attribution_target_id=gateway,
                corrupted_observer_count=1,
            )
        )
    if include_gateway_fabrication:
        for gateway in gateways:
            scenarios.append(
                build_scenario_record(
                    name=f"gateway_fabrication_{gateway}",
                    expected_label="gateway_fabrication",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="gateway_fabrication",
                        gateway=gateway,
                        fabricate_fraction=0.08,
                        fabricate_shift_db=9.0,
                        seed=19,
                    ),
                    severity=8.0,
                    severity_bucket="strong",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                )
            )
    for gateway in gateways:
        scenarios.append(
            build_scenario_record(
                name=f"counter_corruption_{gateway}",
                expected_label="counter_corruption",
                group="attack",
                scenario=AttackScenario(
                    attack_type="counter_corruption",
                    gateway=gateway,
                    counter_shift=-9,
                    seed=23,
                ),
                severity=9.0,
                severity_bucket="strong",
                corruption_scope="gateway",
                attribution_target_type="gateway",
                attribution_target_id=gateway,
                corrupted_observer_count=1,
            )
        )
    for sensor in sensors[7:10]:
        scenarios.append(
            build_scenario_record(
                name=f"weak_noise_benign_{sensor}",
                expected_label="none",
                group="weak_noise_benign",
                scenario=AttackScenario(
                    attack_type="random_noise",
                    sensor=sensor,
                    noise_sigma_db=1.5,
                    seed=11,
                ),
                severity=1.5,
                severity_bucket="weak",
                corruption_scope="sensor",
                attribution_target_type="none",
                attribution_target_id=None,
                environment_plausibility="high",
            )
        )
    return scenarios


def make_quick_scenarios(
    server: GatewayCoordinatorServer,
    *,
    include_gateway_fabrication: bool,
) -> list[dict]:
    sensors = sorted(server.metadata["SENSORS_LATLON"].keys())
    gateways = sorted(server.metadata["GATEWAYS_LATLON"].keys())
    sensor_a = sensors[0]
    sensor_b = sensors[min(4, len(sensors) - 1)]
    sensor_c = sensors[min(7, len(sensors) - 1)]
    gateway_a = gateways[0]

    scenarios = [
        build_scenario_record(
            name="baseline_clean",
            expected_label="none",
            group="clean",
            scenario=AttackScenario(attack_type="none"),
        ),
        build_scenario_record(
            name=f"sensor_foil_{sensor_a}",
            expected_label="sensor_foil",
            group="attack",
            scenario=AttackScenario(
                attack_type="sensor_foil",
                sensor=sensor_a,
                rssi_shift_db=-12.0,
            ),
            severity=12.0,
            severity_bucket="strong",
            corruption_scope="sensor",
            attribution_target_type="sensor",
            attribution_target_id=sensor_a,
        ),
        build_scenario_record(
            name=f"gateway_bias_{gateway_a}",
            expected_label="gateway_bias",
            group="attack",
            scenario=AttackScenario(
                attack_type="gateway_bias",
                gateway=gateway_a,
                rssi_shift_db=-10.0,
            ),
            severity=10.0,
            severity_bucket="strong",
            corruption_scope="gateway",
            attribution_target_type="gateway",
            attribution_target_id=gateway_a,
            corrupted_observer_count=1,
        ),
        build_scenario_record(
            name=f"random_noise_{sensor_b}",
            expected_label="random_noise",
            group="attack",
            scenario=AttackScenario(
                attack_type="random_noise",
                sensor=sensor_b,
                noise_sigma_db=6.0,
                seed=7,
            ),
            severity=6.0,
            severity_bucket="strong",
            corruption_scope="sensor",
            attribution_target_type="sensor",
            attribution_target_id=sensor_b,
        ),
        build_scenario_record(
            name=f"packet_drop_{gateway_a}",
            expected_label="packet_drop",
            group="attack",
            scenario=AttackScenario(
                attack_type="packet_drop",
                gateway=gateway_a,
                drop_prob=0.7,
                seed=7,
            ),
            severity=0.7,
            severity_bucket="strong",
            corruption_scope="gateway",
            attribution_target_type="gateway",
            attribution_target_id=gateway_a,
            corrupted_observer_count=1,
        ),
        build_scenario_record(
            name=f"replay_attack_{gateway_a}",
            expected_label="replay_attack",
            group="attack",
            scenario=AttackScenario(
                attack_type="replay_attack",
                gateway=gateway_a,
                replay_fraction=0.10,
                replay_delay_s=60.0,
                seed=13,
            ),
            severity=6.0,
            severity_bucket="medium",
            corruption_scope="gateway",
            attribution_target_type="gateway",
            attribution_target_id=gateway_a,
            corrupted_observer_count=1,
        ),
        build_scenario_record(
            name=f"counter_corruption_{gateway_a}",
            expected_label="counter_corruption",
            group="attack",
            scenario=AttackScenario(
                attack_type="counter_corruption",
                gateway=gateway_a,
                counter_shift=-9,
                seed=23,
            ),
            severity=9.0,
            severity_bucket="strong",
            corruption_scope="gateway",
            attribution_target_type="gateway",
            attribution_target_id=gateway_a,
            corrupted_observer_count=1,
        ),
        build_scenario_record(
            name=f"weak_noise_benign_{sensor_c}",
            expected_label="none",
            group="weak_noise_benign",
            scenario=AttackScenario(
                attack_type="random_noise",
                sensor=sensor_c,
                noise_sigma_db=1.5,
                seed=11,
            ),
            severity=1.5,
            severity_bucket="weak",
            corruption_scope="sensor",
            attribution_target_type="none",
            attribution_target_id=None,
            environment_plausibility="high",
        ),
    ]
    if include_gateway_fabrication:
        scenarios.insert(
            6,
            build_scenario_record(
                name=f"gateway_fabrication_{gateway_a}",
                expected_label="gateway_fabrication",
                group="attack",
                scenario=AttackScenario(
                    attack_type="gateway_fabrication",
                    gateway=gateway_a,
                    fabricate_fraction=0.08,
                    fabricate_shift_db=9.0,
                    seed=19,
                ),
                severity=8.0,
                severity_bucket="strong",
                corruption_scope="gateway",
                attribution_target_type="gateway",
                attribution_target_id=gateway_a,
                corrupted_observer_count=1,
            ),
        )
    return scenarios


def make_benchmark_scenarios(
    server: GatewayCoordinatorServer,
    *,
    include_gateway_fabrication: bool,
    benchmark_split: str,
) -> list[dict]:
    sensors = sorted(server.metadata["SENSORS_LATLON"].keys())
    gateways = sorted(server.metadata["GATEWAYS_LATLON"].keys())
    scenarios: list[dict] = []
    gateway_pairs = list(zip(gateways[:-1], gateways[1:]))

    def severity_bucket_for(value: float, *, weak_cutoff: float, medium_cutoff: float) -> str:
        if value <= weak_cutoff:
            return "weak"
        if value <= medium_cutoff:
            return "medium"
        return "strong"

    def add(item: dict[str, Any]) -> None:
        if benchmark_split != "all" and item["split"] != benchmark_split:
            return
        scenarios.append(item)

    add(
        build_scenario_record(
            name="clean_baseline",
            expected_label="none",
            group="clean",
            scenario=AttackScenario(attack_type="none"),
            benchmark_track="benchmark",
            observer_regime="clean",
            attack_surface="clean",
        )
    )

    for sensor in sensors[:4]:
        for shift in (4.0, 8.0, 12.0):
            add(
                build_scenario_record(
                    name=f"benchmark_sensor_foil_{sensor}_{int(shift)}db",
                    expected_label="sensor_foil",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="sensor_foil",
                        sensor=sensor,
                        rssi_shift_db=-shift,
                    ),
                    severity=shift,
                    severity_bucket="weak" if shift <= 4 else "medium" if shift <= 8 else "strong",
                    benchmark_track="benchmark",
                    corruption_scope="sensor",
                    attribution_target_type="sensor",
                    attribution_target_id=sensor,
                    attack_surface="physical",
                )
            )

    for gateway in gateways:
        for shift in (4.0, 8.0, 12.0):
            add(
                build_scenario_record(
                    name=f"benchmark_gateway_bias_{gateway}_{int(shift)}db",
                    expected_label="gateway_bias",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="gateway_bias",
                        gateway=gateway,
                        rssi_shift_db=-shift,
                    ),
                    severity=shift,
                    severity_bucket=severity_bucket_for(shift, weak_cutoff=4.0, medium_cutoff=8.0),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                    attack_surface="physical",
                )
            )

    for gateway_a, gateway_b in gateway_pairs[: max(len(gateway_pairs) - 1, 1)]:
        for shift, seed in ((5.0, 59), (8.0, 61)):
            add(
                build_scenario_record(
                    name=f"benchmark_gateway_bias_pair_{gateway_a}_{gateway_b}_{int(shift)}db_seed{seed}",
                    expected_label="gateway_bias",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="gateway_bias",
                        gateways=[gateway_a, gateway_b],
                        rssi_shift_db=-shift,
                        seed=seed,
                    ),
                    severity=shift + 1.5,
                    severity_bucket=severity_bucket_for(shift + 1.5, weak_cutoff=4.0, medium_cutoff=8.0),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway_pair",
                    attribution_target_id=f"{gateway_a},{gateway_b}",
                    corrupted_observer_count=2,
                    observer_regime="multi",
                    attack_surface="physical",
                )
            )

    for sensor in sensors[4:8]:
        for sigma, seed in ((1.5, 11), (3.0, 7), (6.0, 17), (4.5, 29)):
            expected_label = "none" if sigma <= 1.5 else "random_noise"
            add(
                build_scenario_record(
                    name=f"benchmark_random_noise_{sensor}_{sigma:.1f}_seed{seed}",
                    expected_label=expected_label,
                    group="weak_noise_benign" if expected_label == "none" else "attack",
                    scenario=AttackScenario(
                        attack_type="random_noise",
                        sensor=sensor,
                        noise_sigma_db=sigma,
                        seed=seed,
                    ),
                    severity=sigma,
                    severity_bucket=severity_bucket_for(sigma, weak_cutoff=1.5, medium_cutoff=3.5),
                    benchmark_track="benchmark",
                    corruption_scope="sensor",
                    attribution_target_type="none" if expected_label == "none" else "sensor",
                    attribution_target_id=None if expected_label == "none" else sensor,
                    environment_plausibility="high" if expected_label == "none" else "neutral",
                    attack_surface="rf",
                )
            )

    for gateway in gateways:
        for drop_prob, seed in ((0.3, 7), (0.5, 13), (0.7, 19), (0.4, 53)):
            add(
                build_scenario_record(
                    name=f"benchmark_packet_drop_{gateway}_{drop_prob:.1f}_seed{seed}",
                    expected_label="packet_drop",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="packet_drop",
                        gateway=gateway,
                        drop_prob=drop_prob,
                        seed=seed,
                    ),
                    severity=drop_prob,
                    severity_bucket=severity_bucket_for(drop_prob, weak_cutoff=0.3, medium_cutoff=0.55),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                    attack_surface="availability",
                )
            )

    for gateway_a, gateway_b in gateway_pairs[: max(len(gateway_pairs) - 1, 1)]:
        for drop_prob, seed in ((0.35, 67), (0.55, 71)):
            add(
                build_scenario_record(
                    name=f"benchmark_packet_drop_pair_{gateway_a}_{gateway_b}_{drop_prob:.2f}_seed{seed}",
                    expected_label="packet_drop",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="packet_drop",
                        gateways=[gateway_a, gateway_b],
                        drop_prob=drop_prob,
                        seed=seed,
                    ),
                    severity=drop_prob + 0.8,
                    severity_bucket=severity_bucket_for(drop_prob + 0.8, weak_cutoff=0.9, medium_cutoff=1.2),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway_pair",
                    attribution_target_id=f"{gateway_a},{gateway_b}",
                    corrupted_observer_count=2,
                    observer_regime="multi",
                    attack_surface="availability",
                )
            )

    for gateway in gateways:
        for replay_fraction, replay_delay_s, seed in (
            (0.05, 30.0, 13),
            (0.10, 60.0, 17),
            (0.15, 120.0, 21),
            (0.20, 180.0, 23),
        ):
            severity = replay_fraction * (1.0 + replay_delay_s / 60.0)
            add(
                build_scenario_record(
                    name=(
                        f"benchmark_replay_attack_{gateway}_frac{replay_fraction:.2f}_"
                        f"delay{int(replay_delay_s)}_seed{seed}"
                    ),
                    expected_label="replay_attack",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="replay_attack",
                        gateway=gateway,
                        replay_fraction=replay_fraction,
                        replay_delay_s=replay_delay_s,
                        seed=seed,
                    ),
                    severity=severity,
                    severity_bucket=severity_bucket_for(replay_fraction, weak_cutoff=0.05, medium_cutoff=0.12),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                    attack_surface="temporal",
                )
            )

    for gateway_a, gateway_b in gateway_pairs[: max(len(gateway_pairs) - 1, 1)]:
        for replay_fraction, replay_delay_s, seed in ((0.08, 60.0, 73), (0.14, 150.0, 79)):
            severity = replay_fraction * (1.0 + replay_delay_s / 60.0) + 0.8
            add(
                build_scenario_record(
                    name=(
                        f"benchmark_replay_attack_pair_{gateway_a}_{gateway_b}_frac{replay_fraction:.2f}_"
                        f"delay{int(replay_delay_s)}_seed{seed}"
                    ),
                    expected_label="replay_attack",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="replay_attack",
                        gateways=[gateway_a, gateway_b],
                        replay_fraction=replay_fraction,
                        replay_delay_s=replay_delay_s,
                        seed=seed,
                    ),
                    severity=severity,
                    severity_bucket=severity_bucket_for(severity, weak_cutoff=1.0, medium_cutoff=1.4),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway_pair",
                    attribution_target_id=f"{gateway_a},{gateway_b}",
                    corrupted_observer_count=2,
                    observer_regime="multi",
                    attack_surface="temporal",
                )
            )

    if include_gateway_fabrication:
        for gateway in gateways:
            for fabricate_fraction, fabricate_shift_db, seed in (
                (0.04, 6.0, 19),
                (0.08, 9.0, 23),
                (0.16, 12.0, 29),
            ):
                add(
                    build_scenario_record(
                        name=(
                            f"benchmark_gateway_fabrication_{gateway}_frac{fabricate_fraction:.2f}_"
                            f"shift{int(fabricate_shift_db)}_seed{seed}"
                        ),
                        expected_label="gateway_fabrication",
                        group="attack",
                        scenario=AttackScenario(
                            attack_type="gateway_fabrication",
                            gateway=gateway,
                            fabricate_fraction=fabricate_fraction,
                            fabricate_shift_db=fabricate_shift_db,
                            seed=seed,
                        ),
                        severity=fabricate_fraction * fabricate_shift_db,
                        severity_bucket=severity_bucket_for(fabricate_fraction, weak_cutoff=0.04, medium_cutoff=0.08),
                        benchmark_track="benchmark",
                        corruption_scope="gateway",
                        attribution_target_type="gateway",
                        attribution_target_id=gateway,
                        corrupted_observer_count=1,
                        attack_surface="trust",
                    )
                )

    for gateway in gateways:
        for counter_shift, seed in ((-3, 23), (-6, 31), (-9, 37), (-12, 83)):
            add(
                build_scenario_record(
                    name=f"benchmark_counter_corruption_{gateway}_{abs(counter_shift)}_seed{seed}",
                    expected_label="counter_corruption",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="counter_corruption",
                        gateway=gateway,
                        counter_shift=counter_shift,
                        seed=seed,
                    ),
                    severity=abs(float(counter_shift)),
                    severity_bucket=severity_bucket_for(abs(float(counter_shift)), weak_cutoff=3.0, medium_cutoff=7.0),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                    attack_surface="temporal",
                )
            )

    for gateway_a, gateway_b in gateway_pairs[: max(len(gateway_pairs) - 1, 1)]:
        for counter_shift, seed in ((-4, 89), (-8, 97)):
            add(
                build_scenario_record(
                    name=f"benchmark_counter_corruption_pair_{gateway_a}_{gateway_b}_{abs(counter_shift)}_seed{seed}",
                    expected_label="counter_corruption",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="counter_corruption",
                        gateways=[gateway_a, gateway_b],
                        counter_shift=counter_shift,
                        seed=seed,
                    ),
                    severity=abs(float(counter_shift)) + 0.5,
                    severity_bucket=severity_bucket_for(abs(float(counter_shift)) + 0.5, weak_cutoff=3.0, medium_cutoff=7.0),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway_pair",
                    attribution_target_id=f"{gateway_a},{gateway_b}",
                    corrupted_observer_count=2,
                    observer_regime="multi",
                    attack_surface="temporal",
                )
            )

    for gateway in gateways:
        for drop_prob, seed in ((0.25, 41), (0.45, 43), (0.65, 47), (0.55, 101)):
            add(
                build_scenario_record(
                    name=f"benchmark_selective_suppression_{gateway}_{drop_prob:.2f}_seed{seed}",
                    expected_label="selective_suppression",
                    group="attack",
                    scenario=AttackScenario(
                        attack_type="selective_suppression",
                        gateway=gateway,
                        drop_prob=drop_prob,
                        seed=seed,
                    ),
                    severity=drop_prob,
                    severity_bucket=severity_bucket_for(drop_prob, weak_cutoff=0.25, medium_cutoff=0.5),
                    benchmark_track="benchmark",
                    corruption_scope="gateway",
                    attribution_target_type="gateway",
                    attribution_target_id=gateway,
                    corrupted_observer_count=1,
                    attack_surface="availability",
                )
            )

    return scenarios


def summarize_run(rows: list[dict]) -> dict:
    total = len(rows)
    ambiguous = [row for row in rows if row["supervisor_triggered"]]
    cleanish = [row for row in rows if row["group"] in {"clean", "weak_noise_benign"}]
    case_studies = [
        row
        for row in rows
        if row["supervisor_triggered"] or row["llm_invoked"] or not row["correct"]
    ][:3]
    binary_metrics = summarize_binary_attack_detection(rows)
    attribution_metrics = summarize_attribution(rows)
    uncertainty_metrics = summarize_bootstrap_uncertainty(rows)
    return {
        "overall_accuracy": round(sum(row["correct"] for row in rows) / total, 3),
        "attack_detection_accuracy": round(
            sum(row["attack_detected"] == row["expected_attack_detected"] for row in rows) / total,
            3,
        ),
        "ambiguous_case_accuracy": round(
            sum(row["correct"] for row in ambiguous) / max(len(ambiguous), 1),
            3,
        ),
        "false_positive_rate_clean_and_weak_noise": round(
            sum(row["attack_detected"] for row in cleanish) / max(len(cleanish), 1),
            3,
        ),
        "llm_invocation_rate": round(sum(row["llm_invoked"] for row in rows) / total, 3),
        "faithful_explanations_rate": round(
            sum(row["llm_faithful"] for row in rows if row["llm_invoked"])
            / max(sum(row["llm_invoked"] for row in rows), 1),
            3,
        ),
        "mean_agent_claim_count": round(sum(row["agent_claim_count"] for row in rows) / total, 3),
        "mean_rebuttal_claim_count": round(sum(row["rebuttal_claim_count"] for row in rows) / total, 3),
        "mean_supervisor_rounds": round(sum(row["agent_round_count"] for row in rows) / total, 3),
        **binary_metrics,
        **attribution_metrics,
        **uncertainty_metrics,
        "case_studies": case_studies,
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_binary_attack_detection(rows: list[dict]) -> dict:
    true_positive = sum(
        bool(row["expected_attack_detected"]) and bool(row["attack_detected"])
        for row in rows
    )
    false_positive = sum(
        not bool(row["expected_attack_detected"]) and bool(row["attack_detected"])
        for row in rows
    )
    true_negative = sum(
        not bool(row["expected_attack_detected"]) and not bool(row["attack_detected"])
        for row in rows
    )
    false_negative = sum(
        bool(row["expected_attack_detected"]) and not bool(row["attack_detected"])
        for row in rows
    )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )
    false_negative_rate = (
        false_negative / (false_negative + true_positive)
        if false_negative + true_positive
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "attack_detection_tp": true_positive,
        "attack_detection_fp": false_positive,
        "attack_detection_tn": true_negative,
        "attack_detection_fn": false_negative,
        "attack_detection_precision": round(precision, 3),
        "attack_detection_recall": round(recall, 3),
        "attack_detection_f1": round(f1, 3),
        "attack_detection_fpr": round(false_positive_rate, 3),
        "attack_detection_fnr": round(false_negative_rate, 3),
    }


def summarize_attribution(rows: list[dict]) -> dict:
    attribution_rows = [row for row in rows if row.get("attribution_required")]
    strict_correct = sum(bool(row.get("attribution_target_strict_correct")) for row in attribution_rows)
    partial_correct = sum(bool(row.get("attribution_target_partial_correct")) for row in attribution_rows)
    attempted = sum(bool(row.get("predicted_attribution_target_id")) for row in attribution_rows)
    total = len(attribution_rows)
    return {
        "attribution_required_count": total,
        "attribution_attempt_rate": round(ratio(attempted, total), 3),
        "attribution_strict_accuracy": round(ratio(strict_correct, total), 3),
        "attribution_partial_accuracy": round(ratio(partial_correct, total), 3),
    }


def attribution_summary_for_rows(rows: list[dict]) -> dict[str, float | int]:
    attribution_rows = [row for row in rows if row.get("attribution_required")]
    total = len(attribution_rows)
    return {
        "attribution_required_count": total,
        "attribution_strict_accuracy": round(
            ratio(sum(bool(row.get("attribution_target_strict_correct")) for row in attribution_rows), total),
            3,
        ),
        "attribution_partial_accuracy": round(
            ratio(sum(bool(row.get("attribution_target_partial_correct")) for row in attribution_rows), total),
            3,
        ),
    }


def summarize_bootstrap_uncertainty(
    rows: list[dict],
    *,
    iterations: int = 1000,
    seed: int = 1729,
) -> dict:
    if not rows:
        return {}

    rng = random.Random(seed)
    metric_fns = {
        "overall_accuracy": lambda sample: ratio(sum(bool(row["correct"]) for row in sample), len(sample)),
        "attack_detection_accuracy": lambda sample: ratio(
            sum(bool(row["attack_detected"]) == bool(row["expected_attack_detected"]) for row in sample),
            len(sample),
        ),
        "attack_detection_fpr": lambda sample: _binary_rate(sample, numerator="fp", denominator="negative"),
        "attack_detection_fnr": lambda sample: _binary_rate(sample, numerator="fn", denominator="positive"),
        "attack_detection_f1": lambda sample: _binary_f1(sample),
        "attribution_strict_accuracy": lambda sample: ratio(
            sum(bool(row.get("attribution_target_strict_correct")) for row in sample if row.get("attribution_required")),
            sum(bool(row.get("attribution_required")) for row in sample),
        ),
    }
    distributions: dict[str, list[float]] = {name: [] for name in metric_fns}
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        for name, metric_fn in metric_fns.items():
            distributions[name].append(metric_fn(sample))

    summary: dict[str, float] = {"bootstrap_iterations": iterations}
    for name, values in distributions.items():
        values.sort()
        low = values[int(0.025 * (len(values) - 1))]
        high = values[int(0.975 * (len(values) - 1))]
        summary[f"{name}_ci95_low"] = round(low, 3)
        summary[f"{name}_ci95_high"] = round(high, 3)
    return summary


def _binary_counts(rows: list[dict]) -> dict[str, int]:
    return {
        "tp": sum(bool(row["expected_attack_detected"]) and bool(row["attack_detected"]) for row in rows),
        "fp": sum(not bool(row["expected_attack_detected"]) and bool(row["attack_detected"]) for row in rows),
        "tn": sum(not bool(row["expected_attack_detected"]) and not bool(row["attack_detected"]) for row in rows),
        "fn": sum(bool(row["expected_attack_detected"]) and not bool(row["attack_detected"]) for row in rows),
    }


def _binary_rate(rows: list[dict], *, numerator: str, denominator: str) -> float:
    counts = _binary_counts(rows)
    if denominator == "negative":
        return ratio(counts[numerator], counts["fp"] + counts["tn"])
    if denominator == "positive":
        return ratio(counts[numerator], counts["tp"] + counts["fn"])
    raise ValueError(f"unknown denominator: {denominator}")


def _binary_f1(rows: list[dict]) -> float:
    counts = _binary_counts(rows)
    precision = ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = ratio(counts["tp"], counts["tp"] + counts["fn"])
    return ratio(2.0 * precision * recall, precision + recall)


def summarize_by_family(rows: list[dict]) -> list[dict]:
    family_names = sorted({row["scenario_family"] for row in rows})
    summaries: list[dict] = []
    for family_name in family_names:
        family_rows = [row for row in rows if row["scenario_family"] == family_name]
        total = len(family_rows)
        summaries.append(
            {
                "architecture": family_rows[0]["architecture"],
                "method": family_rows[0].get("method", family_rows[0]["architecture"]),
                "mode": family_rows[0]["mode"],
                "role_reasoning": family_rows[0].get("role_reasoning", "rules"),
                "scenario_family": family_name,
                "scenario_count": total,
                "overall_accuracy": round(sum(row["correct"] for row in family_rows) / total, 3),
                "attack_detection_accuracy": round(
                    sum(row["attack_detected"] == row["expected_attack_detected"] for row in family_rows) / total,
                    3,
                ),
                "false_positive_rate": round(
                    sum(row["attack_detected"] for row in family_rows if not row["expected_attack_detected"])
                    / max(sum(not row["expected_attack_detected"] for row in family_rows), 1),
                    3,
                ),
                "mean_final_confidence": round(
                    sum(row["final_confidence"] for row in family_rows) / total,
                    3,
                ),
                "mean_agent_claim_count": round(
                    sum(row["agent_claim_count"] for row in family_rows) / total,
                    3,
                ),
                "mean_supervisor_rounds": round(
                    sum(row["agent_round_count"] for row in family_rows) / total,
                    3,
                ),
                **attribution_summary_for_rows(family_rows),
            }
        )
    return summaries


def summarize_by_difficulty(rows: list[dict]) -> list[dict]:
    difficulty_names = sorted({row["difficulty"] for row in rows})
    summaries: list[dict] = []
    for difficulty_name in difficulty_names:
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty_name]
        total = len(difficulty_rows)
        summaries.append(
            {
                "architecture": difficulty_rows[0]["architecture"],
                "method": difficulty_rows[0].get("method", difficulty_rows[0]["architecture"]),
                "mode": difficulty_rows[0]["mode"],
                "role_reasoning": difficulty_rows[0].get("role_reasoning", "rules"),
                "difficulty": difficulty_name,
                "scenario_count": total,
                "overall_accuracy": round(sum(row["correct"] for row in difficulty_rows) / total, 3),
                "attack_detection_accuracy": round(
                    sum(row["attack_detected"] == row["expected_attack_detected"] for row in difficulty_rows) / total,
                    3,
                ),
                "attribution_target_accuracy": round(
                    ratio(
                        sum(bool(row.get("attribution_target_strict_correct")) for row in difficulty_rows if row.get("attribution_required")),
                        sum(bool(row.get("attribution_required")) for row in difficulty_rows),
                    ),
                    3,
                ),
                **attribution_summary_for_rows(difficulty_rows),
            }
        )
    return summaries


def summarize_by_split(rows: list[dict]) -> list[dict]:
    split_names = sorted({row["split"] for row in rows})
    summaries: list[dict] = []
    for split_name in split_names:
        split_rows = [row for row in rows if row["split"] == split_name]
        total = len(split_rows)
        summaries.append(
            {
                "architecture": split_rows[0]["architecture"],
                "method": split_rows[0].get("method", split_rows[0]["architecture"]),
                "mode": split_rows[0]["mode"],
                "role_reasoning": split_rows[0].get("role_reasoning", "rules"),
                "split": split_name,
                "scenario_count": total,
                "overall_accuracy": round(sum(row["correct"] for row in split_rows) / total, 3),
                "attack_detection_accuracy": round(
                    sum(row["attack_detected"] == row["expected_attack_detected"] for row in split_rows) / total,
                    3,
                ),
                "attribution_target_accuracy": round(
                    ratio(
                        sum(bool(row.get("attribution_target_strict_correct")) for row in split_rows if row.get("attribution_required")),
                        sum(bool(row.get("attribution_required")) for row in split_rows),
                    ),
                    3,
                ),
                **attribution_summary_for_rows(split_rows),
            }
        )
    return summaries


def build_benchmark_manifest_rows(scenarios: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in scenarios:
        scenario = item["scenario"]
        rows.append(
            {
                "name": item["name"],
                "expected_label": item["expected_label"],
                "group": item["group"],
                "scenario_family": scenario_family(item),
                "severity": item.get("severity", 0.0),
                "severity_bucket": item.get("severity_bucket", "none"),
                "benchmark_track": item.get("benchmark_track", "legacy"),
                "difficulty": item.get("difficulty", "unknown"),
                "split": item.get("split", "train"),
                "corruption_scope": item.get("corruption_scope", "none"),
                "attribution_target_type": item.get("attribution_target_type", "none"),
                "attribution_target_id": item.get("attribution_target_id"),
                "corrupted_observer_count": item.get("corrupted_observer_count", 0),
                "environment_plausibility": item.get("environment_plausibility", "neutral"),
                "observer_regime": item.get("observer_regime", "single"),
                "attack_surface": item.get("attack_surface", "rf"),
                "attack_type": scenario.attack_type,
                "sensor": scenario.sensor,
                "sensors": ",".join(scenario.sensors),
                "gateway": scenario.gateway,
                "gateways": ",".join(scenario.gateways),
                "rssi_shift_db": scenario.rssi_shift_db,
                "noise_sigma_db": scenario.noise_sigma_db,
                "drop_prob": scenario.drop_prob,
                "replay_fraction": scenario.replay_fraction,
                "replay_delay_s": scenario.replay_delay_s,
                "fabricate_fraction": scenario.fabricate_fraction,
                "fabricate_shift_db": scenario.fabricate_shift_db,
                "counter_shift": scenario.counter_shift,
                "seed": scenario.seed,
            }
        )
    return rows


def attribution_fields(item: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    target_type = item.get("attribution_target_type", "none")
    target_id = item.get("attribution_target_id")
    required = target_type not in {None, "none"} and bool(target_id)
    predicted_sensor = report.get("suspicious_sensor")
    predicted_gateway = report.get("suspicious_gateway")

    predicted_type = "none"
    predicted_id = None
    if target_type == "sensor":
        predicted_type = "sensor" if predicted_sensor else "none"
        predicted_id = predicted_sensor
    elif target_type in {"gateway", "gateway_pair"}:
        predicted_type = "gateway" if predicted_gateway else "none"
        predicted_id = predicted_gateway
    elif predicted_gateway:
        predicted_type = "gateway"
        predicted_id = predicted_gateway
    elif predicted_sensor:
        predicted_type = "sensor"
        predicted_id = predicted_sensor

    expected_ids = {
        part.strip()
        for part in str(target_id or "").split(",")
        if part.strip()
    }
    partial_correct = (not required) or (bool(predicted_id) and predicted_id in expected_ids)
    strict_correct = partial_correct
    if target_type == "gateway_pair":
        strict_correct = bool(predicted_id) and str(predicted_id) == str(target_id)

    return {
        "attribution_required": required,
        "predicted_attribution_target_type": predicted_type,
        "predicted_attribution_target_id": predicted_id,
        "attribution_target_partial_correct": partial_correct,
        "attribution_target_strict_correct": strict_correct,
        "attribution_target_correct": strict_correct,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_cache_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return stem or "scenario"


def build_energy_calibration_examples(
    *,
    server: GatewayCoordinatorServer,
    enabled_roles: set[str],
    calibration_scenarios: list[dict],
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_agent = ReActGatewayAgent(
        server,
        llm_client=None,
        llm_mode="off",
        architecture="loramas",
        enabled_roles=set(enabled_roles),
        supervisor_max_rounds=1,
    )
    examples: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    cache_hits = 0
    cache_misses = 0
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for item in calibration_scenarios:
        cache_payload = None
        cache_file = None
        if cache_dir is not None:
            cache_file = cache_dir / f"{_safe_cache_stem(item['name'])}.json"
            if cache_file.exists():
                cache_payload = json.loads(cache_file.read_text())
                if cache_payload.get("name") == item["name"]:
                    cache_hits += 1
        if cache_payload is None:
            report = calibration_agent.investigate(item["scenario"])
            cache_payload = {
                "name": item["name"],
                "label": item["expected_label"],
                "evidence": dict(report.evidence),
            }
            cache_misses += 1
            if cache_file is not None:
                cache_file.write_text(json.dumps(cache_payload, indent=2))
        examples.append(cache_payload)
        label_counts[item["expected_label"]] = label_counts.get(item["expected_label"], 0) + 1
    return examples, {
        "calibration_example_count": len(examples),
        "calibration_label_counts": label_counts,
        "calibration_cache_hits": cache_hits,
        "calibration_cache_misses": cache_misses,
    }


def fit_energy_verifier(
    *,
    agent: ReActGatewayAgent,
    calibration_examples: list[dict[str, Any]],
    calibration_summary: dict[str, Any],
) -> dict[str, Any]:
    agent.energy_consistency_verifier.fit(calibration_examples)
    return dict(calibration_summary)


def tune_energy_verifier(
    *,
    agent: ReActGatewayAgent,
    scenarios: list[dict],
) -> dict[str, Any]:
    original = agent.energy_consistency_verifier.get_tunable_params()
    best_params = dict(original)
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}

    grid = list(
        itertools.product(
            [0.8, 1.0, 1.2],   # claim_scale
            [0.8, 1.0, 1.2],   # template_scale
            [0.10, 0.25, 0.40],  # centralized_scale
            [-0.6, -0.3, 0.0],   # none_label_offset
        )
    )

    for claim_scale, template_scale, centralized_scale, none_label_offset in grid:
        candidate = {
            **original,
            "claim_scale": claim_scale,
            "template_scale": template_scale,
            "centralized_scale": centralized_scale,
            "none_label_offset": none_label_offset,
        }
        agent.energy_consistency_verifier.set_tunable_params(**candidate)

        correct = 0
        detect_correct = 0
        for item in scenarios:
            report = agent.investigate(item["scenario"]).to_dict()
            predicted_label = report["predicted_attack_type"]
            attack_detected = bool(report["attack_detected"])
            expected_attack_detected = item["expected_label"] != "none"
            correct += int(predicted_label == item["expected_label"])
            detect_correct += int(attack_detected == expected_attack_detected)

        overall = correct / max(len(scenarios), 1)
        detect = detect_correct / max(len(scenarios), 1)
        score = (0.65 * detect) + (0.35 * overall)
        if score > best_score:
            best_score = score
            best_params = candidate
            best_metrics = {
                "tuning_objective": round(score, 3),
                "tuning_overall_accuracy": round(overall, 3),
                "tuning_attack_detection_accuracy": round(detect, 3),
            }

    agent.energy_consistency_verifier.set_tunable_params(**best_params)
    return {
        "best_params": best_params,
        **best_metrics,
        "grid_size": len(grid),
    }


def load_or_build_energy_calibration_bundle(
    *,
    cache_path: Path,
    example_cache_dir: Path | None,
    server: GatewayCoordinatorServer,
    enabled_roles: set[str],
    calibration_scenarios: list[dict],
) -> dict[str, Any]:
    expected_names = [item["name"] for item in calibration_scenarios]
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        cached_names = [item.get("name") for item in payload.get("examples", [])]
        if cached_names == expected_names:
            return payload

    examples, summary = build_energy_calibration_examples(
        server=server,
        enabled_roles=enabled_roles,
        calibration_scenarios=calibration_scenarios,
        cache_dir=example_cache_dir,
    )
    payload = {
        "scenario_names": expected_names,
        "examples": examples,
        "summary": summary,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2))
    return payload


def evaluate_run(
    *,
    server: GatewayCoordinatorServer,
    architecture_name: str,
    mode: str,
    role_reasoning: str,
    scenarios: list[dict],
    llm_model: str,
    ollama_endpoint: str,
    run_index: int,
    total_runs: int,
    progress_every: int,
    global_start_s: float,
    calibration_examples: list[dict[str, Any]] | None = None,
    calibration_summary: dict[str, Any] | None = None,
    tune_energy_on_scenarios: bool = False,
    fixed_energy_params: dict[str, float] | None = None,
) -> dict:
    llm_client = None
    if mode != "off" or role_reasoning == "llm":
        llm_client = OllamaAdjudicator(OllamaConfig(model=llm_model, endpoint=ollama_endpoint))
    preset = ARCHITECTURE_PRESETS[architecture_name]
    agent = ReActGatewayAgent(
        server,
        llm_client=llm_client,
        llm_mode=mode,
        architecture=str(preset["architecture"]),
        enabled_roles=set(preset["enabled_roles"]),
        supervisor_max_rounds=int(preset["supervisor_max_rounds"]),
        role_backend=role_reasoning,
    )
    calibration_summary = dict(calibration_summary or {})
    uses_energy_verifier = str(preset["architecture"]) == "energy_graph"
    if uses_energy_verifier and calibration_examples:
        calibration_summary = fit_energy_verifier(
            agent=agent,
            calibration_examples=calibration_examples,
            calibration_summary=calibration_summary or {},
        )
        if fixed_energy_params:
            agent.energy_consistency_verifier.set_tunable_params(**fixed_energy_params)
            calibration_summary = {
                **calibration_summary,
                "best_params": dict(fixed_energy_params),
                "frozen_params": True,
            }
        elif tune_energy_on_scenarios and scenarios:
            calibration_summary = {
                **calibration_summary,
                **tune_energy_verifier(
                    agent=agent,
                    scenarios=scenarios,
                ),
            }

    run_start_s = time.time()
    scenario_count = len(scenarios)
    print(
        f"[start] run {run_index}/{total_runs} architecture={architecture_name} mode={mode} "
        f"scenarios={scenario_count}",
        flush=True,
    )

    rows: list[dict] = []
    for scenario_index, item in enumerate(scenarios, start=1):
        scenario_start_s = time.time()
        scenario = item["scenario"]
        report = agent.investigate(item["scenario"]).to_dict()
        scenario_elapsed_s = time.time() - scenario_start_s
        claims = report.get("agent_claims", [])
        rebuttal_claim_count = sum(int(claim.get("round_index", 1)) > 1 for claim in claims)
        agent_round_count = max([int(claim.get("round_index", 1)) for claim in claims], default=1)
        row = {
            "architecture": architecture_name,
            "method": paper_method_name(architecture_name, role_reasoning),
            "mode": mode,
            "role_reasoning": role_reasoning,
            "name": item["name"],
            "group": item["group"],
            "scenario_family": scenario_family(item),
            "severity": item.get("severity", 0.0),
            "severity_bucket": item.get("severity_bucket", "none"),
            "benchmark_track": item.get("benchmark_track", "legacy"),
            "difficulty": item.get("difficulty", "unknown"),
            "split": item.get("split", "train"),
            "corruption_scope": item.get("corruption_scope", "none"),
            "attribution_target_type": item.get("attribution_target_type", "none"),
            "attribution_target_id": item.get("attribution_target_id"),
            "corrupted_observer_count": item.get("corrupted_observer_count", 0),
            "environment_plausibility": item.get("environment_plausibility", "neutral"),
            "observer_regime": item.get("observer_regime", "single"),
            "attack_surface": item.get("attack_surface", "rf"),
            "scenario_attack_type": scenario.attack_type,
            "scenario_sensor": scenario.sensor,
            "scenario_sensors": ",".join(scenario.sensors),
            "scenario_gateway": scenario.gateway,
            "scenario_gateways": ",".join(scenario.gateways),
            "scenario_rssi_shift_db": scenario.rssi_shift_db,
            "scenario_noise_sigma_db": scenario.noise_sigma_db,
            "scenario_drop_prob": scenario.drop_prob,
            "scenario_replay_fraction": scenario.replay_fraction,
            "scenario_replay_delay_s": scenario.replay_delay_s,
            "scenario_fabricate_fraction": scenario.fabricate_fraction,
            "scenario_fabricate_shift_db": scenario.fabricate_shift_db,
            "scenario_counter_shift": scenario.counter_shift,
            "scenario_seed": scenario.seed,
            "expected_label": item["expected_label"],
            "predicted_label": report["predicted_attack_type"],
            "correct": report["predicted_attack_type"] == item["expected_label"],
            "attack_detected": report["attack_detected"],
            "expected_attack_detected": item["expected_label"] != "none",
            "llm_invoked": report["llm"]["invoked"],
            "llm_available": report["llm"]["available"],
            "llm_faithful": report["llm"]["faithful"],
            "request_more_evidence": report["llm"]["request_more_evidence"],
            "supervisor_triggered": report["heuristic"]["should_invoke_llm"],
            "trigger_reasons": "|".join(report["heuristic"]["trigger_reasons"]),
            "supervisor_prediction": report["heuristic"]["predicted_attack_type"],
            "supervisor_confidence": report["heuristic"]["confidence"],
            "heuristic_top_factors": "|".join(report["heuristic"]["evidence"].get("energy_verifier_top_factors", [])),
            "heuristic_feature_snapshot": json.dumps(
                report["heuristic"]["evidence"].get("energy_verifier_feature_snapshot", {}),
                sort_keys=True,
            ),
            "final_confidence": report["confidence"],
            "agent_claim_count": len(claims),
            "rebuttal_claim_count": rebuttal_claim_count,
            "agent_round_count": agent_round_count,
            "roles_present": "|".join(sorted({claim["role"] for claim in claims})),
            "predicted_suspicious_sensor": report["suspicious_sensor"],
            "predicted_suspicious_gateway": report["suspicious_gateway"],
            "scenario_elapsed_s": round(scenario_elapsed_s, 3),
        }
        row.update(attribution_fields(item, report))
        rows.append(row)

        if scenario_index % max(progress_every, 1) == 0 or scenario_index == scenario_count:
            run_elapsed_s = time.time() - run_start_s
            run_eta_s = (run_elapsed_s / scenario_index) * (scenario_count - scenario_index)
            overall_elapsed_s = time.time() - global_start_s
            approx_total_scenarios_done = (run_index - 1) * scenario_count + scenario_index
            approx_total_scenarios = total_runs * scenario_count
            overall_eta_s = (overall_elapsed_s / approx_total_scenarios_done) * (
                approx_total_scenarios - approx_total_scenarios_done
            )
            print(
                f"[progress] run {run_index}/{total_runs} scenario {scenario_index}/{scenario_count} "
                f"name={item['name']} predicted={row['predicted_label']} correct={row['correct']} "
                f"scenario_time={format_duration(scenario_elapsed_s)} "
                f"run_elapsed={format_duration(run_elapsed_s)} run_eta={format_duration(run_eta_s)} "
                f"overall_eta={format_duration(overall_eta_s)}",
                flush=True,
            )

    run_summary = summarize_run(rows)
    run_elapsed_s = time.time() - run_start_s
    print(
        f"[done] run {run_index}/{total_runs} architecture={architecture_name} mode={mode} "
        f"overall_accuracy={run_summary['overall_accuracy']:.3f} "
        f"attack_detection_accuracy={run_summary['attack_detection_accuracy']:.3f} "
        f"elapsed={format_duration(run_elapsed_s)}",
        flush=True,
    )
    return {
        "architecture": architecture_name,
        "method": paper_method_name(architecture_name, role_reasoning),
        "mode": mode,
        "role_reasoning": role_reasoning,
        "elapsed_s": round(run_elapsed_s, 3),
        "calibration": calibration_summary,
        **run_summary,
        "family_summaries": summarize_by_family(rows),
        "difficulty_summaries": summarize_by_difficulty(rows),
        "split_summaries": summarize_by_split(rows),
        "rows": rows,
    }


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json_out) if args.json_out else out_dir / "react_eval_summary.json"

    server = GatewayCoordinatorServer(ServerConfig(use_environment_context=args.use_environment_context))
    server.ensure_metadata()
    scenarios = (
        make_quick_scenarios(server, include_gateway_fabrication=args.include_gateway_fabrication)
        if args.scenario_set == "quick"
        else make_benchmark_scenarios(
            server,
            include_gateway_fabrication=args.include_gateway_fabrication,
            benchmark_split=args.benchmark_split,
        )
        if args.scenario_set == "benchmark"
        else make_scenarios(server, include_gateway_fabrication=args.include_gateway_fabrication)
    )
    calibration_scenarios = (
        make_benchmark_scenarios(
            server,
            include_gateway_fabrication=args.include_gateway_fabrication,
            benchmark_split="train",
        )
        if args.scenario_set == "benchmark"
        else []
    )
    if args.energy_calibration_limit and args.energy_calibration_limit > 0:
        calibration_scenarios = calibration_scenarios[: int(args.energy_calibration_limit)]
    energy_calibration_bundle = None
    fixed_energy_params = None
    if (
        any(architecture_uses_energy(name) for name in args.architectures)
        and calibration_scenarios
        and not args.energy_fixed_params_json
    ):
        energy_calibration_bundle = load_or_build_energy_calibration_bundle(
            cache_path=out_dir / "energy_calibration_examples.json",
            example_cache_dir=out_dir / "energy_calibration_reports",
            server=server,
            enabled_roles=set(ARCHITECTURE_PRESETS["energy_graph"]["enabled_roles"]),
            calibration_scenarios=calibration_scenarios,
        )
    if args.energy_fixed_params_json:
        fixed_energy_params = json.loads(Path(args.energy_fixed_params_json).read_text())

    run_plan = [
        (architecture_name, mode, role_reasoning)
        for architecture_name in args.architectures
        for mode in args.modes
        for role_reasoning in args.role_reasoning
    ]
    total_runs = len(run_plan)
    global_start_s = time.time()

    runs = [
        evaluate_run(
            server=server,
            architecture_name=architecture_name,
            mode=mode,
            role_reasoning=role_reasoning,
            scenarios=scenarios,
            llm_model=args.llm_model,
            ollama_endpoint=args.ollama_endpoint,
            run_index=run_index,
            total_runs=total_runs,
            progress_every=args.progress_every,
            global_start_s=global_start_s,
            calibration_examples=None if energy_calibration_bundle is None else energy_calibration_bundle["examples"],
            calibration_summary=None if energy_calibration_bundle is None else energy_calibration_bundle["summary"],
            tune_energy_on_scenarios=(
                architecture_uses_energy(architecture_name)
                and args.scenario_set == "benchmark"
                and args.benchmark_split == "val"
                and fixed_energy_params is None
            ),
            fixed_energy_params=(
                fixed_energy_params
                if architecture_uses_energy(architecture_name)
                else None
            ),
        )
        for run_index, (architecture_name, mode, role_reasoning) in enumerate(run_plan, start=1)
    ]

    rows = [row for run in runs for row in run["rows"]]
    family_summaries = [row for run in runs for row in run["family_summaries"]]
    difficulty_summaries = [row for run in runs for row in run["difficulty_summaries"]]
    split_summaries = [row for run in runs for row in run["split_summaries"]]
    high_level_rows = []
    for run in runs:
        high_level_rows.append(
            {
                "architecture": run["architecture"],
                "method": run["method"],
                "mode": run["mode"],
                "role_reasoning": run["role_reasoning"],
                "elapsed_s": run["elapsed_s"],
                "overall_accuracy": run["overall_accuracy"],
                "attack_detection_accuracy": run["attack_detection_accuracy"],
                "ambiguous_case_accuracy": run["ambiguous_case_accuracy"],
                "false_positive_rate_clean_and_weak_noise": run["false_positive_rate_clean_and_weak_noise"],
                "attack_detection_tp": run["attack_detection_tp"],
                "attack_detection_fp": run["attack_detection_fp"],
                "attack_detection_tn": run["attack_detection_tn"],
                "attack_detection_fn": run["attack_detection_fn"],
                "attack_detection_fpr": run["attack_detection_fpr"],
                "attack_detection_fnr": run["attack_detection_fnr"],
                "attack_detection_precision": run["attack_detection_precision"],
                "attack_detection_recall": run["attack_detection_recall"],
                "attack_detection_f1": run["attack_detection_f1"],
                "attribution_required_count": run["attribution_required_count"],
                "attribution_attempt_rate": run["attribution_attempt_rate"],
                "attribution_strict_accuracy": run["attribution_strict_accuracy"],
                "attribution_partial_accuracy": run["attribution_partial_accuracy"],
                "overall_accuracy_ci95_low": run["overall_accuracy_ci95_low"],
                "overall_accuracy_ci95_high": run["overall_accuracy_ci95_high"],
                "attack_detection_accuracy_ci95_low": run["attack_detection_accuracy_ci95_low"],
                "attack_detection_accuracy_ci95_high": run["attack_detection_accuracy_ci95_high"],
                "attack_detection_fpr_ci95_low": run["attack_detection_fpr_ci95_low"],
                "attack_detection_fpr_ci95_high": run["attack_detection_fpr_ci95_high"],
                "attack_detection_fnr_ci95_low": run["attack_detection_fnr_ci95_low"],
                "attack_detection_fnr_ci95_high": run["attack_detection_fnr_ci95_high"],
                "attack_detection_f1_ci95_low": run["attack_detection_f1_ci95_low"],
                "attack_detection_f1_ci95_high": run["attack_detection_f1_ci95_high"],
                "attribution_strict_accuracy_ci95_low": run["attribution_strict_accuracy_ci95_low"],
                "attribution_strict_accuracy_ci95_high": run["attribution_strict_accuracy_ci95_high"],
                "bootstrap_iterations": run["bootstrap_iterations"],
                "llm_invocation_rate": run["llm_invocation_rate"],
                "faithful_explanations_rate": run["faithful_explanations_rate"],
                "mean_agent_claim_count": run["mean_agent_claim_count"],
                "mean_rebuttal_claim_count": run["mean_rebuttal_claim_count"],
                "mean_supervisor_rounds": run["mean_supervisor_rounds"],
                "calibration_example_count": run.get("calibration", {}).get("calibration_example_count", 0),
                "tuning_objective": run.get("calibration", {}).get("tuning_objective"),
            }
        )

    total_elapsed_s = time.time() - global_start_s
    summary = {
        "scenario_set": args.scenario_set,
        "scenario_count": len(scenarios),
        "benchmark_split": args.benchmark_split,
        "include_gateway_fabrication": args.include_gateway_fabrication,
        "use_environment_context": args.use_environment_context,
        "architectures": args.architectures,
        "modes": args.modes,
        "role_reasoning": args.role_reasoning,
        "elapsed_s": round(total_elapsed_s, 3),
        "benchmark_manifest": build_benchmark_manifest_rows(scenarios),
        "energy_calibration": None if energy_calibration_bundle is None else energy_calibration_bundle["summary"],
        "runs": runs,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))
    write_csv(out_dir / "react_eval_run_summary.csv", high_level_rows)
    write_csv(out_dir / "react_eval_family_summary.csv", family_summaries)
    write_csv(out_dir / "react_eval_difficulty_summary.csv", difficulty_summaries)
    write_csv(out_dir / "react_eval_split_summary.csv", split_summaries)
    write_csv(out_dir / "react_eval_rows.csv", rows)
    write_csv(out_dir / "benchmark_manifest.csv", build_benchmark_manifest_rows(scenarios))

    print(
        f"[done] all runs completed in {format_duration(total_elapsed_s)}. "
        f"json={json_path} csv_dir={out_dir}",
        flush=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
