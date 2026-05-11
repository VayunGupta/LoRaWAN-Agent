from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        description="Compare baseline vs OpenStreetMap-augmented heuristic and LLM adjudication."
    )
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument(
        "--ollama-endpoint",
        default="http://127.0.0.1:11434/api/generate",
    )
    parser.add_argument("--out-dir", default="outputs/environment_context")
    return parser


def evaluate_condition(
    *,
    name: str,
    llm_mode: str,
    use_environment_context: bool,
    llm_model: str,
    ollama_endpoint: str,
    scenarios: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    server = GatewayCoordinatorServer(ServerConfig(use_environment_context=use_environment_context))
    server.ensure_metadata()
    scenarios = scenarios or build_scenarios(server)
    llm_client = None
    if llm_mode != "off":
        llm_client = OllamaAdjudicator(
            OllamaConfig(model=llm_model, endpoint=ollama_endpoint, timeout_s=45.0)
        )
    agent = ReActGatewayAgent(server, llm_client=llm_client, llm_mode=llm_mode)

    rows: list[dict] = []
    for item in scenarios:
        report = agent.investigate(item["scenario"]).to_dict()
        rows.append(
            {
                "condition": name,
                "llm_mode": llm_mode,
                "environment_context": use_environment_context,
                "scenario_name": item["scenario_name"],
                "attack_family": item["attack_family"],
                "severity": item["severity"],
                "expected_label": item["expected_label"],
                "predicted_label": report["predicted_attack_type"],
                "correct": report["predicted_attack_type"] == item["expected_label"],
                "attack_detected": report["attack_detected"],
                "expected_attack_detected": item["expected_label"] != "none",
                "heuristic_triggered": report["heuristic"]["should_invoke_llm"],
                "heuristic_confidence": report["heuristic"]["confidence"],
                "llm_invoked": report["llm"]["invoked"],
                "llm_faithful": report["llm"]["faithful"],
                "sensor_context_fragility": report["evidence"].get("sensor_context_fragility", 0.0),
                "gateway_context_fragility": report["evidence"].get("gateway_context_fragility", 0.0),
                "sensor_blocked_link_fraction": report["evidence"].get("sensor_blocked_link_fraction", 0.0),
                "gateway_blocked_link_fraction": report["evidence"].get("gateway_blocked_link_fraction", 0.0),
                "top_sensor_score": report["evidence"]["top_sensor_score"],
                "top_gateway_score": report["evidence"]["top_gateway_score"],
                "top_sensor_score_raw": report["evidence"].get("top_sensor_score_raw", report["evidence"]["top_sensor_score"]),
                "top_gateway_score_raw": report["evidence"].get("top_gateway_score_raw", report["evidence"]["top_gateway_score"]),
            }
        )

    df = pd.DataFrame(rows)
    ambiguous = df[df["heuristic_triggered"]]
    cleanish = df[df["expected_label"] == "none"]
    summary = {
        "condition": name,
        "llm_mode": llm_mode,
        "environment_context": use_environment_context,
        "scenario_count": int(df.shape[0]),
        "overall_accuracy": float(df["correct"].mean()),
        "attack_detection_accuracy": float(
            (df["attack_detected"] == df["expected_attack_detected"]).mean()
        ),
        "ambiguous_case_accuracy": float(ambiguous["correct"].mean()) if not ambiguous.empty else 0.0,
        "false_positive_rate_on_none": float(cleanish["attack_detected"].mean()) if not cleanish.empty else 0.0,
        "heuristic_trigger_rate": float(df["heuristic_triggered"].mean()),
        "llm_invocation_rate": float(df["llm_invoked"].mean()),
        "llm_faithfulness_rate": float(df.loc[df["llm_invoked"], "llm_faithful"].mean()) if df["llm_invoked"].any() else 0.0,
        "mean_sensor_fragility": float(df["sensor_context_fragility"].mean()),
        "mean_gateway_fragility": float(df["gateway_context_fragility"].mean()),
        "mean_sensor_blocked_fraction": float(df["sensor_blocked_link_fraction"].mean()),
        "mean_gateway_blocked_fraction": float(df["gateway_blocked_link_fraction"].mean()),
    }
    return df, summary


def plot_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "overall_accuracy",
        "ambiguous_case_accuracy",
        "false_positive_rate_on_none",
        "llm_invocation_rate",
    ]
    labels = ["Overall Acc.", "Ambiguous Acc.", "False Pos.", "LLM Invoke"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "heuristic_base": "#355070",
        "heuristic_osm": "#6d597a",
        "llm_base_ambiguous": "#b56576",
        "llm_osm_ambiguous": "#e56b6f",
    }
    for idx, row in enumerate(summary_df.to_dict(orient="records")):
        values = [float(row[metric]) for metric in metrics]
        ax.bar(
            x + (idx - (len(summary_df) - 1) / 2) * width,
            values,
            width=width,
            label=row["condition"],
            color=colors.get(row["condition"], "#457b9d"),
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Effect of OpenStreetMap Context on Heuristic and LLM Modes")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "environment_mode_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fragility(summary_rows: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(summary_rows.shape[0])
    ax.bar(x - 0.17, summary_rows["mean_sensor_fragility"], width=0.34, label="Sensor fragility", color="#52796f")
    ax.bar(x + 0.17, summary_rows["mean_gateway_fragility"], width=0.34, label="Gateway fragility", color="#84a98c")
    ax.set_xticks(x, summary_rows["condition"], rotation=15)
    ax.set_ylabel("Mean context score")
    ax.set_title("Average OSM-Derived Path Fragility Exposed to Each Mode")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "environment_fragility_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_server = GatewayCoordinatorServer(ServerConfig(use_environment_context=False))
    base_server.ensure_metadata()
    all_scenarios = build_scenarios(base_server)

    frames = []
    summaries = []

    heuristic_conditions = [
        ("heuristic_base", "off", False, all_scenarios),
        ("heuristic_osm", "off", True, all_scenarios),
    ]
    heuristic_rows: dict[str, pd.DataFrame] = {}
    for name, llm_mode, use_environment_context, scenarios in heuristic_conditions:
        df, summary = evaluate_condition(
            name=name,
            llm_mode=llm_mode,
            use_environment_context=use_environment_context,
            llm_model=args.llm_model,
            ollama_endpoint=args.ollama_endpoint,
            scenarios=scenarios,
        )
        heuristic_rows[name] = df
        frames.append(df)
        summaries.append(summary)

    ambiguous_names = set(
        heuristic_rows["heuristic_base"]
        .loc[heuristic_rows["heuristic_base"]["heuristic_triggered"], "scenario_name"]
        .tolist()
    )
    ambiguous_scenarios = [item for item in all_scenarios if item["scenario_name"] in ambiguous_names]

    llm_conditions = [
        ("llm_base_ambiguous", "adjudicate", False, ambiguous_scenarios),
        ("llm_osm_ambiguous", "adjudicate", True, ambiguous_scenarios),
    ]
    for name, llm_mode, use_environment_context, scenarios in llm_conditions:
        df, summary = evaluate_condition(
            name=name,
            llm_mode=llm_mode,
            use_environment_context=use_environment_context,
            llm_model=args.llm_model,
            ollama_endpoint=args.ollama_endpoint,
            scenarios=scenarios,
        )
        frames.append(df)
        summaries.append(summary)

    all_rows = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    all_rows.to_csv(out_dir / "environment_rows.csv", index=False)
    summary_df.to_csv(out_dir / "environment_summary.csv", index=False)
    plot_summary(summary_df, out_dir)
    plot_fragility(summary_df, out_dir)

    payload = {
        "conditions": summaries,
        "output_dir": str(out_dir),
    }
    (out_dir / "environment_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
