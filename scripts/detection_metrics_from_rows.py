from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build binary attack-detection metrics from react_eval_rows.csv files."
    )
    parser.add_argument(
        "rows_csv",
        nargs="+",
        help="One or more react_eval_rows.csv files to combine.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "csv"],
        default="markdown",
        help="Output table format.",
    )
    return parser


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def method_label(architecture: str) -> str:
    labels = {
        "localization_only": "RF/Trilateration",
        "energy_graph": "LoRaMAS-Rules",
        "consistency_graph": "LoRaMAS-Graph",
        "centralized_trust": "Trust Baseline",
        "loramas": "LoRaMAS-Rules",
    }
    return labels.get(architecture, architecture)


def method_order(architecture: str) -> int:
    order = {
        "localization_only": 0,
        "centralized_trust": 1,
        "consistency_graph": 2,
        "energy_graph": 3,
        "loramas": 4,
    }
    return order.get(architecture, 99)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    keys = sorted(
        {
            (
                row["architecture"],
                row["mode"],
                row.get("role_reasoning", "rules"),
                row.get("method") or method_label(row["architecture"]),
            )
            for row in rows
        },
        key=lambda key: (method_order(key[0]), key[0], key[1], key[2]),
    )
    for architecture, mode, role_reasoning, method in keys:
        subset = [
            row
            for row in rows
            if row["architecture"] == architecture
            and row["mode"] == mode
            and row.get("role_reasoning", "rules") == role_reasoning
        ]
        tp = sum(
            truthy(row["expected_attack_detected"]) and truthy(row["attack_detected"])
            for row in subset
        )
        fp = sum(
            not truthy(row["expected_attack_detected"]) and truthy(row["attack_detected"])
            for row in subset
        )
        tn = sum(
            not truthy(row["expected_attack_detected"]) and not truthy(row["attack_detected"])
            for row in subset
        )
        fn = sum(
            truthy(row["expected_attack_detected"]) and not truthy(row["attack_detected"])
            for row in subset
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        fnr = fn / (fn + tp) if fn + tp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        attribution_subset = [
            row
            for row in subset
            if truthy(row.get("attribution_required", "false"))
        ]
        attribution_strict_accuracy = ratio(
            sum(truthy(row.get("attribution_target_strict_correct", "false")) for row in attribution_subset),
            len(attribution_subset),
        )
        overall_accuracy = (
            sum(truthy(row["correct"]) for row in subset) / len(subset)
            if subset
            else 0.0
        )
        attack_detection_accuracy = (
            sum(
                truthy(row["expected_attack_detected"]) == truthy(row["attack_detected"])
                for row in subset
            )
            / len(subset)
            if subset
            else 0.0
        )
        summaries.append(
            {
                "method": method,
                "architecture": architecture,
                "mode": mode,
                "role_reasoning": role_reasoning,
                "n": str(len(subset)),
                "tp": str(tp),
                "fp": str(fp),
                "tn": str(tn),
                "fn": str(fn),
                "overall_accuracy": f"{overall_accuracy:.3f}",
                "attack_detection_accuracy": f"{attack_detection_accuracy:.3f}",
                "attribution_required": str(len(attribution_subset)),
                "attribution_strict_accuracy": (
                    f"{attribution_strict_accuracy:.3f}" if attribution_subset else ""
                ),
                "fpr": f"{fpr:.3f}",
                "fnr": f"{fnr:.3f}",
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "f1": f"{f1:.3f}",
                **bootstrap_summary(subset),
            }
        )
    return summaries


def bootstrap_summary(
    rows: list[dict[str, str]],
    *,
    iterations: int = 1000,
    seed: int = 1729,
) -> dict[str, str]:
    if not rows:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "overall_accuracy": [],
        "attack_detection_accuracy": [],
        "f1": [],
    }
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        tp = sum(truthy(row["expected_attack_detected"]) and truthy(row["attack_detected"]) for row in sample)
        fp = sum(not truthy(row["expected_attack_detected"]) and truthy(row["attack_detected"]) for row in sample)
        fn = sum(truthy(row["expected_attack_detected"]) and not truthy(row["attack_detected"]) for row in sample)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        values["overall_accuracy"].append(ratio(sum(truthy(row["correct"]) for row in sample), len(sample)))
        values["attack_detection_accuracy"].append(
            ratio(
                sum(truthy(row["expected_attack_detected"]) == truthy(row["attack_detected"]) for row in sample),
                len(sample),
            )
        )
        values["f1"].append(ratio(2.0 * precision * recall, precision + recall))

    summary: dict[str, str] = {}
    for name, metric_values in values.items():
        metric_values.sort()
        low = metric_values[int(0.025 * (len(metric_values) - 1))]
        high = metric_values[int(0.975 * (len(metric_values) - 1))]
        summary[f"{name}_ci95"] = f"[{low:.3f}, {high:.3f}]"
    return summary


def print_markdown(rows: list[dict[str, str]]) -> None:
    headers = [
        "Method",
        "N",
        "Overall Acc.",
        "Attack Detect.",
        "Attr. Strict",
        "TP",
        "FP",
        "TN",
        "FN",
        "FPR",
        "FNR",
        "F1",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["n"],
                    row["overall_accuracy"],
                    row["attack_detection_accuracy"],
                    row["attribution_strict_accuracy"],
                    row["tp"],
                    row["fp"],
                    row["tn"],
                    row["fn"],
                    row["fpr"],
                    row["fnr"],
                    row["f1"],
                ]
            )
            + " |"
        )


def print_latex(rows: list[dict[str, str]]) -> None:
    print("\\begin{tabular}{l c c c c c}")
    print("\\toprule")
    print("Method & Overall Acc. & Attack Detect. & FPR & FNR & F1 \\\\")
    print("\\midrule")
    for row in rows:
        print(
            " & ".join(
                [
                    row["method"].replace("_", "\\_"),
                    row["overall_accuracy"],
                    row["attack_detection_accuracy"],
                    row["fpr"],
                    row["fnr"],
                    row["f1"],
                ]
            )
            + " \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")


def print_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "method",
        "architecture",
        "mode",
        "role_reasoning",
        "n",
        "overall_accuracy",
        "attack_detection_accuracy",
        "attribution_required",
        "attribution_strict_accuracy",
        "tp",
        "fp",
        "tn",
        "fn",
        "fpr",
        "fnr",
        "precision",
        "recall",
        "f1",
        "overall_accuracy_ci95",
        "attack_detection_accuracy_ci95",
        "f1_ci95",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    rows = summarize(read_rows(args.rows_csv))
    if args.format == "latex":
        print_latex(rows)
    elif args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
