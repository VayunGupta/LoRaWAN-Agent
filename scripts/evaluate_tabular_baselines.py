from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train dependency-light tabular and anomaly baselines from react_eval_rows.csv "
            "and export summary plus robustness curves."
        )
    )
    parser.add_argument("rows_csv", help="Input react_eval_rows.csv file.")
    parser.add_argument(
        "--architecture",
        default="auto",
        help="Architecture rows to use as the feature source. Use auto to prefer LoRaMAS rows.",
    )
    parser.add_argument("--mode", default="off", help="LLM mode rows to use.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--out-dir",
        default="outputs/tabular_baselines",
        help="Directory for baseline CSV outputs.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also include non-label scenario metadata features such as severity and corrupted observer count.",
    )
    return parser


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_rows(path: Path, *, architecture: str, mode: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(path.open()))
    if architecture == "auto":
        available = {row.get("architecture") for row in rows}
        if "loramas" in available:
            architecture = "loramas"
        elif "energy_graph" in available:
            architecture = "energy_graph"
    filtered = [
        row
        for row in rows
        if row.get("architecture") == architecture and row.get("mode", "off") == mode
    ]
    if filtered:
        return filtered
    return rows


def parse_snapshot(row: dict[str, Any]) -> dict[str, float]:
    try:
        payload = json.loads(row.get("heuristic_feature_snapshot") or "{}")
    except json.JSONDecodeError:
        return {}
    parsed: dict[str, float] = {}
    for key, value in payload.items():
        try:
            parsed[f"evidence.{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def metadata_features(row: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for key in ["severity", "corrupted_observer_count"]:
        try:
            features[f"meta.{key}"] = float(row.get(key) or 0.0)
        except ValueError:
            features[f"meta.{key}"] = 0.0
    for key in ["attack_surface", "corruption_scope", "observer_regime", "severity_bucket"]:
        value = str(row.get(key) or "unknown")
        features[f"meta.{key}={value}"] = 1.0
    return features


def vectorize(
    rows: list[dict[str, Any]],
    *,
    include_metadata: bool,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    feature_dicts: list[dict[str, float]] = []
    for row in rows:
        features = parse_snapshot(row)
        if include_metadata:
            features.update(metadata_features(row))
        feature_dicts.append(features)

    if feature_names is None:
        feature_names = sorted({key for features in feature_dicts for key in features})
    if not feature_names:
        feature_names = ["bias"]
        feature_dicts = [{"bias": 1.0} for _ in rows]

    matrix = np.zeros((len(rows), len(feature_names)), dtype=float)
    for row_index, features in enumerate(feature_dicts):
        for col_index, name in enumerate(feature_names):
            matrix[row_index, col_index] = float(features.get(name, 0.0))
    return matrix, feature_names


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def majority_label(train_y: list[str], test_count: int) -> list[str]:
    label = Counter(train_y).most_common(1)[0][0] if train_y else "none"
    return [label] * test_count


def nearest_centroid(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    labels = sorted(set(train_y))
    centroids = {
        label: train_x[[idx for idx, value in enumerate(train_y) if value == label]].mean(axis=0)
        for label in labels
    }
    predictions: list[str] = []
    for vector in test_x:
        predictions.append(
            min(labels, key=lambda label: float(np.linalg.norm(vector - centroids[label])))
        )
    return predictions


def gaussian_centroid(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    labels = sorted(set(train_y))
    params: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for label in labels:
        subset = train_x[[idx for idx, value in enumerate(train_y) if value == label]]
        mean = subset.mean(axis=0)
        var = subset.var(axis=0) + 1e-3
        prior = max(len(subset) / max(len(train_y), 1), 1e-6)
        params[label] = (mean, var, math.log(prior))

    predictions: list[str] = []
    for vector in test_x:
        scores = {}
        for label, (mean, var, log_prior) in params.items():
            log_likelihood = -0.5 * np.sum(np.log(var) + ((vector - mean) ** 2) / var)
            scores[label] = float(log_prior + log_likelihood)
        predictions.append(max(scores, key=scores.get))
    return predictions


def anomaly_zscore(train_x: np.ndarray, train_y: list[str], test_x: np.ndarray) -> list[str]:
    clean_indices = [idx for idx, label in enumerate(train_y) if label == "none"]
    reference = train_x[clean_indices] if clean_indices else train_x
    center = reference.mean(axis=0)
    scores_train = np.linalg.norm(train_x - center, axis=1)
    scores_test = np.linalg.norm(test_x - center, axis=1)
    thresholds = sorted(set(float(value) for value in scores_train))
    if not thresholds:
        thresholds = [0.0]

    best_threshold = thresholds[0]
    best_f1 = -1.0
    for threshold in thresholds:
        predicted = ["attack" if score > threshold else "none" for score in scores_train]
        f1 = binary_f1(train_y, predicted)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return ["attack" if score > best_threshold else "none" for score in scores_test]


def binary_f1(expected_labels: list[str], predicted_labels: list[str]) -> float:
    tp = sum(expected != "none" and predicted != "none" for expected, predicted in zip(expected_labels, predicted_labels))
    fp = sum(expected == "none" and predicted != "none" for expected, predicted in zip(expected_labels, predicted_labels))
    fn = sum(expected != "none" and predicted == "none" for expected, predicted in zip(expected_labels, predicted_labels))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return ratio(2.0 * precision * recall, precision + recall)


def summarize_predictions(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    expected = [row["expected_label"] for row in rows]
    tp = sum(exp != "none" and pred != "none" for exp, pred in zip(expected, predictions))
    fp = sum(exp == "none" and pred != "none" for exp, pred in zip(expected, predictions))
    tn = sum(exp == "none" and pred == "none" for exp, pred in zip(expected, predictions))
    fn = sum(exp != "none" and pred == "none" for exp, pred in zip(expected, predictions))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "scenario_count": len(rows),
        "overall_accuracy": round(ratio(sum(exp == pred for exp, pred in zip(expected, predictions)), len(rows)), 3),
        "attack_detection_accuracy": round(ratio(tp + tn, len(rows)), 3),
        "attack_detection_tp": tp,
        "attack_detection_fp": fp,
        "attack_detection_tn": tn,
        "attack_detection_fn": fn,
        "attack_detection_fpr": round(ratio(fp, fp + tn), 3),
        "attack_detection_fnr": round(ratio(fn, fn + tp), 3),
        "attack_detection_precision": round(precision, 3),
        "attack_detection_recall": round(recall, 3),
        "attack_detection_f1": round(ratio(2.0 * precision * recall, precision + recall), 3),
    }


def choose_train_test(
    rows: list[dict[str, Any]],
    *,
    train_split: str,
    test_split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    train_rows = [row for row in rows if row.get("split") == train_split]
    test_rows = [row for row in rows if row.get("split") == test_split]
    if train_rows and test_rows:
        return train_rows, test_rows, False
    return rows, rows, True


def seed_from_row(row: dict[str, Any]) -> str:
    if row.get("scenario_seed") not in {None, ""}:
        return str(row.get("scenario_seed"))
    match = re.search(r"_seed(\d+)", str(row.get("name") or ""))
    return match.group(1) if match else str(row.get("seed") or "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def robustness_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("severity_bucket", "severity_bucket"),
        ("attack_surface", "attack_surface"),
        ("scenario_family", "scenario_family"),
        ("seed", "seed"),
    ]
    output: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in prediction_rows}):
        method_rows = [row for row in prediction_rows if row["method"] == method]
        for group_name, key in groups:
            for value in sorted({str(row.get(key) or "") for row in method_rows}):
                if not value:
                    continue
                subset = [row for row in method_rows if str(row.get(key) or "") == value]
                summary = summarize_predictions(subset, [row["predicted_label"] for row in subset])
                output.append(
                    {
                        "method": method,
                        "group": group_name,
                        "value": value,
                        **summary,
                    }
                )
    return output


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    rows = load_rows(Path(args.rows_csv), architecture=args.architecture, mode=args.mode)
    train_rows, test_rows, leave_one_out = choose_train_test(
        rows,
        train_split=args.train_split,
        test_split=args.test_split,
    )

    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    if leave_one_out:
        methods = ["majority", "nearest_centroid", "gaussian_centroid", "zscore_anomaly"]
        predictions_by_method = {method: [] for method in methods}
        for heldout_index, test_row in enumerate(test_rows):
            fold_train = [row for idx, row in enumerate(train_rows) if idx != heldout_index]
            fold_test = [test_row]
            train_x, feature_names = vectorize(fold_train, include_metadata=args.include_metadata)
            test_x, _ = vectorize(fold_test, include_metadata=args.include_metadata, feature_names=feature_names)
            train_x, test_x = standardize(train_x, test_x)
            train_y = [row["expected_label"] for row in fold_train]
            predictions_by_method["majority"].extend(majority_label(train_y, 1))
            predictions_by_method["nearest_centroid"].extend(nearest_centroid(train_x, train_y, test_x))
            predictions_by_method["gaussian_centroid"].extend(gaussian_centroid(train_x, train_y, test_x))
            predictions_by_method["zscore_anomaly"].extend(anomaly_zscore(train_x, train_y, test_x))
    else:
        train_x, feature_names = vectorize(train_rows, include_metadata=args.include_metadata)
        test_x, _ = vectorize(test_rows, include_metadata=args.include_metadata, feature_names=feature_names)
        train_x, test_x = standardize(train_x, test_x)
        train_y = [row["expected_label"] for row in train_rows]
        predictions_by_method = {
            "majority": majority_label(train_y, len(test_rows)),
            "nearest_centroid": nearest_centroid(train_x, train_y, test_x),
            "gaussian_centroid": gaussian_centroid(train_x, train_y, test_x),
            "zscore_anomaly": anomaly_zscore(train_x, train_y, test_x),
        }

    for method, predictions in predictions_by_method.items():
        summary_rows.append(
            {
                "method": method,
                "architecture_feature_source": args.architecture,
                "mode": args.mode,
                "feature_count": len(vectorize(train_rows, include_metadata=args.include_metadata)[1]),
                "train_count": len(train_rows),
                "test_count": len(test_rows),
                "evaluation_protocol": "leave_one_out" if leave_one_out else f"{args.train_split}_to_{args.test_split}",
                **summarize_predictions(test_rows, predictions),
            }
        )
        for row, prediction in zip(test_rows, predictions):
            prediction_rows.append(
                {
                    "method": method,
                    "name": row.get("name"),
                    "split": row.get("split"),
                    "expected_label": row.get("expected_label"),
                    "predicted_label": prediction,
                    "attack_detected": prediction != "none",
                    "expected_attack_detected": row.get("expected_label") != "none",
                    "correct": row.get("expected_label") == prediction,
                    "scenario_family": row.get("scenario_family"),
                    "attack_surface": row.get("attack_surface"),
                    "severity": row.get("severity"),
                    "severity_bucket": row.get("severity_bucket"),
                    "seed": seed_from_row(row),
                }
            )

    write_csv(out_dir / "tabular_baseline_summary.csv", summary_rows)
    write_csv(out_dir / "tabular_baseline_rows.csv", prediction_rows)
    write_csv(out_dir / "tabular_baseline_robustness.csv", robustness_rows(prediction_rows))
    print(json.dumps({"summary": summary_rows, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
