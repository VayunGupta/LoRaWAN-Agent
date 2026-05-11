from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.helpers import haversine_m, load_and_prepare_packets
from src.traditional.pathloss import TraditionalParams, estimate_pair_distances_traditional
from src.traditional.trilateration import trilaterate_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and summarize the result set needed by paper/main.tex."
    )
    parser.add_argument("--out-dir", default="outputs/main_tex_results")
    parser.add_argument("--data-dir", default="dataset/lorawan_metadata")
    parser.add_argument("--metadata", default="models/metadata.json")
    parser.add_argument("--traditional-params", default="models/traditional_params.json")
    parser.add_argument("--sam-max-side", type=int, default=768)
    parser.add_argument("--quick", action="store_true", help="Use reduced scenario sets for a fast smoke run.")
    parser.add_argument("--skip-satellite", action="store_true")
    parser.add_argument("--skip-agent-eval", action="store_true")
    parser.add_argument("--skip-localization", action="store_true")
    parser.add_argument("--max-packets-per-pair", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def run_command(args: list[str], *, log_path: Path | None = None) -> None:
    print(f"\n[run] {' '.join(args)}", flush=True)
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    code = process.wait()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("".join(lines))
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}: {' '.join(args)}")


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def true_pair_distances(metadata: dict[str, Any]) -> dict[tuple[str, str], float]:
    sensors = metadata["SENSORS_LATLON"]
    gateways = metadata["GATEWAYS_LATLON"]
    distances: dict[tuple[str, str], float] = {}
    for sensor, sinfo in sensors.items():
        for gateway, ginfo in gateways.items():
            distances[(sensor, gateway)] = haversine_m(
                float(sinfo["lat"]),
                float(sinfo["lon"]),
                float(ginfo["lat"]),
                float(ginfo["lon"]),
            )
    return distances


def sample_packets_for_regression(
    packets: pd.DataFrame,
    *,
    max_per_pair: int,
    seed: int,
) -> pd.DataFrame:
    chunks = []
    rng = np.random.default_rng(seed)
    for _, group in packets.groupby(["sensor", "gateway"], sort=False):
        if group.shape[0] > max_per_pair:
            take = rng.choice(group.index.to_numpy(), size=max_per_pair, replace=False)
            chunks.append(group.loc[take])
        else:
            chunks.append(group)
    return pd.concat(chunks, ignore_index=True)


def add_distance_labels(packets: pd.DataFrame, distances: dict[tuple[str, str], float]) -> pd.DataFrame:
    packets = packets.copy()
    packets["true_distance_m"] = [
        distances[(str(sensor), str(gateway))]
        for sensor, gateway in zip(packets["sensor"], packets["gateway"])
    ]
    return packets


def regression_features(packets: pd.DataFrame, *, gateway_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    frame = packets[["rssi_dbm", "snr_db", "gateway"]].copy()
    encoded = pd.get_dummies(frame, columns=["gateway"], prefix="gateway")
    if gateway_columns is None:
        gateway_columns = [col for col in encoded.columns if col.startswith("gateway_")]
    for col in gateway_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    columns = ["rssi_dbm", "snr_db", *gateway_columns]
    return encoded[columns].astype(float), gateway_columns


def fit_regression_model(
    packets: pd.DataFrame,
    *,
    distances: dict[tuple[str, str], float],
    max_per_pair: int,
    seed: int,
) -> tuple[GradientBoostingRegressor, list[str]]:
    train = sample_packets_for_regression(
        add_distance_labels(packets, distances),
        max_per_pair=max_per_pair,
        seed=seed,
    )
    features, gateway_columns = regression_features(train)
    model = GradientBoostingRegressor(random_state=seed)
    model.fit(features, train["true_distance_m"].to_numpy())
    return model, gateway_columns


def regression_pair_distances(
    packets: pd.DataFrame,
    *,
    model: GradientBoostingRegressor,
    gateway_columns: list[str],
) -> pd.DataFrame:
    if packets.empty:
        return pd.DataFrame(columns=["sensor", "gateway", "d_est_m"])
    features, _ = regression_features(packets, gateway_columns=gateway_columns)
    frame = packets[["sensor", "gateway"]].copy()
    frame["d_pred_m"] = model.predict(features)
    return (
        frame.groupby(["sensor", "gateway"], as_index=False)["d_pred_m"]
        .median()
        .rename(columns={"d_pred_m": "d_est_m"})
    )


def summarize_localization(estimates: pd.DataFrame, method: str) -> dict[str, Any]:
    errors = estimates["error_m"].dropna().astype(float)
    return {
        "method": method,
        "sensor_count": int(errors.shape[0]),
        "median_error_m": round(float(errors.median()), 3),
        "p90_error_m": round(float(errors.quantile(0.90)), 3),
        "mean_error_m": round(float(errors.mean()), 3),
        "max_error_m": round(float(errors.max()), 3),
    }


def binary_metrics(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    tp = sum(row["expected_attack"] and row["predicted_attack"] for row in rows)
    fp = sum((not row["expected_attack"]) and row["predicted_attack"] for row in rows)
    tn = sum((not row["expected_attack"]) and (not row["predicted_attack"]) for row in rows)
    fn = sum(row["expected_attack"] and (not row["predicted_attack"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "method": method,
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / max(len(rows), 1), 3),
        "fpr": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "fnr": round(fn / (fn + tp), 3) if fn + tp else 0.0,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_Skipped._"
    columns = sorted({key for row in rows for key in row})
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def evaluate_clean_localization(
    packets: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    traditional_params_path: Path,
    distances: dict[tuple[str, str], float],
    max_per_pair: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = TraditionalParams.load(str(traditional_params_path))
    gateway_xy = {key: tuple(value) for key, value in metadata["GW_XY"].items()}
    sensor_xy = {key: tuple(value) for key, value in metadata["S_XY_TRUE"].items()}

    trad_pair = estimate_pair_distances_traditional(packets, params)
    trad_est = trilaterate_all(trad_pair, gateway_xy, sensor_xy)

    model, gateway_columns = fit_regression_model(
        packets,
        distances=distances,
        max_per_pair=max_per_pair,
        seed=seed,
    )
    reg_pair = regression_pair_distances(packets, model=model, gateway_columns=gateway_columns)
    reg_est = trilaterate_all(reg_pair, gateway_xy, sensor_xy)

    rows = [
        *trad_est.assign(method="Trilateration").to_dict(orient="records"),
        *reg_est.assign(method="GradientBoostingRegression").to_dict(orient="records"),
    ]
    bundle = {
        "traditional_params": params,
        "regression_model": model,
        "regression_gateway_columns": gateway_columns,
        "gateway_xy": gateway_xy,
        "sensor_xy": sensor_xy,
        "clean_trilateration": trad_est,
        "clean_regression": reg_est,
    }
    return rows, bundle


def localization_detection_scenarios(*, quick: bool) -> list[dict[str, Any]]:
    sensors = ["sensor05", "sensor08"] if quick else [f"sensor{i:02d}" for i in range(1, 11)]
    scenarios: list[dict[str, Any]] = []
    for sensor in sensors:
        scenarios.append({"name": f"clean_{sensor}", "attack_type": "none", "sensor": sensor, "expected_attack": False})
    weak_sigmas = [1.0] if quick else [0.5, 1.0, 1.5]
    for sigma in weak_sigmas:
        for sensor in sensors:
            scenarios.append(
                {
                    "name": f"benign_noise_{sensor}_{sigma}",
                    "attack_type": "random_noise",
                    "sensor": sensor,
                    "noise_sigma_db": sigma,
                    "seed": 7,
                    "expected_attack": False,
                }
            )
    shifts = [-8.0] if quick else [-6.0, -8.0, -10.0, -12.0]
    for shift in shifts:
        for sensor in sensors:
            scenarios.append(
                {
                    "name": f"sensor_foil_{sensor}_{abs(shift):.0f}db",
                    "attack_type": "sensor_foil",
                    "sensor": sensor,
                    "rssi_shift_db": shift,
                    "expected_attack": True,
                }
            )
    strong_sigmas = [4.5] if quick else [3.0, 4.5, 6.0]
    for sigma in strong_sigmas:
        for sensor in sensors:
            scenarios.append(
                {
                    "name": f"attack_noise_{sensor}_{sigma}",
                    "attack_type": "random_noise",
                    "sensor": sensor,
                    "noise_sigma_db": sigma,
                    "seed": 29,
                    "expected_attack": True,
                }
            )
    return scenarios


def estimate_for_scenario(
    scenario: dict[str, Any],
    *,
    data_dir: str,
    bundle: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kwargs = {
        "data_dir": data_dir,
        "attack_type": scenario["attack_type"],
        "attack_scope": "sensor" if scenario["attack_type"] != "none" else "global",
        "attack_sensor": scenario.get("sensor"),
        "rssi_shift_db": float(scenario.get("rssi_shift_db", 0.0)),
        "rssi_noise_sigma_db": float(scenario.get("noise_sigma_db", 0.0)),
        "seed": int(scenario.get("seed", 0)),
    }
    packets = load_and_prepare_packets(**kwargs)
    trad_pair = estimate_pair_distances_traditional(packets, bundle["traditional_params"])
    trad_est = trilaterate_all(trad_pair, bundle["gateway_xy"], bundle["sensor_xy"])
    reg_pair = regression_pair_distances(
        packets,
        model=bundle["regression_model"],
        gateway_columns=bundle["regression_gateway_columns"],
    )
    reg_est = trilaterate_all(reg_pair, bundle["gateway_xy"], bundle["sensor_xy"])
    return trad_est, reg_est


def evaluate_localization_detection(
    *,
    data_dir: str,
    clean_metrics: list[dict[str, Any]],
    bundle: dict[str, Any],
    quick: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = {row["method"]: float(row["median_error_m"]) for row in clean_metrics}
    rows: list[dict[str, Any]] = []
    for scenario in localization_detection_scenarios(quick=quick):
        trad_est, reg_est = estimate_for_scenario(scenario, data_dir=data_dir, bundle=bundle)
        target = scenario["sensor"]
        for method, estimates in [
            ("Trilateration", trad_est),
            ("GradientBoostingRegression", reg_est),
        ]:
            match = estimates.loc[estimates["sensor"] == target]
            if match.empty:
                continue
            error_m = float(match.iloc[0]["error_m"])
            rows.append(
                {
                    "method": method,
                    "scenario": scenario["name"],
                    "sensor": target,
                    "expected_attack": bool(scenario["expected_attack"]),
                    "error_m": round(error_m, 3),
                    "threshold_m": round(thresholds[method], 3),
                    "predicted_attack": bool(error_m > thresholds[method]),
                }
            )
    metric_rows = [
        binary_metrics([row for row in rows if row["method"] == method], method)
        for method in ["Trilateration", "GradientBoostingRegression"]
    ]
    return rows, metric_rows


def compute_link_stats(metadata: dict[str, Any], satellite_csv: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    pairs = [
        ("sensor05", "gatewayA"),
        ("sensor05", "gatewayB"),
        ("sensor04", "gatewayA"),
        ("sensor10", "gatewayB"),
        ("sensor03", "gatewayB"),
        ("sensor06", "gatewayB"),
    ]
    sat = pd.read_csv(satellite_csv) if satellite_csv and satellite_csv.exists() else pd.DataFrame()
    for sensor, gateway in pairs:
        path = ROOT / "dataset" / "lorawan_metadata" / f"{sensor}_{gateway}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        row = {
            "sensor": sensor,
            "gateway": gateway,
            "packet_count": int(frame.shape[0]),
            "mean_rssi_dbm": round(float(frame["RSSI (dBm)"].mean()), 3),
            "mean_snr_db": round(float(frame["SNR (dB)"].mean()), 3),
        }
        if not sat.empty:
            match = sat.loc[(sat["sensor"] == sensor) & (sat["gateway"] == gateway)]
            if not match.empty:
                for key in [
                    "satellite_los_blocked",
                    "sam_building_fraction",
                    "sam_vegetation_fraction",
                    "satellite_expected_extra_attenuation_db",
                    "satellite_context_score",
                ]:
                    row[key] = float(match.iloc[0][key])
        rows.append(row)
    return rows


def summarize_agent_tables(agent_out_dir: Path) -> tuple[list[dict[str, Any]], str]:
    run_summary = agent_out_dir / "react_eval_run_summary.csv"
    if not run_summary.exists():
        return [], ""
    summary = pd.read_csv(run_summary)
    keep = [
        "method",
        "architecture",
        "scenario_count",
        "overall_accuracy",
        "attack_detection_accuracy",
        "attack_detection_fpr",
        "attack_detection_fnr",
        "attack_detection_f1",
        "mean_supervisor_rounds",
    ]
    rows = summary[[col for col in keep if col in summary.columns]].to_dict(orient="records")
    return rows, markdown_table(rows)


def write_markdown_report(
    *,
    path: Path,
    localization_metrics: list[dict[str, Any]],
    localization_detection: list[dict[str, Any]],
    link_stats: list[dict[str, Any]],
    agent_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    lines = [
        "# Main.tex Result Checklist",
        "",
        "Generated by `scripts/run_main_tex_results.py`.",
        "",
        "## Baseline Localization Error",
        "",
        markdown_table(localization_metrics),
        "",
        "Use `median_error_m` and `p90_error_m` for the missing Table 1 values.",
        "",
        "## Localization-Threshold Spoof Detection",
        "",
        markdown_table(localization_detection),
        "",
        "This is the Ragini-requested detector that flags a claim when localization error exceeds the clean median error threshold.",
        "",
        "## Link-Level Environment Examples",
        "",
        markdown_table(link_stats) if link_stats else "_No link stats written._",
        "",
        "Use these values in the qualitative environmental-feature paragraph and case study.",
        "",
        "## Agentic Verification Summary",
        "",
        markdown_table(agent_rows) if agent_rows else "_Skipped or not yet run._",
        "",
        "## Files",
        "",
        f"- Results directory: `{out_dir}`",
        "- `localization_error_summary.csv`",
        "- `localization_detection_summary.csv`",
        "- `localization_detection_rows.csv`",
        "- `link_environment_stats.csv`",
        "- `agent_eval/react_eval_run_summary.csv`",
        "- `agent_eval/react_eval_rows.csv`",
        "- `satellite/satellite_context_by_pair.csv`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"

    satellite_dir = out_dir / "satellite"
    if not args.skip_satellite:
        checkpoint = ROOT / "models/sam/sam_vit_b_01ec64.pth"
        if not checkpoint.exists():
            raise SystemExit(
                "Missing SAM checkpoint. Run:\n"
                "mkdir -p models/sam\n"
                "curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "
                "-o models/sam/sam_vit_b_01ec64.pth"
            )
        run_command(
            [
                sys.executable,
                "scripts/build_satellite_context.py",
                "--out-dir",
                str(satellite_dir.relative_to(ROOT)),
                "--segmentation-mode",
                "sam",
                "--sam-max-side",
                str(args.sam_max_side),
            ],
            log_path=logs_dir / "satellite_context.log",
        )

    metadata = load_metadata(ROOT / args.metadata)
    distances = true_pair_distances(metadata)
    localization_metrics: list[dict[str, Any]] = []
    localization_detection: list[dict[str, Any]] = []
    link_stats = compute_link_stats(
        metadata,
        satellite_dir / "satellite_context_by_pair.csv"
        if (satellite_dir / "satellite_context_by_pair.csv").exists()
        else ROOT / "outputs/satellite_context/satellite_context_by_pair.csv",
    )
    write_csv(out_dir / "link_environment_stats.csv", link_stats)

    if not args.skip_localization:
        packets = load_and_prepare_packets(args.data_dir)
        clean_rows, bundle = evaluate_clean_localization(
            packets,
            metadata=metadata,
            traditional_params_path=ROOT / args.traditional_params,
            distances=distances,
            max_per_pair=args.max_packets_per_pair,
            seed=args.seed,
        )
        write_csv(out_dir / "localization_error_rows.csv", clean_rows)
        localization_metrics = [
            summarize_localization(
                pd.DataFrame([row for row in clean_rows if row["method"] == method]),
                method,
            )
            for method in ["Trilateration", "GradientBoostingRegression"]
        ]
        write_csv(out_dir / "localization_error_summary.csv", localization_metrics)
        detection_rows, localization_detection = evaluate_localization_detection(
            data_dir=args.data_dir,
            clean_metrics=localization_metrics,
            bundle=bundle,
            quick=args.quick,
        )
        write_csv(out_dir / "localization_detection_rows.csv", detection_rows)
        write_csv(out_dir / "localization_detection_summary.csv", localization_detection)

    agent_rows: list[dict[str, Any]] = []
    agent_dir = out_dir / "agent_eval"
    if not args.skip_agent_eval:
        if not (agent_dir / "react_eval_run_summary.csv").exists():
            eval_args = [
                sys.executable,
                "-m",
                "src.evaluate_react_agent",
                "--scenario-set",
                "quick" if args.quick else "benchmark",
                "--benchmark-split",
                "all",
                "--architectures",
                "localization_only",
                "centralized_trust",
                "loramas",
                "loramas_no_temporal",
                "loramas_no_physical",
                "--role-reasoning",
                "rules",
                "--modes",
                "off",
                "--use-environment-context",
                "--out-dir",
                str(agent_dir.relative_to(ROOT)),
            ]
            run_command(eval_args, log_path=logs_dir / "agent_eval.log")
        else:
            print(f"[reuse] Found existing agent results in {agent_dir}")
        agent_rows, _ = summarize_agent_tables(agent_dir)
        write_csv(out_dir / "agent_detection_summary.csv", agent_rows)

    write_markdown_report(
        path=out_dir / "RESULTS_FOR_MAIN_TEX.md",
        localization_metrics=localization_metrics,
        localization_detection=localization_detection,
        link_stats=link_stats,
        agent_rows=agent_rows,
        out_dir=out_dir,
    )
    print(f"\n[done] Main paper result bundle written to {out_dir}")
    print(f"[done] Start with {out_dir / 'RESULTS_FOR_MAIN_TEX.md'}")


if __name__ == "__main__":
    main()
