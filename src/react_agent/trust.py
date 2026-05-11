from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import deque

import numpy as np
import pandas as pd


@dataclass
class TrustConfig:
    timestamp_quantile: float = 0.99
    timestamp_floor_s: float = 1.5
    witness_cluster_window_s: float = 5.0
    replay_window_s: float = 900.0
    frequency_tolerance_hz: float = 200_000.0
    rssi_z_threshold: float = 3.5
    snr_z_threshold: float = 3.5
    replay_gap_threshold: float = 3.0


class WitnessTrustAnalyzer:
    """
    Build packet-witness consistency features from multi-gateway LoRaWAN logs.

    The baseline is calibrated from clean traffic and then reused to score
    scenario snapshots for replay-like, fabrication-like, and suppression-like
    inconsistencies.
    """

    def __init__(self, baseline_packets: pd.DataFrame, config: TrustConfig | None = None):
        self.config = config or TrustConfig()
        self.baseline_packets = self._prepare_packets(baseline_packets)
        self.reference = self._build_reference(self.baseline_packets)

    @staticmethod
    def _prepare_packets(packets: pd.DataFrame) -> pd.DataFrame:
        if packets is None or packets.empty:
            return pd.DataFrame()
        frame = packets.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if "counter" in frame.columns:
            frame["counter"] = pd.to_numeric(frame["counter"], errors="coerce")
        for column in ["rssi_dbm", "snr_db", "freq_mhz", "n_rx_gw"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=[c for c in ["sensor", "gateway", "timestamp", "counter"] if c in frame.columns]).copy()

    def _build_reference(self, packets: pd.DataFrame) -> dict[str, Any]:
        if packets.empty:
            return {
                "time_tolerance_s": self.config.timestamp_floor_s,
                "sensor_multiplicity": pd.DataFrame(columns=["sensor", "mean_gateways", "std_gateways"]),
                "link_baseline": pd.DataFrame(columns=["sensor", "gateway"]),
                "counter_step": pd.DataFrame(columns=["sensor", "median_step"]),
            }

        witness = self.build_witness_sets(packets)
        skew = float(witness["time_span_s"].quantile(self.config.timestamp_quantile)) if not witness.empty else 0.0
        time_tolerance_s = max(self.config.timestamp_floor_s, skew + 0.25)

        sensor_multiplicity = (
            witness.groupby("sensor", as_index=False)
            .agg(
                mean_gateways=("gateway_count", "mean"),
                std_gateways=("gateway_count", "std"),
                median_claimed_receivers=("claimed_receivers_median", "median"),
            )
            .fillna({"std_gateways": 0.0, "median_claimed_receivers": 1.0})
        )

        link_baseline = (
            packets.groupby(["sensor", "gateway"], as_index=False)
            .agg(
                rssi_median=("rssi_dbm", "median"),
                rssi_std=("rssi_dbm", "std"),
                snr_median=("snr_db", "median"),
                snr_std=("snr_db", "std"),
            )
            .fillna({"rssi_std": 1.0, "snr_std": 1.0, "snr_median": 0.0})
        )
        link_baseline["rssi_std"] = link_baseline["rssi_std"].clip(lower=1.0)
        link_baseline["snr_std"] = link_baseline["snr_std"].clip(lower=1.0)

        counter_rows: list[dict[str, Any]] = []
        witness_events = witness.sort_values("event_time")
        for sensor, frame in witness_events.groupby("sensor"):
            counter_series = frame["counter"].dropna()
            if counter_series.shape[0] < 2:
                median_step = 1.0
            else:
                steps = counter_series.diff().dropna()
                steps = steps[steps > 0]
                median_step = float(steps.median()) if not steps.empty else 1.0
            counter_rows.append({"sensor": sensor, "median_step": max(median_step, 1.0)})

        return {
            "time_tolerance_s": time_tolerance_s,
            "sensor_multiplicity": sensor_multiplicity,
            "link_baseline": link_baseline,
            "counter_step": pd.DataFrame(counter_rows),
        }

    def _attach_witness_keys(self, packets: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_packets(packets)
        if frame.empty:
            return frame
        frame = frame.sort_values(["sensor", "counter", "timestamp", "gateway"]).reset_index(drop=True)
        time_delta = (
            frame.groupby(["sensor", "counter"])["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
        )
        new_cluster = (
            (frame["sensor"] != frame["sensor"].shift(1))
            | (frame["counter"] != frame["counter"].shift(1))
            | (time_delta > self.config.witness_cluster_window_s)
        )
        frame["witness_id"] = new_cluster.cumsum()
        return frame

    def build_witness_sets(self, packets: pd.DataFrame) -> pd.DataFrame:
        frame = self._attach_witness_keys(packets)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "witness_id",
                    "sensor",
                    "counter",
                    "gateway_count",
                    "time_span_s",
                    "freq_span_hz",
                    "claimed_receivers_median",
                    "rssi_std_across_gateways",
                    "snr_std_across_gateways",
                    "event_time",
                ]
            )

        witness = (
            frame.groupby(["witness_id", "sensor", "counter"], as_index=False)
            .agg(
                event_time=("timestamp", "min"),
                gateway_count=("gateway", "nunique"),
                time_min=("timestamp", "min"),
                time_max=("timestamp", "max"),
                freq_min=("freq_mhz", "min"),
                freq_max=("freq_mhz", "max"),
                claimed_receivers_median=("n_rx_gw", "median"),
                rssi_std_across_gateways=("rssi_dbm", "std"),
                snr_std_across_gateways=("snr_db", "std"),
            )
            .fillna(
                {
                    "claimed_receivers_median": 1.0,
                    "rssi_std_across_gateways": 0.0,
                    "snr_std_across_gateways": 0.0,
                }
            )
        )
        witness["time_span_s"] = (
            witness["time_max"] - witness["time_min"]
        ).dt.total_seconds().fillna(0.0)
        witness["freq_span_hz"] = (witness["freq_max"] - witness["freq_min"]) * 1_000_000.0
        return witness.drop(columns=["time_min", "time_max", "freq_min", "freq_max"])

    def analyze(self, packets: pd.DataFrame) -> dict[str, Any]:
        frame = self._attach_witness_keys(packets)
        if frame.empty:
            return {
                "time_tolerance_s": self.reference["time_tolerance_s"],
                "suspicious_gateway": None,
                "suspicious_sensor": None,
                "max_replay_gap": 0.0,
                "replay_event_count": 0,
                "counter_regression_count": 0,
                "delayed_duplicate_count": 0,
                "max_duplicate_gap_s": 0.0,
                "high_skew_witness_count": 0,
                "fabricated_witness_count": 0,
                "multiplicity_anomaly_count": 0,
                "gateway_trust_scores": {},
                "sensor_replay_scores": {},
                "event_inconsistency_score": 0.0,
            }

        witness = self.build_witness_sets(frame)
        frame = frame.merge(
            self.reference["link_baseline"],
            on=["sensor", "gateway"],
            how="left",
        )
        frame["rssi_z"] = (
            (frame["rssi_dbm"] - frame["rssi_median"]).abs() / frame["rssi_std"].fillna(1.0).clip(lower=1.0)
        ).fillna(0.0)
        frame["snr_z"] = (
            (frame["snr_db"] - frame["snr_median"]).abs() / frame["snr_std"].fillna(1.0).clip(lower=1.0)
        ).fillna(0.0)
        frame["physically_plausible"] = (
            (frame["rssi_z"] <= self.config.rssi_z_threshold)
            & (frame["snr_z"] <= self.config.snr_z_threshold)
        )

        witness = witness.merge(self.reference["sensor_multiplicity"], on="sensor", how="left")
        witness["mean_gateways"] = witness["mean_gateways"].fillna(witness["gateway_count"].median() if not witness.empty else 1.0)
        witness["std_gateways"] = witness["std_gateways"].fillna(0.0)
        witness["multiplicity_z"] = (
            (witness["gateway_count"] - witness["mean_gateways"])
            / (witness["std_gateways"] + 0.5)
        )
        witness["multiplicity_anomaly"] = witness["multiplicity_z"].abs() >= 1.5
        witness["skew_anomaly"] = witness["time_span_s"] > self.reference["time_tolerance_s"]
        witness["fabrication_anomaly"] = witness["freq_span_hz"] > self.config.frequency_tolerance_hz

        counter_profile = self.reference["counter_step"]
        sensor_replay_scores: dict[str, float] = {}
        sensor_regression_counts: dict[str, int] = {}
        replay_event_count = 0
        counter_regression_count = 0
        max_replay_gap = 0.0
        delayed_duplicate_count = 0
        max_duplicate_gap_s = 0.0

        for sensor, sensor_frame in witness.sort_values("event_time").groupby("sensor"):
            expected_step = 1.0
            row = counter_profile.loc[counter_profile["sensor"] == sensor]
            if not row.empty:
                expected_step = max(float(row.iloc[0]["median_step"]), 1.0)
            regressions = 0
            replay_gaps: list[float] = []
            recent_max: deque[tuple[float, float]] = deque()
            for _, event in sensor_frame.iterrows():
                counter = float(event["counter"])
                if not np.isfinite(counter):
                    continue
                event_ts = float(pd.Timestamp(event["event_time"]).timestamp())
                while recent_max and event_ts - recent_max[0][0] > self.config.replay_window_s:
                    recent_max.popleft()
                if recent_max and counter < recent_max[0][1]:
                    regressions += 1
                    replay_gaps.append((recent_max[0][1] - counter) / expected_step)
                while recent_max and counter >= recent_max[-1][1]:
                    recent_max.pop()
                recent_max.append((event_ts, counter))
            score = max(replay_gaps) if replay_gaps else 0.0
            sensor_replay_scores[str(sensor)] = round(float(score), 3)
            sensor_regression_counts[str(sensor)] = regressions
            replay_event_count += sum(gap >= self.config.replay_gap_threshold for gap in replay_gaps)
            counter_regression_count += regressions
            max_replay_gap = max(max_replay_gap, score)

        duplicate_groups = (
            frame.groupby(["sensor", "gateway", "counter"], as_index=False)
            .agg(
                observation_count=("timestamp", "size"),
                first_seen=("timestamp", "min"),
                last_seen=("timestamp", "max"),
            )
        )
        duplicate_groups["gap_s"] = (
            duplicate_groups["last_seen"] - duplicate_groups["first_seen"]
        ).dt.total_seconds().fillna(0.0)
        delayed_duplicates = duplicate_groups[
            (duplicate_groups["observation_count"] >= 2)
            & (duplicate_groups["gap_s"] > max(self.reference["time_tolerance_s"] * 4.0, 10.0))
        ].copy()
        delayed_duplicate_count = int(delayed_duplicates.shape[0])
        max_duplicate_gap_s = float(delayed_duplicates["gap_s"].max()) if not delayed_duplicates.empty else 0.0

        per_gateway = (
            frame.groupby("gateway", as_index=False)
            .agg(
                observation_count=("sensor", "size"),
                physics_rate=("physically_plausible", "mean"),
                mean_rssi_z=("rssi_z", "mean"),
                mean_snr_z=("snr_z", "mean"),
            )
        )
        phys_by_witness = (
            frame.groupby("witness_id", as_index=False)
            .agg(
                physically_implausible_fraction=("physically_plausible", lambda s: 1.0 - float(np.mean(s))),
            )
        )
        witness = witness.merge(phys_by_witness, on="witness_id", how="left").fillna({"physically_implausible_fraction": 0.0})
        witness["fabrication_anomaly"] = witness["fabrication_anomaly"] | (
            (witness["physically_implausible_fraction"] >= 1.0)
            & (witness["gateway_count"] <= 1)
            & witness["multiplicity_anomaly"]
        )
        witness_gateway = frame.merge(
            witness[
                [
                    "witness_id",
                    "sensor",
                    "counter",
                    "skew_anomaly",
                    "multiplicity_anomaly",
                    "fabrication_anomaly",
                ]
            ],
            on=["witness_id", "sensor", "counter"],
            how="left",
        )
        gateway_flags = (
            witness_gateway.groupby("gateway", as_index=False)
            .agg(
                skew_rate=("skew_anomaly", "mean"),
                multiplicity_clean_rate=("multiplicity_anomaly", lambda s: 1.0 - float(np.mean(s))),
                fabrication_clean_rate=("fabrication_anomaly", lambda s: 1.0 - float(np.mean(s))),
            )
            .fillna(0.0)
        )
        gateway_replay = []
        for gateway, gateway_frame in frame.sort_values("timestamp").groupby("gateway"):
            regressions = 0
            total = 0
            dedup = gateway_frame.drop_duplicates(subset=["sensor", "counter", "witness_id"])
            for sensor, sensor_frame in dedup.groupby("sensor"):
                recent_max: deque[tuple[float, float]] = deque()
                for _, event in sensor_frame.sort_values("timestamp").iterrows():
                    counter = float(event["counter"])
                    if not np.isfinite(counter):
                        continue
                    total += 1
                    event_ts = float(pd.Timestamp(event["timestamp"]).timestamp())
                    while recent_max and event_ts - recent_max[0][0] > self.config.replay_window_s:
                        recent_max.popleft()
                    if recent_max and counter < recent_max[0][1]:
                        regressions += 1
                    while recent_max and counter >= recent_max[-1][1]:
                        recent_max.pop()
                    recent_max.append((event_ts, counter))
            gateway_replay.append(
                {
                    "gateway": gateway,
                    "replay_clean_rate": 1.0 - (regressions / max(total, 1)),
                    "duplicate_clean_rate": 1.0
                    - (
                        float(
                            delayed_duplicates.loc[delayed_duplicates["gateway"] == gateway].shape[0]
                        )
                        / max(float(gateway_frame["sensor"].nunique()), 1.0)
                    ),
                }
            )
        per_gateway = (
            per_gateway.merge(gateway_flags, on="gateway", how="left")
            .merge(pd.DataFrame(gateway_replay), on="gateway", how="left")
            .fillna(
                {
                    "skew_rate": 0.0,
                    "multiplicity_clean_rate": 1.0,
                    "fabrication_clean_rate": 1.0,
                    "replay_clean_rate": 1.0,
                    "duplicate_clean_rate": 1.0,
                }
            )
        )
        per_gateway["alignment_rate"] = (
            0.5 * per_gateway["multiplicity_clean_rate"] + 0.5 * per_gateway["fabrication_clean_rate"]
        )
        per_gateway["timing_rate"] = 1.0 - per_gateway["skew_rate"]
        per_gateway["trust_score"] = (
            0.22 * per_gateway["alignment_rate"]
            + 0.18 * per_gateway["timing_rate"]
            + 0.25 * per_gateway["physics_rate"]
            + 0.15 * per_gateway["multiplicity_clean_rate"]
            + 0.15 * per_gateway["replay_clean_rate"]
            + 0.05 * per_gateway["duplicate_clean_rate"].clip(lower=0.0, upper=1.0)
        ).clip(lower=0.0, upper=1.0)

        gateway_trust_scores = {
            str(row["gateway"]): round(float(row["trust_score"]), 3)
            for _, row in per_gateway.sort_values("trust_score").iterrows()
        }

        suspicious_gateway = None
        if gateway_trust_scores:
            suspicious_gateway = min(gateway_trust_scores, key=gateway_trust_scores.get)

        suspicious_sensor = None
        if sensor_replay_scores:
            suspicious_sensor = max(sensor_replay_scores, key=sensor_replay_scores.get)

        event_inconsistency_score = (
            1.8 * min(max_replay_gap / max(self.config.replay_gap_threshold, 1.0), 3.0)
            + 1.5 * min(max_duplicate_gap_s / max(self.config.replay_gap_threshold * 60.0, 1.0), 3.0)
            + 1.0 * min(delayed_duplicate_count / 250.0, 3.0)
            + 1.2 * float(witness["skew_anomaly"].mean())
            + 1.2 * float(witness["fabrication_anomaly"].mean())
            + 1.0 * float(witness["multiplicity_anomaly"].mean())
            + 1.0 * float((~frame["physically_plausible"]).mean())
            + 1.0 * (1.0 - min(gateway_trust_scores.values()) if gateway_trust_scores else 0.0)
        )

        return {
            "time_tolerance_s": round(float(self.reference["time_tolerance_s"]), 3),
            "suspicious_gateway": suspicious_gateway,
            "suspicious_sensor": suspicious_sensor,
            "max_replay_gap": round(float(max_replay_gap), 3),
            "replay_event_count": int(replay_event_count),
            "counter_regression_count": int(counter_regression_count),
            "delayed_duplicate_count": delayed_duplicate_count,
            "max_duplicate_gap_s": round(max_duplicate_gap_s, 3),
            "high_skew_witness_count": int(witness["skew_anomaly"].sum()),
            "fabricated_witness_count": int(witness["fabrication_anomaly"].sum()),
            "multiplicity_anomaly_count": int(witness["multiplicity_anomaly"].sum()),
            "gateway_trust_scores": gateway_trust_scores,
            "sensor_replay_scores": sensor_replay_scores,
            "event_inconsistency_score": round(float(event_inconsistency_score), 3),
        }
