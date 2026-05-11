from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.helpers import load_and_prepare_packets
from src.traditional.pathloss import TraditionalParams, estimate_pair_distances_traditional
from src.traditional.trilateration import trilaterate_all

from .environment import OSMEnvironmentContext
from .trust import TrustConfig, WitnessTrustAnalyzer
from .types import AttackScenario


@dataclass
class ServerConfig:
    data_dir: str = "dataset/lorawan_metadata"
    metadata_path: str = "models/metadata.json"
    traditional_params_path: str = "models/traditional_params.json"
    target_freq_mhz: float = 915.0
    outlier_db: float = 20.0
    min_pkts: int = 10
    use_environment_context: bool = False
    satellite_context_path: str = "outputs/satellite_context/satellite_context_by_pair.csv"
    osm_cache_path: str = "outputs/osm/osm_context_cache.json"
    osm_overpass_endpoint: str = "https://overpass-api.de/api/interpreter"
    osm_timeout_s: float = 25.0
    osm_margin_m: float = 250.0
    trust_timestamp_quantile: float = 0.99
    trust_timestamp_floor_s: float = 1.5
    trust_frequency_tolerance_hz: float = 200000.0
    trust_rssi_z_threshold: float = 3.5
    trust_snr_z_threshold: float = 3.5
    trust_replay_gap_threshold: float = 3.0


class GatewayCoordinatorServer:
    """
    Server-side coordinator that exposes read-only tools over gateway observations.

    The ReAct agent uses these tools to investigate whether a synthetic attack
    was injected into the dataset-backed network snapshot.
    """

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        self.params = TraditionalParams.load(self.config.traditional_params_path)
        with open(self.config.metadata_path, "r") as handle:
            self.metadata = json.load(handle)

        self.gateway_xy = {name: tuple(xy) for name, xy in self.metadata["GW_XY"].items()}
        self.sensor_xy_true = {
            name: tuple(xy) for name, xy in self.metadata.get("S_XY_TRUE", {}).items()
        }
        self._snapshot_cache: dict[str, dict[str, Any]] = {}
        self._investigation_bundle_cache: dict[str, dict[str, Any]] = {}
        self.environment_context = self._build_environment_context()
        self._baseline_packets = self._load_packets(AttackScenario())
        self._baseline_summary = self._summarize_pairs(self._baseline_packets)
        self.trust_analyzer = WitnessTrustAnalyzer(
            self._baseline_packets,
            TrustConfig(
                timestamp_quantile=self.config.trust_timestamp_quantile,
                timestamp_floor_s=self.config.trust_timestamp_floor_s,
                frequency_tolerance_hz=self.config.trust_frequency_tolerance_hz,
                rssi_z_threshold=self.config.trust_rssi_z_threshold,
                snr_z_threshold=self.config.trust_snr_z_threshold,
                replay_gap_threshold=self.config.trust_replay_gap_threshold,
            ),
        )

    def _load_packets(self, scenario: AttackScenario) -> pd.DataFrame:
        return load_and_prepare_packets(
            data_dir=self.config.data_dir,
            target_freq_mhz=self.config.target_freq_mhz,
            outlier_db=self.config.outlier_db,
            min_pkts=self.config.min_pkts,
            **scenario.load_kwargs(),
        )

    @staticmethod
    def _summarize_pairs(packets: pd.DataFrame) -> pd.DataFrame:
        if packets.empty:
            return pd.DataFrame(
                columns=[
                    "sensor",
                    "gateway",
                    "packet_count",
                    "rssi_median_dbm",
                    "rssi_mean_dbm",
                    "rssi_std_db",
                    "snr_median_db",
                ]
            )

        summary = (
            packets.groupby(["sensor", "gateway"], as_index=False)
            .agg(
                packet_count=("rssi_dbm", "size"),
                rssi_median_dbm=("rssi_dbm", "median"),
                rssi_mean_dbm=("rssi_dbm", "mean"),
                rssi_std_db=("rssi_dbm", "std"),
                snr_median_db=("snr_db", "median"),
            )
            .fillna({"rssi_std_db": 0.0})
        )
        return summary.sort_values(["sensor", "gateway"]).reset_index(drop=True)

    @staticmethod
    def _pairwise_delta(
        baseline_summary: pd.DataFrame, scenario_summary: pd.DataFrame, env_context: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        merged = baseline_summary.merge(
            scenario_summary,
            on=["sensor", "gateway"],
            how="left",
            suffixes=("_baseline", "_scenario"),
        )
        fill_from_baseline = [
            "rssi_median_dbm",
            "rssi_mean_dbm",
            "rssi_std_db",
            "snr_median_db",
        ]
        for column in fill_from_baseline:
            merged[f"{column}_scenario"] = merged[f"{column}_scenario"].fillna(
                merged[f"{column}_baseline"]
            )
        merged["packet_count_scenario"] = merged["packet_count_scenario"].fillna(0)
        merged["rssi_shift_db"] = (
            merged["rssi_median_dbm_scenario"] - merged["rssi_median_dbm_baseline"]
        )
        merged["packet_ratio"] = (
            merged["packet_count_scenario"] / merged["packet_count_baseline"].clip(lower=1)
        )
        merged["std_delta_db"] = merged["rssi_std_db_scenario"] - merged["rssi_std_db_baseline"]
        merged["snr_shift_db"] = (
            merged["snr_median_db_scenario"] - merged["snr_median_db_baseline"]
        )
        if env_context is not None and not env_context.empty:
            merged = merged.merge(env_context, on=["sensor", "gateway"], how="left")
        return merged

    def get_network_snapshot(self, scenario: AttackScenario) -> dict[str, Any]:
        scenario_key = json.dumps(scenario.to_dict(), sort_keys=True)
        cached = self._snapshot_cache.get(scenario_key)
        if cached is not None:
            return cached
        packets = self._load_packets(scenario)
        summary = self._summarize_pairs(packets)
        delta = self._pairwise_delta(self._baseline_summary, summary, self.environment_context)
        trust = self.trust_analyzer.analyze(packets)
        snapshot = {"packets": packets, "summary": summary, "delta": delta, "trust": trust}
        self._snapshot_cache[scenario_key] = snapshot
        return snapshot

    def rank_sensors(self, delta: pd.DataFrame) -> pd.DataFrame:
        if delta.empty:
            return pd.DataFrame(columns=["sensor", "anomaly_score"])

        sensor_rank = (
            delta.groupby("sensor", as_index=False)
            .agg(
                mean_abs_rssi_shift=("rssi_shift_db", lambda s: float(np.mean(np.abs(s)))),
                worst_packet_ratio=("packet_ratio", "min"),
                mean_std_delta=("std_delta_db", "mean"),
            )
        )
        sensor_rank["base_anomaly_score"] = (
            sensor_rank["mean_abs_rssi_shift"] * 1.4
            + (1.0 - sensor_rank["worst_packet_ratio"].clip(lower=0.0, upper=1.0)) * 10.0
            + sensor_rank["mean_std_delta"].clip(lower=0.0) * 1.2
        )
        sensor_rank["anomaly_score"] = sensor_rank["base_anomaly_score"]
        if self.config.use_environment_context and "context_fragility_score" in delta.columns:
            env_rank = (
                delta.groupby("sensor", as_index=False)
                .agg(
                    mean_context_fragility=("context_fragility_score", "mean"),
                    blocked_link_fraction=("line_of_sight_blocked", "mean"),
                    indoor_link_fraction=("sensor_indoor", "mean"),
                    mean_expected_extra_attenuation_db=("expected_extra_attenuation_db", "mean"),
                )
            )
            sensor_rank = sensor_rank.merge(env_rank, on="sensor", how="left").fillna(0.0)
            weak_factor = ((6.5 - sensor_rank["mean_abs_rssi_shift"]).clip(lower=0.0, upper=6.5)) / 6.5
            packet_guard = ((sensor_rank["worst_packet_ratio"] - 0.5).clip(lower=0.0, upper=0.5)) / 0.5
            plausibility_discount = (
                sensor_rank["mean_expected_extra_attenuation_db"].clip(lower=0.0, upper=8.0) * 0.12
            )
            discount = (
                plausibility_discount
                + sensor_rank["blocked_link_fraction"] * 0.35
                + sensor_rank["indoor_link_fraction"] * 0.2
            ) * weak_factor * packet_guard
            sensor_rank["anomaly_score"] = (sensor_rank["base_anomaly_score"] - discount).clip(lower=0.0)
        return sensor_rank.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    def rank_gateways(self, delta: pd.DataFrame) -> pd.DataFrame:
        if delta.empty:
            return pd.DataFrame(columns=["gateway", "anomaly_score"])

        gateway_rank = (
            delta.groupby("gateway", as_index=False)
            .agg(
                mean_abs_rssi_shift=("rssi_shift_db", lambda s: float(np.mean(np.abs(s)))),
                worst_packet_ratio=("packet_ratio", "min"),
                mean_std_delta=("std_delta_db", "mean"),
            )
        )
        gateway_rank["base_anomaly_score"] = (
            gateway_rank["mean_abs_rssi_shift"] * 1.5
            + (1.0 - gateway_rank["worst_packet_ratio"].clip(lower=0.0, upper=1.0)) * 8.0
            + gateway_rank["mean_std_delta"].clip(lower=0.0)
        )
        gateway_rank["anomaly_score"] = gateway_rank["base_anomaly_score"]
        if self.config.use_environment_context and "context_fragility_score" in delta.columns:
            env_rank = (
                delta.groupby("gateway", as_index=False)
                .agg(
                    mean_context_fragility=("context_fragility_score", "mean"),
                    blocked_link_fraction=("line_of_sight_blocked", "mean"),
                    mean_expected_extra_attenuation_db=("expected_extra_attenuation_db", "mean"),
                )
            )
            gateway_rank = gateway_rank.merge(env_rank, on="gateway", how="left").fillna(0.0)
            weak_factor = ((6.5 - gateway_rank["mean_abs_rssi_shift"]).clip(lower=0.0, upper=6.5)) / 6.5
            packet_guard = ((gateway_rank["worst_packet_ratio"] - 0.5).clip(lower=0.0, upper=0.5)) / 0.5
            plausibility_discount = (
                gateway_rank["mean_expected_extra_attenuation_db"].clip(lower=0.0, upper=8.0) * 0.08
            )
            discount = (
                plausibility_discount
                + gateway_rank["blocked_link_fraction"] * 0.2
            ) * weak_factor * packet_guard
            gateway_rank["anomaly_score"] = (gateway_rank["base_anomaly_score"] - discount).clip(lower=0.0)
        return gateway_rank.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def sensor_view(delta: pd.DataFrame, sensor: str) -> pd.DataFrame:
        cols = [
            "sensor",
            "gateway",
            "rssi_shift_db",
            "packet_ratio",
            "std_delta_db",
            "snr_shift_db",
            "packet_count_baseline",
            "packet_count_scenario",
        ]
        for extra in [
            "path_distance_m",
            "sensor_indoor",
            "building_intersections",
            "vegetation_intersections",
            "line_of_sight_blocked",
            "expected_extra_attenuation_db",
            "context_fragility_score",
        ]:
            if extra in delta.columns:
                cols.append(extra)
        return delta.loc[delta["sensor"] == sensor, cols].sort_values("gateway").reset_index(drop=True)

    @staticmethod
    def gateway_view(delta: pd.DataFrame, gateway: str) -> pd.DataFrame:
        cols = [
            "sensor",
            "gateway",
            "rssi_shift_db",
            "packet_ratio",
            "std_delta_db",
            "snr_shift_db",
        ]
        for extra in [
            "path_distance_m",
            "sensor_indoor",
            "building_intersections",
            "vegetation_intersections",
            "line_of_sight_blocked",
            "expected_extra_attenuation_db",
            "context_fragility_score",
        ]:
            if extra in delta.columns:
                cols.append(extra)
        return delta.loc[delta["gateway"] == gateway, cols].sort_values("sensor").reset_index(drop=True)

    def trilateration_view(self, packets: pd.DataFrame, sensor: str) -> dict[str, Any]:
        sensor_packets = packets.loc[packets["sensor"] == sensor].copy()
        if sensor_packets.empty:
            return {
                "sensor": sensor,
                "available": False,
                "reason": "No packets were available for this sensor in the scenario snapshot.",
            }

        pair_pred = estimate_pair_distances_traditional(sensor_packets, self.params)
        loc_df = trilaterate_all(
            pair_pred,
            GW_XY=self.gateway_xy,
            S_XY_TRUE=self.sensor_xy_true,
            min_gateways=3,
        )
        if loc_df.empty:
            return {
                "sensor": sensor,
                "available": False,
                "reason": "Fewer than three gateways survived preprocessing for this sensor.",
            }

        row = loc_df.iloc[0].to_dict()
        x_est = float(row["x_est"])
        y_est = float(row["y_est"])
        residuals = []
        for _, pair in pair_pred.iterrows():
            gx, gy = self.gateway_xy[pair["gateway"]]
            modeled = float(np.hypot(x_est - gx, y_est - gy))
            residuals.append(modeled - float(pair["d_est_m"]))

        row["available"] = True
        row["gateway_distance_estimates_m"] = {
            pair["gateway"]: float(pair["d_est_m"]) for _, pair in pair_pred.iterrows()
        }
        row["residual_rmse_m"] = float(np.sqrt(np.mean(np.square(residuals))))
        row["pair_count"] = int(pair_pred.shape[0])
        return row

    def baseline_trilateration_view(self, sensor: str) -> dict[str, Any]:
        return self.trilateration_view(self._baseline_packets, sensor)

    def ensure_metadata(self) -> None:
        for path in [self.config.metadata_path, self.config.traditional_params_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required model metadata is missing: {path}")

    def _build_environment_context(self) -> pd.DataFrame:
        if not self.config.use_environment_context:
            return pd.DataFrame()
        satellite_path = Path(self.config.satellite_context_path)
        if satellite_path.exists():
            context = pd.read_csv(satellite_path)
            for source, target in {
                "satellite_los_blocked": "line_of_sight_blocked",
                "satellite_expected_extra_attenuation_db": "expected_extra_attenuation_db",
                "satellite_context_score": "context_fragility_score",
            }.items():
                if source in context.columns:
                    context[target] = context[source]
            return context
        provider = OSMEnvironmentContext(
            sensor_meta=self.metadata.get("SENSORS_LATLON", {}),
            gateway_meta=self.metadata.get("GATEWAYS_LATLON", {}),
            cache_path=self.config.osm_cache_path,
            overpass_endpoint=self.config.osm_overpass_endpoint,
            timeout_s=self.config.osm_timeout_s,
            margin_m=self.config.osm_margin_m,
        )
        return provider.build_pair_contexts()
