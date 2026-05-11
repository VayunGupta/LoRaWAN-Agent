from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib import parse, request

import pandas as pd


class OSMEnvironmentContext:
    """
    Fetch lightweight OpenStreetMap context and convert it into static
    sensor-gateway path features that the heuristic can use as priors.

    The provider caches raw Overpass responses so later runs can work offline.
    """

    def __init__(
        self,
        *,
        sensor_meta: dict[str, dict[str, Any]],
        gateway_meta: dict[str, dict[str, Any]],
        cache_path: str,
        overpass_endpoint: str,
        timeout_s: float,
        margin_m: float,
    ):
        self.sensor_meta = sensor_meta
        self.gateway_meta = gateway_meta
        self.cache_path = Path(cache_path)
        self.overpass_endpoint = overpass_endpoint
        self.timeout_s = timeout_s
        self.margin_m = margin_m
        self.ref_lat = self._mean([float(item["lat"]) for item in sensor_meta.values()])
        self.ref_lon = self._mean([float(item["lon"]) for item in gateway_meta.values()])

    def build_pair_contexts(self) -> pd.DataFrame:
        elements = self._load_or_fetch_elements()
        if not elements:
            return self._fallback_pair_contexts()

        features = self._extract_features(elements)
        rows: list[dict[str, Any]] = []
        for sensor, sensor_info in self.sensor_meta.items():
            s_xy = self._latlon_to_xy(float(sensor_info["lat"]), float(sensor_info["lon"]))
            sensor_indoor = 1.0 if str(sensor_info.get("env", "")).lower() == "indoor" else 0.0
            for gateway, gateway_info in self.gateway_meta.items():
                g_xy = self._latlon_to_xy(float(gateway_info["lat"]), float(gateway_info["lon"]))
                building_hits = 0
                vegetation_hits = 0
                for feature in features:
                    if self._segment_hits_feature(s_xy, g_xy, feature["points"], feature["closed"]):
                        if feature["kind"] == "building":
                            building_hits += 1
                        elif feature["kind"] == "vegetation":
                            vegetation_hits += 1

                distance_m = float(math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]))
                expected_extra_attenuation_db = (
                    building_hits * 2.8
                    + vegetation_hits * 1.4
                    + sensor_indoor * 3.5
                    + min(distance_m / 400.0, 3.0) * 0.6
                )
                fragility = (
                    building_hits * 1.2
                    + vegetation_hits * 0.7
                    + sensor_indoor * 1.4
                    + min(distance_m / 250.0, 4.0) * 0.5
                )
                rows.append(
                    {
                        "sensor": sensor,
                        "gateway": gateway,
                        "path_distance_m": round(distance_m, 3),
                        "sensor_indoor": sensor_indoor,
                        "building_intersections": float(building_hits),
                        "vegetation_intersections": float(vegetation_hits),
                        "line_of_sight_blocked": float((building_hits + vegetation_hits) > 0),
                        "expected_extra_attenuation_db": round(expected_extra_attenuation_db, 3),
                        "context_fragility_score": round(fragility, 3),
                    }
                )

        return pd.DataFrame(rows).sort_values(["sensor", "gateway"]).reset_index(drop=True)

    def _load_or_fetch_elements(self) -> list[dict[str, Any]]:
        cached = self._read_cache()
        if cached:
            return cached

        try:
            payload = self._fetch_elements()
        except Exception:
            return []

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({"elements": payload}, indent=2))
        return payload

    def _read_cache(self) -> list[dict[str, Any]]:
        if not self.cache_path.exists():
            return []
        try:
            data = json.loads(self.cache_path.read_text())
        except json.JSONDecodeError:
            return []
        elements = data.get("elements")
        return elements if isinstance(elements, list) else []

    def _fetch_elements(self) -> list[dict[str, Any]]:
        min_lat, max_lat, min_lon, max_lon = self._bounding_box()
        query = (
            "[out:json][timeout:25];("
            f'way["building"]({min_lat},{min_lon},{max_lat},{max_lon});'
            f'way["landuse"="forest"]({min_lat},{min_lon},{max_lat},{max_lon});'
            f'way["natural"="wood"]({min_lat},{min_lon},{max_lat},{max_lon});'
            f'way["natural"="tree_row"]({min_lat},{min_lon},{max_lat},{max_lon});'
            ");(._;>;);out body;"
        )
        data = parse.urlencode({"data": query}).encode("utf-8")
        req = request.Request(
            self.overpass_endpoint,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with request.urlopen(req, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        elements = payload.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError("Unexpected Overpass response.")
        return elements

    def _bounding_box(self) -> tuple[float, float, float, float]:
        lats = [float(item["lat"]) for item in self.sensor_meta.values()] + [
            float(item["lat"]) for item in self.gateway_meta.values()
        ]
        lons = [float(item["lon"]) for item in self.sensor_meta.values()] + [
            float(item["lon"]) for item in self.gateway_meta.values()
        ]
        lat_margin = self.margin_m / 111_320.0
        lon_margin = self.margin_m / (111_320.0 * max(math.cos(math.radians(self.ref_lat)), 0.2))
        return (
            min(lats) - lat_margin,
            max(lats) + lat_margin,
            min(lons) - lon_margin,
            max(lons) + lon_margin,
        )

    def _extract_features(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nodes = {
            int(item["id"]): (float(item["lat"]), float(item["lon"]))
            for item in elements
            if item.get("type") == "node" and "lat" in item and "lon" in item
        }
        features: list[dict[str, Any]] = []
        for item in elements:
            if item.get("type") != "way":
                continue
            tags = item.get("tags", {})
            kind = self._feature_kind(tags)
            if kind is None:
                continue
            node_ids = item.get("nodes", [])
            latlon = [nodes[int(node_id)] for node_id in node_ids if int(node_id) in nodes]
            if len(latlon) < 2:
                continue
            points = [self._latlon_to_xy(lat, lon) for lat, lon in latlon]
            closed = len(points) >= 3 and points[0] == points[-1]
            features.append({"kind": kind, "points": points, "closed": closed})
        return features

    @staticmethod
    def _feature_kind(tags: dict[str, Any]) -> str | None:
        if "building" in tags:
            return "building"
        if tags.get("landuse") == "forest":
            return "vegetation"
        if tags.get("natural") in {"wood", "tree_row"}:
            return "vegetation"
        return None

    def _fallback_pair_contexts(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for sensor, sensor_info in self.sensor_meta.items():
            s_xy = self._latlon_to_xy(float(sensor_info["lat"]), float(sensor_info["lon"]))
            sensor_indoor = 1.0 if str(sensor_info.get("env", "")).lower() == "indoor" else 0.0
            for gateway, gateway_info in self.gateway_meta.items():
                g_xy = self._latlon_to_xy(float(gateway_info["lat"]), float(gateway_info["lon"]))
                distance_m = float(math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]))
                fragility = sensor_indoor * 1.4 + min(distance_m / 250.0, 4.0) * 0.5
                expected_extra_attenuation_db = sensor_indoor * 3.5 + min(distance_m / 400.0, 3.0) * 0.6
                rows.append(
                    {
                        "sensor": sensor,
                        "gateway": gateway,
                        "path_distance_m": round(distance_m, 3),
                        "sensor_indoor": sensor_indoor,
                        "building_intersections": 0.0,
                        "vegetation_intersections": 0.0,
                        "line_of_sight_blocked": 0.0,
                        "expected_extra_attenuation_db": round(expected_extra_attenuation_db, 3),
                        "context_fragility_score": round(fragility, 3),
                    }
                )
        return pd.DataFrame(rows).sort_values(["sensor", "gateway"]).reset_index(drop=True)

    def _latlon_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        x = (lon - self.ref_lon) * 111_320.0 * math.cos(math.radians(self.ref_lat))
        y = (lat - self.ref_lat) * 111_320.0
        return (x, y)

    @staticmethod
    def _segment_hits_feature(
        a: tuple[float, float],
        b: tuple[float, float],
        points: list[tuple[float, float]],
        closed: bool,
    ) -> bool:
        if closed and OSMEnvironmentContext._point_in_polygon(a, points):
            return True
        if closed and OSMEnvironmentContext._point_in_polygon(b, points):
            return True

        edge_count = len(points) if closed else len(points) - 1
        for idx in range(edge_count):
            c = points[idx]
            d = points[(idx + 1) % len(points)]
            if OSMEnvironmentContext._segments_intersect(a, b, c, d):
                return True
        return False

    @staticmethod
    def _segments_intersect(
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> bool:
        def orient(p, q, r) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r) -> bool:
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)

        if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
            return True
        if abs(o1) < 1e-9 and on_segment(a, c, b):
            return True
        if abs(o2) < 1e-9 and on_segment(a, d, b):
            return True
        if abs(o3) < 1e-9 and on_segment(c, a, d):
            return True
        if abs(o4) < 1e-9 and on_segment(c, b, d):
            return True
        return False

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        inside = False
        x, y = point
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)
