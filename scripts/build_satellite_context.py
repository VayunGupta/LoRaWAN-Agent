from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib import parse, request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import GATEWAYS, SENSORS
from src.react_agent.environment import OSMEnvironmentContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch satellite imagery for the UVA LoRaWAN deployment and derive LOS context."
    )
    parser.add_argument("--out-dir", default="outputs/satellite_context")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--margin-m", type=float, default=180.0)
    parser.add_argument("--sam-checkpoint", default="models/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam-max-side", type=int, default=1024)
    parser.add_argument(
        "--segmentation-mode",
        choices=["sam", "hybrid"],
        default="sam",
        help="Use SAM-derived masks, or hybrid SAM+map-footprint masks.",
    )
    parser.add_argument(
        "--imagery-url",
        default="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
        help="ArcGIS REST export endpoint for aerial/satellite imagery.",
    )
    return parser


def deployment_bbox(margin_m: float) -> tuple[float, float, float, float]:
    lats = [float(item["lat"]) for item in SENSORS.values()] + [
        float(item["lat"]) for item in GATEWAYS.values()
    ]
    lons = [float(item["lon"]) for item in SENSORS.values()] + [
        float(item["lon"]) for item in GATEWAYS.values()
    ]
    ref_lat = float(np.mean(lats))
    lat_margin = margin_m / 111_320.0
    lon_margin = margin_m / (111_320.0 * max(math.cos(math.radians(ref_lat)), 0.2))
    return (
        min(lons) - lon_margin,
        min(lats) - lat_margin,
        max(lons) + lon_margin,
        max(lats) + lat_margin,
    )


def fetch_satellite_image(
    *,
    endpoint: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    if out_path.exists():
        return
    params = {
        "bbox": ",".join(f"{value:.8f}" for value in bbox),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "png",
        "transparent": "false",
        "f": "image",
    }
    url = f"{endpoint}?{parse.urlencode(params)}"
    req = request.Request(url, headers={"User-Agent": "LoRaMAS research script"})
    with request.urlopen(req, timeout=45.0) as response:
        data = response.read()
    out_path.write_bytes(data)


def latlon_to_pixel(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    x = (lon - min_lon) / (max_lon - min_lon) * width
    y = (max_lat - lat) / (max_lat - min_lat) * height
    return x, y


def pixel_to_latlon(
    x: float,
    y: float,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + x / width * (max_lon - min_lon)
    lat = max_lat - y / height * (max_lat - min_lat)
    return lat, lon


def feature_polygons(context: OSMEnvironmentContext) -> list[dict]:
    elements = context._load_or_fetch_elements()
    return context._extract_features(elements) if elements else []


def xy_to_latlon(
    x: float,
    y: float,
    *,
    ref_lat: float,
    ref_lon: float,
) -> tuple[float, float]:
    lat = y / 111_320.0 + ref_lat
    lon = x / (111_320.0 * max(math.cos(math.radians(ref_lat)), 0.2)) + ref_lon
    return lat, lon


def image_green_fraction(
    image: Image.Image,
    *,
    bbox: tuple[float, float, float, float],
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    samples: int = 120,
) -> float:
    rgb = np.asarray(image.convert("RGB")).astype(float)
    h, w = rgb.shape[:2]
    greenish = 0
    observed = 0
    for t in np.linspace(0.0, 1.0, samples):
        lat = a_lat + (b_lat - a_lat) * t
        lon = a_lon + (b_lon - a_lon) * t
        x, y = latlon_to_pixel(lat, lon, bbox, w - 1, h - 1)
        px = int(np.clip(round(x), 0, w - 1))
        py = int(np.clip(round(y), 0, h - 1))
        r, g, b = rgb[py, px]
        # Lightweight vegetation cue for aerial imagery: green channel dominance.
        if g > r * 1.05 and g > b * 1.05 and g > 55:
            greenish += 1
        observed += 1
    return greenish / max(observed, 1)


def resize_for_sam(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    if scale >= 1.0:
        return image, 1.0
    resized = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
    return resized, scale


def build_sam_masks(
    image: Image.Image,
    *,
    checkpoint_path: Path,
    model_type: str,
    max_side: int,
) -> dict[str, np.ndarray]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing SAM checkpoint: {checkpoint_path}. "
            "Download sam_vit_b_01ec64.pth into models/sam first."
        )
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam_image, scale = resize_for_sam(image.convert("RGB"), max_side)
    image_arr = np.asarray(sam_image)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device="cpu")
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=80,
    )
    masks = generator.generate(image_arr)
    height, width = image_arr.shape[:2]
    building = np.zeros((height, width), dtype=bool)
    vegetation = np.zeros((height, width), dtype=bool)
    rgb = image_arr.astype(float)
    for mask_item in masks:
        mask = mask_item["segmentation"].astype(bool)
        area_fraction = float(mask.mean())
        if area_fraction < 0.00015 or area_fraction > 0.18:
            continue
        pixels = rgb[mask]
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0).mean()
        r, g, b = mean
        green_score = (g - max(r, b)) / 255.0
        brightness = float(mean.mean())
        if green_score > 0.025 and g > 55:
            vegetation |= mask
        elif 35 <= brightness <= 215 and std < 75:
            # SAM supplies object boundaries; this color/texture rule labels roof/pavement-like
            # regions as hard obstructions while excluding high-variance tree canopy.
            building |= mask

    if scale != 1.0:
        original_size = image.size
        building = np.asarray(
            Image.fromarray((building.astype(np.uint8) * 255)).resize(original_size, Image.Resampling.NEAREST)
        ) > 0
        vegetation = np.asarray(
            Image.fromarray((vegetation.astype(np.uint8) * 255)).resize(original_size, Image.Resampling.NEAREST)
        ) > 0
    return {"building": building, "vegetation": vegetation}


def sample_mask_fraction(
    mask: np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    a_lat: float,
    a_lon: float,
    b_lat: float,
    b_lon: float,
    samples: int = 160,
    trim_endpoint_fraction: float = 0.08,
) -> tuple[float, float]:
    height, width = mask.shape[:2]
    mid_hits = 0
    mid_count = 0
    endpoint_hits = 0
    endpoint_count = 0
    for t in np.linspace(0.0, 1.0, samples):
        lat = a_lat + (b_lat - a_lat) * t
        lon = a_lon + (b_lon - a_lon) * t
        x, y = latlon_to_pixel(lat, lon, bbox, width - 1, height - 1)
        px = int(np.clip(round(x), 0, width - 1))
        py = int(np.clip(round(y), 0, height - 1))
        hit = bool(mask[py, px])
        if t <= trim_endpoint_fraction or t >= 1.0 - trim_endpoint_fraction:
            endpoint_hits += int(hit)
            endpoint_count += 1
        else:
            mid_hits += int(hit)
            mid_count += 1
    return mid_hits / max(mid_count, 1), endpoint_hits / max(endpoint_count, 1)


def build_context_rows(
    *,
    osm_context: OSMEnvironmentContext,
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    sam_masks: dict[str, np.ndarray] | None = None,
    segmentation_mode: str = "sam",
) -> pd.DataFrame:
    features = feature_polygons(osm_context)
    rows = []
    for sensor, sensor_info in SENSORS.items():
        s_xy = osm_context._latlon_to_xy(float(sensor_info["lat"]), float(sensor_info["lon"]))
        sensor_indoor = 1.0 if str(sensor_info.get("env", "")).lower() == "indoor" else 0.0
        for gateway, gateway_info in GATEWAYS.items():
            g_xy = osm_context._latlon_to_xy(float(gateway_info["lat"]), float(gateway_info["lon"]))
            map_building_hits = 0
            map_endpoint_building_hits = 0
            map_vegetation_hits = 0
            for feature in features:
                if not osm_context._segment_hits_feature(s_xy, g_xy, feature["points"], feature["closed"]):
                    continue
                contains_endpoint = False
                if feature["closed"]:
                    contains_endpoint = (
                        osm_context._point_in_polygon(s_xy, feature["points"])
                        or osm_context._point_in_polygon(g_xy, feature["points"])
                    )
                if feature["kind"] == "building":
                    if contains_endpoint:
                        map_endpoint_building_hits += 1
                    else:
                        map_building_hits += 1
                elif feature["kind"] == "vegetation":
                    map_vegetation_hits += 1

            distance_m = float(math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]))
            green_fraction = image_green_fraction(
                image,
                bbox=bbox,
                a_lat=float(sensor_info["lat"]),
                a_lon=float(sensor_info["lon"]),
                b_lat=float(gateway_info["lat"]),
                b_lon=float(gateway_info["lon"]),
            )
            sam_building_fraction = sam_endpoint_building_fraction = 0.0
            sam_vegetation_fraction = 0.0
            if sam_masks:
                sam_building_fraction, sam_endpoint_building_fraction = sample_mask_fraction(
                    sam_masks["building"],
                    bbox=bbox,
                    a_lat=float(sensor_info["lat"]),
                    a_lon=float(sensor_info["lon"]),
                    b_lat=float(gateway_info["lat"]),
                    b_lon=float(gateway_info["lon"]),
                )
                sam_vegetation_fraction, _ = sample_mask_fraction(
                    sam_masks["vegetation"],
                    bbox=bbox,
                    a_lat=float(sensor_info["lat"]),
                    a_lon=float(sensor_info["lon"]),
                    b_lat=float(gateway_info["lat"]),
                    b_lon=float(gateway_info["lon"]),
                )

            if segmentation_mode == "hybrid":
                building_signal = float(map_building_hits) + sam_building_fraction * 3.0
                endpoint_building_signal = float(map_endpoint_building_hits) + sam_endpoint_building_fraction * 2.0
                vegetation_signal = float(map_vegetation_hits) + sam_vegetation_fraction * 3.0
            else:
                building_signal = sam_building_fraction * 4.0
                endpoint_building_signal = sam_endpoint_building_fraction * 2.0
                vegetation_signal = sam_vegetation_fraction * 4.0

            satellite_blocked = float(sam_building_fraction >= 0.045 or (segmentation_mode == "hybrid" and map_building_hits > 0))
            expected_extra = (
                building_signal * 3.0
                + endpoint_building_signal * 0.7
                + vegetation_signal * 1.0
                + green_fraction * 2.4
                + sensor_indoor * 3.5
                + min(distance_m / 500.0, 2.5) * 0.4
            )
            context_score = (
                building_signal * 1.25
                + endpoint_building_signal * 0.3
                + vegetation_signal * 0.5
                + green_fraction * 1.6
                + sensor_indoor * 1.4
                + min(distance_m / 300.0, 3.0) * 0.25
            )
            rows.append(
                {
                    "sensor": sensor,
                    "gateway": gateway,
                    "path_distance_m": round(distance_m, 3),
                    "sensor_indoor": sensor_indoor,
                    "building_intersections": round(float(building_signal), 3),
                    "endpoint_building_intersections": round(float(endpoint_building_signal), 3),
                    "vegetation_intersections": round(float(vegetation_signal), 3),
                    "map_building_intersections": float(map_building_hits),
                    "map_endpoint_building_intersections": float(map_endpoint_building_hits),
                    "map_vegetation_intersections": float(map_vegetation_hits),
                    "sam_building_fraction": round(sam_building_fraction, 3),
                    "sam_endpoint_building_fraction": round(sam_endpoint_building_fraction, 3),
                    "sam_vegetation_fraction": round(sam_vegetation_fraction, 3),
                    "line_of_sight_blocked": satellite_blocked,
                    "expected_extra_attenuation_db": round(expected_extra, 3),
                    "context_fragility_score": round(context_score, 3),
                    "satellite_green_fraction": round(green_fraction, 3),
                    "satellite_los_blocked": satellite_blocked,
                    "satellite_expected_extra_attenuation_db": round(expected_extra, 3),
                    "satellite_context_score": round(context_score, 3),
                }
            )
    return pd.DataFrame(rows)


def draw_overlay(
    *,
    image_path: Path,
    out_path: Path,
    bbox: tuple[float, float, float, float],
    rows: pd.DataFrame,
    osm_context: OSMEnvironmentContext,
    sam_masks: dict[str, np.ndarray] | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    if sam_masks:
        building_overlay = np.zeros((height, width, 4), dtype=float)
        building_overlay[sam_masks["building"]] = [1.0, 0.76, 0.05, 0.22]
        veg_overlay = np.zeros((height, width, 4), dtype=float)
        veg_overlay[sam_masks["vegetation"]] = [0.1, 0.7, 0.2, 0.22]
        ax.imshow(building_overlay)
        ax.imshow(veg_overlay)

    ref_lat = osm_context.ref_lat
    ref_lon = osm_context.ref_lon
    for feature in feature_polygons(osm_context):
        color = "#ffcc00" if feature["kind"] == "building" else "#39b54a"
        points = [
            latlon_to_pixel(*xy_to_latlon(x, y, ref_lat=ref_lat, ref_lon=ref_lon), bbox, width, height)
            for x, y in feature["points"]
        ]
        if len(points) >= 2:
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.7)

    for row in rows.to_dict(orient="records"):
        sensor_info = SENSORS[row["sensor"]]
        gateway_info = GATEWAYS[row["gateway"]]
        sx, sy = latlon_to_pixel(float(sensor_info["lat"]), float(sensor_info["lon"]), bbox, width, height)
        gx, gy = latlon_to_pixel(float(gateway_info["lat"]), float(gateway_info["lon"]), bbox, width, height)
        blocked = bool(row["satellite_los_blocked"])
        ax.plot(
            [sx, gx],
            [sy, gy],
            color="#d62828" if blocked else "#2a9d8f",
            alpha=0.28 if blocked else 0.38,
            linewidth=1.0,
        )

    for sensor, info in SENSORS.items():
        x, y = latlon_to_pixel(float(info["lat"]), float(info["lon"]), bbox, width, height)
        ax.scatter(x, y, s=48, c="#005f73", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(x + 7, y - 7, sensor.replace("sensor", "S"), color="white", fontsize=7, weight="bold")

    for gateway, info in GATEWAYS.items():
        x, y = latlon_to_pixel(float(info["lat"]), float(info["lon"]), bbox, width, height)
        ax.scatter(x, y, s=110, marker="^", c="#ae2012", edgecolors="white", linewidths=1.0, zorder=6)
        ax.text(x + 8, y + 10, gateway.replace("gateway", "G"), color="white", fontsize=9, weight="bold")

    ax.set_axis_off()
    ax.set_title("Satellite-backed LOS context for UVA LoRaWAN deployment", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    paper_dir = Path(args.paper_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    bbox = deployment_bbox(args.margin_m)
    image_path = out_dir / "uva_satellite.png"
    fetch_satellite_image(
        endpoint=args.imagery_url,
        bbox=bbox,
        width=args.width,
        height=args.height,
        out_path=image_path,
    )

    osm_context = OSMEnvironmentContext(
        sensor_meta=SENSORS,
        gateway_meta=GATEWAYS,
        cache_path="outputs/osm/osm_context_cache.json",
        overpass_endpoint="https://overpass-api.de/api/interpreter",
        timeout_s=25.0,
        margin_m=args.margin_m,
    )
    image = Image.open(image_path).convert("RGB")
    sam_masks = None
    if args.segmentation_mode in {"sam", "hybrid"}:
        sam_masks = build_sam_masks(
            image,
            checkpoint_path=Path(args.sam_checkpoint),
            model_type=args.sam_model_type,
            max_side=args.sam_max_side,
        )
        Image.fromarray((sam_masks["building"].astype(np.uint8) * 255)).save(out_dir / "sam_building_mask.png")
        Image.fromarray((sam_masks["vegetation"].astype(np.uint8) * 255)).save(out_dir / "sam_vegetation_mask.png")

    rows = build_context_rows(
        osm_context=osm_context,
        image=image,
        bbox=bbox,
        sam_masks=sam_masks,
        segmentation_mode=args.segmentation_mode,
    )
    rows.to_csv(out_dir / "satellite_context_by_pair.csv", index=False)
    (out_dir / "satellite_context_meta.json").write_text(
        json.dumps(
            {
                "bbox_lonlat": bbox,
                "image": str(image_path),
                "pair_count": int(rows.shape[0]),
                "blocked_pair_count": int(rows["satellite_los_blocked"].sum()),
                "mean_satellite_context_score": float(rows["satellite_context_score"].mean()),
                "segmentation_mode": args.segmentation_mode,
                "sam_checkpoint": args.sam_checkpoint if sam_masks else None,
                "sam_building_mask_fraction": float(sam_masks["building"].mean()) if sam_masks else 0.0,
                "sam_vegetation_mask_fraction": float(sam_masks["vegetation"].mean()) if sam_masks else 0.0,
            },
            indent=2,
        )
    )

    overlay_path = out_dir / "satellite_los_overlay.png"
    draw_overlay(
        image_path=image_path,
        out_path=overlay_path,
        bbox=bbox,
        rows=rows,
        osm_context=osm_context,
        sam_masks=sam_masks,
    )
    # Keep a copy beside the TeX source so \includegraphics works from paper/.
    (paper_dir / "satellite_los_overlay.png").write_bytes(overlay_path.read_bytes())
    print(f"Wrote {rows.shape[0]} satellite context rows to {out_dir}")
    print(f"Copied overlay figure to {paper_dir / 'satellite_los_overlay.png'}")


if __name__ == "__main__":
    main()
