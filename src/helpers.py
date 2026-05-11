# src/helpers.py
from __future__ import annotations

import os, glob, json, numpy as np, pandas as pd
from math import radians, cos, sin, asin, sqrt

RADIUS_OF_EARTH = 6371000.0
OUTLIER_DB = 20.0
MIN_PKTS = 10
TARGET_FREQ_MHZ = 915

def haversine_m(lat1, lon1, lat2, lon2):
    R = RADIUS_OF_EARTH
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def drop_rssi_outliers(group, thresh_db=OUTLIER_DB):
    med = group['rssi_dbm'].median()
    return group[(group['rssi_dbm'] >= med - thresh_db) & (group['rssi_dbm'] <= med + thresh_db)]

def load_pair_parquet(path):
    """
    Load a single sensor–gateway parquet file and normalize column names.

    Tries a known mapping first, then falls back to simple heuristic
    detection (case-insensitive match on substrings like 'sensor', 'gateway',
    'rssi', 'snr', 'freq').
    """
    df = pd.read_parquet(path, engine="pyarrow")

    # 1) Known mapping from the original export format
    col_map = {
        'Sensor Alias'            : 'sensor',
        'Gateway Alias'           : 'gateway',
        'RSSI (dBm)'              : 'rssi_dbm',
        'SNR (dB)'                : 'snr_db',
        'Frequency (Hz)'          : 'freq_hz',
        'Timestamp'               : 'timestamp',
        'Counter (-)'             : 'counter',
        'Spreading Factor (-)'    : 'sf',
        'Bandwidth (Hz)'          : 'bw_hz',
        '# Receiving Gateways (-)': 'n_rx_gw',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 2) Heuristic fallback if required columns are still missing
    cols_lower = {c.lower(): c for c in df.columns}

    def _maybe_rename(substr: str, new_name: str):
        if new_name in df.columns:
            return
        for lower, orig in cols_lower.items():
            if substr in lower:
                df.rename(columns={orig: new_name}, inplace=True)
                return

    _maybe_rename("sensor", "sensor")
    _maybe_rename("gateway", "gateway")
    _maybe_rename("rssi", "rssi_dbm")
    _maybe_rename("snr", "snr_db")
    # frequency can be in Hz or MHz; keep as Hz and convert later
    if "freq_hz" not in df.columns:
        for lower, orig in cols_lower.items():
            if "freq" in lower:
                df.rename(columns={orig: "freq_hz"}, inplace=True)
                break

    # ensure SF is int when present
    if 'sf' in df.columns:
        df['sf'] = pd.to_numeric(df['sf'], errors='coerce')

    keep = [
        c for c in [
            'sensor','gateway','timestamp',
            'rssi_dbm','snr_db',
            'freq_hz','bw_hz','sf','n_rx_gw','counter'
        ] if c in df.columns
    ]

    # If we still can't see the core columns, bail out early with a clear error
    required = {'sensor', 'gateway', 'rssi_dbm', 'freq_hz'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"[load_pair_parquet] Missing expected columns {missing} in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[keep].dropna(subset=['sensor','gateway','rssi_dbm','freq_hz'])


def frequency_filter(df, target=TARGET_FREQ_MHZ):
    df = df.copy()
    df['freq_mhz'] = (df['freq_hz']/1e6).round(0)
    if (df['freq_mhz']==target).any():
        return df[df['freq_mhz']==target].copy()
    mode = df['freq_mhz'].mode()
    return df[df['freq_mhz']==float(mode.iloc[0])].copy() if len(mode) else df.iloc[0:0].copy()

def load_all_pairs(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "sensor*_gateway*.parquet")))
    frames = []
    for p in paths:
        df = load_pair_parquet(p)
        if df.empty: continue
        df = frequency_filter(df)
        if df.empty: continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def clean_and_featurize(raw: pd.DataFrame):
    if raw.empty: return raw
    trimmed = (
        raw.groupby(['sensor','gateway'], group_keys=False)
           .apply(drop_rssi_outliers)
           .reset_index(drop=True)
    )
    # enforce min packets
    pair_sizes = trimmed.groupby(['sensor','gateway']).size()
    valid_pairs = pair_sizes[pair_sizes>=MIN_PKTS].index
    trimmed = trimmed.set_index(['sensor','gateway']).loc[valid_pairs].reset_index()
    if 'snr_db' not in trimmed: trimmed['snr_db']=np.nan
    trimmed['snr_db'] = trimmed['snr_db'].fillna(trimmed['snr_db'].median() if not trimmed['snr_db'].dropna().empty else 0.0)
    trimmed['freq_mhz'] = (trimmed['freq_hz']/1e6).round(0)
    return trimmed

def load_latlon_from_meta(meta_json: str):
    """
    Expects models/metadata.json to include:
      {
        "SENSORS_LATLON": {"sensor01":{"lat":..,"lon":..}, ...},
        "GATEWAYS_LATLON": {"gatewayA":{"lat":..,"lon":..}, ...}
      }
    """
    with open(meta_json, "r") as f:
        meta = json.load(f)
    sensors = meta.get("SENSORS_LATLON")
    gateways = meta.get("GATEWAYS_LATLON")
    if sensors is None or gateways is None:
        raise ValueError("metadata.json must include SENSORS_LATLON and GATEWAYS_LATLON")
    return sensors, gateways

def load_and_prepare_packets(
    data_dir: str,
    target_freq_mhz: float = TARGET_FREQ_MHZ,
    outlier_db: float = OUTLIER_DB,
    min_pkts: int = MIN_PKTS,
    attack_type: str = "none",
    attack_scope: str = "global",
    attack_sensor: str = None,
    attack_gateway: str = None,
    attack_sensors: list[str] | None = None,
    attack_gateways: list[str] | None = None,
    rssi_shift_db: float = 0.0,
    rssi_noise_sigma_db: float = 0.0,
    drop_prob: float = 0.0,
    replay_fraction: float = 0.0,
    replay_delay_s: float = 0.0,
    fabricate_fraction: float = 0.0,
    fabricate_shift_db: float = 8.0,
    counter_shift: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Unified loader used by both traditional and regression paths.
    1) loads all sensor*_gateway*.parquet
    2) frequency-bins to target (or dominant bin)
    3) trims RSSI outliers (±outlier_db around median) per (sensor,gateway)
    4) enforces min_pkts per (sensor,gateway)
    5) fills SNR and returns packet-level rows with ['sensor','gateway','rssi_dbm','snr_db','freq_mhz', 'timestamp'(if present)]
    """
    raw = load_all_pairs(data_dir)
    if raw.empty:
        return raw

    # Copy so we can safely modify RSSI and derived columns
    raw = raw.copy()

        # ----------------------------
    # Attack mask (what rows are targeted)
    # ----------------------------
    scope = attack_scope
    s = attack_sensor
    g = attack_gateway
    sensor_targets = list(attack_sensors or ([] if attack_sensor is None else [attack_sensor]))
    gateway_targets = list(attack_gateways or ([] if attack_gateway is None else [attack_gateway]))

    if scope == "global":
        mask = np.ones(len(raw), dtype=bool)
    elif scope == "sensor":
        if not sensor_targets:
            raise ValueError("attack_scope='sensor' requires attack_sensor or attack_sensors")
        mask = raw["sensor"].isin(sensor_targets).to_numpy()
    elif scope == "gateway":
        if not gateway_targets:
            raise ValueError("attack_scope='gateway' requires attack_gateway or attack_gateways")
        mask = raw["gateway"].isin(gateway_targets).to_numpy()
    elif scope == "link":
        if not s or not g:
            raise ValueError("attack_scope='link' requires attack_sensor and attack_gateway")
        mask = ((raw["sensor"] == s) & (raw["gateway"] == g)).to_numpy()
    else:
        raise ValueError(f"Unknown attack_scope: {scope}")

    # ----------------------------
    # Attack aliases for richer cross-gateway consistency attacks
    # ----------------------------
    if attack_type == "selective_suppression" and drop_prob <= 0.0:
        drop_prob = 0.45
    if attack_type == "replay_attack" and replay_fraction <= 0.0:
        replay_fraction = 0.12
    if attack_type == "delayed_replay":
        if replay_fraction <= 0.0:
            replay_fraction = 0.12
        if replay_delay_s <= 0.0:
            replay_delay_s = 180.0
    if attack_type == "gateway_fabrication" and fabricate_fraction <= 0.0:
        fabricate_fraction = 0.10
    if attack_type == "counter_corruption" and counter_shift == 0:
        counter_shift = -7

    # ----------------------------
    # Packet drop / jamming (drop targeted packets)
    # ----------------------------
    drop_mask = mask.copy()
    if attack_type == "selective_suppression" and mask.any():
        rng = np.random.default_rng(int(seed) + 17)
        targeted = raw.loc[mask].copy()
        targeted_sensors = sorted(targeted["sensor"].unique().tolist())
        if targeted_sensors:
            select_count = max(1, int(np.ceil(len(targeted_sensors) / 3.0)))
            selected_sensors = set(
                rng.choice(targeted_sensors, size=min(select_count, len(targeted_sensors)), replace=False).tolist()
            )
            selective_mask = mask & raw["sensor"].isin(selected_sensors).to_numpy()
            if "timestamp" in raw.columns and selective_mask.any():
                selective_times = pd.to_datetime(raw.loc[selective_mask, "timestamp"], utc=True)
                median_time = selective_times.median()
                selective_mask &= (pd.to_datetime(raw["timestamp"], utc=True) >= median_time).to_numpy()
            elif "counter" in raw.columns and selective_mask.any():
                selective_mask &= ((raw["counter"].fillna(0).astype(int) % 2) == 0).to_numpy()
            if selective_mask.any():
                drop_mask = selective_mask

    if drop_prob and drop_prob > 0.0:
        if not (0.0 < drop_prob < 1.0):
            raise ValueError("drop_prob must be in (0,1)")
        rng = np.random.default_rng(int(seed))
        keep = np.ones(len(raw), dtype=bool)
        # only drop within mask
        drop_draw = rng.random(drop_mask.sum())
        keep_idx = np.where(drop_mask)[0]
        keep[keep_idx] = drop_draw >= float(drop_prob)
        raw = raw.loc[keep].copy()
        # recompute mask after dropping (indices changed)
        if scope == "global":
            mask = np.ones(len(raw), dtype=bool)
        elif scope == "sensor":
            mask = raw["sensor"].isin(sensor_targets).to_numpy()
        elif scope == "gateway":
            mask = raw["gateway"].isin(gateway_targets).to_numpy()
        else:
            mask = ((raw["sensor"] == s) & (raw["gateway"] == g)).to_numpy()

    # ----------------------------
    # RSSI shift (foil / gateway bias) applied ONLY to targeted packets
    # ----------------------------
    if rssi_shift_db and rssi_shift_db != 0.0:
        raw.loc[mask, "rssi_dbm"] = raw.loc[mask, "rssi_dbm"] + float(rssi_shift_db)

    # ----------------------------
    # Random noise attack (Gaussian per packet, targeted by mask)
    # ----------------------------
    if rssi_noise_sigma_db and rssi_noise_sigma_db > 0.0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(loc=0.0, scale=float(rssi_noise_sigma_db), size=int(mask.sum()))
        raw.loc[mask, "rssi_dbm"] = raw.loc[mask, "rssi_dbm"] + noise

    # ----------------------------
    # Replay-style attacks: duplicate targeted observations later in time
    # ----------------------------
    if replay_fraction and replay_fraction > 0.0 and 'timestamp' in raw.columns:
        candidate = raw.loc[mask].copy()
        if not candidate.empty:
            rng = np.random.default_rng(int(seed) + 101)
            replay_n = max(1, int(round(candidate.shape[0] * float(replay_fraction))))
            replay = candidate.sample(n=min(replay_n, candidate.shape[0]), random_state=int(seed) + 101).copy()
            default_delay = 45.0 if attack_type == "replay_attack" else 180.0
            delay_s = float(replay_delay_s if replay_delay_s > 0.0 else default_delay)
            replay["timestamp"] = pd.to_datetime(replay["timestamp"], utc=True) + pd.to_timedelta(delay_s, unit="s")
            # Make the replay visibly implausible by slightly perturbing the duplicated reception metadata.
            if "rssi_dbm" in replay.columns:
                replay["rssi_dbm"] = replay["rssi_dbm"] + rng.normal(loc=0.0, scale=1.25, size=replay.shape[0])
            if "snr_db" in replay.columns:
                replay["snr_db"] = replay["snr_db"] + rng.normal(loc=0.0, scale=0.75, size=replay.shape[0])
            raw = pd.concat([raw, replay], ignore_index=True)

    # ----------------------------
    # Gateway fabrication: create new gateway-side witnesses for packets that
    # were observed elsewhere, but perturb their physical evidence.
    # ----------------------------
    if fabricate_fraction and fabricate_fraction > 0.0:
        if not gateway_targets:
            raise ValueError("gateway_fabrication requires attack_gateway or attack_gateways")
        source = raw.loc[~raw["gateway"].isin(gateway_targets)].copy()
        if attack_sensor:
            source = source.loc[source["sensor"] == attack_sensor]
        if not source.empty:
            fabricate_n = max(1, int(round(source.shape[0] * float(fabricate_fraction))))
            fabricated = source.sample(
                n=min(fabricate_n, source.shape[0]),
                random_state=int(seed) + 211,
            ).copy()
            rng = np.random.default_rng(int(seed) + 211)
            fabricated["gateway"] = rng.choice(gateway_targets, size=fabricated.shape[0], replace=True)
            if "timestamp" in fabricated.columns:
                fabricated["timestamp"] = pd.to_datetime(fabricated["timestamp"], utc=True) + pd.to_timedelta(
                    rng.uniform(0.5, 3.0, size=fabricated.shape[0]),
                    unit="s",
                )
            if "rssi_dbm" in fabricated.columns:
                fabricated["rssi_dbm"] = fabricated["rssi_dbm"] + float(fabricate_shift_db)
            if "snr_db" in fabricated.columns:
                fabricated["snr_db"] = fabricated["snr_db"] + rng.normal(loc=-1.5, scale=1.0, size=fabricated.shape[0])
            if "n_rx_gw" in fabricated.columns:
                fabricated["n_rx_gw"] = fabricated["n_rx_gw"].clip(lower=1) + 1
            raw = pd.concat([raw, fabricated], ignore_index=True)

    # ----------------------------
    # Counter corruption: break packet identity continuity on the targeted view
    # ----------------------------
    if counter_shift and counter_shift != 0 and "counter" in raw.columns:
        raw.loc[mask, "counter"] = raw.loc[mask, "counter"] + int(counter_shift)

    # Recompute freq_mhz because load_all_pairs already frequency_filter'ed
    raw["freq_mhz"] = (raw["freq_hz"] / 1e6).round(0)

    # Basic schema sanity check before grouping
    required = {"sensor", "gateway", "rssi_dbm", "freq_hz"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"[load_and_prepare_packets] Missing columns {sorted(missing)} in concatenated data. "
            f"Available columns: {list(raw.columns)}"
        )

    # Outlier trim with provided window (override module default if passed).
    # Use groupby+transform so we never lose the 'sensor'/'gateway' columns.
    med = raw.groupby(['sensor', 'gateway'])['rssi_dbm'].transform('median')
    lo = med - outlier_db
    hi = med + outlier_db
    trimmed = raw[(raw['rssi_dbm'] >= lo) & (raw['rssi_dbm'] <= hi)].copy()

    # Enforce min_pkts
    counts = trimmed.groupby(['sensor','gateway']).size()
    keep_idx = counts[counts >= min_pkts].index
    trimmed = trimmed.set_index(['sensor','gateway']).loc[keep_idx].reset_index()

    # SNR fill + freq_mhz (again, ensure present)
    if 'snr_db' not in trimmed.columns:
        trimmed['snr_db'] = np.nan
    trimmed['snr_db'] = trimmed['snr_db'].fillna(
        trimmed['snr_db'].median() if not trimmed['snr_db'].dropna().empty else 0.0
    )
    trimmed['freq_mhz'] = (trimmed['freq_hz'] / 1e6).round(0)

    # Keep a stable set of columns
    keep_cols = [c for c in [
    'sensor','gateway','timestamp',
    'rssi_dbm','snr_db','freq_mhz',
    'bw_hz','sf','n_rx_gw','counter'
    ] if c in trimmed.columns]
    return trimmed[keep_cols]

def summarize_pairs_to_csv(pairs_df: pd.DataFrame, out_csv: str):
    """
    Collapse packets to per-(sensor,gateway) summary used by report.py.
    Produces columns:
      sensor, gateway, n_pkts, snr_db_median, freq_mhz, sensor_bw_hz, sensor_sf
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if pairs_df.empty:
        pd.DataFrame(columns=[
            'sensor','gateway','n_pkts','snr_db_median','freq_mhz','sensor_bw_hz','sensor_sf'
        ]).to_csv(out_csv, index=False)
        return

    # per-(sensor,gateway)
    agg = (pairs_df.groupby(['sensor','gateway'], as_index=False)
                     .agg(n_pkts=('rssi_dbm','size'),
                          snr_db_median=('snr_db','median'),
                          freq_mhz=('freq_mhz','first')))

    # dominant bandwidth per sensor (if present)
    if 'bw_hz' in pairs_df.columns:
        dom_bw = (pairs_df.groupby('sensor')['bw_hz']
                          .agg(lambda s: pd.Series.mode(s.dropna())[0] if s.dropna().size else np.nan)
                          .rename('sensor_bw_hz'))
        agg = agg.merge(dom_bw, on='sensor', how='left')

    # dominant spreading factor per sensor (if present)
    sf_col = 'sf' if 'sf' in pairs_df.columns else None
    if sf_col:
        dom_sf = (pairs_df.groupby('sensor')[sf_col]
                          .agg(lambda s: pd.Series.mode(s.dropna())[0] if s.dropna().size else np.nan)
                          .rename('sensor_sf'))
        agg = agg.merge(dom_sf, on='sensor', how='left')

    agg.to_csv(out_csv, index=False)
