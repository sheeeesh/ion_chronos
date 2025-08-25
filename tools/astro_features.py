"""
High-precision astrology feature computation using Swiss Ephemeris (tropical, geocentric) with Placidus houses and major aspects. Supports minute and daily timestamp data.

- If Swiss Ephemeris (`pyswisseph`) is not available, returns the same columns filled with NaN to keep the feature schema stable for ML/RL.
- Time handling: expects a tz-aware UTC index; input is localized or converted to UTC.
- Houses: Placidus by default; requires a location (lat, lon).
- Caching: optional parquet cache to avoid recomputation (by date index).
"""
from __future__ import annotations

from typing import Dict, List, Optional
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Global warning flag for Swiss Ephemeris
_WARNED_NO_SWE = False

# Attempt to import Swiss Ephemeris for high-precision calculations
try:
    import swisseph as swe  # pip install pyswisseph
    _HAS_SWE = True
    # Set ephemeris path if provided via environment
    _EPHE_PATH = os.getenv("SWISSEPH_EPHE_PATH")
    if _EPHE_PATH:
        try:
            swe.set_ephe_path(_EPHE_PATH)
        except Exception:
            pass
except Exception:
    swe = None
    _HAS_SWE = False

# Swiss Ephemeris body IDs (using mean node for stability; use True Node if desired)
_BODIES = {
    "sun": 0,
    "moon": 1,
    "mercury": 2,
    "venus": 3,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
    "pluto": 9,
    "node": 10,  # MEAN_NODE (11 would be TRUE_NODE)
}

# Major aspects and exact angles (degrees)
_ASPECTS = {
    "conj": 0.0,
    "sext": 60.0,
    "sq":   90.0,
    "tri":  120.0,
    "opp":  180.0,
}

# ------------------------------ Helper Functions ------------------------------

def _mod360(x: float) -> float:
    return x % 360.0

def _ang_diff(a: float, b: float) -> float:
    """Shortest signed angular difference (a - b) in [-180, 180) degrees."""
    return (a - b + 180.0) % 360.0 - 180.0

def _delta_to_aspect(a: float, b: float, asp_deg: float) -> float:
    """Signed distance from (a - b) to aspect angle asp_deg on a circle (degrees)."""
    return _ang_diff(a - b, asp_deg)

def _sin_deg(x: float) -> float:
    return math.sin(math.radians(x))

def _cos_deg(x: float) -> float:
    return math.cos(math.radians(x))

def _predeclare_columns() -> List[str]:
    """List all output columns in stable order."""
    body_cols: List[str] = []
    for name in _BODIES.keys():
        body_cols += [
            f"astro_lon_{name}_deg",
            f"astro_speed_{name}_degpd",
            f"astro_retro_{name}",
            f"astro_sin_lon_{name}",
            f"astro_cos_lon_{name}",
        ]
    asp_cols: List[str] = []
    primary_A = ["sun", "moon"]
    primary_B = [b for b in _BODIES.keys() if b not in ("sun", "moon", "node")]
    for a in primary_A:
        for b in primary_B:
            for asp_name in _ASPECTS.keys():
                asp_cols.append(f"astro_asp_{a}_{b}_{asp_name}")
                asp_cols.append(f"astro_aspd_{a}_{b}_{asp_name}")
    house_cols = [f"astro_house_cusp_{i}_deg" for i in range(1, 13)] + ["astro_asc_deg", "astro_mc_deg"]
    return body_cols + ["astro_phase_deg", "astro_phase_sin", "astro_phase_cos"] + asp_cols + house_cols

def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a DatetimeIndex to UTC (tz-aware)."""
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

def _to_jd_array(idx_utc: pd.DatetimeIndex) -> np.ndarray:
    """Convert UTC timestamps to Julian Day (UT) for Swiss Ephemeris."""
    y = idx_utc.year
    m = idx_utc.month
    d = idx_utc.day
    # Fractional day from time of day
    frac = idx_utc.hour + idx_utc.minute / 60.0 + idx_utc.second / 3600.0 + idx_utc.microsecond / 3.6e9
    jd_vals = np.empty(len(idx_utc), dtype=float)
    for i in range(len(idx_utc)):
        try:
            jd_vals[i] = swe.julday(int(y[i]), int(m[i]), int(d[i]), float(frac[i]))
        except Exception:
            jd_vals[i] = np.nan
    return jd_vals

def _compute_row(jd_ut: float, lat: float, lon: float, orb_deg: float, base_cols: List[str]) -> Dict[str, float]:
    """Compute astro features for a single timestamp (Julian Day UTC)."""
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED  # high-precision ephemeris with speed
    row: Dict[str, float] = {}
    # Planets: longitudes, speeds, retrograde flags, sin/cos of longitude
    for name, pid in _BODIES.items():
        lon_deg = np.nan
        speed = np.nan
        try:
            lon, lat_ecl, dist, dlon, dlat, ddist = swe.calc_ut(jd_ut, pid, flags)
            lon_deg = _mod360(float(lon))
            speed = float(dlon)  # degrees per day
        except Exception:
            pass
        row[f"astro_lon_{name}_deg"] = lon_deg
        row[f"astro_speed_{name}_degpd"] = speed
        row[f"astro_retro_{name}"] = 1.0 if (not np.isnan(speed) and speed < 0.0) else 0.0
        row[f"astro_sin_lon_{name}"] = _sin_deg(lon_deg) if not np.isnan(lon_deg) else np.nan
        row[f"astro_cos_lon_{name}"] = _cos_deg(lon_deg) if not np.isnan(lon_deg) else np.nan
    # Moon phase (Moon - Sun)
    sun_lon = row.get("astro_lon_sun_deg", np.nan)
    moon_lon = row.get("astro_lon_moon_deg", np.nan)
    phase = _mod360(moon_lon - sun_lon) if (not np.isnan(sun_lon) and not np.isnan(moon_lon)) else np.nan
    row["astro_phase_deg"] = phase
    row["astro_phase_sin"] = _sin_deg(phase) if not np.isnan(phase) else np.nan
    row["astro_phase_cos"] = _cos_deg(phase) if not np.isnan(phase) else np.nan
    # Aspects: Sun/Moon vs other planets
    for a in ["sun", "moon"]:
        a_lon = row.get(f"astro_lon_{a}_deg", np.nan)
        for b in ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]:
            b_lon = row.get(f"astro_lon_{b}_deg", np.nan)
            for asp_name, deg in _ASPECTS.items():
                key_f = f"astro_asp_{a}_{b}_{asp_name}"
                key_d = f"astro_aspd_{a}_{b}_{asp_name}"
                if np.isnan(a_lon) or np.isnan(b_lon):
                    row[key_f] = np.nan
                    row[key_d] = np.nan
                else:
                    delta = _delta_to_aspect(a_lon, b_lon, deg)
                    row[key_f] = 1.0 if abs(delta) <= orb_deg else 0.0
                    row[key_d] = float(delta)
    # Placidus houses (requires lat/lon)
    try:
        cusps, ascmc = swe.houses(jd_ut, float(lat), float(lon), b'P')
        for i in range(1, 13):
            row[f"astro_house_cusp_{i}_deg"] = _mod360(float(cusps[i]))
        row["astro_asc_deg"] = _mod360(float(ascmc[0]))
        row["astro_mc_deg"] = _mod360(float(ascmc[1]))
    except Exception:
        for i in range(1, 13):
            row[f"astro_house_cusp_{i}_deg"] = np.nan
        row["astro_asc_deg"] = np.nan
        row["astro_mc_deg"] = np.nan
    # Ensure all expected columns exist
    for col in base_cols:
        row.setdefault(col, np.nan)
    return row

# ------------------------------ Public API ------------------------------

def compute_astro_features(dates_utc: pd.DatetimeIndex,
                           lat: float = 40.7128,
                           lon: float = -74.0060,
                           house_system: str = "P",
                           orb_deg: float = 3.0,
                           cache_path: Optional[str] = None,
                           max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Compute astro features for each timestamp in `dates_utc` using Swiss Ephemeris.

    If the Swiss Ephemeris is unavailable, returns a DataFrame of NaNs with the same columns for consistency.

    Args:
        dates_utc: A tz-aware DatetimeIndex of timestamps (any frequency). Will be converted to UTC.
        lat, lon: Latitude and longitude for house cusp calculations (Placidus houses).
        house_system: House system (only 'P' is used here for Placidus).
        orb_deg: Orb threshold in degrees for considering an aspect exact.
        cache_path: Optional path to a parquet file for caching results.
        max_workers: max threads for parallel computation (None uses default).

    Returns:
        A DataFrame indexed by dates_utc with astro feature columns (dtype float32).
    """
    base_cols = _predeclare_columns()
    index = dates_utc if isinstance(dates_utc, pd.DatetimeIndex) else pd.to_datetime(dates_utc, errors="coerce")
    index = _to_utc_index(index)
    
    # ---------------- input validation ----------------
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    if not index.is_unique:
        raise ValueError("dates_utc must have a unique DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError("dates_utc must be monotonic increasing.")
    
    # Try cache first
    if cache_path and os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            output = cached.reindex(index)
            missing_idx = output[output.isna().all(axis=1)].index
            if len(missing_idx) == 0:
                return output.astype("float32")
            # Compute missing rows and fill in
            fill_df = compute_astro_features(missing_idx, lat=lat, lon=lon,
                                             house_system=house_system, orb_deg=orb_deg, 
                                             cache_path=None, max_workers=max_workers)
            output.loc[missing_idx, :] = fill_df
            output = output.astype("float32")
            # Update cache
            try:
                output.to_parquet(cache_path)
            except Exception:
                pass
            return output
        except Exception:
            pass
        
    if not _HAS_SWE:
        # No SwissEph? warn once and return NaNs with same shape
        global _WARNED_NO_SWE
        if not _WARNED_NO_SWE:
            warnings.warn(
                "Swiss Ephemeris not found – astro features filled with NaNs",
                RuntimeWarning,
            )
            _WARNED_NO_SWE = True
        # Swiss Ephemeris unavailable: return NaN frame
        return pd.DataFrame(np.nan, index=index, columns=base_cols).astype("float32")
    # Compute features for each timestamp
    jd_arr = _to_jd_array(index)
    rows: List[Dict[str, float]] = [None] * len(jd_arr)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, jd_ut in enumerate(jd_arr):
            if np.isnan(jd_ut):
                rows[i] = {c: np.nan for c in base_cols}
            else:
                futures[i] = executor.submit(
                    _compute_row, float(jd_ut), lat, lon, orb_deg, base_cols
                )
        for i, fut in futures.items():
            rows[i] = fut.result()
    result = pd.DataFrame(rows, index=index)[base_cols].astype("float32")
    # Save to cache if requested
    if cache_path:
        try:
            result.to_parquet(cache_path)
        except Exception:
            pass
    return result
