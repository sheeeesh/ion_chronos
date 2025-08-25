# tools/astro_features.py
"""
High-precision astrology features using Swiss Ephemeris (tropical, geocentric),
with Placidus houses and major aspects. Minute & daily timestamps supported.

- If Swiss Ephemeris (pyswisseph) is NOT available, returns the SAME columns
  filled with NaN to keep the feature schema stable for ML/RL.
- Time handling: expects a tz-aware UTC index; will localize/convert to UTC.
- Houses: Placidus by default; requires a location (lat, lon).
- Caching: optional parquet cache to avoid recomputation.

Env (optional):
- SWISSEPH_EPHE_PATH: folder with ephemeris files (e.g., "sepl_18.se1")
"""

from __future__ import annotations

from typing import Dict, List, Optional
import math
import os

import numpy as np
import pandas as pd

# ---------------- Swiss Ephemeris ----------------
try:
    import swisseph as swe  # pip install pyswisseph
    _HAS_SWE = True
    # set ephemeris path if provided
    _EPHE_PATH = os.getenv("SWISSEPH_EPHE_PATH")
    if _EPHE_PATH:
        try:
            swe.set_ephe_path(_EPHE_PATH)
        except Exception:
            pass
except Exception:
    swe = None
    _HAS_SWE = False

# Bodies (SwissEph IDs); names used in column prefixes
# Note: we use MEAN_NODE for stability; switch to TRUE_NODE if you prefer.
_BODIES = {
    "sun":     0,   # swe.SUN
    "moon":    1,   # swe.MOON
    "mercury": 2,
    "venus":   3,
    "mars":    4,
    "jupiter": 5,
    "saturn":  6,
    "uranus":  7,
    "neptune": 8,
    "pluto":   9,
    "node":    10,  # swe.MEAN_NODE (change to 11 for TRUE_NODE)
}

# Major aspects and exact angles (degrees)
_ASPECTS = {
    "conj": 0.0,
    "sext": 60.0,
    "sq":   90.0,
    "tri":  120.0,
    "opp":  180.0,
}

# ---------------- helpers ----------------

def _mod360(x: float) -> float:
    return x % 360.0

def _ang_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0

def _delta_to_aspect(a: float, b: float, asp_deg: float) -> float:
    """Signed distance from (a-b) to asp_deg on a circle."""
    return _ang_diff(a - b, asp_deg)

def _sin_deg(x: float) -> float:
    return math.sin(math.radians(x))

def _cos_deg(x: float) -> float:
    return math.cos(math.radians(x))

def _predeclare_columns() -> List[str]:
    """All output columns, in a stable order."""
    body_cols: List[str] = []
    for b in _BODIES.keys():
        body_cols += [
            f"astro_lon_{b}_deg",
            f"astro_speed_{b}_degpd",
            f"astro_retro_{b}",
            f"astro_sin_lon_{b}",
            f"astro_cos_lon_{b}",
        ]

    asp_cols: List[str] = []
    primary_A = ["sun", "moon"]
    primary_B = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
    for a in primary_A:
        for b in primary_B:
            for name in _ASPECTS.keys():
                asp_cols.append(f"astro_asp_{a}_{b}_{name}")   # boolean (0/1)
                asp_cols.append(f"astro_aspd_{a}_{b}_{name}")  # angular distance to exact aspect

    # Placidus houses: cusps 1..12, plus Asc & MC
    house_cols = [f"astro_house_cusp_{i}_deg" for i in range(1, 13)] + [
        "astro_asc_deg",
        "astro_mc_deg",
    ]

    base_cols = (
        body_cols
        + ["astro_phase_deg", "astro_phase_sin", "astro_phase_cos"]
        + asp_cols
        + house_cols
    )
    return base_cols

def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

def _to_jd_array(idx_utc: pd.DatetimeIndex) -> np.ndarray:
    """Convert UTC timestamps to SwissEph UT Julian Days."""
    y = idx_utc.year
    m = idx_utc.month
    d = idx_utc.day
    frac = (
        idx_utc.hour
        + idx_utc.minute / 60.0
        + idx_utc.second / 3600.0
        + idx_utc.microsecond / 3.6e9
    )
    out = np.empty(len(idx_utc), dtype=float)
    for i in range(len(idx_utc)):
        try:
            out[i] = swe.julday(int(y[i]), int(m[i]), int(d[i]), float(frac[i]))
        except Exception:
            out[i] = np.nan
    return out

def _compute_row(jd_ut: float, lat: float, lon: float, orb_deg: float, base_cols: List[str]) -> Dict[str, float]:
    """Compute a single timestamp of astro features using SwissEph."""
    # Tropical, geocentric, ecliptic longitude with speed
    flags = 256 | 1024  # swe.FLG_SWIEPH | swe.FLG_SPEED
    row: Dict[str, float] = {}

    # Planetary longitudes, speeds, retro flags, sin/cos
    for name, pid in _BODIES.items():
        lon_deg = np.nan
        spd = np.nan
        try:
            lon, lat_ecl, dist, dlon, dlat, ddist = swe.calc_ut(jd_ut, pid, flags)
            lon_deg = _mod360(float(lon))
            spd = float(dlon)  # deg/day
        except Exception:
            pass
        row[f"astro_lon_{name}_deg"] = lon_deg
        row[f"astro_speed_{name}_degpd"] = spd
        row[f"astro_retro_{name}"] = 1.0 if (not np.isnan(spd) and spd < 0.0) else 0.0
        row[f"astro_sin_lon_{name}"] = _sin_deg(lon_deg) if not np.isnan(lon_deg) else np.nan
        row[f"astro_cos_lon_{name}"] = _cos_deg(lon_deg) if not np.isnan(lon_deg) else np.nan

    # Moon phase (Moon - Sun)
    sun = row.get("astro_lon_sun_deg", np.nan)
    moon = row.get("astro_lon_moon_deg", np.nan)
    phase = _mod360((moon - sun)) if (not np.isnan(sun) and not np.isnan(moon)) else np.nan
    row["astro_phase_deg"] = phase
    row["astro_phase_sin"] = _sin_deg(phase) if not np.isnan(phase) else np.nan
    row["astro_phase_cos"] = _cos_deg(phase) if not np.isnan(phase) else np.nan

    # Aspects: Sun/Moon vs other planets
    primary_A = ["sun", "moon"]
    primary_B = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
    for a in primary_A:
        a_lon = row.get(f"astro_lon_{a}_deg", np.nan)
        for b in primary_B:
            b_lon = row.get(f"astro_lon_{b}_deg", np.nan)
            for name, deg in _ASPECTS.items():
                key_f = f"astro_asp_{a}_{b}_{name}"
                key_d = f"astro_aspd_{a}_{b}_{name}"
                if np.isnan(a_lon) or np.isnan(b_lon):
                    row[key_f] = np.nan
                    row[key_d] = np.nan
                    continue
                delta = _delta_to_aspect(a_lon, b_lon, deg)
                row[key_f] = 1.0 if abs(delta) <= float(orb_deg) else 0.0
                row[key_d] = float(delta)

    # Placidus houses (requires location)
    try:
        # returns (cusps[1..12], ascmc[asc, mc, ...])
        cusps, ascmc = swe.houses(jd_ut, float(lat), float(lon), b'P')  # 'P' = Placidus
        for i in range(1, 13):
            row[f"astro_house_cusp_{i}_deg"] = _mod360(float(cusps[i]))
        row["astro_asc_deg"] = _mod360(float(ascmc[0]))
        row["astro_mc_deg"] = _mod360(float(ascmc[1]))
    except Exception:
        for i in range(1, 13):
            row[f"astro_house_cusp_{i}_deg"] = np.nan
        row["astro_asc_deg"] = np.nan
        row["astro_mc_deg"] = np.nan

    # guarantee all columns exist
    for k in base_cols:
        row.setdefault(k, np.nan)

    return row

# ---------------- public API ----------------

def compute_astro_features(
    dates_utc: pd.DatetimeIndex,
    lat: float = 40.7128,            # New York City
    lon: float = -74.0060,
    house_system: str = "P",         # Placidus
    orb_deg: float = 3.0,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute Swiss Ephemeris astro features for given UTC timestamps.
    If pyswisseph is missing, returns NaNs with the same columns.

    Args:
      dates_utc: tz-aware DatetimeIndex (any frequency). Converted to UTC.
      lat, lon: location for house cusps (Placidus).
      house_system: currently only 'P' used by SwissEph houses call here.
      orb_deg: orb threshold for aspects (degrees).
      cache_path: optional parquet file path to read/write cached results.

    Returns:
      DataFrame[dates_utc, features...] with float32 dtype.
    """
    base_cols = _predeclare_columns()
    index = dates_utc
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index, errors="coerce")
    index = _to_utc_index(index)

    # Cache: if provided, try to load and reindex
    if cache_path and os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index, utc=True)
            out = cached.reindex(index)
            # if any timestamps completely missing, compute only those
            missing = out[out.isna().all(axis=1)].index
            if len(missing) == 0:
                return out.astype("float32")
            # compute the missing rows and fill
            fill = compute_astro_features(missing, lat=lat, lon=lon,
                                          house_system=house_system, orb_deg=orb_deg,
                                          cache_path=None)
            out.loc[missing, :] = fill
            out = out.astype("float32")
            # write-back
            try:
                out.to_parquet(cache_path)
            except Exception:
                pass
            return out
        except Exception:
            # fall through to fresh compute
            pass

    # No SwissEph? return NaNs with same shape
    if not _HAS_SWE:
        return pd.DataFrame(np.nan, index=index, columns=base_cols).astype("float32")

    # Compute row-by-row (SwissEph expects scalar times)
    jd_arr = _to_jd_array(index)
    rows: List[Dict[str, float]] = []
    for jd_ut in jd_arr:
        if np.isnan(jd_ut):
            rows.append({c: np.nan for c in base_cols})
        else:
            rows.append(_compute_row(float(jd_ut), lat=lat, lon=lon, orb_deg=orb_deg, base_cols=base_cols))

    out = pd.DataFrame(rows, index=index)[base_cols].astype("float32")

    # Save cache if requested
    if cache_path:
        try:
            out.to_parquet(cache_path)
        except Exception:
            pass

    return out
