"""
Build an astro-enhanced price dataset for a given symbol and date range.

- Fetches OHLCV via yfinance with a UTC index.
- Merges Swiss Ephemeris features (tropical/geocentric), including Placidus houses & aspects.
- Normalizes price column names (adds standard aliases like 'Close' if only suffixed names exist).
- Saves a stable CSV schema to workspace/{TICKER}_astro_dataset.csv.

If the astro engine is unavailable, returns the same columns filled with NaN to keep the schema stable for ML/RL pipelines.
"""
from __future__ import annotations

import os
import re
from typing import Optional, List


from tools.io_paths import WORKSPACE

import numpy as np
import pandas as pd
import yfinance as yf

from tools.io_paths import WORKSPACE

# Try to import real astro_features implementation if available (Swiss Ephemeris)
try:
    from tools.astro_features import compute_astro_features as _real_compute_astro_features  # type: ignore
    _HAS_REAL_ASTRO = True
except Exception:
    _HAS_REAL_ASTRO = False
    _real_compute_astro_features = None  # type: ignore

# ------------------------------ Helpers ------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns are not a MultiIndex (flatten MultiIndex if present)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(x) for x in tup if x not in ("", None)).strip("_")
            for tup in df.columns.to_list()
        ]
    return df

def _ensure_utc_index(df: pd.DataFrame, name: str = "Date") -> pd.DataFrame:
    """Ensure a single-level UTC DatetimeIndex with a given name."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    df.index.name = name
    return df

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())

def _find_prefixed_col(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
    """Return a column whose normalized name starts with any target (e.g., 'Close_SPY')."""
    target_norms = [_norm_name(t) for t in targets]
    for col in df.columns:
        if any(_norm_name(col).startswith(tn) for tn in target_norms):
            return col
    return None

def _ensure_standard_price_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    If price columns only exist with suffixes (e.g., 'Close_SPY'), mirror them to standard names
    ('Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume').
    """
    df = df.copy()
    close_like = _find_prefixed_col(df, ["Close", "Adj Close"])
    if close_like and "Close" not in df.columns:
        df["Close"] = df[close_like]
    for base in ["Open", "High", "Low", "Adj Close", "Volume"]:
        like = _find_prefixed_col(df, [base])
        if like and base not in df.columns:
            df[base] = df[like]
    return df

# ------------------------------ Astro Schema ------------------------------

_PLANETS = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
_ASPECTS = ["conj", "sext", "sq", "tri", "opp"]  # major aspects: conjunction, sextile, square, trine, opposition

def _standard_astro_columns() -> List[str]:
    cols: List[str] = []
    # Planet positions, speeds, retrograde flags, sin/cos of longitude
    for p in _PLANETS:
        cols += [
            f"astro_lon_{p}_deg",
            f"astro_speed_{p}_degpd",
            f"astro_retro_{p}",
            f"astro_sin_lon_{p}",
            f"astro_cos_lon_{p}",
        ]
    # Lunar phase
    cols += ["astro_phase_deg", "astro_phase_sin", "astro_phase_cos"]
    # Aspects: Sun/Moon vs other planets
    for left in ("sun", "moon"):
        for right in [p for p in _PLANETS if p not in (left, "sun")]:
            for aspect in _ASPECTS:
                cols.append(f"astro_asp_{left}_{right}_{aspect}")
                cols.append(f"astro_aspd_{left}_{right}_{aspect}")
    # Placidus house cusps (1–12), plus Ascendant and Midheaven
    cols += [f"astro_house_cusp_{i}_deg" for i in range(1, 13)]
    cols += ["astro_asc_deg", "astro_mc_deg"]
    return cols

def _empty_astro_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a DataFrame of NaN with the standard astro columns, indexed by `index`."""
    return pd.DataFrame(index=index, columns=_standard_astro_columns(), dtype="float32")

def _compute_astro_features_safe(index: pd.DatetimeIndex,
                                 lat: float,
                                 lon: float,
                                 house_system: str,
                                 orb_deg: float,
                                 cache_path: Optional[str]) -> pd.DataFrame:
    """
    Compute astro features (Swiss Ephemeris) for the given index, or return a stable NaN-filled frame on failure.
    If a real implementation returns only a subset of columns, fill missing standard columns with NaN.
    """
    if _HAS_REAL_ASTRO and callable(_real_compute_astro_features):
        try:
            astro = _real_compute_astro_features(index=index, lat=lat, lon=lon,
                                                house_system=house_system, orb_deg=orb_deg,
                                                cache_path=cache_path)
            astro = _ensure_utc_index(_flatten_columns(astro), name="Date").copy()
            # Ensure float32 dtype and add missing columns
            for c in astro.columns:
                if np.issubdtype(np.asarray(astro[c]).dtype, np.number):
                    astro[c] = astro[c].astype("float32", copy=False)
            for col in _standard_astro_columns():
                if col not in astro.columns:
                    astro[col] = np.nan
            return astro.reindex(index)
        except Exception:
            # On any failure, fall through to NaN frame
            pass
    return _empty_astro_frame(index)

# ------------------------------ Prices ------------------------------

def _fetch_prices(ticker: str,
                  start_date: str,
                  end_date: Optional[str],
                  timeframe: str) -> pd.DataFrame:
    """
    Fetch price data via yfinance for the given ticker and date range.
    The `timeframe` can be '1d', '5m', '15m', '1h', etc.
    Ensures a UTC tz-aware index and float32 numeric columns. Adds a 'Return' column.
    """
    params = dict(interval=timeframe, auto_adjust=True, progress=False)
    df = yf.download(ticker, start=start_date, end=end_date, **params)
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker} [{start_date}..{end_date or 'present'} @ {timeframe}]")
    df = _ensure_utc_index(_flatten_columns(df))
    df = _ensure_standard_price_aliases(df)
    # Set standard numeric dtypes
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = df[c].astype("float32", copy=False)
    # Add 'Return' column if missing
    if "Return" not in df.columns:
        base = "Close" if "Close" in df.columns else _find_prefixed_col(df, ["Adj Close"])
        if base is None:
            raise ValueError("Dataset must include 'Close' or 'Adj Close' to compute returns.")
        df["Return"] = df[base].pct_change().fillna(0.0).astype("float32")
    # Drop rows with missing price/return and sort
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close", "Return"]).sort_index()

# ------------------------------ Public API ------------------------------

def build_astro_dataset(ticker: str,
                        start_date: str,
                        end_date: Optional[str] = None,
                        timeframe: str = "1d",
                        lat: float = 40.7128,    # default latitude (NYC)
                        lon: float = -74.0060,   # default longitude (NYC)
                        orb_deg: float = 3.0,
                        cache_parquet: Optional[str] = None) -> str:
    """
    Build and save an astro-enhanced dataset for the given ticker and date range.

    Args:
        ticker: Symbol or ticker (e.g., 'BTC-USD', 'SPY').
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD) or None for present.
        timeframe: Bar interval (e.g., '1d', '5m', '1h').
        lat, lon: Location coordinates for house calculations (Placidus houses).
        orb_deg: Orb threshold (degrees) for major aspects.
        cache_parquet: Optional path to a parquet file for caching astro features.

    Returns:
        A message indicating the output path and dataset dimensions.
    """
    os.makedirs(WORKSPACE, exist_ok=True)
    # 1) Fetch price data
    df = _fetch_prices(ticker, start_date, end_date, timeframe)
    if df.empty:
        raise ValueError("No data retrieved for the given range.")
    # 2) Compute astro features or use placeholders
    astro = _compute_astro_features_safe(index=df.index, lat=lat, lon=lon,
                                        house_system="P", orb_deg=orb_deg,
                                        cache_path=cache_parquet)
    # 3) Merge price data with astro features
    df = _ensure_utc_index(_flatten_columns(df), name="Date").sort_index()
    astro = _ensure_utc_index(_flatten_columns(astro), name="Date").sort_index()
    astro = astro.reindex(df.index)
    out = pd.concat([df, astro], axis=1)
    out = _ensure_standard_price_aliases(out)
    out.index.name = "Date"
    # Save to CSV
    out_path = os.path.join(WORKSPACE, f"{ticker}_astro_dataset.csv")
    out.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
    message = (
        f"Astro dataset written: {os.path.relpath(out_path)} "
        f"(rows={len(out)}, cols={len(out.columns)})"
    )
    # Log action to experiments.log
    try:
        with open(os.path.join(WORKSPACE, "experiments.log"), "a", encoding="utf-8") as log:
            log.write(f"{pd.Timestamp.utcnow()} - astro_dataset {ticker} [{start_date} → {end_date or 'present'}] rows={len(out)}\n")
    except Exception:
        pass
    return message
