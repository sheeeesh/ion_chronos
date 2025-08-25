# tools/astro_dataset.py
"""
Build an astro-enhanced price dataset for a symbol and date range.

- Fetches OHLCV via yfinance with UTC index.
- Merges Swiss Ephemeris features (tropical/geocentric), incl. Placidus houses & aspects.
- Normalizes price column names (adds standard aliases like 'Close' if only 'Close_SPY' exists).
- Saves a stable CSV schema to workspace/{TICKER}_astro_dataset.csv

If the astro engine is unavailable, a consistent set of astro columns is created and
filled with NaN so the schema stays stable for RL/backtests.
"""
from __future__ import annotations

import os
import re
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

# Prefer a real implementation if you have one:
#   from tools.astro_features import compute_astro_features
# Otherwise this stub is used — and we ensure a stable column schema.
try:
    from tools.astro_features import compute_astro_features as _real_compute_astro_features  # type: ignore
    _HAS_REAL_ASTRO = True
except Exception:
    _HAS_REAL_ASTRO = False
    _real_compute_astro_features = None  # type: ignore


# ============================== helpers ==============================

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure columns aren’t MultiIndex (yfinance sometimes returns them)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(x) for x in tup if x not in (None, "")]).strip("_")
            for tup in df.columns.to_list()
        ]
    return df


def _ensure_utc_index(df: pd.DataFrame, name: str = "Date") -> pd.DataFrame:
    """Force a single-level UTC DatetimeIndex with a consistent name."""
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
    """Return a column whose normalized name starts with any target (handles Close_SPY, Adj Close-BTC, etc.)."""
    tnorms = [_norm_name(t) for t in targets]
    for c in df.columns:
        cn = _norm_name(c)
        for tn in tnorms:
            if cn.startswith(tn):
                return c
    return None


def _ensure_standard_price_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    If price columns only exist with suffixes (e.g., Close_SPY), mirror them to standard names
    (Close/Open/High/Low/Adj Close/Volume) so downstream code is happy.
    """
    df = df.copy()
    # Close or Adj Close
    close_like = _find_prefixed_col(df, ["Close", "Adj Close"])
    if close_like and "Close" not in df.columns:
        df["Close"] = df[close_like]
    # Others (best-effort)
    for base in ["Open", "High", "Low", "Adj Close", "Volume"]:
        like = _find_prefixed_col(df, [base])
        if like and base not in df.columns:
            df[base] = df[like]
    return df


# ============================== astro schema ==============================

_PLANETS = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node"]
_ASPECTS = ["conj", "sext", "sq", "tri", "opp"]  # conjunction/sextile/square/trine/opposition

def _standard_astro_columns() -> List[str]:
    cols: List[str] = []
    # planet positions, speeds, retro flags, sin/cos of longitude
    for p in _PLANETS:
        cols += [
            f"astro_lon_{p}_deg",
            f"astro_speed_{p}_degpd",
            f"astro_retro_{p}",
            f"astro_sin_lon_{p}",
            f"astro_cos_lon_{p}",
        ]
    # lunar phase
    cols += ["astro_phase_deg", "astro_phase_sin", "astro_phase_cos"]
    # aspects: sun↔planet, moon↔planet
    for left in ("sun", "moon"):
        for right in [p for p in _PLANETS if p not in (left, "sun")]:
            for a in _ASPECTS:
                cols.append(f"astro_asp_{left}_{right}_{a}")   # boolean-ish
                cols.append(f"astro_aspd_{left}_{right}_{a}")  # angular distance from exact
    # Placidus houses (1..12), asc & mc
    cols += [f"astro_house_cusp_{i}_deg" for i in range(1, 13)]
    cols += ["astro_asc_deg", "astro_mc_deg"]
    return cols


def _empty_astro_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    cols = _standard_astro_columns()
    out = pd.DataFrame(index=index, columns=cols, dtype="float32")
    return out  # NaNs everywhere; schema is stable


def _compute_astro_features_safe(
    index: pd.DatetimeIndex,
    lat: float,
    lon: float,
    house_system: str,
    orb_deg: float,
    cache_path: Optional[str],
) -> pd.DataFrame:
    """
    Try to compute astro features; on failure, return a stable NaN-filled schema.
    If a real implementation returns a subset of standard columns, we add the missing ones.
    """
    if _HAS_REAL_ASTRO and callable(_real_compute_astro_features):
        try:
            astro = _real_compute_astro_features(
                index=index,
                lat=lat,
                lon=lon,
                house_system=house_system,
                orb_deg=orb_deg,
                cache_path=cache_path,
            )
            astro = _flatten_columns(astro)
            astro = astro.copy()
            # Ensure float32 and UTC index
            for c in astro.columns:
                if np.issubdtype(np.asarray(astro[c]).dtype, np.number):
                    astro[c] = astro[c].astype("float32", copy=False)
            astro = _ensure_utc_index(astro, name="Date")
            # Add missing standard columns to keep schema stable
            for col in _standard_astro_columns():
                if col not in astro.columns:
                    astro[col] = np.nan
            return astro.reindex(index)
        except Exception:
            # Fall through to NaN frame
            pass
    return _empty_astro_frame(index)


# ============================== prices ==============================

def _fetch_prices(
    ticker: str,
    start_date: str,
    end_date: Optional[str],
    timeframe: str,
) -> pd.DataFrame:
    """
    Fetch price data via yfinance. timeframe like '1d', '5m', '15m', '1h', etc.
    Enforces UTC tz-aware index and float32 numeric cols. Adds 'Return'.
    """
    kw = dict(interval=timeframe, auto_adjust=True, progress=False)
    if end_date:
        df = yf.download(ticker, start=start_date, end=end_date, **kw)
    else:
        df = yf.download(ticker, start=start_date, **kw)

    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker} [{start_date}..{end_date or 'present'} @ {timeframe}]")

    # Normalize/clean
    df = _flatten_columns(df)
    df = _ensure_utc_index(df)  # name='Date'
    df = _ensure_standard_price_aliases(df)

    # Standard numeric dtypes
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = df[c].astype("float32", copy=False)

    # 'Return' column
    if "Return" not in df.columns:
        base = "Close" if "Close" in df.columns else _find_prefixed_col(df, ["Adj Close"])
        if base is None:
            raise ValueError("Dataset must include Close or Adj Close. "
                             f"Found: {list(df.columns)}")
        df["Return"] = df[base].pct_change().fillna(0.0).astype("float32")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close", "Return"]).sort_index()
    return df


# ============================== public API ==============================

def build_astro_dataset(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    timeframe: str = "1d",
    lat: float = 40.7128,        # default NYC
    lon: float = -74.0060,
    orb_deg: float = 3.0,
    cache_parquet: Optional[str] = None,
) -> str:
    """
    Build & save an astro-enhanced dataset.

    Args:
      ticker, start_date, end_date, timeframe: price data selection
      lat, lon: location for Placidus houses
      orb_deg: orb threshold for major aspects
      cache_parquet: optional cache path for astro features (parquet)

    Returns:
      str: message with output path
    """
    os.makedirs("workspace", exist_ok=True)

    # 1) Prices
    df = _fetch_prices(ticker, start_date, end_date, timeframe)

    # 2) Astro features (Swiss Ephemeris tropical/geocentric + Placidus/Aspects)
    astro = _compute_astro_features_safe(
        index=df.index,
        lat=lat,
        lon=lon,
        house_system="P",  # Placidus
        orb_deg=orb_deg,
        cache_path=cache_parquet,
    )

    # 3) Join and finalize (avoid MultiIndex; align strictly by index)
    df = _flatten_columns(df)
    astro = _flatten_columns(astro)
    df = _ensure_utc_index(df, name="Date").sort_index()
    astro = _ensure_utc_index(astro, name="Date").sort_index()

    astro = astro.reindex(df.index)  # 1:1 alignment to prices
    out = pd.concat([df, astro], axis=1)

    # Ensure standard price aliases exist even after merge
    out = _ensure_standard_price_aliases(out)

    # Final cleanup
    assert not isinstance(out.index, pd.MultiIndex), "Index should be single-level"
    out.index.name = "Date"

    out_path = f"workspace/{ticker}_astro_dataset.csv"
    # Use ISO-like format; tz-aware offset is preserved (%z)
    out.to_csv(out_path, date_format="%Y-%m-%d %H:%M:%S%z")
    return f"Astro dataset written: {out_path} (rows={len(out)}, cols={len(out.columns)})"
