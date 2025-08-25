# tools/pipeline.py
"""
Ion Chronos — One-shot RL + Astrology pipeline

Steps:
1) Build astro-enhanced dataset (Swiss Ephemeris features; tropical/geocentric, Placidus houses)
2) Train PPO RL agent (with OOS eval) on price + astro features
3) (Optional) Run a baseline backtest for comparison (e.g., ma_cross)

Artifacts (by ticker, under workspace/):
- {T}_astro_dataset.csv
- {T}_ppo_model.zip
- {T}_rl_eval.csv, {T}_rl_summary.txt, {T}_trades.csv
- {T}_equity.png, {T}_drawdown.png
- (optional baseline) {T}_ma_cross_metrics.csv, {T}_ma_cross_trades.csv, plots, summary
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from tools.io_paths import WORKSPACE

from tools.astro_dataset import build_astro_dataset
from tools.rl_train import train_rl_agent
from tools.backtest import backtest_signal


# ------------------------- Helpers -------------------------


def _resolve_start_date(start_date: Optional[str], end_date: Optional[str], period: Optional[str]) -> str:
    """
    If start_date is missing, compute it from `period` relative to today (UTC).
    Accepts 'Nd', 'Nmo'/'Nm', 'Ny', or 'max'. Defaults to 5y if unparseable.
    """
    if start_date:
        return start_date

    today = dt.date.today()
    per = (period or "5y").strip().lower()

    # 'max' → 1970-01-01
    if per in ("max", "all"):
        return "1970-01-01"

    m = re.match(r"^\s*(\d+)\s*([a-z]+)\s*$", per)
    if not m:
        # fallback 5 years
        years = 5
        try:
            return today.replace(year=max(1970, today.year - years)).isoformat()
        except ValueError:
            return (today.replace(year=max(1970, today.year - years)) - dt.timedelta(days=1)).isoformat()

    n = int(m.group(1))
    unit = m.group(2)
    unit = {
        "mos": "m", "mo": "m", "mon": "m", "month": "m", "months": "m",
        "yrs": "y", "yr": "y", "year": "y", "years": "y",
        "d": "d", "day": "d", "days": "d",
        "m": "m", "y": "y",
    }.get(unit, unit)

    if unit == "y":
        year = max(1970, today.year - n)
        try:
            return today.replace(year=year).isoformat()
        except ValueError:
            return (today.replace(year=year) - dt.timedelta(days=1)).isoformat()
    elif unit == "m":
        year = today.year
        month = today.month - n
        while month <= 0:
            month += 12
            year -= 1
        day = min(today.day, 28)  # keep it simple for EOM
        return dt.date(year, month, day).isoformat()
    else:
        # days
        return (today - dt.timedelta(days=n)).isoformat()


# ------------------------- Schema -------------------------


class PipelineArgs(BaseModel):
    # Dataset
    ticker: str = Field(..., description="Ticker symbol, e.g., BTC-USD or SPY")
    # UPDATED: dates optional; `period` used if start_date not provided
    start_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD (optional if 'period' provided)")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD (optional)")
    period: Optional[str] = Field("5y", description="Used if start_date missing, e.g. '3mo', '1y', '5y', '30d', 'max'")
    timeframe: str = Field("1d", description="Bar interval, e.g., '1d', '5m', '15m', '1h'")
    lat: float = Field(40.7128, description="Latitude for Placidus houses (default NYC)")
    lon: float = Field(-74.0060, description="Longitude for Placidus houses (default NYC)")
    orb_deg: float = Field(3.0, description="Aspect orb in degrees")
    cache_parquet: Optional[str] = Field(None, description="Optional parquet cache path for astro features")

    # RL (mirrors tools/rl_train.py)
    total_timesteps: int = Field(200_000, description="PPO training steps")
    window_size: int = Field(30, description="Observation window length")
    cost: float = Field(0.0005, description="Transaction cost per position change")
    slippage: float = Field(0.0005, description="Slippage cost per position change")
    continuous: bool = Field(False, description="Use continuous action space [-1,1]")
    n_envs: int = Field(1, description="Number of vectorized envs")
    use_vecnorm: bool = Field(True, description="Enable VecNormalize for obs/reward")
    lr: float = Field(3e-4, description="Learning rate (start of linear schedule)")
    ent_coef: float = Field(0.01, description="Entropy bonus coefficient")
    gamma: float = Field(0.99, description="Discount factor")
    seed: int = Field(42, description="Random seed")
    max_trades_per_day: int = Field(30, description="Daily trade cap")
    day_stop_dd: float = Field(0.02, description="Daily stop after 2% drawdown")
    train_split: float = Field(0.70, ge=0.5, le=0.95, description="Train/OOS split fraction")
    extra_timesteps_after: int = Field(0, description="Extra PPO steps to continue after main training")
    resume_from: Optional[str] = Field(None, description="Path to an existing model.zip to resume training")

    # Baseline
    run_baseline: bool = Field(True, description="Also run a baseline backtest for comparison")
    baseline_strategy: str = Field("ma_cross", description="Baseline strategy (e.g., 'ma_cross' or 'buy_and_hold')")


# ------------------------- Pipeline -------------------------


def run_rl_astro_pipeline(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = "5y",
    timeframe: str = "1d",
    lat: float = 40.7128,
    lon: float = -74.0060,
    orb_deg: float = 3.0,
    cache_parquet: Optional[str] = None,
    total_timesteps: int = 200_000,
    window_size: int = 30,
    cost: float = 0.0005,
    slippage: float = 0.0005,
    continuous: bool = False,
    n_envs: int = 1,
    use_vecnorm: bool = True,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    seed: int = 42,
    max_trades_per_day: int = 30,
    day_stop_dd: float = 0.02,
    train_split: float = 0.70,
    extra_timesteps_after: int = 0,
    resume_from: Optional[str] = None,
    run_baseline: bool = True,
    baseline_strategy: str = "ma_cross",
) -> str:
    """
    Execute the full pipeline and return a human-readable summary string.
    If start_date is not provided, a start date is derived from `period` (default '5y').
    """
    msgs = []

    # Resolve date range
    sd = _resolve_start_date(start_date, end_date, period)
    ed = end_date
    used_range = f"{sd} → {ed or 'present'} @ {timeframe}"
    msgs.append(f"[range] {used_range}")

    # 1) Dataset
    try:
        ds_msg = build_astro_dataset(
            ticker=ticker,
            start_date=sd,
            end_date=ed,
            timeframe=timeframe,
            lat=lat,
            lon=lon,
            orb_deg=orb_deg,
            cache_parquet=cache_parquet,
        )
        msgs.append(f"[dataset] {ds_msg}")
    except Exception as e:
        msgs.append(f"[dataset:ERROR] {e}")
        return "\n".join(msgs)  # Bail out early if dataset fails

    # 2) RL training + OOS eval
    try:
        rl_msg = train_rl_agent(
            ticker=ticker,
            total_timesteps=total_timesteps,
            window_size=window_size,
            cost=cost,
            slippage=slippage,
            continuous=continuous,
            max_trades_per_day=max_trades_per_day,
            day_stop_dd=day_stop_dd,
            n_envs=n_envs,
            use_vecnorm=use_vecnorm,
            lr=lr,
            ent_coef=ent_coef,
            gamma=gamma,
            seed=seed,
            train_split=train_split,
            extra_timesteps_after=extra_timesteps_after,
            resume_from=resume_from,
        )
        msgs.append(f"[rl] {rl_msg}")
    except Exception as e:
        msgs.append(f"[rl:ERROR] {e}")

    # 3) Optional baseline
    if run_baseline:
        try:
            bt_msg = backtest_signal(
                ticker=ticker,
                start_date=sd,
                end_date=ed,
                strategy=baseline_strategy,
                cost=cost,
                slippage=slippage,
            )
            msgs.append(f"[baseline] {bt_msg}")
        except Exception as e:
            msgs.append(f"[baseline:ERROR] {e}")

    # 4) Quick artifact recap
    root = WORKSPACE
    recap = f"[artifacts] (root: {root})\n"
    artifacts = [
        f"{ticker}_astro_dataset.csv",
        f"{ticker}_ppo_model.zip",
        f"{ticker}_rl_eval.csv",
        f"{ticker}_rl_summary.txt",
        f"{ticker}_trades.csv",
        f"{ticker}_equity.png",
        f"{ticker}_drawdown.png",
    ]
    for art in artifacts:
        recap += f" - {os.path.relpath(os.path.join(root, art), root)}\n"
    
    if run_baseline:
        base_artifacts = [
            f"{ticker}_{baseline_strategy}_metrics.csv",
            f"{ticker}_{baseline_strategy}_trades.csv",
            f"{ticker}_{baseline_strategy}_equity.png",
            f"{ticker}_{baseline_strategy}_drawdown.png",
            f"{ticker}_{baseline_strategy}_summary.txt",
        ]
        for art in base_artifacts:
            recap += f" - {os.path.relpath(os.path.join(root, art), root)}\n"
    msgs.append(recap)

    return "\n".join(msgs)
