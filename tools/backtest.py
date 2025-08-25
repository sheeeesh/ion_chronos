"""
Baseline signal backtester for Ion Chronos (optimized for headless environments).

Features:
- Loads price data from workspace/{TICKER}_astro_dataset.csv if present; otherwise fetches via yfinance.
- Handles suffixed price columns (e.g., 'Close_SPY' → 'Close') and enforces UTC timestamps.
- Supports daily or intraday bars; infers an annualization factor from timestamp spacing.
- Applies transaction costs & slippage on position changes.
- Saves outputs: equity/drawdown PNGs, metrics CSV, trades CSV, and a text summary.
- Appends a one-line summary to workspace/experiments.log for tracking.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

# Use a headless backend for plotting (no GUI required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.io_paths import WORKSPACE

# ----------------------------- Data I/O Helpers -----------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present (e.g., from yfinance)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(x) for x in tup if x not in ("", None)).strip("_")
            for tup in df.columns.to_list()
        ]
    return df

def _ensure_utc_index(df: pd.DataFrame, name: str = "Date") -> pd.DataFrame:
    """Ensure the DataFrame index is a UTC DatetimeIndex with a given name."""
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
    """Find a column whose normalized name starts with any of `targets`."""
    target_norms = [_norm_name(t) for t in targets]
    for col in df.columns:
        col_norm = _norm_name(col)
        if any(col_norm.startswith(tn) for tn in target_norms):
            return col
    return None

def _ensure_standard_price_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    If price columns only exist with suffixes (e.g., 'Close_SPY'), add standard columns 
    (Close, Open, High, Low, Adj Close, Volume) for compatibility.
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

def _read_workspace_csv(ticker: str) -> Optional[pd.DataFrame]:
    """Read an existing astro dataset CSV from the workspace (if available)."""
    path = os.path.join(WORKSPACE, f"{ticker}_astro_dataset.csv")
    if not os.path.exists(path):
        return None
    header_df = pd.read_csv(path, nrows=1)
    cols = list(header_df.columns)
    date_col = next((c for c in cols if c.lower() in ("date", "datetime", "timestamp")), cols[0])
    df = pd.read_csv(path, parse_dates=[date_col])
    ts = df[date_col]
    df[date_col] = ts.dt.tz_localize("UTC") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("UTC")
    df = df.set_index(date_col)
    df = _ensure_utc_index(_flatten_columns(df))
    df = _ensure_standard_price_aliases(df)
    return df

def _fetch_prices_yf(ticker: str, start_date: str, end_date: Optional[str], timeframe: str = "1d") -> pd.DataFrame:
    """Fetch price data from yfinance for the given period."""
    import yfinance as yf
    params = dict(interval=timeframe, auto_adjust=True, progress=False)
    df = yf.download(ticker, start=start_date, end=end_date, **params)
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker} [{start_date}..{end_date or 'present'} @ {timeframe}].")
    df = _ensure_utc_index(_flatten_columns(df))
    df = _ensure_standard_price_aliases(df)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in df.columns:
            df[c] = df[c].astype("float32", copy=False)
    if "Return" not in df.columns:
        base = "Close" if "Close" in df.columns else _find_prefixed_col(df, ["Adj Close"])
        if base is None:
            raise ValueError("Missing Close/Adj Close for returns calculation.")
        df["Return"] = df[base].pct_change().fillna(0.0).astype("float32")
    return df

def _load_price_frame(ticker: str, start_date: str, end_date: Optional[str], timeframe: str = "1d") -> pd.DataFrame:
    """Load price data from workspace or fetch via yfinance, then filter to [start_date, end_date]."""
    df = _read_workspace_csv(ticker)
    if df is None:
        df = _fetch_prices_yf(ticker, start_date, end_date, timeframe=timeframe)
    else:
        df = _ensure_standard_price_aliases(df)
        if "Return" not in df.columns:
            base = "Close" if "Close" in df.columns else _find_prefixed_col(df, ["Adj Close"])
            if base is None:
                raise ValueError("Dataset missing Close/Adj Close to compute returns.")
            df["Return"] = df[base].pct_change().fillna(0.0).astype("float32")
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True) if end_date else df.index.max()
    df = df.loc[start_ts:end_ts].copy()
    required_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Return"] if c in df.columns]
    df = df.dropna(subset=required_cols).sort_index()
    return df

# ----------------------------- Metrics & Plotting -----------------------------

def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """Estimate bars-per-day from median spacing, then derive annualization factor."""
    if len(index) < 2:
        return 252.0
    diffs_ns = np.diff(index.asi8.astype(np.int64))
    median_s = float(np.median(diffs_ns)) / 1e9
    if median_s <= 0:
        return 252.0
    bars_per_day = 86400.0 / median_s
    return bars_per_day * 365.25

def _metrics(equity: pd.Series, per_step: pd.Series, ann_factor: float) -> Dict[str, float]:
    total_ret = float(equity.iloc[-1] - 1.0)
    mu = float(per_step.mean())
    sigma = float(per_step.std(ddof=0))
    cagr = (1 + mu) ** ann_factor - 1 if sigma == sigma else np.nan  # avoid NaN comparison issues
    ann_vol = sigma * np.sqrt(ann_factor) if sigma == sigma else np.nan
    sharpe = (mu / sigma) * np.sqrt(ann_factor) if sigma > 0 else np.nan
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    dd_duration = int((end - start).days) if isinstance(end, pd.Timestamp) else np.nan
    return {
        "total_return": total_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "dd_duration_days": dd_duration,
    }

def _save_plots(equity: pd.Series, out_prefix: str) -> None:
    """Save equity curve and drawdown plots to files (PNG)."""
    equity_path = f"{out_prefix}_equity.png"
    dd_path = f"{out_prefix}_drawdown.png"
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Equity Curve"); plt.xlabel("Time"); plt.ylabel("Equity (start=1.0)")
    plt.tight_layout(); plt.savefig(equity_path); plt.close()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    plt.figure(figsize=(10, 4))
    drawdown.plot()
    plt.title("Drawdown"); plt.xlabel("Time"); plt.ylabel("Drawdown")
    plt.tight_layout(); plt.savefig(dd_path); plt.close()

# ----------------------------- Strategies -----------------------------

def _signal_buy_and_hold(index: pd.DatetimeIndex) -> pd.Series:
    """Trivial strategy: always long (buy-and-hold)."""
    return pd.Series(1.0, index=index, dtype="float32")

def _signal_ma_cross(close: pd.Series, fast: int = 10, slow: int = 50, lag: int = 1) -> pd.Series:
    """Moving average crossover strategy signal (1.0 when fast MA > slow MA, else 0.0)."""
    fast = int(fast); slow = int(slow); lag = int(lag)
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    signal = (sma_fast > sma_slow).astype("float32")
    if lag > 0:
        signal = signal.shift(lag).fillna(0.0)
    return signal

def _build_positions(df: pd.DataFrame, strategy: str, **kwargs) -> pd.Series:
    """Generate a position time series (0.0 or 1.0) for the given strategy."""
    strat = (strategy or "ma_cross").lower()
    if strat == "buy_and_hold":
        return _signal_buy_and_hold(df.index)
    if strat == "ma_cross":
        close_series = df["Close"] if "Close" in df.columns else df["Adj Close"]
        fast = int(kwargs.get("fast", 10))
        slow = int(kwargs.get("slow", 50))
        lag = int(kwargs.get("lag", 1))
        return _signal_ma_cross(close_series, fast=fast, slow=slow, lag=lag)
    raise ValueError(f"Unknown strategy '{strategy}'. Use 'ma_cross' or 'buy_and_hold'.")

# ----------------------------- Backtest Engine -----------------------------

def _run_backtest(df: pd.DataFrame, positions: pd.Series,
                 cost: float = 0.0005, slippage: float = 0.0005) -> Tuple[pd.Series, pd.Series, pd.DataFrame, Dict[str, float]]:
    """Run the core backtest logic given price DataFrame and position series."""
    positions = positions.reindex(df.index).fillna(0.0).astype("float32")
    returns = df["Return"].astype("float32")
    position_changes = positions.diff().fillna(positions.iloc[0])
    trade_costs = (cost + slippage) * position_changes.abs()
    step_returns = positions * returns - trade_costs
    equity = (1.0 + step_returns).cumprod()
    equity.index = df.index
    # Build trade list for reporting
    trades_list: List[Dict] = []
    in_trade = False
    entry_time = None
    entry_price = None
    close_series = df["Close"] if "Close" in df.columns else df["Adj Close"]
    for t in df.index:
        delta_pos = float(position_changes.loc[t])
        if not in_trade and delta_pos > 0:
            in_trade = True
            entry_time = t
            entry_price = float(close_series.loc[t])
        elif in_trade and delta_pos < 0:
            exit_time = t
            exit_price = float(close_series.loc[t])
            gross_ret = (exit_price / max(entry_price, 1e-12)) - 1.0
            net_ret = gross_ret - 2.0 * (cost + slippage)
            trades_list.append({
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_ret,
                "net_return": net_ret,
            })
            in_trade = False
            entry_time = None
            entry_price = None
    trades_df = pd.DataFrame(trades_list)
    win_rate = float((trades_df["net_return"] > 0).mean()) if not trades_df.empty else np.nan
    n_trades = int(len(trades_df))
    turnover = float(position_changes.abs().sum())
    ann_factor = _annualization_factor(df.index)
    metrics = _metrics(equity, step_returns, ann_factor)
    metrics.update({
        "win_rate": win_rate,
        "trades": n_trades,
        "turnover": turnover,
        "ann_factor": ann_factor,
    })
    return equity, step_returns, trades_df, metrics

# ----------------------------- Public API -----------------------------

def backtest_signal(ticker: str,
                    start_date: str,
                    end_date: Optional[str] = None,
                    strategy: str = "ma_cross",
                    cost: float = 0.0005,
                    slippage: float = 0.0005,
                    timeframe: str = "1d",
                    **kwargs) -> str:
    """
    Run a simple backtest for the given ticker over the specified date range.

    If workspace/{ticker}_astro_dataset.csv exists, it will be used as the data source.
    Otherwise, price data is fetched via yfinance (using the specified timeframe).

    Outputs:
      - {ticker}_{strategy}_equity.png (equity curve plot)
      - {ticker}_{strategy}_drawdown.png (drawdown plot)
      - {ticker}_{strategy}_metrics.csv (CSV of performance metrics)
      - {ticker}_{strategy}_trades.csv (CSV of trade entries/exits)
      - {ticker}_{strategy}_summary.txt (text summary of results)
      - Appends one line to workspace/experiments.log summarizing the run.

    Returns:
        A summary string listing the generated artifact paths.
    """
    os.makedirs(WORKSPACE, exist_ok=True)
    df = _load_price_frame(ticker, start_date, end_date, timeframe=timeframe)
    if df.empty:
        raise ValueError("No data available for backtesting after applying date range.")
    positions = _build_positions(df, strategy=strategy, **kwargs)
    equity, step_returns, trades_df, metrics = _run_backtest(df, positions, cost=cost, slippage=slippage)
    strat = strategy.lower()
    prefix_abs = os.path.join(WORKSPACE, f"{ticker}_{strat}")
    prefix_rel = f"workspace/{ticker}_{strat}"
    _save_plots(equity, prefix_abs)
    # Save metrics and trades
    pd.DataFrame([metrics]).to_csv(f"{prefix_abs}_metrics.csv", index=False)
    trades_df.to_csv(f"{prefix_abs}_trades.csv", index=False)
    # Save summary text
    with open(f"{prefix_abs}_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Backtest Summary — {ticker} [{df.index.min()} → {df.index.max()}]\n"
            f"Strategy: {strategy}\n"
            f"Params: {kwargs}\n"
            f"Costs: cost={cost}, slippage={slippage}\n\n"
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"CAGR: {metrics['cagr']:.2%}\n"
            f"Ann.Vol: {metrics['ann_vol']:.2%}\n"
            f"Sharpe (rf=0): {metrics['sharpe']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"DD Duration (days): {metrics['dd_duration_days']}\n"
            f"Win rate: {metrics.get('win_rate', np.nan):.2%}\n"
            f"Trades: {metrics.get('trades', 0)}\n"
            f"Turnover: {metrics.get('turnover', 0):.4f}\n"
            f"Ann.Factor: {metrics.get('ann_factor', np.nan):.1f}\n"
        )
    # Append summary to experiments log
    try:
        with open(os.path.join(WORKSPACE, "experiments.log"), "a", encoding="utf-8") as log:
            log.write(
                f"{pd.Timestamp.utcnow()} - backtest {ticker} {strategy} "
                f"start={df.index.min()} end={df.index.max()} "
                f"cost={cost} slip={slippage} "
                f"-> ret={metrics['total_return']:.4f} sharpe={metrics['sharpe']:.3f} "
                f"mdd={metrics['max_drawdown']:.4f} trades={metrics.get('trades', 0)} "
                f"win={metrics.get('win_rate', np.nan):.3f}\n"
            )
    except Exception:
        pass
    return (
        "Backtest complete. "
        f"Metrics: {prefix_rel}_metrics.csv | Trades: {prefix_rel}_trades.csv | "
        f"Equity/Drawdown: {prefix_rel}_equity.png, {prefix_rel}_drawdown.png | "
        f"Summary: {prefix_rel}_summary.txt"
    )
