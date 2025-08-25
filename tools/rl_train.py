# tools/rl_train.py
"""
Ion Chronos — Reinforcement Learning Trainer (PPO), optimized

Changes:
- Headless Matplotlib backend (no Tk errors).
- Robust column mapping (Close_SPY → Close), adds Return if missing.
- Broad astro-feature detection (astro_, planets, aspects, houses, etc.).
- Realistic env: cost + slippage, daily 2% stop, max 30 trades/day.
- Discrete {Flat, Long} or Continuous [-1,1] sizing.
- VecNormalize support; device=auto; linear LR schedule; net_arch tuned.
- Train/test split (OOS evaluation), resume + extra-timesteps options.
- Observation-shape assertions, stale VecNormalize auto-reset.
- NEW: Safe split logic so both train & test have ≥ (window+2) bars; otherwise fall back to
       overlap eval on the last window+2 bars to avoid index errors.
"""

from __future__ import annotations

import os
import re
import random
import datetime as dt
from typing import Optional, List, Dict, Tuple, Callable, Union

from tools.io_paths import WORKSPACE

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

# Torch / SB3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Headless plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# For inspecting saved VecNormalize to detect shape mismatch
import cloudpickle  # pip install cloudpickle


# ========================== Data Loading ==========================

def _read_csv_with_dates(path: str) -> pd.DataFrame:
    hdr = pd.read_csv(path, nrows=1)
    cols = list(hdr.columns)
    candidates = [c for c in cols if c.lower() in ("date", "datetime", "timestamp")]
    date_col = candidates[0] if candidates else cols[0]

    df = pd.read_csv(path, parse_dates=[date_col])
    ts = df[date_col]
    if getattr(ts.dt, "tz", None) is None:
        df[date_col] = ts.dt.tz_localize("UTC")
    else:
        df[date_col] = ts.dt.tz_convert("UTC")
    df = df.set_index(date_col).sort_index()
    return df


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def _find_prefixed_col(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
    tnorms = [_norm_name(t) for t in targets]
    for c in df.columns:
        cn = _norm_name(c)
        for tn in tnorms:
            if cn.startswith(tn):
                return c
    return None


def _ensure_standard_price_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close_like = _find_prefixed_col(df, ["Close", "Adj Close"])
    if close_like and "Close" not in df.columns:
        df["Close"] = df[close_like]
    for base in ["Open", "High", "Low", "Adj Close", "Volume"]:
        like = _find_prefixed_col(df, [base])
        if like and base not in df.columns:
            df[base] = df[like]
    return df


def _load_dataframe(ticker: str) -> pd.DataFrame:
    ws_path = os.path.join(WORKSPACE, f"{ticker}_astro_dataset.csv")
    if os.path.exists(ws_path):
        df = _read_csv_with_dates(ws_path)
    else:
        import yfinance as yf
        df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No data for {ticker}")
        if getattr(df.index, "tz", None) is None:
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            df.index = df.index.tz_convert("UTC")

    df = _ensure_standard_price_aliases(df)

    lower = {c.lower(): c for c in df.columns}
    close_col = lower.get("close") or lower.get("adj close") or _find_prefixed_col(df, ["Close", "Adj Close"])
    if close_col is None:
        raise ValueError(f"Dataset must include Close or Adj Close. Found: {list(df.columns)}")

    if "Return" not in df.columns:
        df["Return"] = df[close_col].pct_change().fillna(0.0)

    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Return"]:
        if c in df.columns:
            df[c] = df[c].astype("float32", copy=False)

    df = df.dropna(subset=["Close", "Return"]).sort_index()
    return df


# ========================== Features ==========================

_ASTRO_PREFIXES = (
    "astro_",
    "phase",
    "sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto", "node",
    "house", "asc", "mc",
    "aspect", "asp", "aspd",
    "retro", "speed", "zodiac", "sign", "eclipse", "ingress",
)


def _is_astro_col(name: str) -> bool:
    cl = name.lower()
    if cl.startswith("astro_"):
        return True
    if cl in ("open", "high", "low", "close", "adj close", "volume", "return"):
        return False
    return any(cl.startswith(p) for p in _ASTRO_PREFIXES if p != "astro_")


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build features without fragmenting the DataFrame."""
    cols: Dict[str, pd.Series] = {}
    cols["ret"] = df["Return"].astype("float32")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            cols[f"pct_{col.lower()}"] = df[col].pct_change().fillna(0.0).astype("float32")

    astro_like: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "adj close", "volume", "return"):
            continue
        if _is_astro_col(c):
            astro_like.append(c)

    for c in astro_like:
        cols[c] = (
            df[c].astype("float32", copy=False)
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0.0)
        )

    feats = pd.DataFrame(cols, index=df.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return feats.astype("float32")


# ========================== VecNormalize guards ==========================

def _expected_obs_dim(window_size: int, feats_2d: np.ndarray) -> int:
    return int(window_size) * int(feats_2d.shape[1])


def _maybe_delete_stale_vecnorm(path: str, expected_dim: int) -> Optional[str]:
    """Delete saved VecNormalize if its obs_dim differs from current."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            vn = cloudpickle.load(f)
        rms = getattr(vn, "obs_rms", None)
        if rms is None:
            os.remove(path); return f"Deleted {path} (missing obs_rms)."
        old_mean = getattr(rms, "mean", None)
        if old_mean is None:
            os.remove(path); return f"Deleted {os.path.relpath(path)} (no obs_rms.mean)."
        old_dim = int(np.prod(np.shape(old_mean)))
        if old_dim != int(expected_dim):
            os.remove(path); return f"Deleted {os.path.relpath(path)} (old_dim={old_dim} != expected={expected_dim})."
    except Exception:
        try:
            os.remove(path); return f"Deleted unreadable {os.path.relpath(path)}."
        except Exception:
            return None
    return None


# ========================== Environment ==========================

class IonChronosEnv(gym.Env):
    """
    Long/Flat or Continuous position env with costs & daily constraints.
    Reward: position * return_t − (cost+slippage)*|Δposition|
    Daily rules: stop if day DD ≥ day_stop_dd; cap trades/day.
    Uses a rolling (window, features) buffer to guarantee fixed obs length.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        feats: np.ndarray,
        rets: np.ndarray,
        prices: np.ndarray,
        times: pd.DatetimeIndex,
        window_size: int = 30,
        cost: float = 0.0005,
        slippage: float = 0.0005,
        continuous: bool = False,
        max_trades_per_day: int = 30,
        day_stop_dd: float = 0.02,
        use_log_returns: bool = False,
    ):
        super().__init__()
        assert feats.ndim == 2, f"feats must be (T, F), got {feats.shape}"
        assert len(rets) == len(feats) == len(prices) == len(times), "length mismatch"

        self.feats = feats.astype(np.float32, copy=False)
        self.rets = rets.astype(np.float32, copy=False)
        self.prices = prices.astype(np.float32, copy=False)
        self.times = times
        self.window_size = int(window_size)
        self.cost = float(cost)
        self.slippage = float(slippage)
        self.continuous = bool(continuous)
        self.max_trades = int(max_trades_per_day)
        self.day_stop_dd = float(day_stop_dd)
        self.use_log_returns = bool(use_log_returns)

        self.num_features = int(self.feats.shape[1])
        self.expected_obs_dim = int(self.window_size * self.num_features)

        # Timeline bounds
        self.start = self.window_size
        self.end = len(self.rets) - 1

        # Spaces
        self.action_space = (
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            if self.continuous else spaces.Discrete(2)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.expected_obs_dim,),
            dtype=np.float32,
        )

        # Rolling buffer (filled on reset)
        self._buf: Optional[np.ndarray] = None  # (W, F)

        # State
        self._t: Optional[int] = None
        self._pos: float = 0.0
        self._last_action: float = 0.0
        self._equity: float = 1.0
        self._day_key: Optional[pd.Timestamp] = None
        self._day_peak: float = 1.0
        self._day_trades: int = 0
        self._day_blocked: bool = False

    # ----- rolling buffer helpers -----

    def _init_buffer(self, t: int):
        """Create a window buffer ending at index t-1 (exclusive). Pad if needed."""
        w = self.window_size
        s = max(0, t - w)
        window = self.feats[s:t, :]
        if window.shape[0] < w:
            if window.shape[0] == 0:
                first = self.feats[0:1, :]
                window = np.repeat(first, w, axis=0)
            else:
                pad = np.repeat(window[:1, :], w - window.shape[0], axis=0)
                window = np.vstack([pad, window])
        assert window.shape == (w, self.num_features), f"Init buffer shape {window.shape} != ({w},{self.num_features})"
        self._buf = window.astype(np.float32, copy=False)

    def _push_frame(self, frame: np.ndarray):
        self._buf[:-1, :] = self._buf[1:, :]
        self._buf[-1, :] = frame

    def _obs_from_buffer(self) -> np.ndarray:
        flat = self._buf.reshape(-1).astype(np.float32)
        if flat.size != self.expected_obs_dim:
            raise RuntimeError(
                f"Obs shape mismatch: got {flat.size}, expected {self.expected_obs_dim}. "
                f"window={self.window_size}, features={self.num_features}"
            )
        return flat

    # ----- daily controls -----

    def _date_key(self, t: int) -> pd.Timestamp:
        return self.times[t].normalize()

    def _apply_daily_controls(self):
        dk = self._date_key(self._t)
        if (self._day_key is None) or (dk != self._day_key):
            self._day_key = dk
            self._day_peak = self._equity
            self._day_trades = 0
            self._day_blocked = False

        if self._equity > self._day_peak:
            self._day_peak = self._equity
        day_dd = self._equity / max(self._day_peak, 1e-12) - 1.0
        if day_dd <= -self.day_stop_dd:
            self._day_blocked = True

    # ----- gym API -----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._t = int(self.start)
        self._pos = 0.0
        self._last_action = 0.0
        self._equity = 1.0
        self._day_key = None
        self._day_peak = 1.0
        self._day_trades = 0
        self._day_blocked = False

        # Guard: env must have at least (window_size + 2) bars
        if len(self.rets) < self.window_size + 2:
            raise RuntimeError(
                f"Env needs ≥ window_size+2 bars, got {len(self.rets)} with window={self.window_size}"
            )

        self._init_buffer(self._t)
        obs = self._obs_from_buffer()
        return obs, {}

    def step(self, action: Union[int, float, np.ndarray, List[float]]):
        if self.continuous:
            if isinstance(action, (list, tuple, np.ndarray)):
                action = float(np.asarray(action).reshape(-1)[0])
            action = float(np.clip(action, -1.0, 1.0))
        else:
            action = int(action)
            assert self.action_space.contains(action)
            action = float(action)

        self._apply_daily_controls()
        effective_action = action
        if self._day_blocked or self._day_trades >= self.max_trades:
            effective_action = self._last_action

        # Reward uses return at t
        r = float(self.rets[self._t])
        delta_pos = float(effective_action) - float(self._last_action)
        trade_cost = (self.cost + self.slippage) * abs(delta_pos)
        pnl = float(effective_action) * r - trade_cost
        self._equity *= (1.0 + pnl)

        if abs(delta_pos) > 1e-9 and (not self._day_blocked):
            self._day_trades += 1

        self._last_action = effective_action
        self._pos = effective_action

        # Advance and build next observation
        self._t += 1
        terminated = self._t >= self.end
        truncated = False

        next_idx = min(self._t - 1, len(self.feats) - 1)
        next_frame = self.feats[next_idx, :]
        self._push_frame(next_frame)

        obs = self._obs_from_buffer()
        info = {
            "equity": self._equity,
            "blocked": self._day_blocked,
            "day_trades": self._day_trades,
            "price": float(self.prices[next_idx]),
            "time": self.times[next_idx],
            "position": self._pos,
            "pnl": pnl,
        }
        return obs, pnl, terminated, truncated, info


# ========================== Metrics & Plotting ==========================

def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    diffs_ns = np.diff(index.asi8.astype(np.int64))
    median_s = float(np.median(diffs_ns)) / 1e9
    if median_s <= 0:
        return 252.0
    bars_per_day = 86400.0 / median_s
    return float(bars_per_day * 365.25)


def _metrics(equity: pd.Series, step_returns: pd.Series, ann_factor: float) -> Dict[str, float]:
    total_ret = float(equity.iloc[-1] - 1.0)
    mu = float(step_returns.mean())
    sigma = float(step_returns.std(ddof=0))
    cagr = (1 + mu) ** ann_factor - 1 if sigma == sigma else np.nan
    ann_vol = sigma * np.sqrt(ann_factor) if sigma == sigma else np.nan
    sharpe = (mu / sigma) * np.sqrt(ann_factor) if sigma > 0 else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = float(dd.min())
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    dd_dur = int((end - start).days) if isinstance(end, pd.Timestamp) else np.nan

    return dict(total_return=total_ret, cagr=cagr, ann_vol=ann_vol,
                sharpe=sharpe, max_drawdown=mdd, dd_duration_days=dd_dur)


def _save_plots(equity: pd.Series, out_prefix: str) -> None:
    eq_path = f"{out_prefix}_equity.png"
    dd_path = f"{out_prefix}_drawdown.png"

    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Time"); plt.ylabel("Equity (start=1.0)")
    plt.tight_layout(); plt.savefig(eq_path); plt.close()

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    plt.figure(figsize=(10, 4))
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Time"); plt.ylabel("Drawdown")
    plt.tight_layout(); plt.savefig(dd_path); plt.close()


# ========================== Split Utils ==========================

def _safe_train_test_split(
    rets: np.ndarray, prices: np.ndarray, feats: np.ndarray, idx: pd.DatetimeIndex,
    window_size: int, train_split: float
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex],
    Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex],
    str
]:
    """
    Ensure both train and test have at least (window+2) rows.
    If the full series is too short for a clean OOS split, we:
      - Train on the full series
      - Evaluate on the last (window+2) bars (overlap eval)
    Returns: (train_tuple, test_tuple, split_note)
    """
    N = len(rets)
    min_len = int(window_size) + 2
    if N < min_len:
        raise ValueError(f"Not enough rows after cleaning ({N}) for window_size={window_size} (need ≥ {min_len}).")

    # Case 1: enough to keep both sides ≥ min_len
    if N >= 2 * min_len:
        split_i = int(N * float(train_split))
        split_i = max(min_len, min(split_i, N - min_len))
        note = f"standard split @ {split_i}/{N}"
        tr = (rets[:split_i], prices[:split_i], feats[:split_i, :], idx[:split_i])
        te = (rets[split_i:], prices[split_i:], feats[split_i:, :], idx[split_i:])
        return tr, te, note

    # Case 2: short series — overlap eval on tail window+2
    split_i = N  # train on all
    tail_start = N - min_len
    note = f"short series (N={N}) — overlap eval on last {min_len} bars"
    tr = (rets[:split_i], prices[:split_i], feats[:split_i, :], idx[:split_i])
    te = (rets[tail_start:], prices[tail_start:], feats[tail_start:, :], idx[tail_start:])
    return tr, te, note


# ========================== Env Utils ==========================

def _make_env_factory(
    feats: np.ndarray, rets: np.ndarray, prices: np.ndarray, times: pd.DatetimeIndex,
    window_size: int, cost: float, slippage: float, continuous: bool,
    max_trades_per_day: int, day_stop_dd: float, use_log_returns: bool, seed: int
) -> Callable[[], gym.Env]:
    def _thunk():
        env = IonChronosEnv(
            feats=feats, rets=rets, prices=prices, times=times,
            window_size=window_size, cost=cost, slippage=slippage,
            continuous=continuous, max_trades_per_day=max_trades_per_day,
            day_stop_dd=day_stop_dd, use_log_returns=use_log_returns,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _thunk


def _deterministic_rollout_vec(model: PPO, venv, idx: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    obs = venv.reset()
    equity_vals: List[float] = [1.0]
    step_rewards: List[float] = []
    actions: List[float] = []
    prices: List[float] = []
    times: List[pd.Timestamp] = []
    positions: List[float] = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        done = bool(dones[0])
        info = infos[0]
        equity_vals.append(float(info.get("equity", equity_vals[-1])))
        step_rewards.append(float(rewards[0]))
        if isinstance(action, (list, tuple, np.ndarray)):
            actions.append(float(np.asarray(action).reshape(-1)[0]))
        else:
            actions.append(float(action))
        prices.append(float(info.get("price", np.nan)))
        times.append(info.get("time", None))
        positions.append(float(info.get("position", np.nan)))

    eq_index = pd.to_datetime(idx[: len(equity_vals)])
    equity_series = pd.Series(equity_vals, index=eq_index)
    ret_index = eq_index[1:]
    step_return_series = pd.Series(step_rewards, index=ret_index)

    trades = pd.DataFrame({
        "time": times,
        "price": prices,
        "action": actions,
        "position": positions,
        "reward": step_rewards,
        "equity_after": equity_vals[1:],
    })
    return equity_series, step_return_series, trades


# ========================== Training & Evaluation ==========================

def _select_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps and getattr(mps, "is_available", lambda: False)():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_rl_agent(
    ticker: str,
    total_timesteps: int = 200_000,
    window_size: int = 30,
    cost: float = 0.0005,
    slippage: float = 0.0005,
    continuous: bool = False,
    max_trades_per_day: int = 30,
    day_stop_dd: float = 0.02,
    n_envs: int = 1,
    use_vecnorm: bool = True,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    seed: int = 42,
    # NEW:
    train_split: float = 0.70,
    extra_timesteps_after: int = 0,
    resume_from: Optional[str] = None,
) -> str:
    """Train PPO and save metrics/artifacts with an OOS evaluation."""
    os.makedirs("workspace", exist_ok=True)

    # Repro / perf
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    device = _select_device()

    # 1) Data & features
    df = _load_dataframe(ticker)
    feats_df = _build_feature_matrix(df)

    rets = df["Return"].to_numpy(dtype=np.float32)
    prices = df[[c for c in df.columns if c.lower() == "close"][0]].to_numpy(dtype=np.float32)
    feats = feats_df.to_numpy(dtype=np.float32)
    idx_all = df.index

    drop = max(int(window_size), 1)
    rets, prices, feats, idx_all = rets[drop:], prices[drop:], feats[drop:, :], idx_all[drop:]

    # Safe split: guarantee viable train & test (or overlap fallback)
    (rets_tr, prices_tr, feats_tr, idx_tr), (rets_te, prices_te, feats_te, idx_te), split_note = \
        _safe_train_test_split(rets, prices, feats, idx_all, window_size, train_split)

    # 2) Envs (train & eval)
    make_env_tr = _make_env_factory(
        feats_tr, rets_tr, prices_tr, idx_tr,
        window_size, cost, slippage, continuous,
        max_trades_per_day, day_stop_dd,
        use_log_returns=False, seed=seed
    )
    venv = DummyVecEnv([make_env_tr for _ in range(max(1, int(n_envs)))])

    vecnorm_path = os.path.join(WORKSPACE, f"{ticker}_vecnorm.pkl")

    # Guard against stale VecNormalize (obs_dim mismatch)
    if use_vecnorm:
        obs_dim = _expected_obs_dim(window_size, feats_tr)
        note = _maybe_delete_stale_vecnorm(vecnorm_path, obs_dim)
        if note:
            print("[rl_train] VecNormalize reset:", note)

        if resume_from and os.path.exists(vecnorm_path) and note is None:
            venv = VecNormalize.load(vecnorm_path, venv)
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3) PPO with linear LR schedule + modest net arch
    def lr_schedule(progress_remaining: float) -> float:
        return lr * progress_remaining

    policy_kwargs = dict(net_arch=[128, 128], ortho_init=False)

    if resume_from and os.path.exists(resume_from):
        model = PPO.load(resume_from, env=venv, device=device)
        model.learning_rate = lr_schedule
        model.ent_coef = ent_coef
        model.gamma = gamma
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=lr_schedule,
            ent_coef=ent_coef,
            gamma=gamma,
            verbose=0,
            seed=seed,
            batch_size=2048,
            n_epochs=10,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            device=device,
        )

    model.learn(total_timesteps=int(total_timesteps))
    if extra_timesteps_after and int(extra_timesteps_after) > 0:
        model.learn(total_timesteps=int(extra_timesteps_after))

    if use_vecnorm and isinstance(venv, VecNormalize):
        venv.save(vecnorm_path)

    # 4) Deterministic OOS (or overlap) evaluation
    make_env_te = _make_env_factory(
        feats_te, rets_te, prices_te, idx_te,
        window_size, cost, slippage, continuous,
        max_trades_per_day, day_stop_dd,
        use_log_returns=False, seed=seed
    )
    eval_env = DummyVecEnv([make_env_te])

    if use_vecnorm and os.path.exists(vecnorm_path):
        try:
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        except Exception as e:
            print(f"[rl_train] Warning: could not load VecNormalize for eval ({e}); evaluating without it.")

    equity_series, step_return_series, trades = _deterministic_rollout_vec(model, eval_env, idx_te)

    # 5) Metrics & artifacts
    ann_factor = _annualization_factor(equity_series.index)
    met = _metrics(equity_series, step_return_series, ann_factor)

    model_path = os.path.join(WORKSPACE, f"{ticker}_ppo_model.zip")
    model.save(model_path)

    out_prefix = os.path.join(WORKSPACE, f"{ticker}")
    _save_plots(equity_series, out_prefix)

    eval_csv = os.path.join(WORKSPACE, f"{ticker}_rl_eval.csv")
    row = dict(
        ticker=ticker,
        steps=int(total_timesteps),
        extra_steps=int(extra_timesteps_after or 0),
        window=int(window_size),
        cost=float(cost),
        slippage=float(slippage),
        continuous=bool(continuous),
        day_stop_dd=float(day_stop_dd),
        max_trades_per_day=int(max_trades_per_day),
        n_envs=int(n_envs),
        use_vecnorm=bool(use_vecnorm),
        lr=float(lr),
        ent_coef=float(ent_coef),
        gamma=float(gamma),
        device=device,
        train_split=float(train_split),
        split_note=split_note,
        ann_factor=float(ann_factor),
        bars=int(len(step_return_series)),
        **met,
    )
    pd.DataFrame([row]).to_csv(eval_csv, index=False)

    trades_path = os.path.join(WORKSPACE, f"{ticker}_trades.csv")
    trades.to_csv(trades_path, index=False)

    summary_txt = os.path.join(WORKSPACE, f"{ticker}_rl_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(
            "RL Evaluation Summary\n"
            f"Ticker: {ticker}\n"
            f"Split: {split_note}\n"
            f"Total Return: {met['total_return']:.2%}\n"
            f"CAGR: {met['cagr']:.2%}\n"
            f"Ann.Vol: {met['ann_vol']:.2%}\n"
            f"Sharpe (rf=0): {met['sharpe']:.2f}\n"
            f"Max Drawdown: {met['max_drawdown']:.2%}\n"
            f"DD Duration (days): {met['dd_duration_days']}\n"
            f"Model: {os.path.relpath(model_path)}\n"
            f"VecNormalize: {os.path.relpath(vecnorm_path) if use_vecnorm else 'disabled'}\n"
            f"Equity PNG: {os.path.relpath(out_prefix + '_equity.png')}\n"
            f"Drawdown PNG: {os.path.relpath(out_prefix + '_drawdown.png')}\n"
            f"Trades CSV: {os.path.relpath(trades_path)}\n"
        )

    with open(os.path.join(WORKSPACE, "experiments.log"), "a", encoding="utf-8") as log:
        log.write(
            f"{dt.datetime.now()} - RL train {ticker} "
            f"steps={total_timesteps} +extra={extra_timesteps_after} "
            f"window={window_size} cost={cost} slip={slippage} cont={continuous} "
            f"split='{split_note}' "
            f"-> ret={met['total_return']:.4f} sharpe={met['sharpe']:.3f} mdd={met['max_drawdown']:.4f}\n"
        )

    return (
        f"RL training completed for {ticker}. "
        f"Saved model to {os.path.relpath(model_path)}, "
        f"vecnorm to {os.path.relpath(vecnorm_path) if use_vecnorm else 'N/A'}, "
        f"metrics to {os.path.relpath(eval_csv)}, "
        f"summary to {os.path.relpath(summary_txt)}, "
        f"trades to {os.path.relpath(trades_path)}, "
        f"and plots to {os.path.relpath(out_prefix + '_equity.png')} / "
        f"{os.path.relpath(out_prefix + '_drawdown.png')}."
    )
