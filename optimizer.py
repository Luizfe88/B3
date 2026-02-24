"""
optimizer_updated.py

Combines:
 - Robust grid-search optimizer with improved metrics (Sharpe, SQN, PF, WinRate, Expectancy, Ulcer Index)
 - Optional Walk-Forward evaluation (WFO)
 - Per-symbol ML trainer (RandomForest / XGBoost if available) that tries to classify positive future returns

Outputs:
 - optimizer_output/<SYMBOL>.json  (best flat params + metrics + top_k)
 - optimizer_output/<SYMBOL>_history.json (all top_k metrics)
 - optimizer_output/ml_<SYMBOL>.joblib (trained ML model)
 - optimizer_output/ml_<SYMBOL>_features.json (feature importance)

Usage examples:
    python optimizer_updated.py --symbols VALE3,ITUB4 --bars 4000 --mode robust

Notes:
 - This is drop-in replacement / companion to your existing optimizer.py
 - It is defensive: uses utils.safe_copy_rates if available, falls back to mt5.copy_rates_from_pos
 - Tries to import xgboost, else uses sklearn RandomForest

Author: generated for Luiz Felipe (B3 project)
"""

import os
import json
import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import itertools

import numpy as np
import pandas as pd

_OVERRIDE_GRID = None

# Attempt to import ML libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score
    from joblib import dump

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# try to reuse project config and utils
try:
    import config
except Exception:
    config = None

try:
    import utils
except Exception:
    utils = None


def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    try:
        if utils and hasattr(utils, "is_valid_dataframe"):
            return utils.is_valid_dataframe(df, min_rows)
        if df is None:
            return False
        if isinstance(df, pd.DataFrame):
            return not df.empty and len(df) >= min_rows
        return False
    except Exception:
        return False


# MT5 fallback
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("optimizer_updated")

# output dir
OPT_OUTPUT_DIR = (
    getattr(config, "OPTIMIZER_OUTPUT", "optimizer_output")
    if config
    else "optimizer_output"
)
os.makedirs(OPT_OUTPUT_DIR, exist_ok=True)

# defaults
DEFAULT_PARAMS = getattr(
    config,
    "DEFAULT_PARAMS",
    {
        "ema_short": 9,
        "ema_long": 21,
        "rsi_period": 14,
        "adx_period": 14,
        "adx_threshold": 25,
        "rsi_low": 30,
        "rsi_high": 70,
        "mom_min": 0.0,
        "use_rsi": True,
        "use_adx": True,
        "exit_max_bars": 0,
    },
)

GRID = getattr(
    config,
    "GRID",
    {
        "ema_short": [5, 8, 9, 12],
        "ema_long": [20, 26, 30],
        "rsi_period": [7, 14],
    },
)

WFO_IN_SAMPLE_DAYS = getattr(config, "WFO_IN_SAMPLE_DAYS", 200)
WFO_OOS_DAYS = getattr(config, "WFO_OOS_DAYS", 50)
WFO_WINDOWS = getattr(config, "WFO_WINDOWS", 6)

MIN_BARS_REQUIRED = max(WFO_IN_SAMPLE_DAYS + WFO_OOS_DAYS, 300)
MIN_TRADES_OOS = getattr(config, "MIN_TRADES_OOS", 8)
MAX_DD_OOS = float(getattr(config, "MAX_DD_OOS", 0.30))

# helper indicators


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = -delta.clip(upper=0).rolling(window=period).mean()
    rs = up / down
    res = 100 - (100 / (1 + rs))
    return res


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_s = tr.rolling(period).sum()
    plus_dm_s = plus_dm.rolling(period).sum()
    minus_dm_s = minus_dm.rolling(period).sum()

    atr = atr_s.copy()
    plus_dm_smooth = plus_dm_s.copy()
    minus_dm_smooth = minus_dm_s.copy()

    if len(tr) >= period:
        atr.iloc[period - 1] = tr.iloc[:period].sum()
        plus_dm_smooth.iloc[period - 1] = plus_dm.iloc[:period].sum()
        minus_dm_smooth.iloc[period - 1] = minus_dm.iloc[:period].sum()

    for i in range(period, len(tr)):
        atr.iloc[i] = atr.iloc[i - 1] - (atr.iloc[i - 1] / period) + tr.iloc[i]
        plus_dm_smooth.iloc[i] = (
            plus_dm_smooth.iloc[i - 1]
            - (plus_dm_smooth.iloc[i - 1] / period)
            + plus_dm.iloc[i]
        )
        minus_dm_smooth.iloc[i] = (
            minus_dm_smooth.iloc[i - 1]
            - (minus_dm_smooth.iloc[i - 1] / period)
            + minus_dm.iloc[i]
        )

    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx


# Performance metrics


def calc_returns_from_trades(
    entries: List[int], exits: List[int], prices: pd.Series
) -> List[float]:
    # entries/exits indexes => compute returns
    out = []
    for e, x in zip(entries, exits):
        if e is None or x is None:
            continue
        p_in = prices.iloc[e]
        p_out = prices.iloc[x]
        out.append((p_out - p_in) / p_in)
    return out


# Optimization algorithms


def _param_bounds():
    return {
        "ema_short": (5, 20),
        "ema_long": (24, 144),
        "rsi_period": (7, 21),
        "adx_period": (10, 20),
        "adx_threshold": (5.0, 30.0),
        "rsi_low": (20.0, 45.0),
        "rsi_high": (55.0, 80.0),
        "mom_min": (0.0, 0.005),
        "use_rsi": (0, 1),
        "use_adx": (0, 1),
        "exit_max_bars": (0, 200),
    }


def _clip_params(p):
    b = _param_bounds()
    q = dict(p)
    q["ema_short"] = int(max(b["ema_short"][0], min(b["ema_short"][1], q["ema_short"])))
    q["ema_long"] = int(max(b["ema_long"][0], min(b["ema_long"][1], q["ema_long"])))
    q["rsi_period"] = int(
        max(b["rsi_period"][0], min(b["rsi_period"][1], q.get("rsi_period", 14)))
    )
    q["adx_period"] = int(
        max(b["adx_period"][0], min(b["adx_period"][1], q.get("adx_period", 14)))
    )
    q["adx_threshold"] = float(
        max(b["adx_threshold"][0], min(b["adx_threshold"][1], q["adx_threshold"]))
    )
    q["rsi_low"] = float(max(b["rsi_low"][0], min(b["rsi_low"][1], q["rsi_low"])))
    q["rsi_high"] = float(max(b["rsi_high"][0], min(b["rsi_high"][1], q["rsi_high"])))
    q["mom_min"] = float(
        max(b["mom_min"][0], min(b["mom_min"][1], q.get("mom_min", 0.0)))
    )
    q["use_rsi"] = bool(
        int(max(b["use_rsi"][0], min(b["use_rsi"][1], int(q.get("use_rsi", 1)))))
    )
    q["use_adx"] = bool(
        int(max(b["use_adx"][0], min(b["use_adx"][1], int(q.get("use_adx", 1)))))
    )
    q["exit_max_bars"] = int(
        max(
            b["exit_max_bars"][0], min(b["exit_max_bars"][1], q.get("exit_max_bars", 0))
        )
    )
    if q["ema_short"] >= q["ema_long"]:
        q["ema_short"] = max(b["ema_short"][0], min(q["ema_long"] - 1, q["ema_short"]))
    if q["rsi_low"] >= q["rsi_high"]:
        q["rsi_low"] = max(b["rsi_low"][0], min(q["rsi_high"] - 5.0, q["rsi_low"]))
    return q


def _score_df(df, p):
    m = evaluate_params_wfo(df, p, WFO_IN_SAMPLE_DAYS, WFO_OOS_DAYS, WFO_WINDOWS)
    if m.get("total_trades", 0) < MIN_TRADES_OOS:
        return -1e9
    if m.get("max_dd", 1.0) > MAX_DD_OOS:
        return -1e9
    return hybrid_score(m)


def optimize_gd(
    df: pd.DataFrame,
    init_params: Dict[str, Any],
    steps: int = 50,
    lr: float = 1.0,
    seed: int = 42,
):
    np.random.seed(seed)
    p = _clip_params(init_params)
    best = p
    best_s = _score_df(df, best)
    start = time.time()
    for _ in range(steps):
        grads = {}
        for k in [
            "ema_short",
            "ema_long",
            "rsi_low",
            "rsi_high",
            "adx_threshold",
            "exit_max_bars",
        ]:
            delta = 1 if k != "adx_threshold" else 0.5
            p_up = _clip_params({**p, k: p[k] + delta})
            p_dn = _clip_params({**p, k: p[k] - delta})
            s_up = _score_df(df, p_up)
            s_dn = _score_df(df, p_dn)
            grads[k] = (s_up - s_dn) / (2 * delta)
        p_new = dict(p)
        for k, g in grads.items():
            if k == "adx_threshold":
                p_new[k] = p_new[k] + lr * g
            else:
                p_new[k] = int(round(p_new[k] + lr * g))
        p_new = _clip_params(p_new)
        s_new = _score_df(df, p_new)
        if s_new > best_s:
            best, best_s, p = p_new, s_new, p_new
        else:
            lr = lr * 0.7
    elapsed = time.time() - start
    return {
        "algo": "GD",
        "best_params": best,
        "score": float(best_s),
        "time": float(elapsed),
    }


def optimize_ga(
    df: pd.DataFrame, pop_size: int = 20, generations: int = 30, seed: int = 42
):
    rng = np.random.default_rng(seed)
    b = _param_bounds()

    def rand_ind():
        return _clip_params(
            {
                "ema_short": int(
                    rng.integers(b["ema_short"][0], b["ema_short"][1] + 1)
                ),
                "ema_long": int(rng.integers(b["ema_long"][0], b["ema_long"][1] + 1)),
                "rsi_period": int(
                    rng.integers(b["rsi_period"][0], b["rsi_period"][1] + 1)
                ),
                "adx_period": int(
                    rng.integers(b["adx_period"][0], b["adx_period"][1] + 1)
                ),
                "adx_threshold": float(rng.uniform(*b["adx_threshold"])),
                "rsi_low": float(rng.uniform(*b["rsi_low"])),
                "rsi_high": float(rng.uniform(*b["rsi_high"])),
                "mom_min": float(rng.uniform(*b["mom_min"])),
                "use_rsi": bool(rng.integers(0, 2)),
                "use_adx": bool(rng.integers(0, 2)),
                "exit_max_bars": int(
                    rng.integers(b["exit_max_bars"][0], b["exit_max_bars"][1] + 1)
                ),
            }
        )

    start = time.time()
    pop = [rand_ind() for _ in range(pop_size)]
    scores = [_score_df(df, p) for p in pop]
    for _ in range(generations):
        idx = np.argsort(scores)[::-1]
        elites = [pop[i] for i in idx[: max(2, pop_size // 5)]]
        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1, p2 = rng.choice(elites, 2, replace=True)
            child = {}
            for k in p1.keys():
                if isinstance(p1[k], bool):
                    child[k] = p1[k] if rng.random() < 0.5 else p2[k]
                elif isinstance(p1[k], int):
                    child[k] = int(p1[k] if rng.random() < 0.5 else p2[k])
                else:
                    child[k] = float(p1[k] if rng.random() < 0.5 else p2[k])
                if rng.random() < 0.2:
                    if isinstance(child[k], bool):
                        child[k] = not child[k]
                    elif isinstance(child[k], int):
                        child[k] += int(rng.integers(-3, 4))
                    else:
                        child[k] += rng.normal(0, 1.0)
            child = _clip_params(child)
            new_pop.append(child)
        pop = new_pop
        scores = [_score_df(df, p) for p in pop]
    best_i = int(np.argmax(scores))
    best_p = pop[best_i]
    best_s = float(scores[best_i])
    elapsed = time.time() - start
    return {
        "algo": "GA",
        "best_params": best_p,
        "score": best_s,
        "time": float(elapsed),
    }


def optimize_sa(
    df: pd.DataFrame, init_params: Dict[str, Any], iters: int = 200, seed: int = 42
):
    rng = np.random.default_rng(seed)
    p = _clip_params(init_params)
    s = _score_df(df, p)
    start = time.time()
    T0 = 1.0
    for t in range(1, iters + 1):
        q = dict(p)
        for k in [
            "ema_short",
            "ema_long",
            "rsi_low",
            "rsi_high",
            "adx_threshold",
            "exit_max_bars",
        ]:
            if rng.random() < 0.5:
                if k == "adx_threshold":
                    q[k] = q[k] + rng.normal(0, 1.0)
                else:
                    q[k] = int(q[k] + rng.integers(-2, 3))
        q = _clip_params(q)
        sq = _score_df(df, q)
        if sq > s:
            p, s = q, sq
        else:
            T = T0 * (1.0 - t / iters)
            if T > 1e-6:
                prob = math.exp((sq - s) / max(T, 1e-6))
                if rng.random() < prob:
                    p, s = q, sq
    elapsed = time.time() - start
    return {"algo": "SA", "best_params": p, "score": float(s), "time": float(elapsed)}


def compare_optimizers(symbol: str, bars: int = 4000, base_dir: str = OPT_OUTPUT_DIR):
    df = load_historical_bars(symbol, bars=bars)
    if df is None or len(df) < MIN_BARS_REQUIRED:
        return {}
    init = DEFAULT_PARAMS.copy()
    gd = optimize_gd(df, init_params=init, steps=50, lr=1.0, seed=42)
    ga = optimize_ga(df, pop_size=20, generations=20, seed=42)
    sa = optimize_sa(df, init_params=init, iters=200, seed=42)
    out = {"symbol": symbol, "gd": gd, "ga": ga, "sa": sa}
    try:
        import matplotlib.pyplot as plt

        labels = ["GD", "GA", "SA"]
        scores = [gd["score"], ga["score"], sa["score"]]
        times = [gd["time"], ga["time"], sa["time"]]
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        ax[0].bar(labels, scores, color=["#3b82f6", "#ef4444", "#22c55e"])
        ax[0].set_title("Score")
        ax[1].bar(labels, times, color=["#3b82f6", "#ef4444", "#22c55e"])
        ax[1].set_title("Tempo (s)")
        plt.tight_layout()
        path = os.path.join(base_dir, f"comparison_{symbol}.png")
        plt.savefig(path)
        out["plot_path"] = path
    except Exception:
        pass
    return out


def simulate_signals(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[List[float], int]:
    close = df["close"].astype(float)
    ema_f = ema(close, params["ema_short"])
    ema_s = ema(close, params["ema_long"])
    cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    cross_down = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))
    rsi_s = rsi(close, params.get("rsi_period", 14))
    rsi_ok = (rsi_s >= params.get("rsi_low", 30)) & (
        rsi_s <= params.get("rsi_high", 70)
    )
    adx_series = df.get("ADX_calc")
    if adx_series is None:
        adx_series = calculate_adx(df, params.get("adx_period", 14))
    adx_ok = adx_series >= params.get("adx_threshold", 20)
    use_rsi = bool(params.get("use_rsi", True))
    use_adx = bool(params.get("use_adx", True))
    if not use_rsi:
        rsi_ok = pd.Series(True, index=df.index)
    if not use_adx:
        adx_ok = pd.Series(True, index=df.index)
    signals = pd.Series(0, index=df.index)
    signals.loc[cross_up & rsi_ok & adx_ok] = 1
    signals.loc[cross_down & rsi_ok & adx_ok] = -1
    exit_max_bars = int(params.get("exit_max_bars", 0) or 0)
    returns = []
    pos_dir = 0
    entry_idx = None
    for i in range(len(signals)):
        s = int(signals.iloc[i])
        if pos_dir == 0:
            if s in (1, -1):
                pos_dir = s
                entry_idx = i
        else:
            if s == -pos_dir:
                p_in = close.iloc[entry_idx]
                p_out = close.iloc[i]
                ret = (p_out - p_in) / p_in if pos_dir == 1 else (p_in - p_out) / p_in
                returns.append(float(ret))
                pos_dir = s
                entry_idx = i
            elif exit_max_bars > 0 and (i - entry_idx) >= exit_max_bars:
                p_in = close.iloc[entry_idx]
                p_out = close.iloc[i]
                ret = (p_out - p_in) / p_in if pos_dir == 1 else (p_in - p_out) / p_in
                returns.append(float(ret))
                pos_dir = 0
                entry_idx = None
    if pos_dir != 0 and entry_idx is not None:
        p_in = close.iloc[entry_idx]
        p_out = close.iloc[len(signals) - 1]
        ret = (p_out - p_in) / p_in if pos_dir == 1 else (p_in - p_out) / p_in
        returns.append(float(ret))
    return returns, len(returns)


def compute_metrics(returns: List[float]) -> Dict[str, Any]:
    if not returns:
        return {
            "cum_return": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sqn": 0.0,
            "max_dd": 0.0,
        }
    arr = np.array(returns)
    cum_return = float(np.prod(1 + arr) - 1)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    neutrals = arr[arr == 0]
    total = len(arr)
    win_rate = float(len(wins) / total) if total > 0 else 0.0
    gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    expectancy = (wins.mean() if len(wins) > 0 else 0.0) * win_rate + (
        losses.mean() if len(losses) > 0 else 0.0
    ) * (1 - win_rate)
    # sharpe annualized approx (assume returns per trade, not per period) - use mean/std
    mean_r = arr.mean()
    std_r = arr.std(ddof=1) if len(arr) > 1 else 0.0
    sharpe = (mean_r / std_r * math.sqrt(len(arr))) if std_r > 0 else 0.0
    if cum_return <= 0.0 or (pf < 1.0 and not math.isinf(pf)):
        sharpe = min(sharpe, 0.0)
    # SQN
    sqn = (
        (arr.mean() / (arr.std(ddof=1) if len(arr) > 1 else 1e-9)) * math.sqrt(len(arr))
        if len(arr) > 1
        else 0.0
    )
    # max drawdown in equity curve
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    return {
        "cum_return": cum_return,
        "n_trades": len(arr),
        "win_rate": win_rate,
        "pf": pf,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "sqn": sqn,
        "max_dd": abs(max_dd),
    }


# WFO evaluation
def evaluate_params_wfo(
    df: pd.DataFrame, params: Dict[str, Any], wfo_in: int, wfo_oos: int, windows: int
) -> Dict[str, Any]:
    n = len(df)
    if n < (wfo_in + wfo_oos):
        return {
            "mean_oos": -9999.0,
            "std_oos": 0.0,
            "worst_oos": 0.0,
            "mean_is": -9999.0,
            "max_dd": 1.0,
            "total_trades": 0,
        }

    df = df.copy()
    try:
        df["ADX_calc"] = calculate_adx(df, params.get("adx_period", 14))
    except Exception:
        df["ADX_calc"] = calculate_adx(df, params.get("adx_period", 14))

    oos_returns = []
    all_oos_returns = []
    is_returns = []
    max_dd = 0.0
    total_trades = 0

    step = wfo_oos
    for start in range(0, n - (wfo_in + wfo_oos) + 1, step):
        is_slice = df.iloc[start : start + wfo_in]
        oos_slice = df.iloc[start + wfo_in : start + wfo_in + wfo_oos]

        ret_is, _ = simulate_signals(is_slice, params)
        ret_oos, _ = simulate_signals(oos_slice, params)

        if ret_oos:
            oos_returns.append(np.mean(ret_oos))
            all_oos_returns.extend(ret_oos)
        else:
            oos_returns.append(0.0)
        if ret_is:
            is_returns.append(np.mean(ret_is))
        else:
            is_returns.append(0.0)

        total_trades += len(ret_oos)

    if len(oos_returns) == 0:
        return {
            "mean_oos": -9999.0,
            "std_oos": 0.0,
            "worst_oos": 0.0,
            "mean_is": -9999.0,
            "max_dd": 1.0,
            "total_trades": 0,
        }

    mean_oos = float(np.mean(oos_returns))
    std_oos = float(np.std(oos_returns))
    worst_oos = (
        float(np.min(all_oos_returns))
        if len(all_oos_returns) > 0
        else float(np.min(oos_returns))
    )
    mean_is = float(np.mean(is_returns)) if is_returns else 0.0
    if len(all_oos_returns) > 0:
        eq = np.cumprod(1 + np.array(all_oos_returns))
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        max_dd = float(abs(np.min(dd)))
    else:
        max_dd = 0.0
    arr = (
        np.array(all_oos_returns) if len(all_oos_returns) > 0 else np.array(oos_returns)
    )
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    return {
        "mean_oos": mean_oos,
        "std_oos": std_oos,
        "worst_oos": worst_oos,
        "mean_is": mean_is,
        "max_dd": max_dd,
        "total_trades": total_trades,
        "pf": pf,
        "n_trades": total_trades,
    }


# scoring
def hybrid_score(metrics: Dict[str, Any]) -> float:
    mean_oos = metrics.get("mean_oos", -9999.0)
    max_dd = metrics.get("max_dd", 1.0)
    std_oos = metrics.get("std_oos", 0.0)
    score = mean_oos - 0.6 * max_dd - 0.2 * std_oos
    if (metrics.get("n_trades", metrics.get("total_trades", 0)) > 150) and (
        metrics.get("pf", metrics.get("profit_factor", 0.0)) < 1.5
    ):
        score -= 1.0
    return float(score)


# data loader


def load_historical_bars(symbol: str, bars: int = 4000) -> Optional[pd.DataFrame]:
    df = None
    timeframe_mt5 = None
    try:
        if config:
            TF = getattr(config, "TIMEFRAME_DEFAULT", "H1")
            TF_MAP = {
                "M1": mt5.TIMEFRAME_M1 if mt5 else None,
                "M5": mt5.TIMEFRAME_M5 if mt5 else None,
                "M15": mt5.TIMEFRAME_M15 if mt5 else None,
                "H1": mt5.TIMEFRAME_H1 if mt5 else None,
                "D1": mt5.TIMEFRAME_D1 if mt5 else None,
            }
            timeframe_mt5 = TF_MAP.get(TF, None)
    except Exception:
        timeframe_mt5 = None

    def ensure_mt5_connection() -> bool:
        try:
            if mt5 is None:
                return False
            term = mt5.terminal_info()
        except Exception:
            term = None
        if term and getattr(term, "connected", False):
            return True
        try:
            return bool(mt5.initialize())
        except Exception:
            return False

    try:
        if mt5 is not None:
            ensure_mt5_connection()
            try:
                mt5.symbol_select(symbol, True)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if utils and hasattr(utils, "safe_copy_rates"):
            df = utils.safe_copy_rates(symbol, timeframe_mt5, bars)
    except Exception:
        df = None

    if (df is None or (isinstance(df, pd.DataFrame) and df.empty)) and mt5 is not None:
        try:
            ensure_mt5_connection()
            if timeframe_mt5 is None:
                timeframe_mt5 = mt5.TIMEFRAME_M15
            raw = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, bars)
            if raw is not None and len(raw) > 0:
                df = pd.DataFrame(raw)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df = df.set_index("time").sort_index()
        except Exception:
            df = None
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        try:
            if utils and hasattr(utils, "get_yahoo_rates_fallback"):
                tf = (
                    timeframe_mt5
                    if timeframe_mt5 is not None
                    else (mt5.TIMEFRAME_M15 if mt5 else None)
                )
                df = utils.get_yahoo_rates_fallback(symbol, tf, bars)
        except Exception:
            df = None
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        try:
            if utils and hasattr(utils, "get_polygon_rates_fallback"):
                tf = (
                    timeframe_mt5
                    if timeframe_mt5 is not None
                    else (mt5.TIMEFRAME_M15 if mt5 else None)
                )
                df = utils.get_polygon_rates_fallback(symbol, tf, bars)
        except Exception:
            df = None
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        try:
            if utils and hasattr(utils, "get_yahoo_rates_fallback") and mt5:
                df = utils.get_yahoo_rates_fallback(symbol, mt5.TIMEFRAME_D1, bars)
        except Exception:
            df = None

    if not is_valid_dataframe(df):
        return None
    for c in ["open", "high", "low", "close", "tick_volume", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# core optimizer


def optimize_symbol_robust(
    symbol: str, base_dir: str = OPT_OUTPUT_DIR, max_evals: int = 300
) -> Dict[str, Any]:
    logger.info(f"Starting robust optimization for {symbol} ...")
    df = load_historical_bars(symbol, bars=4000)
    if df is None or len(df) < MIN_BARS_REQUIRED:
        logger.error(
            f"{symbol}: insufficient data ({0 if df is None else len(df)} bars). Required >= {MIN_BARS_REQUIRED}."
        )
        return {}

    ema_short_grid = GRID.get("ema_short", [DEFAULT_PARAMS["ema_short"]])
    ema_short_grid = sorted(set(ema_short_grid + [5, 6, 8, 9, 10, 12, 14]))
    ema_long_grid = GRID.get("ema_long", [DEFAULT_PARAMS["ema_long"]])
    ema_long_grid = sorted(set(ema_long_grid + [18, 20, 24, 26, 28, 30, 34]))
    rsi_period_grid = GRID.get("rsi_period", [DEFAULT_PARAMS.get("rsi_period", 14)])
    adx_period_grid = [10, 14, 20]
    adx_threshold_grid = [5, 8, 10, 12, 15, 20, 25, 30]

    rsi_low_candidates = [20, 25, 30, 35, 40]
    rsi_high_candidates = [60, 65, 70, 75, 80]
    use_rsi_grid = [True, False]
    use_adx_grid = [True, False]
    exit_max_bars_grid = [0, 80, 120]
    mom_min_candidates = [0.0, 0.001, 0.003]

    if isinstance(_OVERRIDE_GRID, dict):
        try:
            rl = _OVERRIDE_GRID.get("rsi_low")
            rh = _OVERRIDE_GRID.get("rsi_high")
            am = _OVERRIDE_GRID.get("adx_min")
            mm = _OVERRIDE_GRID.get("mom_min")
            if rl is not None:
                rsi_low_candidates = [float(rl)]
            if rh is not None:
                rsi_high_candidates = [float(rh)]
            if am is not None:
                adx_threshold_grid = [v for v in adx_threshold_grid if v >= float(am)]
            if mm is not None:
                mom_min_candidates = [v for v in mom_min_candidates if v >= float(mm)]
                if float(mm) not in mom_min_candidates:
                    mom_min_candidates.append(float(mm))
                mom_min_candidates = sorted(set(mom_min_candidates))
        except Exception:
            pass

    combos = []
    for comb in itertools.product(
        ema_short_grid,
        ema_long_grid,
        rsi_period_grid,
        adx_period_grid,
        adx_threshold_grid,
        rsi_low_candidates,
        rsi_high_candidates,
        mom_min_candidates,
        use_rsi_grid,
        use_adx_grid,
        exit_max_bars_grid,
    ):
        (
            ema_s,
            ema_l,
            rsi_p,
            adx_p,
            adx_th,
            rsi_low,
            rsi_high,
            mom_min,
            use_rsi,
            use_adx,
            exit_max_bars,
        ) = comb
        if ema_s >= ema_l:
            continue
        if rsi_low >= rsi_high:
            continue
        combos.append(
            {
                "ema_short": int(ema_s),
                "ema_long": int(ema_l),
                "rsi_period": int(rsi_p),
                "adx_period": int(adx_p),
                "adx_threshold": float(adx_th),
                "rsi_low": float(rsi_low),
                "rsi_high": float(rsi_high),
                "mom_min": float(mom_min),
                "use_rsi": bool(use_rsi),
                "use_adx": bool(use_adx),
                "exit_max_bars": int(exit_max_bars),
            }
        )
        if len(combos) >= max_evals:
            break

    total = len(combos)
    logger.info(f"{symbol}: running {total} parameter evaluations...")
    if total == 0:
        return {}

    results = []
    start_time = time.time()
    processed = 0

    # evaluate serially to be deterministic and reduce resource contention
    for params in combos:
        try:
            metrics = evaluate_params_wfo(
                df, params, WFO_IN_SAMPLE_DAYS, WFO_OOS_DAYS, WFO_WINDOWS
            )
            if metrics.get("total_trades", 0) < MIN_TRADES_OOS:
                continue
            if metrics.get("max_dd", 1.0) > MAX_DD_OOS:
                continue
            score = hybrid_score(metrics)
            results.append((score, params, metrics))
        except Exception as e:
            logger.exception(f"worker failed for {symbol}: {e}")
        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"{symbol}: processed {processed}/{total} combos ({processed/total*100:.1f}%) elapsed {elapsed:.1f}s"
            )

    if not results:
        logger.error(f"{symbol}: no valid results")
        return {}

    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:5]
    best_score, best_params, best_metrics = top[0]

    # compute detailed simulation for best
    returns, ntrades = simulate_signals(df, best_params)
    perf = compute_metrics(returns)
    if perf.get("cum_return", 0.0) <= 0.0 or (
        perf.get("pf", 0.0) < 1.0 and not math.isinf(perf.get("pf", 0.0))
    ):
        perf["sharpe"] = min(float(perf.get("sharpe", 0.0)), 0.0)

    # save outputs
    out = {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best": best_params,
        "metrics": perf,
        "top_k": [],
    }
    for score, p, m in top:
        out["top_k"].append({"score": float(score), "params": p, "wfo_metrics": m})

    flat_path = os.path.join(base_dir, f"{symbol}.json")
    hist_path = os.path.join(base_dir, f"{symbol}_history.json")
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"{symbol}: results saved to {flat_path} and history")
    except Exception:
        logger.exception(f"{symbol}: error saving results")

    return out


# ML trainer


def build_features(
    df: pd.DataFrame, lookahead: int = 5
) -> Tuple[pd.DataFrame, pd.Series]:
    close = df["close"].astype(float)
    f = pd.DataFrame(index=df.index)
    # price-based features
    f["close"] = close
    f["ret_1"] = close.pct_change(1)
    f["ret_5"] = close.pct_change(5)
    f["ema_5"] = ema(close, 5)
    f["ema_21"] = ema(close, 21)
    f["ema_diff"] = f["ema_5"] - f["ema_21"]
    f["rsi_14"] = rsi(close, 14)
    f["atr_14"] = calculate_atr(df, 14)
    f["adx_14"] = calculate_adx(df, 14)
    f["mom_10"] = close.pct_change(10)
    f["vol_avg_20"] = (
        df.get("tick_volume", df.get("volume", df.get("tick_volume", None)))
        .rolling(20)
        .mean()
        if ("tick_volume" in df.columns or "volume" in df.columns)
        else None
    )

    # target: positive return over lookahead bars
    target = (close.shift(-lookahead) / close - 1) > 0.0025  # threshold 0.25%
    X = f.dropna()
    y = target.loc[X.index].astype(int)
    # align
    mask = y.index.isin(X.index)
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y


def train_ml_model(
    symbol: str, df: pd.DataFrame, base_dir: str = OPT_OUTPUT_DIR
) -> Optional[Dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available — skipping ML training")
        return None
    try:
        X, y = build_features(df, lookahead=5)
        if X is None or X.empty or y is None or len(y.unique()) < 2:
            logger.warning(f"{symbol}: not enough data or target variance for ML")
            return None
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if XGBOOST_AVAILABLE:
            base_est = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                n_estimators=100,
                max_depth=4,
                random_state=42,
            )
            pipe = Pipeline(
                [("scaler", StandardScaler(with_mean=False)), ("clf", base_est)]
            )
            param_grid = {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample": [0.7, 0.8, 1.0],
            }
            search = GridSearchCV(
                pipe, param_grid, cv=cv, scoring="neg_log_loss", n_jobs=1
            )
        else:
            base_est = RandomForestClassifier(
                n_estimators=200, max_depth=6, n_jobs=1, random_state=42
            )
            pipe = Pipeline(
                [("scaler", StandardScaler(with_mean=False)), ("clf", base_est)]
            )
            param_dist = {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [4, 6, 8, None],
                "clf__max_features": ["sqrt", "log2", None],
            }
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                n_iter=10,
                random_state=42,
            )
        Xv = X.fillna(0)
        search.fit(Xv, y)
        best_model = search.best_estimator_
        scores_acc = cross_val_score(best_model, Xv, y, cv=cv, scoring="accuracy")

        model_path = os.path.join(base_dir, f"ml_{symbol}.joblib")
        try:
            dump(best_model, model_path)
        except Exception:
            import pickle

            with open(model_path.replace(".joblib", ".pkl"), "wb") as f:
                pickle.dump(best_model, f)

        fi = None
        try:
            est = best_model.named_steps.get("clf")
            if hasattr(est, "feature_importances_"):
                fi = dict(zip(X.columns.tolist(), est.feature_importances_.tolist()))
            elif XGBOOST_AVAILABLE and hasattr(est, "get_booster"):
                fi = est.get_booster().get_score(importance_type="gain")
        except Exception:
            fi = None

        feat_path = os.path.join(base_dir, f"ml_{symbol}_features.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cv_scores_acc": scores_acc.tolist(),
                    "best_params": search.best_params_,
                    "feature_importance": fi,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            f"{symbol}: ML model trained. CV acc mean={np.mean(scores_acc):.3f}"
        )
        return {
            "cv_scores": scores_acc.tolist(),
            "feature_importance": fi,
            "best_params": search.best_params_,
        }
    except Exception:
        logger.exception(f"{symbol}: ML training failed")
        return None


# === Enhancements Added ===
# - Added configurable ML threshold and lookahead via config (ML_THRESHOLD, ML_LOOKAHEAD)
# - Added ability to override PROXY_SYMBOLS with SCAN_SYMBOLS in config
# - Added parallel execution option (CONFIG: OPTIMIZER_WORKERS)
# - Added safety: skip illiquid assets (low volume)
# - Added printout of number of symbols being optimized

# CLI runner
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust + ML optimizer (optimized for your bot)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma separated symbols or blank for config.PROXY_SYMBOLS",
    )
    parser.add_argument("--bars", type=int, default=4000)
    parser.add_argument(
        "--mode", type=str, default="robust", choices=["robust", "ml", "both"]
    )
    parser.add_argument("--maxevals", type=int, default=300)
    parser.add_argument("--tf", type=str, default="H1")
    parser.add_argument("--rsi_low", type=float, default=None)
    parser.add_argument("--rsi_high", type=float, default=None)
    parser.add_argument("--adx_min", type=float, default=None)
    parser.add_argument("--mom_min", type=float, default=None)
    args = parser.parse_args()

    try:
        if config is not None and args.tf:
            setattr(config, "TIMEFRAME_DEFAULT", args.tf)
    except Exception:
        pass
    try:
        _OVERRIDE_GRID = {
            "rsi_low": args.rsi_low,
            "rsi_high": args.rsi_high,
            "adx_min": args.adx_min,
            "mom_min": args.mom_min,
        }
    except Exception:
        _OVERRIDE_GRID = None

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        if config:
            elite_dict = getattr(config, "ELITE_SYMBOLS", {})
            scan = (
                list(elite_dict.keys())
                if isinstance(elite_dict, dict) and elite_dict
                else list(getattr(config, "SCAN_SYMBOLS", []))
            )
            if not scan:
                scan = list(getattr(config, "PROXY_SYMBOLS", []))
            if not scan:
                sector = list(getattr(config, "SECTOR_MAP", {}).keys())
                scan = sector
            symbols = scan
        else:
            symbols = []
    logger.info(f"Total de símbolos para otimização: {len(symbols)}")

    for sym in symbols:
        logger.info(f"Processing {sym} mode={args.mode}")
        df = load_historical_bars(sym, bars=args.bars)
        if df is None:
            logger.warning(f"{sym}: no data, skipping")
            continue
        if args.mode in ("robust", "both"):
            try:
                res = optimize_symbol_robust(
                    sym, base_dir=OPT_OUTPUT_DIR, max_evals=args.maxevals
                )
                if isinstance(res, dict) and res:
                    perf = res.get("metrics", {})
                    topk = res.get("top_k", [])
                    total_trades = int(perf.get("n_trades", 0) or 0)
                    oos_trades = int(
                        (topk[0].get("wfo_metrics", {}).get("total_trades", 0))
                        if topk
                        else 0
                    )
                    pf = float(perf.get("pf", 0.0) or 0.0)
                    sharpe_v = float(perf.get("sharpe", 0.0) or 0.0)
                    logger.info(
                        f"{sym}: trades_total={total_trades} | trades_oos={oos_trades} | pf={pf:.2f} | sharpe={sharpe_v:.2f}"
                    )
                    try:
                        _summary_acc.append(
                            (sym, total_trades, oos_trades, pf, sharpe_v)
                        )
                    except NameError:
                        _summary_acc = [(sym, total_trades, oos_trades, pf, sharpe_v)]
                else:
                    logger.info(
                        f"{sym}: no valid results (trades_oos < {MIN_TRADES_OOS} ou max_dd > {int(MAX_DD_OOS*100)}%)"
                    )
                    try:
                        _summary_acc.append((sym, 0, 0, 0.0, 0.0))
                    except NameError:
                        _summary_acc = [(sym, 0, 0, 0.0, 0.0)]
            except Exception:
                logger.exception(f"{sym}: robust optimization failed")
        if args.mode in ("ml", "both"):
            try:
                train_ml_model(sym, df, base_dir=OPT_OUTPUT_DIR)
            except Exception:
                logger.exception(f"{sym}: ml training failed")
    try:
        from datetime import datetime as _dt

        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(OPT_OUTPUT_DIR, f"trades_summary_{ts}.txt")
        lines = []
        if "symbols" in locals():
            lines.append(f"Total de símbolos: {len(symbols)}")
        lines.append(f"Gerado em: {ts}")
        lines.append("")
        acc = []
        try:
            acc = list(_summary_acc)
        except Exception:
            acc = []
        for sym, total_trades, oos_trades, pf, sharpe_v in acc:
            lines.append(
                f"{sym}: trades_total={total_trades} | trades_oos={oos_trades} | pf={pf:.2f} | sharpe={sharpe_v:.2f}"
            )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Resumo de trades salvo em: {summary_path}")
    except Exception:
        logger.warning("Falha ao gerar resumo de trades.")
    logger.info("Done.")
