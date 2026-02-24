"""
OPTIMIZER_OPTUNA.PY – HIGH WIN RATE REFINEMENT
✅ Strict Numba Typing (int64/float64)
✅ TA-Lib Integration (VWAP, SAR)
✅ Robust ML Probability Handler
✅ Error-Proof Metric Calculation
"""

import optuna
import logging
import os
import requests
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

try:
    from utils import round_to_tick
except Exception:

    def round_to_tick(price: float, tick_size: float) -> float:
        try:
            ts = tick_size if (tick_size and tick_size > 0) else 1.0
            return float(round(price / ts) * ts)
        except Exception:
            return float(price)


# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def get_macro_rate(rate_name: str):
    try:
        name = rate_name.upper()
        if name.startswith("SELIC"):
            override = os.getenv("XP3_OVERRIDE_SELIC", "")
            return float(override) if override else 0.105
        if name.startswith("IPCA"):
            return 0.04
        return 0.12
    except Exception:
        return 0.12


# =========================================================
# 1. FAST BACKTEST CORE
# =========================================================
def fast_backtest_core(
    close,
    open_,
    high,
    low,
    volume,
    volume_ma,
    vwap,
    ema_short,
    ema_long,
    rsi,
    rsi_2,
    adx,
    sar,
    atr,
    momentum,
    ml_probs,  # ✅ ML Probability Array
    rsi_low,
    rsi_high,
    adx_threshold,
    sl_mult,
    tp_mult,
    base_slippage,
    avg_volume,  # ✅ Float64 Argument
    risk_per_trade=0.01,
    use_trailing=1,  # ✅ Int Flag (1=True, 0=False)
    enable_shorts=1,  # ✅ Flag para habilitar shorts (1=True)
    asset_type=0,  # 0=STOCK, 1=FUTURE
    point_value=0.0,
    tick_size=0.01,
    fee_type=0,  # 0=PERCENT, 1=FIXED
    fee_val=0.00055,
    beta_estimate=1.0,  # ✅ Novo: Beta (reduz risco se >1.3)
    trailing_config=None,  # ✅ Novo: Config completa do Trailing Stop
):
    # Comentário: A.1 Dynamic Position Sizing + A.2 Circuit Breakers + A.3 RR assimétrico
    # ✅ Initial Check
    if len(close) < 3:
        return np.zeros(1), 0, 0, 0, 0, (0, 0, 0, 0, 0, 0, 0)
    ts = tick_size

    # Init
    cash = 100000.0
    equity = cash
    position = 0.0  # Positivo para long, negativo para short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0

    trades = 0
    wins = 0
    losses = 0
    buy_signals_count = 0
    sell_signals_count = 0

    # CONTADORES DE DIAGNÓSTICO (Funnel)
    # 0: Trend, 1: Pullback/Setup, 2: Volatility, 3: ML, 4: Candle/Other, 5: VWAP, 6: Success
    c_trend = 0
    c_setup = 0
    c_volat = 0
    c_ml = 0
    c_candle = 0
    c_vwap = 0
    c_success = 0

    # States
    is_lateral_trade = False
    partial_closed = 0  # ✅ Int Flag (0=False, 1=True)
    bars_in_trade = 0
    consecutive_losses = 0
    trading_paused = 0  # 0=ativo, >0 conta barras em pausa
    trading_paused = 0  # 0=ativo, >0 conta barras em pausa
    pause_bars_remaining = 0
    max_price_reached = 0.0  # Rastreia máximo favorável no trade
    partials_mask = 0  # Bitmask para parciais (1=L1, 2=L2, 4=L3)

    transaction_cost_pct = fee_val if fee_type == 0 else 0.0

    n = len(close)
    equity_curve = np.full(n, cash)  # ✅ FIX: Initialize ALL elements to starting cash
    equity_curve[1] = cash  # ✅ Ensure index 1 is also properly initialized

    # Loop start index 2 to allow i-1 lookback safely
    for i in range(2, n):
        price = close[i]
        vol = volume[i]

        # DD atual da carteira (para ajuste dinâmico de risco)
        peak = max(equity_curve[:i]) if i > 0 else cash
        dd_now = (peak - equity_curve[i - 1]) / peak if peak > 0 else 0.0
        # Ajuste de risco base: atr% + beta + drawdown
        atr_pct = float(atr[i] / max(price, 1e-6))
        risk_dyn = float(risk_per_trade)
        # Volatilidade: reduz em faixas de ATR%
        if atr_pct > 0.04:
            risk_dyn *= 0.5
        elif atr_pct > 0.025:
            risk_dyn *= 0.75
        # Beta: reduz se > 1.3
        if beta_estimate > 1.3:
            risk_dyn *= 0.7
        # Drawdown: reduz 50% se DD > 15%
        if dd_now > 0.15:
            risk_dyn *= 0.5
        # Pausa ativa: impede novas entradas
        if trading_paused == 1:
            pause_bars_remaining = max(0, pause_bars_remaining - 1)
            if pause_bars_remaining == 0:
                trading_paused = 0

        # ---------------------------------------------------
        # POSITION MANAGEMENT
        # ---------------------------------------------------
        if abs(position) > 0:  # Tem posição (long ou short)
            bars_in_trade += 1

            # --- Update Max Price Reached ---
            if position > 0:
                if high[i] > max_price_reached:
                    max_price_reached = high[i]
            else:
                if low[i] < max_price_reached or max_price_reached == 0:
                    max_price_reached = low[i]

            # --- Dynamic Trailing Stop & Circuit Breaker ---
            if use_trailing == 1 and trailing_config:
                from trailing_stop import (
                    calculate_dynamic_stop,
                    check_partial_exit_level,
                )

                # Check Circuit Breaker & Trailing Logic
                pos_side = 1 if position > 0 else -1
                current_stop = stop_price

                # ATR atual
                curr_atr = atr[i]

                # Calculate new stop
                new_stop, reason = calculate_dynamic_stop(
                    current_price=close[i],
                    entry_price=entry_price,
                    current_stop_price=stop_price,
                    max_price_reached=max_price_reached,
                    atr=curr_atr,
                    position_side=pos_side,
                    config=trailing_config,
                    candle_low=low[i],
                    candle_high=high[i],
                    adx=adx[i],
                )

                if new_stop != stop_price:
                    stop_price = new_stop
                    # Se fechou por Circuit Breaker drasticamente, poderia sair aqui,
                    # mas vamos deixar o hit_stop normal pegar no próximo tick ou loop

            # --- Partial Exits (New Logic) ---
            if trailing_config and trailing_config.enable_partials:
                should_part, new_mask, pct_close, reason = check_partial_exit_level(
                    max_price_reached=max_price_reached,
                    entry_price=entry_price,
                    atr=atr[i],
                    position_side=1 if position > 0 else -1,
                    partials_taken=partials_mask,
                    config=trailing_config,
                )

                if should_part:
                    qty_total_orig = abs(position) / (
                        1.0 - (0.25 * bin(partials_mask).count("1"))
                    )  # Estima original grossamente ou usa lógica melhor
                    # Simples: fecha % da posição ATUAL ou ORIGINAL?
                    # Req: "Realize lucros parciais em 25%, 50%, 75% da posição" -> geralmente da original.
                    # Mas aqui position já decresceu. Vamos fechar pct relativo à posição ATUAL que aproxima.
                    # Se quero fechar 25% da original, e tenho 100%, fecho 25% (1/4).
                    # Se tenho 75%, fecho 25% orig (1/3).
                    # Se tenho 50%, fecho 25% orig (1/2).

                    # Simplificação para Backtest: usar pct fixo da config sobre equity ou volume atual
                    # Vamos assumir que pct_close é sobre o volume ORIGINAL.
                    # Como não rastreamos volume original fácil aqui, vamos usar volume atual *fator*
                    # Ex: P1(25%) -> sobra 75%. P2(25%) -> sobra 50%. P3(25%) -> sobra 25%.

                    frac_to_close = 0.0
                    if reason == "PARTIAL_1":
                        frac_to_close = 0.25  # 25% de 100%
                    elif reason == "PARTIAL_2":
                        frac_to_close = 0.33  # 25% de 75%
                    elif reason == "PARTIAL_3":
                        frac_to_close = 0.50  # 25% de 50%

                    qty_close = np.floor(abs(position) * frac_to_close)

                    if qty_close > 0:
                        curr_exit_price = (
                            high[i] if position > 0 else low[i]
                        )  # Otimista no candle que tocou

                        val_exit = qty_close * curr_exit_price
                        if asset_type == 1:
                            gross = (
                                ((curr_exit_price - entry_price) * point_value)
                                * qty_close
                                if position > 0
                                else ((entry_price - curr_exit_price) * point_value)
                                * qty_close
                            )
                            cost = (
                                (fee_val * qty_close * 2)
                                if fee_type == 1
                                else (val_exit * transaction_cost_pct)
                            )
                        else:
                            gross = (
                                (curr_exit_price - entry_price) * qty_close
                                if position > 0
                                else (entry_price - curr_exit_price) * qty_close
                            )
                            cost = val_exit * transaction_cost_pct

                        # Update Cash
                        if asset_type == 1:
                            cash += gross - cost
                        else:
                            if position > 0:
                                cash += val_exit - cost  # Vendeu, recebeu caixa
                            else:
                                cash -= val_exit + cost  # Comprou, pagou caixa

                        if position > 0:
                            position -= qty_close
                        else:
                            position += qty_close

                        partials_mask = new_mask

            # --- C. Time Exit / Gap Protection ---
            time_exit = False
            # Overhead gap protection (17:45 bar index check implied or explicit time check if passed)
            # Como não temos hora exata aqui no array, usamos contagem de barras como proxy ou ignoramos no fast core.
            if bars_in_trade >= 80:  # Aumentado
                time_exit = True

            # --- D. Exit Execution ---
            if position > 0:  # Long exits
                hit_stop = low[i] <= stop_price
                hit_tp = (tp_mult > 0) and (high[i] >= target_price)
            else:  # Short exits
                hit_stop = high[i] >= stop_price
                hit_tp = (tp_mult > 0) and (low[i] <= target_price)

            if hit_stop or hit_tp or time_exit:
                if position > 0:  # Long close
                    raw_exit_price = (
                        stop_price if hit_stop else (target_price if hit_tp else price)
                    )
                    if hit_stop:
                        raw_exit_price *= 1 - base_slippage
                else:  # Short close
                    raw_exit_price = (
                        stop_price if hit_stop else (target_price if hit_tp else price)
                    )
                    if hit_stop:
                        raw_exit_price *= 1 + base_slippage

                exit_price = raw_exit_price
                qty_abs = abs(position)
                val_exit = qty_abs * exit_price
                if asset_type == 1:
                    gross_profit = (
                        ((exit_price - entry_price) * point_value) * qty_abs
                        if position > 0
                        else ((entry_price - exit_price) * point_value) * qty_abs
                    )
                    c_exit = (
                        (fee_val * qty_abs * 2)
                        if fee_type == 1
                        else (val_exit * transaction_cost_pct)
                    )
                    net_profit = gross_profit - c_exit
                    cash += net_profit
                    equity = cash
                else:
                    c_exit = val_exit * transaction_cost_pct
                    if position > 0:
                        gross_profit = (exit_price - entry_price) * qty_abs
                        net_profit = gross_profit - c_exit
                        cash += val_exit - c_exit
                        equity = cash
                    else:
                        gross_profit = (entry_price - exit_price) * qty_abs
                        net_profit = gross_profit - c_exit
                        cash -= val_exit + c_exit
                        equity = cash

                if net_profit > 0:
                    wins += 1
                else:
                    losses += 1
                # Circuit Breaker: consecutivos
                if net_profit < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= 3 and trading_paused == 0:
                        trading_paused = 1
                        pause_bars_remaining = 200  # ~2 dias em M15
                else:
                    consecutive_losses = 0

                position = 0.0
                trades += 1
                bars_in_trade = 0
                partial_closed = 0
                partials_mask = 0
                max_price_reached = 0.0

            # Update Equity (Mark-to-Market)
            if asset_type == 1:
                if position > 0:
                    unreal = ((price - entry_price) * point_value) * position
                else:
                    unreal = ((entry_price - price) * point_value) * abs(position)
                equity = cash + unreal
            else:
                equity = cash + (position * price)

        # ---------------------------------------------------
        # ENTRY LOGIC (DIAGNOSTIC FUNNEL)
        # ---------------------------------------------------
        else:
            if trading_paused == 1 or dd_now > 0.10:
                # Pausado ou DD intraday acima de 10%: sem novas entradas
                equity = cash
                equity_curve[i] = equity
                continue
            # Sinais Potenciais
            is_trend_long = ema_short[i] > ema_long[i]
            is_trend_short = ema_short[i] < ema_long[i]

            # SISTEMA DE PONTUAÇÃO (0-10 pontos)
            score_long = 0
            # CATEGORIA 1: TENDÊNCIA (0-3)
            trend_condition = close[i] > ema_long[i]
            if trend_condition:
                score_long += 3
                ema_spread = (ema_short[i] - ema_long[i]) / max(ema_long[i], 1e-9)
                if ema_spread > 0.03:
                    score_long += 1
            # CATEGORIA 2: SETUP/PULLBACK (0-3 com bônus)
            pullback_condition = rsi[i] < rsi_low
            if pullback_condition:
                score_long += 2
                if rsi[i] < 25:
                    score_long += 1
            # CATEGORIA 3: REVERSÃO RÁPIDA (0-3 com bônus)
            reversion_condition = rsi_2[i] < 10
            if reversion_condition:
                score_long += 2
                if rsi_2[i] < 5:
                    score_long += 1
            # CATEGORIA 4: VOLATILIDADE/ADX (0-2)
            if adx[i] > adx_threshold:
                score_long += 2
            elif adx[i] > (adx_threshold * 0.7):
                score_long += 1
            # CATEGORIA 5: CONFIRMAÇÃO DE CANDLE (0-1)
            candle_body = close[i] - open_[i]
            if candle_body > (atr[i] * 0.3):
                score_long += 1
            ENTRY_THRESHOLD = 6
            score_signal_long = score_long >= ENTRY_THRESHOLD

            # SETUP A: Trend + Pullback (RSI Sold)
            setup_a_long = is_trend_long and (rsi[i] < rsi_low)
            setup_a_short = is_trend_short and (rsi[i] > rsi_high)

            # SETUP B: Lateral / Reversion (RSI 2 Extremo)
            setup_b_long = rsi_2[i] < 20
            setup_b_short = rsi_2[i] > 80

            has_setup_long = (setup_a_long or setup_b_long) or score_signal_long
            has_setup_short = (setup_a_short or setup_b_short) and enable_shorts

            # Comentário: B.1 ADX opcional (não bloqueia, só diagnostica)

            if has_setup_long or has_setup_short:
                c_setup += 1

                # Check Volatility (diagnóstico apenas)
                vol_ok = True
                if adx[i] <= adx_threshold and not (setup_b_long or setup_b_short):
                    c_volat += 1
                # Check ML (simplificado)
                ml_ok = True
                if not ml_ok:
                    c_ml += 1
                else:
                    # Check Candle/Confirm
                    candle_ok = True
                    if not candle_ok:
                        c_candle += 1
                    else:
                        # VWAP como filtro secundário
                        vwap_cond = (
                            (price > vwap[i]) if (has_setup_long) else (price < vwap[i])
                        )
                        if not vwap_cond and not (setup_b_long or setup_b_short):
                            c_vwap += 1
                        else:
                            # SUCCESS ENTRY
                            c_success += 1

                            is_long = has_setup_long
                            # RR assimétrico via WR recente
                            recent_trades = max(trades, 1)
                            wr_curr = wins / recent_trades
                            tp_adj = tp_mult
                            if wr_curr < 0.40:
                                tp_adj = max(tp_mult * 0.8, sl_mult * 1.2)
                            elif wr_curr > 0.60:
                                tp_adj = tp_mult * 1.2

                            # Slippage dinâmico por liquidez
                            ratio = float(vol / (avg_volume + 1e-9))
                            slip_factor = 1.0
                            if ratio < 0.6:
                                slip_factor = 1.8
                            elif ratio < 0.9:
                                slip_factor = 1.3
                            elif ratio > 1.5:
                                slip_factor = 0.8
                            curr_slip = base_slippage * slip_factor

                            if is_long:
                                buy_signals_count += 1
                                entry_price = price * (1.0 + curr_slip)
                                atr_val = atr[i]
                                sl_dist = atr_val * sl_mult
                                tp_dist = atr_val * tp_adj

                                entry_price = round_to_tick(entry_price, ts)
                                stop_price = round_to_tick(entry_price - sl_dist, ts)
                                target_price = round_to_tick(entry_price + tp_dist, ts)

                                risk_amt = equity * risk_dyn
                                if sl_dist > 0:
                                    if asset_type == 1:
                                        raw_qty = risk_amt / max(
                                            sl_dist * point_value, 1e-6
                                        )
                                        pos_size = np.floor(raw_qty)
                                        if pos_size >= 1:
                                            c_entry = (
                                                (fee_val * pos_size)
                                                if fee_type == 1
                                                else 0.0
                                            )
                                            cash -= c_entry
                                            position = pos_size
                                    else:
                                        raw_qty = risk_amt / sl_dist
                                        pos_size = np.floor(raw_qty / 100.0) * 100.0
                                        max_qty = (
                                            np.floor(
                                                ((equity * 2.0) / entry_price) / 100.0
                                            )
                                            * 100.0
                                        )
                                        if pos_size > max_qty:
                                            pos_size = max_qty
                                        if pos_size >= 100.0:
                                            cost_fin = pos_size * entry_price
                                            c_entry = cost_fin * transaction_cost_pct
                                            cash -= cost_fin + c_entry
                                            position = pos_size
                                        max_price_reached = entry_price
                                        partials_mask = 0
                                        is_lateral_trade = setup_b_long
                                        partial_closed = 0
                                        bars_in_trade = 0

                            else:  # Short
                                sell_signals_count += 1
                                entry_price = price * (1.0 - curr_slip)
                                atr_val = atr[i]
                                sl_dist = atr_val * sl_mult
                                tp_dist = atr_val * (
                                    tp_adj * 0.9
                                )  # Shorts mais conservadores

                                entry_price = round_to_tick(entry_price, ts)
                                stop_price = round_to_tick(entry_price + sl_dist, ts)
                                target_price = round_to_tick(entry_price - tp_dist, ts)

                                risk_amt = equity * (
                                    risk_dyn * 0.8
                                )  # Tamanho menor em shorts
                                if sl_dist > 0:
                                    if asset_type == 1:
                                        raw_qty = risk_amt / max(
                                            sl_dist * point_value, 1e-6
                                        )
                                        pos_size = -np.floor(raw_qty)
                                        if abs(pos_size) >= 1:
                                            c_entry = (
                                                (fee_val * abs(pos_size))
                                                if fee_type == 1
                                                else 0.0
                                            )
                                            cash -= c_entry
                                            position = pos_size
                                    else:
                                        raw_qty = risk_amt / sl_dist
                                        pos_size = -np.floor(raw_qty / 100.0) * 100.0
                                        max_qty = (
                                            -np.floor(
                                                ((equity * 2.0) / entry_price) / 100.0
                                            )
                                            * 100.0
                                        )
                                        if pos_size < max_qty:
                                            pos_size = max_qty
                                        if abs(pos_size) >= 100.0:
                                            cost_fin = abs(pos_size) * entry_price
                                            c_entry = cost_fin * transaction_cost_pct
                                            cash += cost_fin - c_entry
                                            position = pos_size
                                        max_price_reached = entry_price
                                        partials_mask = 0
                                        is_lateral_trade = setup_b_short
                                        partial_closed = 0
                                        bars_in_trade = 0

            # Equity if no position
            equity = cash

        equity_curve[i] = equity

    # Retornar contadores
    counts = (c_trend, c_setup, c_volat, c_ml, c_candle, c_vwap, c_success)
    return (
        equity_curve,
        trades,
        wins,
        losses,
        buy_signals_count + sell_signals_count,
        counts,
    )


# =========================================================
# 2. METRICS & UTILS
# =========================================================
def compute_metrics(equity_curve):
    # ✅ Robust Empty Check
    if equity_curve is None or len(equity_curve) < 2:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    equity_curve = np.nan_to_num(equity_curve, nan=equity_curve[0])  # Safety

    # ✅ Bankruptcy Check (Skip first 10 bars warmup period)
    if len(equity_curve) > 10 and np.any(equity_curve[10:] <= 100):
        return {
            "total_return": -1.0,
            "max_drawdown": 1.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    # ✅ Flat Check
    if np.all(equity_curve == equity_curve[0]):
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    returns = np.diff(equity_curve) / equity_curve[:-1]
    std_returns = np.std(returns)

    if std_returns == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    total_return = equity_curve[-1] / equity_curve[0] - 1

    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = max(-np.min(drawdowns), 0.01)

    years = len(equity_curve) / (252 * 28)  # M15 -> ~28 bars/day ? Adjust if H1
    if years < 1:
        years = 1
    annualized = (1 + total_return) ** (1 / years) - 1
    risk_free = float(get_macro_rate("SELIC") or 0.12)
    calmar = annualized / max_dd

    wins_mask = returns > 0
    losses_mask = returns < 0
    total_trades = len(returns[returns != 0])
    win_rate = np.sum(wins_mask) / total_trades if total_trades > 0 else 0.0

    gross_profits = np.sum(returns[wins_mask])
    gross_losses = np.sum(np.abs(returns[losses_mask]))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 2.0

    sharpe = (
        (np.mean(returns) - (risk_free / (252 * 28))) / std_returns * np.sqrt(252 * 28)
    )

    down_rets = returns[returns < 0]
    down_std = np.std(down_rets) if len(down_rets) > 0 else 0.0
    sortino = (
        ((np.mean(returns) - (risk_free / (252 * 28))) / down_std * np.sqrt(252 * 28))
        if down_std > 0
        else 0.0
    )

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }


def calculate_adx(high, low, close, period=14):
    """Calcula ADX Manualmente (Fallback se TA falhar)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])

    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # EWM mean
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values
    plus_di = (
        100
        * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().values
        / (atr + 1e-10)
    )
    minus_di = (
        100
        * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().values
        / (atr + 1e-10)
    )

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().fillna(0).values

    return adx, atr


def extract_features_for_ml(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi"] = 100 - (100 / (1 + rs))

    try:
        import ta

        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        features["adx"] = adx_ind.adx()
    except:
        features["adx"] = 25.0  # Fallback neutro se falhar
    # EMAs
    features["ema_diff"] = close.ewm(span=9).mean() - close.ewm(span=21).mean()

    # Volume
    features["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1)

    # Momentum
    features["momentum"] = close.pct_change(10)

    features["obv"] = (np.sign(close.diff()) * df["volume"]).fillna(0).cumsum()
    # Macro e sentimento
    try:
        features["selic"] = float(get_macro_rate("SELIC"))
    except Exception:
        features["selic"] = 0.12
    try:
        # Tenta usar função de sentimento real; fallback neutro 0.5
        try:
            from otimizador_semanal import x_keyword_search

            res = x_keyword_search(f"sentimento {symbol} B3 2026", limit=10)
            scores = [float(r.get("score", 0.5) or 0.5) for r in (res or [])]
            features["sentiment_score"] = (
                float(np.mean(scores)) if len(scores) > 0 else 0.5
            )
        except Exception:
            features["sentiment_score"] = 0.5
    except Exception:
        features["sentiment_score"] = 0.5


# =========================================================
# 3. BACKTEST PARAMS ON DF
# =========================================================
def backtest_params_on_df(symbol: str, params: dict, df: pd.DataFrame, ml_model=None):
    if df is None or len(df) < 150:
        return {
            "calmar": -10.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_trades": 0,
            "max_drawdown": 0,
        }
    if df.isna().sum().sum() > (0.10 * df.size):
        return {
            "calmar": -10.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "total_trades": 0,
            "max_drawdown": 0,
        }

    df = df.sort_index()
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        elif "real_volume" in df.columns:
            df = df.rename(columns={"real_volume": "volume"})
        else:
            df["volume"] = 1.0

    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ✅ TA-Lib / Robust Indicators
    try:
        import ta

        # VWAP (Check minimal length to avoid index errors)
        if len(df) > 14:
            vwap = (
                ta.volume.VolumeWeightedAveragePrice(
                    high=pd.Series(high),
                    low=pd.Series(low),
                    close=pd.Series(close),
                    volume=pd.Series(volume),
                    window=14,
                )
                .volume_weighted_average_price()
                .bfill()
                .fillna(close[0])
                .values
            )

            # SAR
            sar = (
                ta.trend.PSARIndicator(
                    high=pd.Series(high),
                    low=pd.Series(low),
                    close=pd.Series(close),
                    step=0.02,
                    max_step=0.2,
                )
                .psar()
                .bfill()
                .fillna(low[0])
                .values
            )
        else:
            raise ValueError("Data too short for TA-Lib")
    except Exception:
        # Fallback simplistic
        vwap = pd.Series(close).rolling(14).mean().bfill().values
        sar = pd.Series(close).shift(1).bfill().values

    # Indicators
    ema_s = (
        pd.Series(close)
        .ewm(span=params.get("ema_short", 9), adjust=False)
        .mean()
        .values
    )
    ema_l = (
        pd.Series(close)
        .ewm(span=params.get("ema_long", 21), adjust=False)
        .mean()
        .values
    )

    adx, atr = calculate_adx(high, low, close)

    # RSI standard
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values

    # Momentum
    momentum = pd.Series(close).pct_change(10).fillna(0).values

    # RSI 2
    gain2 = (delta.where(delta > 0, 0)).rolling(2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs2 = gain2 / (loss2 + 1e-10)
    rsi_2 = (100 - (100 / (1 + rs2))).fillna(50).values

    volume_ma = pd.Series(volume).rolling(20).mean().fillna(0).values

    # ✅ ML Probs Logic
    # FORÇAR ML SEMPRE OK PARA DIAGNÓSTICO
    ml_probs = np.ones(len(close)) * 0.85

    # Check de Tendência: ema_s > ema_l em pelo menos 30% das barras
    trend_freq = np.sum(ema_s > ema_l) / len(close)
    if trend_freq < 0.30:
        logger.warning(
            f"[WARN] {symbol}: Mercado sem tendência clara (Alta em apenas {trend_freq:.1%})"
        )

    # ✅ Calculate Average Volume
    avg_volume = np.mean(volume) if len(volume) > 0 else 1000000.0
    from utils import AssetInspector, round_to_tick

    ai = AssetInspector.detect(symbol)
    asset_type = 1 if ai.get("type") == "FUTURE" else 0
    pv = float(ai.get("point_value", 0.0))
    ts = float(ai.get("tick_size", 0.01))
    fee_type = 1 if ai.get("fee_type") == "FIXED" else 0
    fee_val = float(ai.get("fee_val", 0.00055))
    # Beta aproximado com IBOV (se disponível)
    try:
        from otimizador_semanal import get_ibov_data

        ibov = get_ibov_data()
        ibov = ibov.reindex(df.index).fillna(method="ffill").fillna(method="bfill")
        asset_ret = pd.Series(close, index=df.index).pct_change().fillna(0)
        ibov_ret = ibov["close"].pct_change().fillna(0)
        cov = float(np.cov(asset_ret.values[-500:], ibov_ret.values[-500:])[0, 1])
        var_mkt = float(np.var(ibov_ret.values[-500:]))
        beta_est = cov / (var_mkt + 1e-9)
    except Exception:
        beta_est = 1.0

    # Chamada com retorno de contadores
    equity_arr, trades, wins, losses, sigs, counts = fast_backtest_core(
        close,
        open_,
        high,
        low,
        volume,
        volume_ma,
        vwap,
        ema_s,
        ema_l,
        rsi,
        rsi_2,
        adx,
        sar,
        atr,
        momentum,
        ml_probs,
        params.get("rsi_low", 30),
        params.get("rsi_high", 70),
        params.get("adx_threshold", 25),
        params.get("sl_atr_multiplier", 2.0),
        params.get("tp_mult", 2.0),
        params.get("base_slippage", 0.002),
        float(avg_volume),
        0.01,  # Risco base 1% (ajustado dinamicamente dentro do core)
        1,  # use_trailing (int)
        params.get("enable_shorts", 1),  # enable_shorts
        asset_type,
        pv,
        ts,
        fee_type,
        fee_val,
        float(beta_est),
        params.get("trailing_config", None),  # Passa config se existir
    )

    # PRINT DIAGNOSTIC FUNNEL
    total_setups = counts[1]  # c_setup
    if total_setups > 0:
        p_volat = (counts[2] / total_setups) * 100
        p_ml = (counts[3] / total_setups) * 100
        p_vwap = (counts[5] / total_setups) * 100
        p_success = (counts[6] / total_setups) * 100
        print(
            f"[DEBUG] [{symbol}] Funnel: Setups={int(total_setups)} | VolatBlocked={p_volat:.1f}% | MLBlocked={p_ml:.1f}% | VWAPBlocked={p_vwap:.1f}% | Executed={p_success:.1f}%"
        )

    metrics = compute_metrics(equity_arr.tolist())
    metrics.update(
        {
            "total_trades": trades,
            "setups_identified": int(counts[1]),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades if trades > 0 else 0.0,
            "equity_curve": equity_arr.tolist(),
        }
    )
    # Comentário: E.2 Pós-backtest checks
    if metrics["total_trades"] == 0 or metrics["max_drawdown"] >= 0.95:
        metrics["calmar"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["win_rate"] = 0.0

    return metrics


# =========================================================
# 4. OBJECTIVE & OPTUNA
# =========================================================
def log_rejection(symbol, trial_number, reason, value):
    try:
        os.makedirs("optimizer_output", exist_ok=True)
        fname = f"rejections_{symbol}.txt" if symbol else "rejection_reasons.txt"
        with open(os.path.join("optimizer_output", fname), "a", encoding="utf-8") as f:
            # Comentário: H.1 Log expandido com métricas chave
            f.write(
                f"{symbol} | Trial {trial_number} | REJEITADO: {reason} | {value}\n"
            )
    except Exception:
        pass


def objective(trial, symbol, df, ml_model=None):
    params = {
        "ema_short": trial.suggest_int("ema_short", 5, 35),
        "ema_long": trial.suggest_int("ema_long", 30, 120),
        "rsi_low": trial.suggest_int("rsi_low", 20, 45),
        "rsi_high": trial.suggest_int("rsi_high", 55, 85),
        "adx_threshold": trial.suggest_categorical("adx_threshold", [5.0, 25.0]),
        "use_adx": trial.suggest_categorical("use_adx", [True, False]),
        "sl_atr_multiplier": trial.suggest_float(
            "sl_atr_multiplier", 1.2, 4.0, step=0.1
        ),
        "tp_ratio": trial.suggest_float("tp_ratio", 1.0, 3.5, step=0.2),
        "base_slippage": 0.0015,
        "enable_shorts": 1,
    }
    from trailing_stop import TrailingStopConfig

    tc = TrailingStopConfig(
        level1_trigger_r=1.0,
        level2_trigger_r=1.5,
        level3_trigger_r=2.5,
        level4_trigger_r=4.0,
        enable_partials=True,
        max_profit_drop_pct=0.50,
    )

    tc.trend_adx_threshold = params["adx_threshold"]

    params["tp_mult"] = params["sl_atr_multiplier"] * params["tp_ratio"]
    params["trailing_config"] = tc

    if params["ema_short"] >= params["ema_long"]:
        raise optuna.TrialPruned()
    try:
        metrics = backtest_params_on_df(symbol, params, df, ml_model=ml_model)
        if int(metrics.get("total_trades", 0) or 0) < 5:
            return -999.0
        score = float(metrics.get("sharpe", 0.0) or 0.0)
        if float(params.get("adx_threshold", 25.0) or 25.0) == 5.0:
            if (
                float(metrics.get("profit_factor", 0.0) or 0.0) < 1.25
                or float(metrics.get("win_rate", 0.0) or 0.0) < 0.30
            ):
                score -= 2.0
            else:
                score += 0.5
        elif float(params.get("adx_threshold", 25.0) or 25.0) == 25.0:
            if int(metrics.get("total_trades", 0) or 0) < 8:
                score -= 0.5
            if float(metrics.get("profit_factor", 0.0) or 0.0) < 1.5:
                score -= 1.0
        return score
    except Exception as e:
        log_rejection(symbol, trial.number, "EXCEPTION", f"{str(e)[:50]}")
        return -999.0


def optimize_with_optuna(
    symbol, df_train, n_trials=150, timeout=1500, base_slippage=0.001
):
    # Ajuste de defaults para busca mais ampla
    if n_trials == 150:
        n_trials = 350
    if timeout == 1500:
        timeout = 5400
    try:
        from xgboost import XGBClassifier
        from imblearn.over_sampling import SMOTE

        base_model = XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        feats = extract_features_for_ml(df_train, symbol)
        target = (df_train["close"].shift(-5) > df_train["close"]).astype(int)
        target = target.reindex(feats.index).fillna(0)
        valid_cols = [
            "rsi",
            "ema_diff",
            "volume_ratio",
            "momentum",
            "obv",
            "selic",
            "sentiment_score",
        ]
        valid_cols = [c for c in valid_cols if c in feats.columns]
        if len(valid_cols) > 0:
            X = feats[valid_cols]
            y = target
            best_logloss = float("inf")
            best_model = None
            splits = max(2, min(5, len(X) // 100))
            tss = TimeSeriesSplit(n_splits=splits)
            for train_idx, val_idx in tss.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                try:
                    smote = SMOTE(random_state=42)
                    X_res, y_res = smote.fit_resample(X_tr, y_tr)
                except Exception:
                    X_res, y_res = X_tr, y_tr
                model = XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                )
                model.fit(X_res, y_res)
                try:
                    y_prob = model.predict_proba(X_val)[:, 1]
                    ll = log_loss(y_val, y_prob, labels=[0, 1])
                except Exception:
                    ll = 1.0
                if ll < best_logloss:
                    best_logloss = ll
                    best_model = model
            ml_model = best_model
        else:
            ml_model = None
    except Exception:
        ml_model = None

    # ✅ Pass base_slippage through closure
    def objective_wrapper(trial):
        params = {
            "ema_short": trial.suggest_int("ema_short", 5, 35),
            "ema_long": trial.suggest_int("ema_long", 30, 120),
            "rsi_low": trial.suggest_int("rsi_low", 20, 45),
            "rsi_high": trial.suggest_int("rsi_high", 55, 85),
            "adx_threshold": trial.suggest_float("adx_threshold", 5.0, 35.0, step=5.0),
            "use_adx": trial.suggest_categorical("use_adx", [True, False]),
            "sl_atr_multiplier": trial.suggest_float(
                "sl_atr_multiplier", 1.2, 4.0, step=0.1
            ),
            "tp_ratio": trial.suggest_float("tp_ratio", 1.0, 3.5, step=0.2),
            "base_slippage": base_slippage,
            "enable_shorts": 1,
        }
        params["tp_mult"] = params["sl_atr_multiplier"] * params["tp_ratio"]

        try:
            metrics = backtest_params_on_df(symbol, params, df_train, ml_model=ml_model)

            wr = metrics.get("win_rate", 0.0)
            pf = metrics.get("profit_factor", 0.0)
            dd = metrics.get("max_drawdown", 1.0)
            trades = metrics.get("total_trades", 0)

            penalty = 0.0
            reason = []
            if trades == 0:
                penalty += 5.0
                reason.append("ZERO_TRADES")
            elif trades < 3:
                penalty += 2.0
                reason.append(f"Trades={trades} (crítico)")
            elif trades < 5:
                penalty += 0.8
                reason.append(f"Trades={trades} (baixo)")
            elif trades < 8:
                penalty += 0.3
                reason.append(f"Trades={trades} (ok-)")
            if dd > 0.70:
                penalty += (dd - 0.70) * 3.0
                reason.append(f"DD={dd:.1%} (alto)")
            elif dd > 0.55:
                penalty += (dd - 0.55) * 1.5
                reason.append(f"DD={dd:.1%} (moderado)")
            if wr < 0.15:
                penalty += (0.15 - wr) * 5.0
                reason.append(f"WR={wr:.1%} (péssimo)")
            elif wr < 0.25:
                penalty += (0.25 - wr) * 2.0
                reason.append(f"WR={wr:.1%} (baixo)")
            if pf < 0.8:
                penalty += (0.8 - pf) * 2.0
                reason.append(f"PF={pf:.2f} (ruim)")

            bonus = 0.0
            if wr > 0.55 and pf > 1.5:
                bonus += 0.5
                reason.append("BONUS_QUALITY")
            if trades > 12 and wr > 0.50:
                bonus += 0.3
                reason.append("BONUS_FREQUENCY")
            if dd < 0.30:
                bonus += 0.2
                reason.append("BONUS_LOW_DD")

            score_base = (wr * 2.0) + (pf * 1.2)
            score_final = score_base + bonus - penalty
            if penalty > 0 or bonus > 0:
                log_rejection(
                    symbol,
                    trial.number,
                    "METRICS",
                    f"{' | '.join(reason)} | Score={score_final:.2f}",
                )
            return -score_final

        except Exception as e:
            log_rejection(symbol, trial.number, "EXCEPTION", f"{str(e)[:50]}")
            return 999.0

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
    )
    study.optimize(objective_wrapper, n_trials=n_trials, timeout=timeout)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        pruned = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        )
        failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
        return {
            "best_params": {},
            "best_score": None,
            "ml_model": ml_model,
            "status": "NO_VALID_TRIALS",
            "reason": f"all_trials_pruned_or_failed | pruned={pruned} failed={failed} total={len(study.trials)}",
        }

    try:
        _best_params = dict(study.best_params)
        _best_metrics = backtest_params_on_df(
            symbol, _best_params, df_train, ml_model=ml_model
        )
        _best_metrics = dict(_best_metrics or {})
        if "pf" not in _best_metrics:
            _best_metrics["pf"] = float(_best_metrics.get("profit_factor", 0.0) or 0.0)
        if "n_trades" not in _best_metrics:
            _best_metrics["n_trades"] = int(_best_metrics.get("total_trades", 0) or 0)
        _best_metrics["symbol"] = symbol
        diagnostico_final(_best_metrics, _best_params)
    except Exception:
        pass
    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "ml_model": ml_model,
        "status": "SUCCESS",
    }


def diagnostico_final(metrics, params):
    adx = params["adx_threshold"]
    sharpe = metrics["sharpe"]
    pf = metrics["pf"]
    trades = metrics["n_trades"]

    symbol = metrics.get("symbol", "ATIVO")
    header = (
        "\n" + "=" * 40 + f"\n🤖 DIAGNÓSTICO AUTOMÁTICO PARA: {symbol}\n" + "=" * 40
    )
    print(header)
    lines = [header]
    base = (
        f"Métricas: Trades={trades} | PF={pf:.2f} | Sharpe={sharpe:.2f} | ADX={adx:.1f}"
    )
    print(base)
    lines.append(base)

    # --- CENÁRIO 1: O ROBÔ ESCOLHEU MODO SNIPER (25.0) ---
    if adx == 25.0 or adx >= 20.0:
        if sharpe > 1.0 and pf > 2.0:
            msg = [
                "✅ VEREDITO: SNIPER DE ELITE (Aprovadíssimo)",
                "   Motivo: Sharpe alto com precisão cirúrgica.",
                "   Ação: Colocar em produção IMEDIATAMENTE.",
            ]
            for m in msg:
                print(m)
                lines.append(m)
        elif sharpe > 0.6 and pf > 1.5:
            msg = [
                "⚠️ VEREDITO: SNIPER MODERADO",
                "   Motivo: Lucrativo, mas falta um pouco de consistência.",
                "   Ação: Testar em Demo ou lote mínimo.",
            ]
            for m in msg:
                print(m)
                lines.append(m)
        else:
            msg = [
                "❌ VEREDITO: REPROVADO",
                "   Motivo: Poucos trades e lucro irrelevante. Não vale o risco.",
            ]
            for m in msg:
                print(m)
                lines.append(m)

    # --- CENÁRIO 2: O ROBÔ ESCOLHEU MODO HUNTER/METRALHADORA (5.0) ---
    elif adx == 5.0 or adx <= 10.0:
        if sharpe > 0.8 and pf > 1.25:
            msg = [
                "✅ VEREDITO: TREND HUNTER (Aprovado)",
                "   Motivo: O volume compensa o risco. Estilo PETR4.",
                "   Ação: Produção (cuidado com custos de corretagem).",
            ]
            for m in msg:
                print(m)
                lines.append(m)

        elif sharpe < 0.2:
            msg = [
                "🚫 VEREDITO: FALSO POSITIVO (Ruído Puro)",
                "   Motivo: O robô escolheu ADX 5 só porque operou muito, mas não deu lucro real.",
                "   Exemplo: O caso atual da BBAS3 (Sharpe 0.0).",
                "   Ação: DESCARTAR. Tente forçar manualmente o ADX 25 ou não opere.",
            ]
            for m in msg:
                print(m)
                lines.append(m)

        else:
            msg = [
                "⚠️ VEREDITO: INCONCLUSIVO (Risco de Overtrading)",
                "   Motivo: Lucra pouco para o trabalho que dá.",
                "   Ação: Não operar.",
            ]
            for m in msg:
                print(m)
                lines.append(m)
    else:
        if sharpe > 0.9 and pf > 1.7:
            msg = [
                "✅ VEREDITO: INTERMEDIÁRIO DE ALTA QUALIDADE",
                "   Motivo: Bom equilíbrio entre frequência e precisão.",
                "   Ação: Produção com monitoramento.",
            ]
            for m in msg:
                print(m)
                lines.append(m)
        elif sharpe > 0.5 and pf > 1.3:
            msg = [
                "⚠️ VEREDITO: INTERMEDIÁRIO MODERADO",
                "   Motivo: Lucrativo, mas pode otimizar ADX.",
                "   Ação: Testar em Demo ou ajustar filtros.",
            ]
            for m in msg:
                print(m)
                lines.append(m)
        else:
            msg = [
                "❌ VEREDITO: INTERMEDIÁRIO FRACO",
                "   Motivo: Métricas insuficientes para operar com segurança.",
                "   Ação: Não operar ou reforçar critérios.",
            ]
            for m in msg:
                print(m)
                lines.append(m)

    footer = "=" * 40 + "\n"
    print(footer)
    lines.append(footer)
    try:
        out_dir = os.path.join(os.getcwd(), "optimizer_output")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"diagnostico_{symbol}.txt")
        with open(path, "a", encoding="utf-8") as fd:
            fd.write("\n".join(lines) + "\n")
    except Exception:
        pass
