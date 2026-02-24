import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TrailingStopConfig:
    level1_trigger_r: float = 1.0
    level2_trigger_r: float = 1.5
    level3_trigger_r: float = 2.5
    level4_trigger_r: float = 4.0
    level2_dist_atr: float = 1.5
    level3_dist_atr: float = 1.0
    level4_dist_atr: float = 0.5
    enable_partials: bool = True
    partial1_trigger_r: float = 1.5
    partial1_pct: float = 0.25
    partial2_trigger_r: float = 2.5
    partial2_pct: float = 0.25
    partial3_trigger_r: float = 4.0
    partial3_pct: float = 0.25
    trend_adx_threshold: float = 25.0
    super_trend_adx: float = 40.0
    max_profit_drop_pct: float = 0.50


def calculate_dynamic_stop(
    current_price: float,
    entry_price: float,
    current_stop_price: float,
    max_price_reached: float,
    atr: float,
    position_side: int,
    config: TrailingStopConfig,
    candle_low: Optional[float] = None,
    candle_high: Optional[float] = None,
    adx: Optional[float] = None,
) -> Tuple[float, str]:
    if atr <= 0:
        return current_stop_price, "ATR_ZERO"

    new_stop = current_stop_price
    reason = "HOLD"

    # Calculate R-Multiple (How many ATRs in profit we are/ were at max)
    if position_side == 1:  # LONG
        current_r = (current_price - entry_price) / atr
        max_r = (max_price_reached - entry_price) / atr
    else:  # SHORT
        current_r = (entry_price - current_price) / atr
        max_r = (entry_price - max_price_reached) / atr

    # Level 1
    if max_r >= config.level1_trigger_r:
        if position_side == 1:
            be_price = entry_price + (atr * 0.1)
            if be_price > new_stop:
                new_stop = be_price
                reason = "LEVEL1_BE"
        else:
            be_price = entry_price - (atr * 0.1)
            if be_price < new_stop:
                new_stop = be_price
                reason = "LEVEL1_BE"

    # Level 2
    if max_r >= config.level2_trigger_r:
        if position_side == 1:
            trail_price = max_price_reached - (atr * config.level2_dist_atr)
            if trail_price > new_stop:
                new_stop = trail_price
                reason = "LEVEL2_TRAIL"
        else:
            trail_price = max_price_reached + (atr * config.level2_dist_atr)
            if trail_price < new_stop:
                new_stop = trail_price
                reason = "LEVEL2_TRAIL"

    # Level 3
    if max_r >= config.level3_trigger_r:
        if position_side == 1:
            trail_price = max_price_reached - (atr * config.level3_dist_atr)
            if trail_price > new_stop:
                new_stop = trail_price
                reason = "LEVEL3_TRAIL"
        else:
            trail_price = max_price_reached + (atr * config.level3_dist_atr)
            if trail_price < new_stop:
                new_stop = trail_price
                reason = "LEVEL3_TRAIL"

    # Level 4
    if max_r >= config.level4_trigger_r:
        if position_side == 1:
            trail_price = max_price_reached - (atr * config.level4_dist_atr)
            if trail_price > new_stop:
                new_stop = trail_price
                reason = "LEVEL4_TRAIL"
        else:
            trail_price = max_price_reached + (atr * config.level4_dist_atr)
            if trail_price < new_stop:
                new_stop = trail_price
                reason = "LEVEL4_TRAIL"

    # Level 5
    if adx is not None and adx >= config.super_trend_adx:
        if position_side == 1 and candle_low is not None:
            st_price = candle_low
            if st_price > new_stop:
                new_stop = st_price
                reason = "LEVEL5_SUPER_TREND"
        elif position_side == -1 and candle_high is not None:
            st_price = candle_high
            if st_price < new_stop:
                new_stop = st_price
                reason = "LEVEL5_SUPER_TREND"

    if max_r > 2.0:
        if position_side == 1:
            max_profit = max_price_reached - entry_price
            current_profit = current_price - entry_price
            if current_profit < (max_profit * (1.0 - config.max_profit_drop_pct)):
                tightest_stop = current_price - (atr * 0.1)
                if tightest_stop > new_stop:
                    new_stop = tightest_stop
                    reason = "CIRCUIT_BREAKER"
        else:
            max_profit = entry_price - max_price_reached
            current_profit = entry_price - current_price
            if current_profit < (max_profit * (1.0 - config.max_profit_drop_pct)):
                tightest_stop = current_price + (atr * 0.1)
                if tightest_stop < new_stop:
                    new_stop = tightest_stop
                    reason = "CIRCUIT_BREAKER"

    return new_stop, reason


def check_partial_exit_level(
    max_price_reached: float,
    entry_price: float,
    atr: float,
    position_side: int,
    partials_taken: int,  # Bitmask: 1=Level1 taken, 2=Level2 taken, 4=Level3 taken
    config: TrailingStopConfig,
) -> Tuple[bool, int, float, str]:
    """
    Checks if a partial exit level has been reached.
    Returns (should_exit, updated_partials_mask, pct_to_close, reason)
    """
    if not config.enable_partials or atr <= 0:
        return False, partials_taken, 0.0, ""

    if position_side == 1:
        current_r = (max_price_reached - entry_price) / atr
    else:
        current_r = (entry_price - max_price_reached) / atr

    # Level 1 Partial (25% at 1.5R)
    if current_r >= config.partial1_trigger_r and not (partials_taken & 1):
        return True, partials_taken | 1, config.partial1_pct, "PARTIAL_1"

    # Level 2 Partial (25% at 2.5R)
    if current_r >= config.partial2_trigger_r and not (partials_taken & 2):
        return True, partials_taken | 2, config.partial2_pct, "PARTIAL_2"

    # Level 3 Partial (25% at 4.0R)
    if current_r >= config.partial3_trigger_r and not (partials_taken & 4):
        return True, partials_taken | 4, config.partial3_pct, "PARTIAL_3"

    return False, partials_taken, 0.0, ""


def check_partial_exit(
    max_price_reached: float,
    entry_price: float,
    atr: float,
    position_side: int,
    partials_taken: int,
    config: TrailingStopConfig,
) -> Tuple[bool, int, float, str]:
    return check_partial_exit_level(
        max_price_reached=max_price_reached,
        entry_price=entry_price,
        atr=atr,
        position_side=position_side,
        partials_taken=partials_taken,
        config=config,
    )


def check_circuit_breaker(
    current_price: float,
    entry_price: float,
    max_price_reached: float,
    atr: float,
    position_side: int,
    config: TrailingStopConfig,
) -> bool:
    if atr <= 0:
        return False
    if position_side == 1:
        max_r = (max_price_reached - entry_price) / atr
        if max_r <= 2.0:
            return False
        max_profit = max_price_reached - entry_price
        current_profit = current_price - entry_price
        return current_profit < (max_profit * (1.0 - config.max_profit_drop_pct))
    else:
        max_r = (entry_price - max_price_reached) / atr
        if max_r <= 2.0:
            return False
        max_profit = entry_price - max_price_reached
        current_profit = entry_price - current_price
        return current_profit < (max_profit * (1.0 - config.max_profit_drop_pct))
