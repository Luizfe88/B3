"""
CONFIGURAÇÃO DE RISCO XP3 v5

Centraliza parâmetros de gestão de risco, tuning e sanity checks.
"""

# A.1 Dynamic Position Sizing
DYNAMIC_POSITION_SIZING = {
    "atr_high_pct": 0.04,  # Reduz 50% se ATR% > 4%
    "atr_mid_pct": 0.025,  # Reduz 25% se ATR% > 2.5%
    "beta_threshold": 1.30,  # Reduz 30% se Beta > 1.3
    "dd_threshold": 0.15,  # Reduz 50% se DD > 15%
}

# A.2 Circuit Breakers
CIRCUIT_BREAKERS = {
    "max_consecutive_losses": 4,  # Pausa ao atingir 4 perdas seguidas (Moderado)
    "pause_bars_m15": 200,  # Aproximadamente 2 dias em M15
    "intraday_dd_limit": 0.15,  # Pausa se DD intraday > 15% (Permite reversões)
}

# A.3 RR Assimétrico
ASYMMETRIC_RR = {
    "wr_low": 0.40,  # TP reduzido se WR < 40%
    "wr_high": 0.60,  # TP ampliado se WR > 60%
    "short_tp_factor": 0.90,  # Shorts mais conservadores (TP 90%)
    "short_risk_factor": 0.80,  # Menor tamanho em shorts (80%)
}

# C. Optuna Tuning
OPTUNA_TUNING = {
    "ema_short_min": 8,
    "ema_short_max": 30,
    "ema_long_min": 35,
    "ema_long_max": 100,
    "rsi_low_min": 25,
    "rsi_low_max": 40,
    "rsi_high_min": 60,
    "rsi_high_max": 80,
    "adx_min": 15,
    "adx_max": 35,
    "sl_mult_min": 1.5,
    "sl_mult_max": 3.5,
    "sl_mult_step": 0.1,
    "tp_ratio_min": 1.2,
    "tp_ratio_max": 3.0,
    "tp_ratio_step": 0.2,
    "base_slippage": 0.0015,
    "n_trials": 150,
    "timeout_sec": 1500,
}

# F. Markowitz Protegido
MARKOWITZ_RULES = {
    "sector_cap": 0.25,  # Teto por setor
    "blue_min": 0.50,  # Mínimo Blue Chips
    "opp_max": 0.50,  # Máximo Oportunidades
    "prefilter_dd_max": 0.70,  # Exclui DD >= 70% (Mais maleável p/ ações voláteis)
    "prefilter_trades_min": 10,  # Exclui < 10 trades
    "prefilter_liquidity_min": 10_000_000,  # Liquidez mínima
}

# I. Sanity Checks
SANITY_CHECKS = {
    "min_wr_forward": 0.30,
    "min_calmar_forward": 0.0,
    "min_calmar_stress": -0.20,
    "min_ratio_vs_buyhold": 0.50,
}

# J. Hotfix Small Account (R$ 500) - Teste de Estresse
HOTFIX_SMALL_ACCOUNT = {
    "enabled": True,
    "max_daily_loss_brl": 25.00,
    "break_even_trigger_brl": 10.00,
    "profit_shield_activation_brl": 40.00,
    "profit_shield_trailing_pct": 0.25, # Protege 75% (recuo de 25%)
    "min_conviction_threshold": 75,      # Em percentual (0-100)
    "fixed_lot_size": 100,
    "fixed_lot_equity_threshold": 1000.0,
    "force_virtual_equity": 500.0       # Sobrescreve o equity real para fins de sizing
}
