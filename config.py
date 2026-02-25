"""
Configuração Centralizada v6.0 - XP3 PRO QUANT-REFORM
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Carrega configuração única do config.yaml
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Exporta configurações diretamente para compatibilidade
AGGRESSIVE_MODE = config.get("AGGRESSIVE_MODE", False)

# MT5
MT5_TERMINAL_PATH = config["mt5"]["terminal_path"]
MT5_ACCOUNT = config["mt5"]["account"]
MT5_PASSWORD = config["mt5"]["password"]
MT5_SERVER = config["mt5"]["server"]

# TIMEFRAMES
MACRO_TIMEFRAME = config["timeframes"]["macro"]
MICRO_TIMEFRAME = config["timeframes"]["micro"]
SIGNAL_TIMEFRAME = config["timeframes"]["signal"]

# Kelly Fraction
KELLY_BASE = config["kelly_fraction"]["base"]
KELLY_WINRATE_BOOST = config["kelly_fraction"]["winrate_boost"]
KELLY_LOSS_STREAK_PENALTY = config["kelly_fraction"]["loss_streak_penalty"]
MAX_RUIN_PROBABILITY = config["kelly_fraction"]["max_ruin_probability"]

# Risk Limits
MAX_DAILY_LOSS_MONEY = config["risk_limits"]["max_daily_loss_money"]
MAX_CONCURRENT_POSITIONS = config["risk_limits"]["max_concurrent_positions"]
MAX_LOSSES_PER_SYMBOL_DEFAULT = config["risk_limits"]["max_losses_per_symbol_default"]
MAX_LOSSES_PER_SYMBOL_AGGRESSIVE = config["risk_limits"][
    "max_losses_per_symbol_aggressive"
]
MAX_CAPITAL_ALLOCATION_PCT = config["risk_limits"].get("max_capital_allocation_pct", 0.20)
MAX_TOTAL_EXPOSURE_PCT = config["risk_limits"].get("max_total_exposure_pct", 1.50)
MAX_NEW_POSITIONS_PER_HOUR = config["risk_limits"].get("max_new_positions_per_hour", 4)
MARKET_REGIME_FILTER = config["risk_limits"].get("market_regime_filter", True)

# Position Limits (mapeados do risk_limits)
MAX_SYMBOLS = config["risk_limits"][
    "max_concurrent_positions"
]  # Total de posições simultâneas
MAX_PER_SECTOR = 3  # Máximo de posições por setor (padrão)

# Entry Filters
SPREAD_MAX_MULTIPLIER = config["entry_filters"]["spread_filter"][
    "max_spread_multiplier"
]
SPREAD_LOOKBACK_CANDLES = config["entry_filters"]["spread_filter"]["lookback_candles"]
ANTI_CHOP_CONSECUTIVE_LOSSES = config["entry_filters"]["anti_chop_filter"][
    "consecutive_losses_to_activate"
]
NEWS_BLACKOUT_MINS = config["entry_filters"]["news_blackout_mins"]

# ML Model
ML_CONFIDENCE_BASE = config["ml_model"]["confidence_base"]
ML_DYNAMIC_THRESHOLD_MAX = config["ml_model"]["dynamic_threshold_max"]
ML_TRAINING_DAYS = config["ml_model"]["training_days"]

# Safety Mode
SAFETY_MARKET_DRAWDOWN_PCT = config["safety_mode"]["market_drawdown_trigger_pct"]
SAFETY_VIX_BR_TRIGGER = config["safety_mode"]["vix_br_trigger"]

# Backtest Costs
SLIPPAGE_PCT = config["backtest_costs"]["slippage_pct"]
COMMISSION_PER_ORDER = config["backtest_costs"]["commission_per_order"]
TAX_ON_PROFIT_PCT = config["backtest_costs"]["tax_on_profit_pct"]

# Adaptive Intelligence
ADAPTIVE_ENABLED = config["adaptive_intelligence"]["enabled"]
ADAPTIVE_INTELLIGENCE_ENABLED = ADAPTIVE_ENABLED  # Alias para compatibilidade
ADAPTIVE_MONITORING_INTERVAL = config["adaptive_intelligence"]["monitoring_interval"]
ADAPTIVE_ADJUSTMENT_INTERVAL = config["adaptive_intelligence"]["adjustment_interval"]

# Parâmetros adaptativos dinâmicos
ADAPTIVE_ML_THRESHOLD_MIN = config["adaptive_intelligence"]["adjustable_parameters"][
    "ml_threshold_range"
][0]
ADAPTIVE_ML_THRESHOLD_MAX = config["adaptive_intelligence"]["adjustable_parameters"][
    "ml_threshold_range"
][1]
ADAPTIVE_KELLY_MULT_MIN = config["adaptive_intelligence"]["adjustable_parameters"][
    "kelly_multiplier_range"
][0]
ADAPTIVE_KELLY_MULT_MAX = config["adaptive_intelligence"]["adjustable_parameters"][
    "kelly_multiplier_range"
][1]
ADAPTIVE_SPREAD_MULT_MIN = config["adaptive_intelligence"]["adjustable_parameters"][
    "spread_multiplier_range"
][0]
ADAPTIVE_SPREAD_MULT_MAX = config["adaptive_intelligence"]["adjustable_parameters"][
    "spread_multiplier_range"
][1]
ADAPTIVE_MAX_LOSSES_MIN = config["adaptive_intelligence"]["adjustable_parameters"][
    "max_losses_range"
][0]
ADAPTIVE_MAX_LOSSES_MAX = config["adaptive_intelligence"]["adjustable_parameters"][
    "max_losses_range"
][1]

# Triggers emergenciais
ADAPTIVE_LOW_WINRATE_THRESHOLD = config["adaptive_intelligence"]["emergency_triggers"][
    "low_winrate_threshold"
]
ADAPTIVE_HIGH_VOLATILITY_THRESHOLD = config["adaptive_intelligence"][
    "emergency_triggers"
]["high_volatility_threshold"]
ADAPTIVE_DAILY_DRAWDOWN_THRESHOLD = config["adaptive_intelligence"][
    "emergency_triggers"
]["daily_drawdown_threshold"]

# Variáveis dinâmicas do sistema adaptativo
ADAPTIVE_ML_THRESHOLD = ML_CONFIDENCE_BASE
ADAPTIVE_KELLY_MULTIPLIER = 1.0
ADAPTIVE_SPREAD_MULTIPLIER = 1.0
ADAPTIVE_SL_MULTIPLIER = 1.0
ADAPTIVE_TP_MULTIPLIER = 1.0
ADAPTIVE_ADX_MIN = 25.0

# Configurações do Anti-Chop (faltando no YAML)
ANTI_CHOP = {
    "enabled": True,
    "consecutive_losses_to_activate": ANTI_CHOP_CONSECUTIVE_LOSSES,
    "cooldown_after_sl_minutes": 120,
    "block_full_day_on_single_sl": False,
    "cooldown_after_sl_hours": 2,
    "max_symbols_blocked": 5,
}

# Configurações de Time Score (regras por período do dia)
TIME_SCORE_RULES = {
    "OPEN": {"start": "10:00", "end": "11:30", "adx_min": 25},
    "MID": {"start": "11:30", "end": "15:30", "adx_min": 20},
    "CLOSE": {"start": "15:30", "end": "18:00", "adx_min": 30},
}


def get_config() -> Dict[str, Any]:
    """Retorna configuração completa"""
    return config.copy()


def is_aggressive_mode() -> bool:
    """Verifica se está em modo agressivo"""
    return AGGRESSIVE_MODE


def get_max_losses_per_symbol() -> int:
    """Retorna máximo de losses por símbolo baseado no modo"""
    return (
        MAX_LOSSES_PER_SYMBOL_AGGRESSIVE
        if AGGRESSIVE_MODE
        else MAX_LOSSES_PER_SYMBOL_DEFAULT
    )


def update_adaptive_parameters(
    ml_threshold: float = None,
    kelly_multiplier: float = None,
    spread_multiplier: float = None,
    max_losses: int = None,
):
    """Atualiza parâmetros adaptativos dinamicamente"""
    global ADAPTIVE_ML_THRESHOLD, ADAPTIVE_KELLY_MULTIPLIER, ADAPTIVE_SPREAD_MULTIPLIER

    if ml_threshold is not None:
        ADAPTIVE_ML_THRESHOLD = max(
            ADAPTIVE_ML_THRESHOLD_MIN, min(ADAPTIVE_ML_THRESHOLD_MAX, ml_threshold)
        )

    if kelly_multiplier is not None:
        ADAPTIVE_KELLY_MULTIPLIER = max(
            ADAPTIVE_KELLY_MULT_MIN, min(ADAPTIVE_KELLY_MULT_MAX, kelly_multiplier)
        )

    if spread_multiplier is not None:
        ADAPTIVE_SPREAD_MULTIPLIER = max(
            ADAPTIVE_SPREAD_MULT_MIN, min(ADAPTIVE_SPREAD_MULT_MAX, spread_multiplier)
        )

    if max_losses is not None:
        # Atualiza diretamente no config para ser usado pelo risk_manager
        config["risk_limits"]["max_losses_per_symbol_temp"] = max(
            ADAPTIVE_MAX_LOSSES_MIN, min(ADAPTIVE_MAX_LOSSES_MAX, max_losses)
        )


def get_adaptive_ml_threshold() -> float:
    """Retorna threshold ML adaptativo atual"""
    return ADAPTIVE_ML_THRESHOLD


def get_adaptive_kelly_multiplier() -> float:
    """Retorna multiplicador Kelly adaptativo atual"""
    return ADAPTIVE_KELLY_MULTIPLIER


def get_adaptive_spread_multiplier() -> float:
    """Retorna multiplicador de spread adaptativo atual"""
    return ADAPTIVE_SPREAD_MULTIPLIER


def get_current_mode_params() -> dict:
    """Retorna parâmetros baseados no modo de operação atual"""
    base_params = {
        "ml_threshold": ML_MIN_CONFIDENCE,
        "kelly_multiplier": 1.0,
        "max_positions": MAX_CONCURRENT_POSITIONS,
        "max_losses_per_symbol": get_max_losses_per_symbol(),
        "spread_multiplier": 1.0,
    }

    if CURRENT_OPERATION_MODE == "AGGRESSIVE":
        return {
            **base_params,
            "ml_threshold": max(0.55, ML_MIN_CONFIDENCE - 0.05),
            "kelly_multiplier": 1.2,
            "spread_multiplier": 1.5,
            "max_losses_per_symbol": MAX_LOSSES_PER_SYMBOL_AGGRESSIVE,
        }
    elif CURRENT_OPERATION_MODE == "DEFENSIVE":
        return {
            **base_params,
            "ml_threshold": min(0.75, ML_MIN_CONFIDENCE + 0.05),
            "kelly_multiplier": 0.8,
            "spread_multiplier": 0.8,
            "max_losses_per_symbol": MAX_LOSSES_PER_SYMBOL_DEFAULT,
        }
    elif CURRENT_OPERATION_MODE == "PROTECTION":
        return {
            **base_params,
            "ml_threshold": min(0.80, ML_MIN_CONFIDENCE + 0.10),
            "kelly_multiplier": 0.5,
            "spread_multiplier": 0.6,
            "max_positions": max(2, MAX_CONCURRENT_POSITIONS // 2),
            "max_losses_per_symbol": 1,
        }
    else:  # NORMAL
        return base_params


# Configurações de Horário de Trading (adicionadas para corrigir erros)
TRADING_START = "10:20"
TRADING_LUNCH_BREAK_START = "11:45"
TRADING_LUNCH_BREAK_END = "13:05"

# Mapeamento de Setores (para pesos adaptativos)
SECTOR_MAP = {
    # ==========================================
    # 🏦 FINANCEIRO E SEGUROS
    # ==========================================
    "BBAS3": "FINANCEIRO",
    "ITUB4": "FINANCEIRO",
    "BBDC4": "FINANCEIRO",
    "BBDC3": "FINANCEIRO",
    "SANB11": "FINANCEIRO",
    "ITSA4": "FINANCEIRO",
    "BPAC11": "FINANCEIRO",
    "B3SA3": "FINANCEIRO",
    "ABCB4": "FINANCEIRO",
    "BPAN4": "FINANCEIRO",
    "BRSR6": "FINANCEIRO",
    "MODL11": "FINANCEIRO",
    "BBSE3": "SEGUROS",
    "CXSE3": "SEGUROS",
    "PSSA3": "SEGUROS",
    "IRBR3": "SEGUROS",
    "SULA11": "SEGUROS",
    "CIEL3": "FINANCEIRO",
    "CASH3": "FINANCEIRO",
    "BMGB4": "FINANCEIRO",
    "PINE4": "FINANCEIRO",
    # ==========================================
    # 🛢️ ENERGIA E PETRÓLEO (Óleo e Gás)
    # ==========================================
    "PETR3": "PETROLEO",
    "PETR4": "PETROLEO",
    "PRIO3": "PETROLEO",
    "RRRP3": "PETROLEO",
    "ENAT3": "PETROLEO",
    "RECV3": "PETROLEO",
    "CSAN3": "PETROLEO",
    "UGPA3": "PETROLEO",
    "VBBR3": "PETROLEO",
    "RAIZ4": "PETROLEO",
    "OSXB3": "PETROLEO",
    "DMMO3": "PETROLEO",
    # ==========================================
    # ⚡ UTILIDADE PÚBLICA (Eletricidade e Saneamento)
    # ==========================================
    "ELET3": "UTILIDADE_PUBLICA",
    "ELET6": "UTILIDADE_PUBLICA",
    "EQTL3": "UTILIDADE_PUBLICA",
    "ENEV3": "UTILIDADE_PUBLICA",
    "CPLE6": "UTILIDADE_PUBLICA",
    "CMIG4": "UTILIDADE_PUBLICA",
    "TAEE11": "UTILIDADE_PUBLICA",
    "TRPL4": "UTILIDADE_PUBLICA",
    "ALUP11": "UTILIDADE_PUBLICA",
    "EGIE3": "UTILIDADE_PUBLICA",
    "NEOE3": "UTILIDADE_PUBLICA",
    "AURE3": "UTILIDADE_PUBLICA",
    "MEGA3": "UTILIDADE_PUBLICA",
    "AESB3": "UTILIDADE_PUBLICA",
    "SBSP3": "SANEAMENTO",
    "CSMG3": "SANEAMENTO",
    "SAPR11": "SANEAMENTO",
    "SAPR4": "SANEAMENTO",
    "ORVR3": "SANEAMENTO",
    # ==========================================
    # ⛏️ MATERIAIS BÁSICOS (Mineração, Siderurgia, Celulose)
    # ==========================================
    "VALE3": "MINERACAO",
    "BRAP4": "MINERACAO",
    "CMIN3": "MINERACAO",
    "AURA33": "MINERACAO",
    "CBAV3": "MINERACAO",
    "LITH3": "MINERACAO",
    "GGBR4": "SIDERURGIA",
    "GOAU4": "SIDERURGIA",
    "CSNA3": "SIDERURGIA",
    "USIM5": "SIDERURGIA",
    "FESA4": "SIDERURGIA",
    "SUZB3": "PAPEL_CELULOSE",
    "KLBN11": "PAPEL_CELULOSE",
    "RANI3": "PAPEL_CELULOSE",
    "BRKM5": "QUIMICA",
    "UNIP6": "QUIMICA",
    "DEXP3": "MATERIAIS",
    # ==========================================
    # 🥩 AGROPECUÁRIA E PROTEÍNAS
    # ==========================================
    "JBSS3": "PROTEINAS",
    "MRFG3": "PROTEINAS",
    "BEEF3": "PROTEINAS",
    "BRFS3": "PROTEINAS",
    "MDIA3": "ALIMENTOS",
    "CAML3": "ALIMENTOS",
    "SLCE3": "AGRO",
    "AGRO3": "AGRO",
    "SMTO3": "AGRO",
    "TTEN3": "AGRO",
    "SOJA3": "AGRO",
    "JALL3": "AGRO",
    # ==========================================
    # 🛒 VAREJO E CONSUMO
    # ==========================================
    "MGLU3": "VAREJO",
    "BHIA3": "VAREJO",
    "LREN3": "VAREJO",
    "CEAB3": "VAREJO",
    "ARZZ3": "VAREJO",
    "SOMA3": "VAREJO",
    "GUAR3": "VAREJO",
    "AMER3": "VAREJO",
    "PETZ3": "VAREJO",
    "ALPA4": "VAREJO",
    "CGRA4": "VAREJO",
    "SLED4": "VAREJO",
    "CRFB3": "SUPERMERCADOS",
    "ASAI3": "SUPERMERCADOS",
    "PCAR3": "SUPERMERCADOS",
    "GMAT3": "SUPERMERCADOS",
    "ABEV3": "BEBIDAS",
    "VIVA3": "CONSUMO",
    "NTCO3": "CONSUMO",
    "ESPA3": "CONSUMO",
    "PGMN3": "FARMACIA",
    "PNVL3": "FARMACIA",
    # ==========================================
    # 🏥 SAÚDE
    # ==========================================
    "RADL3": "SAUDE",
    "HAPV3": "SAUDE",
    "FLRY3": "SAUDE",
    "RDOR3": "SAUDE",
    "QUAL3": "SAUDE",
    "DASA3": "SAUDE",
    "MATD3": "SAUDE",
    "AALR3": "SAUDE",
    "ODPV3": "SAUDE",
    "BLCA3": "SAUDE",
    "ONCO3": "SAUDE",
    "VVEO3": "SAUDE",
    # ==========================================
    # 🏗️ CONSTRUÇÃO CIVIL E IMOBILIÁRIO
    # ==========================================
    "CYRE3": "CONSTRUCAO",
    "EZTC3": "CONSTRUCAO",
    "MRVE3": "CONSTRUCAO",
    "DIRR3": "CONSTRUCAO",
    "TEND3": "CONSTRUCAO",
    "CURY3": "CONSTRUCAO",
    "PLPL3": "CONSTRUCAO",
    "LAVV3": "CONSTRUCAO",
    "GFSA3": "CONSTRUCAO",
    "HBOR3": "CONSTRUCAO",
    "MDNE3": "CONSTRUCAO",
    "TCSA3": "CONSTRUCAO",
    "JHSF3": "CONSTRUCAO",
    "EVEN3": "CONSTRUCAO",
    "TRIS3": "CONSTRUCAO",
    "MULT3": "SHOPPINGS",
    "IGTI11": "SHOPPINGS",
    "ALOS3": "SHOPPINGS",
    "SYNE3": "SHOPPINGS",
    "LOGG3": "IMOBILIARIO",
    "BRPR3": "IMOBILIARIO",
    # ==========================================
    # 🚚 LOGÍSTICA E TRANSPORTE
    # ==========================================
    "RENT3": "LOGISTICA",
    "RAIL3": "LOGISTICA",
    "CCRO3": "LOGISTICA",
    "ECOR3": "LOGISTICA",
    "STBP3": "LOGISTICA",
    "JSLG3": "LOGISTICA",
    "VAMO3": "LOGISTICA",
    "TGMA3": "LOGISTICA",
    "HBSA3": "LOGISTICA",
    "PORT3": "LOGISTICA",
    "SEQL3": "LOGISTICA",
    "AZUL4": "AVIAÇÃO",
    "GOLL4": "AVIAÇÃO",
    "EMBR3": "AEROESPACIAL",
    # ==========================================
    # ⚙️ BENS INDUSTRIAIS
    # ==========================================
    "WEGE3": "INDUSTRIA",
    "POMO4": "INDUSTRIA",
    "TUPY3": "INDUSTRIA",
    "RAPT4": "INDUSTRIA",
    "MYPK3": "INDUSTRIA",
    "SHUL4": "INDUSTRIA",
    "PTBL3": "INDUSTRIA",
    "ROMI3": "INDUSTRIA",
    "LEVE3": "INDUSTRIA",
    "KEPL3": "INDUSTRIA",
    "FRAS3": "INDUSTRIA",
    # ==========================================
    # 💻 TECNOLOGIA E TELECOMUNICAÇÕES
    # ==========================================
    "VIVT3": "TELECOM",
    "TIMS3": "TELECOM",
    "FIQE3": "TELECOM",
    "DESK3": "TELECOM",
    "TELB4": "TELECOM",
    "TOTS3": "TECNOLOGIA",
    "LWSA3": "TECNOLOGIA",
    "POSI3": "TECNOLOGIA",
    "INTB3": "TECNOLOGIA",
    "CBAV3": "TECNOLOGIA",
    "MLAS3": "TECNOLOGIA",
    "NGRD3": "TECNOLOGIA",
    "PDGR3": "TECNOLOGIA",
    # ==========================================
    # 📚 EDUCAÇÃO
    # ==========================================
    "YDUQ3": "EDUCACAO",
    "COGN3": "EDUCACAO",
    "ANIM3": "EDUCACAO",
    "SEER3": "EDUCACAO",
    "CSED3": "EDUCACAO",
    "BAHI3": "EDUCACAO",
}

# Intervalo de atualização da correlação (segundos)
CORR_UPDATE_INTERVAL = 3600  # 1 hora

# Horários de fechamento e bloqueio de entradas
NO_ENTRY_AFTER = "17:30"
CLOSE_ALL_BY = "17:45"
FRIDAY_NO_ENTRY_AFTER = "17:00"
FRIDAY_CLOSE_ALL_BY = "17:15"
DAILY_RESET_TIME = "10:00"  # Reset diário do circuit breaker

# Modo de operação
DAY_ONLY_MODE = False
CURRENT_OPERATION_MODE = "NORMAL"

# Configurações de teste A/B
AB_TEST_ENABLED = False
AB_TEST_GROUPS = {
    "A": {"ml_threshold": 0.65, "kelly_multiplier": 1.0},
    "B": {"ml_threshold": 0.70, "kelly_multiplier": 0.8},
}

# Parâmetros ML
ML_MIN_CONFIDENCE = 0.65
MIN_SIGNAL_SCORE = 60

# Limites de Drawdown
MAX_DAILY_DRAWDOWN_PCT = 3.0
MAX_DAILY_LOSS_MONEY = 1000.0

# Configurações de Breakout de Volatilidade
VOL_BREAKOUT = {
    "enabled": True,
    "atr_expansion": 1.5,  # ATR deve expandir 50% acima da média
    "volume_ratio": 1.2,  # Volume deve ser 20% acima da média
    "lookback": 20,  # Período de lookback para máximas/mínimas
}

# Configurações de trades
MAX_TRADE_DURATION_CANDLES = 24
SYMBOL_COOLDOWN_HOURS = 24  # Horas de bloqueio após perdas consecutivas

# Parâmetros de Tendência Macro
MACRO_EMA_LONG = 200  # EMA longa para análise macro (candles)
ENABLE_MACRO_FILTER = True  # Ativa filtro de tendência macro

# Limite de Volume para Correlação
MAX_VOL_THRESHOLD = 2.0  # Multiplicador máximo de volume (ex: 2x média)

# Configurações de Proteção de Lucro
PROFIT_LOCK = {
    "enabled": True,  # Ativa proteção de lucro
    "daily_target_pct": 2.0,  # Meta diária em % (ex: 2%)
    "reduce_risk": True,  # Reduz risco após atingir meta
    "min_minutes_between_actions": 5,  # Intervalo mínimo entre ações
}
ENABLE_BREAKEVEN = True
ENABLE_PARTIAL_CLOSE = True
PARTIAL_CLOSE_ATR_MULT = 1.5
ENABLE_TRAILING_STOP = True

# Regras de Take Profit por regime
TP_RULES = {
    "TREND": {"min_tp_atr": 2.0, "max_tp_atr": 4.0},
    "LATERAL": {"min_tp_atr": 1.5, "max_tp_atr": 3.0},
    "VOLATILE": {"min_tp_atr": 2.5, "max_tp_atr": 5.0},
}

# Lista de símbolos monitorados (derivada do mapa de setores)
MONITORED_SYMBOLS = list(SECTOR_MAP.keys())
