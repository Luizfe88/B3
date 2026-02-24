#!/usr/bin/env python3
"""
Teste final (corrigido) para AÇÕES.
Inicializa o MT5 antes de buscar os dados.
"""

import sys

sys.path.append(".")

import MetaTrader5 as mt5
from utils import quick_indicators_custom, safe_copy_rates
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- INICIALIZAÇÃO DO MT5 (ESSENCIAL) ---
try:
    if not mt5.initialize():
        logging.error("❌ initialize() falhou, erro code =", mt5.last_error())
        sys.exit()
    logging.info("✅ MT5 Conectado para o teste")
except Exception as e:
    logging.error(f"❌ Erro fatal na conexão MT5: {e}")
    sys.exit()
# -----------------------------------------

# Testar com ações
print("\n=== Teste Final com Ações (DataFrame injetado e MT5 conectado) ===")
ativos_teste = ["PETR4", "ITUB4"]

for ativo in ativos_teste:
    try:
        logging.info(f"--- Testando {ativo} ---")

        # 1. Buscar um número menor de candles
        df_teste = safe_copy_rates(ativo, mt5.TIMEFRAME_M15, 100)

        if df_teste is None or df_teste.empty or len(df_teste) < 50:
            logging.error(f"Falha ao obter dados base para {ativo}. Pulando teste.")
            continue

        logging.info(
            f"Dados base para {ativo} obtidos com sucesso ({len(df_teste)} candles)."
        )

        # 2. Injetar o DataFrame na função
        indicadores = quick_indicators_custom(ativo, mt5.TIMEFRAME_M15, df=df_teste)

        if indicadores and not indicadores.get("error"):
            print(f"Indicadores {ativo}:")
            for key, valor in indicadores.items():
                if key in ["ema_diff", "rsi", "volume_ratio", "close"]:
                    print(f"  {key}: {valor}")

            if "ema_diff" in indicadores:
                print(
                    f"✅✅✅ SUCESSO! ema_diff calculado para {ativo}: {indicadores['ema_diff']:.4f}"
                )
            else:
                print(f"❌❌❌ FALHA! ema_diff não encontrado para {ativo}!")
        else:
            print(
                f"❌ Erro ao calcular indicadores para {ativo}: {indicadores.get('error')}"
            )

    except Exception as e:
        logging.error(f"❌ Erro fatal no teste de {ativo}: {e}", exc_info=True)

# Desconectar
mt5.shutdown()
print("\n🎯 Teste concluído!")
