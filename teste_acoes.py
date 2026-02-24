#!/usr/bin/env python3
"""
Teste das correções de ema_diff para AÇÕES (PETR4/ITUB4)
"""

import sys

sys.path.append(".")

from utils import quick_indicators_custom
import redis
import logging

# Configurar logging para ver mais detalhes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Limpar cache Redis
try:
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()
    logging.info("✅ Cache Redis limpo")
except Exception as e:
    logging.error(f"❌ Erro ao limpar cache: {e}")

# Testar com ações
print("\n=== Teste com Ações (quick_indicators_custom) ===")
ativos_teste = ["PETR4", "ITUB4"]

for ativo in ativos_teste:
    try:
        logging.info(f"--- Testando {ativo} ---")
        indicadores = quick_indicators_custom(ativo, "M15")

        if indicadores and not indicadores.get("error"):
            print(f"Indicadores {ativo}:")
            for key, valor in indicadores.items():
                # Imprimir apenas alguns valores para não poluir
                if key in ["ema_diff", "rsi", "volume_ratio", "close"]:
                    print(f"  {key}: {valor}")

            # Verificar se ema_diff está presente
            if "ema_diff" in indicadores:
                print(
                    f"✅ ema_diff calculado para {ativo}: {indicadores['ema_diff']:.4f}"
                )
            else:
                print(f"❌ ema_diff não encontrado para {ativo}!")
        else:
            print(
                f"❌ Erro ao buscar indicadores para {ativo}: {indicadores.get('error')}"
            )

    except Exception as e:
        logging.error(f"❌ Erro fatal no teste de {ativo}: {e}", exc_info=True)

print("\n🎯 Teste concluído!")
