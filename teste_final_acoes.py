#!/usr/bin/env python3
"""
Teste final para AÇÕES, contornando o problema da quantidade de dados.
Vamos passar um DataFrame menor diretamente para a função.
"""

import sys

sys.path.append(".")

from utils import quick_indicators_custom, safe_copy_rates
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Testar com ações
print("\n=== Teste Final com Ações (DataFrame injetado) ===")
ativos_teste = ["PETR4", "ITUB4"]

for ativo in ativos_teste:
    try:
        logging.info(f"--- Testando {ativo} ---")

        # 1. Buscar um número menor de candles que sabemos que funciona
        df_teste = safe_copy_rates(ativo, "M15", 100)

        if df_teste is None or df_teste.empty or len(df_teste) < 50:
            logging.error(f"Falha ao obter dados base para {ativo}. Pulando teste.")
            continue

        logging.info(
            f"Dados base para {ativo} obtidos com sucesso ({len(df_teste)} candles). Injetando em quick_indicators_custom..."
        )

        # 2. Injetar o DataFrame na função para testar a lógica de cálculo
        indicadores = quick_indicators_custom(ativo, "M15", df=df_teste)

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

print("\n🎯 Teste concluído!")
