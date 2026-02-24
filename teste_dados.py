#!/usr/bin/env python3
"""
Teste focado na função safe_copy_rates para depurar o erro 'no_data'.
"""

import sys

sys.path.append(".")

import MetaTrader5 as mt5
from utils import safe_copy_rates
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Conectar ao MT5
try:
    if not mt5.initialize():
        logging.error("❌ initialize() falhou, erro code =", mt5.last_error())
        sys.exit()
    logging.info("✅ MT5 Conectado")
except Exception as e:
    logging.error(f"❌ Erro fatal na conexão MT5: {e}")
    sys.exit()

# Ativos para testar
ativos_teste = ["PETR4", "ITUB4", "WINQ24"]  # Adicionei um futuro para comparação

print("\n=== Teste Direto de safe_copy_rates ===")
for ativo in ativos_teste:
    try:
        logging.info(f"--- Buscando dados para {ativo} ---")

        # Tenta garantir que o símbolo está visível
        selected = mt5.symbol_select(ativo, True)
        if not selected:
            logging.warning(
                f"⚠️ Não foi possível selecionar {ativo}, pode não estar visível no Market Watch."
            )

        # Chama a função que está falhando
        df = safe_copy_rates(ativo, mt5.TIMEFRAME_M15, 100)

        if df is not None and not df.empty:
            print(f"✅ {ativo}: Sucesso! {len(df)} candles recebidos.")
            print(df.head(2))
        else:
            print(f"❌ {ativo}: Falha! A função retornou None ou um DataFrame vazio.")
            # Tenta a chamada direta do MT5 para ver o erro
            rates = mt5.copy_rates_from_pos(ativo, mt5.TIMEFRAME_M15, 0, 100)
            if rates is None:
                print(
                    f"  -> Causa: mt5.copy_rates_from_pos retornou None. Erro MT5: {mt5.last_error()}"
                )
            elif len(rates) == 0:
                print(f"  -> Causa: mt5.copy_rates_from_pos retornou 0 candles.")
            else:
                print(
                    f"  -> Causa: Inesperado. mt5.copy_rates_from_pos retornou dados, mas safe_copy_rates falhou."
                )

    except Exception as e:
        logging.error(f"❌ Erro fatal no teste de {ativo}: {e}", exc_info=True)

# Desconectar
mt5.shutdown()
print("\n🎯 Teste concluído!")
