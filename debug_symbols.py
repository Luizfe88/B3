#!/usr/bin/env python3
"""
Debug script para verificar símbolos disponíveis no MT5
"""
import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def debug_symbols():
    """Debug dos símbolos disponíveis"""
    try:
        if not mt5.initialize():
            logger.error(f"Falha ao conectar MT5: {mt5.last_error()}")
            return

        logger.info("✅ MT5 conectado")

        # Pega todos os símbolos
        all_symbols = mt5.symbols_get()
        logger.info(f"📊 Total de símbolos: {len(all_symbols)}")

        # Filtra por diferentes critérios
        b3_symbols = []

        for sym in all_symbols[:100]:  # Primeiros 100 para debug
            name = sym.name.upper()

            # Verifica se é B3 (termina com número)
            is_b3 = name[-1].isdigit() and len(name) <= 6

            # Verifica se é futuro
            is_fut = "$" in name or name.startswith(
                ("WIN", "WDO", "WSP", "IND", "DOL", "SMALL")
            )

            # Verifica se é selecionável
            try:
                info = mt5.symbol_info(name)
                if info:
                    selectable = info.select
                    trade_mode = info.trade_mode
                    logger.info(
                        f"{name:8} | B3: {is_b3:5} | Fut: {is_fut:5} | Select: {selectable:5} | TradeMode: {trade_mode}"
                    )

                    if is_b3 and not is_fut and selectable and trade_mode == 1:
                        b3_symbols.append(name)

            except Exception as e:
                logger.warning(f"Erro ao verificar {name}: {e}")

        logger.info(f"\n🏆 Símbolos B3 encontrados: {len(b3_symbols)}")
        logger.info(f"Exemplos: {b3_symbols[:10]}")

        # Verifica símbolos específicos
        test_symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
        logger.info(f"\n🔍 Verificando símbolos específicos:")
        for symbol in test_symbols:
            info = mt5.symbol_info(symbol)
            if info:
                logger.info(
                    f"{symbol}: Select={info.select}, TradeMode={info.trade_mode}, Visible={info.visible}"
                )
            else:
                logger.info(f"{symbol}: ❌ Não encontrado")

    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback

        traceback.print_exc()

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    debug_symbols()
