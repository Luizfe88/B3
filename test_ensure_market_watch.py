#!/usr/bin/env python3
"""
Testa a função ensure_market_watch_symbols() com o novo Universe Builder
"""
import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_ensure_market_watch():
    """Testa a função ensure_market_watch_symbols()"""
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Testa a função
        logger.info("🧪 Testando ensure_market_watch_symbols()...")
        utils.ensure_market_watch_symbols()

        logger.info("✅ Teste concluído com sucesso!")
        return True

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    test_ensure_market_watch()
