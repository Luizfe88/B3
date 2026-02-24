#!/usr/bin/env python3
"""
Teste simplificado do Universe Builder usando listas pré-definidas
"""
import sys

sys.path.append(".")

import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_universe_builder_simple():
    """Testa o Universe Builder com ativos pré-definidos"""
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Primeiro adiciona os ativos ao Market Watch
        logger.info("📌 Adicionando ativos ELITE ao Market Watch...")
        utils.auto_add_stocks_to_market_watch(utils.ELITE_SYMBOLS, log_name="ELITE")

        logger.info("📌 Adicionando ativos OPORTUNIDADE ao Market Watch...")
        utils.auto_add_stocks_to_market_watch(
            utils.OPORTUNIDADE_SYMBOLS, log_name="OPORTUNIDADE"
        )

        # Agora executa o Universe Builder
        logger.info("🚀 Executando Universe Builder...")
        universe = utils.build_b3_universe(
            min_fin_volume=1_000_000, min_atr_pct=0.5, max_atr_pct=10.0
        )

        if universe:
            logger.info(f"🏆 Resultados:")
            logger.info(f"  ELITE: {len(universe.get('ELITE', []))} ativos")
            logger.info(
                f"  OPORTUNIDADE: {len(universe.get('OPORTUNIDADE', []))} ativos"
            )
            logger.info(f"  TOTAL: {len(universe.get('TOTAL', []))} ativos")

            if universe.get("ELITE"):
                logger.info(f"  📈 Top 10 ELITE: {', '.join(universe['ELITE'][:10])}")

            return True
        else:
            logger.error("❌ Universe Builder retornou None")
            return False

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    success = test_universe_builder_simple()
    sys.exit(0 if success else 1)
