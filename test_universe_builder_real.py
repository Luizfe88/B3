#!/usr/bin/env python3
"""
Testa o Universe Builder real com os parâmetros corretos
"""
import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_real_universe_builder():
    """Testa o universe builder com parâmetros reais"""
    try:
        # Conecta MT5
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Executa o universe builder com parâmetros padrão
        logger.info("🚀 Executando build_b3_universe()...")
        universe = utils.build_b3_universe()

        logger.info(f"🎯 Resultados:")
        logger.info(f"ELITE: {len(universe['ELITE'])} ativos")
        logger.info(f"OPORTUNIDADE: {len(universe['OPORTUNIDADE'])} ativos")
        logger.info(f"TOTAL: {len(universe['TOTAL'])} ativos")

        if universe["ELITE"]:
            logger.info(f"🏆 Top 5 ELITE:")
            for symbol, score in universe["ELITE"][:5]:
                logger.info(f"  {symbol}: {score:.1f}")

        if universe["OPORTUNIDADE"]:
            logger.info(f"🎯 Top 5 OPORTUNIDADE:")
            for symbol, score in universe["OPORTUNIDADE"][:5]:
                logger.info(f"  {symbol}: {score:.1f}")

        # Salva o JSON se tiver resultados
        if universe["ELITE"] or universe["OPORTUNIDADE"]:
            logger.info("💾 Salvando elite_symbols_latest.json...")
            success = utils.atomic_save_json("elite_symbols_latest.json", universe)
            if success:
                logger.info("✅ JSON salvo com sucesso!")
            else:
                logger.error("❌ Falha ao salvar JSON")

        return True

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    test_real_universe_builder()
