#!/usr/bin/env python3
"""
Teste final integrado - Universe Builder completo
"""
import utils
import MetaTrader5 as mt5
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_integrated_universe_builder():
    """Teste final do Universe Builder integrado"""
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # 1. Testa o Universe Builder
        logger.info("🚀 Executando Universe Builder completo...")
        universe = utils.build_b3_universe(
            min_fin_volume=500_000,
            min_atr_pct=0.80,
            max_atr_pct=7.0,
            max_ibov_corr=0.90,
            min_market_cap=300_000_000,
            save_json=True,
        )

        logger.info(f"\n📊 Resultados do Universe Builder:")
        logger.info(f"🏆 ELITE: {len(universe['ELITE'])} ativos")
        logger.info(f"🎯 OPORTUNIDADE: {len(universe['OPORTUNIDADE'])} ativos")
        logger.info(f"📈 TOTAL: {len(universe['TOTAL'])} ativos")

        # 2. Verifica JSON salvo
        logger.info(f"\n📄 Verificando elite_symbols_latest.json...")
        try:
            with open("elite_symbols_latest.json", "r") as f:
                saved_data = json.load(f)

            logger.info(f"✅ JSON salvo com sucesso!")
            logger.info(f"  - ELITE: {len(saved_data.get('ELITE', []))} ativos")
            logger.info(
                f"  - OPORTUNIDADE: {len(saved_data.get('OPORTUNIDADE', []))} ativos"
            )
            logger.info(f"  - Metadata: {bool(saved_data.get('metadata', {}))}")

            # Mostra os top ativos
            if saved_data.get("ELITE"):
                logger.info(f"\n🏆 Top ELITE:")
                for item in saved_data["ELITE"][:3]:
                    symbol, score = item if isinstance(item, list) else (item, "N/A")
                    logger.info(f"  - {symbol}: {score}")

            if saved_data.get("OPORTUNIDADE"):
                logger.info(f"\n🎯 Top OPORTUNIDADE:")
                for item in saved_data["OPORTUNIDADE"][:5]:
                    symbol, score = item if isinstance(item, list) else (item, "N/A")
                    logger.info(f"  - {symbol}: {score}")

        except Exception as e:
            logger.error(f"❌ Erro ao verificar JSON: {e}")
            return False

        # 3. Testa carregamento do JSON
        logger.info(f"\n📂 Testando load_elite_symbols_from_json()...")
        loaded_data = utils.load_elite_symbols_from_json()
        if loaded_data:
            logger.info(f"✅ Dados carregados com sucesso!")
            logger.info(f"  - ELITE: {len(loaded_data.get('ELITE', []))} ativos")
            logger.info(
                f"  - OPORTUNIDADE: {len(loaded_data.get('OPORTUNIDADE', []))} ativos"
            )
        else:
            logger.warning("⚠️ Falha ao carregar dados do JSON")

        # 4. Testa ensure_market_watch_symbols
        logger.info(f"\n📋 Testando ensure_market_watch_symbols()...")
        utils.ensure_market_watch_symbols()

        # 5. Testa auto_add_stocks_to_market_watch
        logger.info(f"\n➕ Testando auto_add_stocks_to_market_watch()...")
        if universe["ELITE"]:
            elite_symbols = [
                item[0] if isinstance(item, list) else item
                for item in universe["ELITE"]
            ]
            utils.auto_add_stocks_to_market_watch(elite_symbols[:5], "Teste ELITE")

        logger.info(f"\n🎉 Teste integrado concluído com sucesso!")
        return True

    except Exception as e:
        logger.error(f"❌ Erro no teste integrado: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    test_integrated_universe_builder()
