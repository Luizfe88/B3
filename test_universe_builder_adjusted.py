#!/usr/bin/env python3
"""
Testa o Universe Builder com thresholds ajustados
"""
import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_universe_builder_adjusted():
    """Testa o universe builder com thresholds mais realistas"""
    try:
        # Conecta MT5
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Executa o universe builder com thresholds ajustados
        logger.info("🚀 Executando build_b3_universe() com thresholds ajustados...")

        # Ajusta os thresholds baseado no debug
        universe = utils.build_b3_universe(
            min_fin_volume=1_000_000,  # Reduz de 15M para 1M
            min_atr_pct=0.80,  # Mantém
            max_atr_pct=7.0,  # Mantém
            max_ibov_corr=0.90,  # Aumenta de 0.82 para 0.90
            min_market_cap=500_000_000,  # Reduz de 1B para 500M
            save_json=True,
        )

        logger.info(f"🎯 Resultados:")
        logger.info(f"ELITE: {len(universe['ELITE'])} ativos")
        logger.info(f"OPORTUNIDADE: {len(universe['OPORTUNIDADE'])} ativos")
        logger.info(f"TOTAL: {len(universe['TOTAL'])} ativos")

        if universe["ELITE"]:
            logger.info(f"🏆 Top 10 ELITE:")
            for symbol, score in universe["ELITE"][:10]:
                logger.info(f"  {symbol}: {score:.1f}")

        if universe["OPORTUNIDADE"]:
            logger.info(f"🎯 Top 10 OPORTUNIDADE:")
            for symbol, score in universe["OPORTUNIDADE"][:10]:
                logger.info(f"  {symbol}: {score:.1f}")

        # Salva o JSON se tiver resultados
        if universe["ELITE"] or universe["OPORTUNIDADE"]:
            logger.info("💾 Salvando elite_symbols_latest.json...")
            success = utils.atomic_save_json("elite_symbols_latest.json", universe)
            if success:
                logger.info("✅ JSON salvo com sucesso!")

                # Mostra o conteúdo do JSON
                import json

                with open("elite_symbols_latest.json", "r") as f:
                    data = json.load(f)

                logger.info(f"📄 JSON salvo com:")
                logger.info(f"  - ELITE: {len(data['ELITE'])} ativos")
                logger.info(f"  - OPORTUNIDADE: {len(data['OPORTUNIDADE'])} ativos")
                logger.info(f"  - TOTAL: {len(data['TOTAL'])} ativos")
                logger.info(f"  - Metadata: {data.get('metadata', {})}")
            else:
                logger.error("❌ Falha ao salvar JSON")
        else:
            logger.warning("⚠️  Nenhum ativo encontrado com os critérios atuais")

        return True

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    test_universe_builder_adjusted()
