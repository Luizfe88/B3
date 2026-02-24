#!/usr/bin/env python3
"""
Testa o Universe Builder e verifica a estrutura dos dados
"""
import utils
import MetaTrader5 as mt5
import logging
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_universe_structure():
    """Testa a estrutura do universe builder"""
    try:
        # Conecta MT5
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Testa apenas alguns símbolos conhecidos
        test_symbols = ["VALE3", "PETR4", "WEGE3", "SUZB3"]

        universe = {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

        for symbol in test_symbols:
            try:
                name = symbol.upper()

                # Info básica
                info = mt5.symbol_info(name)
                if not info or not info.select:
                    if not mt5.symbol_select(name, True):
                        continue
                    info = mt5.symbol_info(name)
                    if not info:
                        continue

                # Dados históricos
                rates = utils.safe_copy_rates(name, mt5.TIMEFRAME_D1, 60)
                if rates is None or len(rates) < 20:
                    continue

                # Calcula indicadores
                avg_fin_volume = (rates["tick_volume"] * rates["close"]).mean()
                atr = utils.get_atr(rates, 14)
                current_price = rates["close"].iloc[-1]
                atr_pct = (atr / current_price) * 100
                ibov_corr = utils.get_ibov_correlation(name)
                spread_pct = (
                    (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
                )

                fund = utils.fundamental_fetcher.get_fundamentals(name)
                mcap = fund.get("market_cap", 0)

                # Calcula score
                score = utils.calculate_asset_score(
                    volume=avg_fin_volume,
                    atr_pct=atr_pct,
                    corr=ibov_corr,
                    mcap=mcap,
                    spread_pct=spread_pct,
                )

                logger.info(f"\n📊 {name}:")
                logger.info(f"  Volume: {avg_fin_volume:,.0f}")
                logger.info(f"  ATR %: {atr_pct:.2f}%")
                logger.info(f"  IBOV Corr: {ibov_corr:.3f}")
                logger.info(f"  Spread: {spread_pct:.3f}%")
                logger.info(f"  Market Cap: {mcap:,.0f}")
                logger.info(f"  Score: {score:.1f}")

                # Classifica
                if score >= 85:
                    universe["ELITE"].append((name, score))
                    logger.info(f"  ➕ Adicionado a ELITE")
                elif score >= 65:
                    universe["OPORTUNIDADE"].append((name, score))
                    logger.info(f"  ➕ Adicionado a OPORTUNIDADE")
                else:
                    universe["TOTAL"].append((name, score))
                    logger.info(f"  ➕ Adicionado a TOTAL")

            except Exception as e:
                logger.error(f"❌ Erro ao analisar {symbol}: {e}")
                continue

        logger.info(f"\n🎯 Resultados:")
        logger.info(f"ELITE: {len(universe['ELITE'])} - {universe['ELITE']}")
        logger.info(
            f"OPORTUNIDADE: {len(universe['OPORTUNIDADE'])} - {universe['OPORTUNIDADE']}"
        )
        logger.info(f"TOTAL: {len(universe['TOTAL'])} - {universe['TOTAL']}")

        # Testa salvar JSON
        logger.info(f"\n💾 Testando salvamento JSON...")
        success = utils.atomic_save_json("test_universe.json", universe)

        if success:
            logger.info("✅ JSON salvo com sucesso!")

            # Lê o JSON salvo
            with open("test_universe.json", "r") as f:
                loaded = json.load(f)

            logger.info(f"📄 JSON carregado:")
            logger.info(f"ELITE: {loaded['ELITE']}")
            logger.info(f"OPORTUNIDADE: {loaded['OPORTUNIDADE']}")
            logger.info(f"TOTAL: {loaded['TOTAL']}")

            # Verifica se os scores foram preservados
            for category in ["ELITE", "OPORTUNIDADE", "TOTAL"]:
                if loaded[category]:
                    logger.info(f"\n🔍 Verificando {category}:")
                    for item in loaded[category]:
                        if isinstance(item, list) and len(item) == 2:
                            symbol, score = item
                            logger.info(f"  {symbol}: {score}")
                        else:
                            logger.warning(f"  Formato inválido: {item}")

        return True

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    test_universe_structure()
