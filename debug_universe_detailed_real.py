#!/usr/bin/env python3
"""
Debug detalhado do Universe Builder para entender por que não encontra ativos
"""
import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def debug_universe_builder_detailed():
    """Debug detalhado do Universe Builder"""
    try:
        # Conecta MT5
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Testa alguns símbolos específicos que sabemos que existem
        test_symbols = ["PETR4", "VALE3", "ITUB4", "ABEV3", "BBDC4", "B3SA3"]

        logger.info(f"🧪 Testando {len(test_symbols)} símbolos conhecidos...")

        approved_symbols = []

        for symbol in test_symbols:
            logger.info(f"\n🔍 Analisando {symbol}...")

            try:
                # Info básica
                info = mt5.symbol_info(symbol)
                if not info:
                    logger.warning(f"  ❌ Símbolo não encontrado")
                    continue

                logger.info(f"  ✅ Símbolo encontrado")
                logger.info(f"  Select: {info.select}")
                logger.info(f"  Trade Mode: {info.trade_mode}")
                logger.info(f"  Ask: {info.ask}")
                logger.info(f"  Bid: {info.bid}")

                # Seleciona no Market Watch se necessário
                if not info.select:
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"  ❌ Falha ao selecionar no Market Watch")
                        continue
                    logger.info("  ✅ Adicionado ao Market Watch")

                # Dados históricos
                rates = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_D1, 60)
                if rates is None or len(rates) < 20:
                    logger.warning(
                        f"  ❌ Dados históricos insuficientes: {len(rates) if rates else 0}"
                    )
                    continue

                logger.info(f"  ✅ Dados históricos: {len(rates)} candles")

                # Calcula indicadores com os parâmetros reais do build_b3_universe
                avg_fin_volume = (rates["tick_volume"] * rates["close"]).mean()
                atr = utils.get_atr(rates, 14)
                current_price = rates["close"].iloc[-1]
                atr_pct = (atr / current_price) * 100
                ibov_corr = utils.get_ibov_correlation(symbol)
                spread_pct = (
                    (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
                )

                fund = utils.fundamental_fetcher.get_fundamentals(symbol)
                mcap = fund.get("market_cap", 0)

                logger.info(f"  📊 Indicadores:")
                logger.info(f"    Volume financeiro médio: {avg_fin_volume:,.0f}")
                logger.info(f"    ATR %: {atr_pct:.2f}%")
                logger.info(f"    Correlação IBOV: {ibov_corr:.3f}")
                logger.info(f"    Spread %: {spread_pct:.3f}%")
                logger.info(f"    Market Cap: R$ {mcap:,.0f}")

                # Verifica critérios do build_b3_universe
                criteria = {
                    "volume": avg_fin_volume >= 15_000_000,
                    "atr_min": atr_pct >= 0.80,
                    "atr_max": atr_pct <= 7.0,
                    "ibov_corr": ibov_corr <= 0.82,
                    "market_cap": mcap >= 1_000_000_000,
                    "spread": spread_pct <= 0.35,
                }

                logger.info(f"  ✅ Critérios:")
                for crit, passed in criteria.items():
                    status = "✅" if passed else "❌"
                    logger.info(f"    {crit}: {status}")

                # Se passou em todos os critérios, calcula score
                if all(criteria.values()):
                    score = utils.calculate_asset_score(
                        volume=avg_fin_volume,
                        atr_pct=atr_pct,
                        corr=ibov_corr,
                        mcap=mcap,
                        spread_pct=spread_pct,
                    )
                    logger.info(f"  🏆 Score: {score:.1f}")

                    if score >= 85:
                        category = "ELITE"
                    elif score >= 65:
                        category = "OPORTUNIDADE"
                    else:
                        category = "TOTAL"

                    logger.info(f"  📈 Categoria: {category}")
                    approved_symbols.append((symbol, score, category))
                else:
                    logger.info(f"  ❌ Reprovado - não atende todos os critérios")

            except Exception as e:
                logger.error(f"  ❌ Erro ao analisar {symbol}: {e}")

        logger.info(f"\n🎯 RESUMO:")
        logger.info(f"Símbolos testados: {len(test_symbols)}")
        logger.info(f"Aprovados: {len(approved_symbols)}")

        for symbol, score, category in approved_symbols:
            logger.info(f"  {symbol}: {score:.1f} ({category})")

        return True

    except Exception as e:
        logger.error(f"❌ Erro geral: {e}")
        return False
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    debug_universe_builder_detailed()
