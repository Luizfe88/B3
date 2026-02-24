#!/usr/bin/env python3
"""
Debug detalhado do Universe Builder
"""
import sys

sys.path.append(".")

import utils
import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def debug_universe_builder():
    """Debug detalhado do Universe Builder"""
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        logger.info("✅ MT5 conectado")

        # Testa alguns símbolos específicos
        test_symbols = ["PETR4", "VALE3", "ITUB4", "ABEV3"]

        for symbol in test_symbols:
            logger.info(f"\n🔍 Analisando {symbol}...")

            # Verifica se é futuro
            is_future = utils.is_future(symbol)
            logger.info(f"  É futuro: {is_future}")

            if is_future:
                continue

            # Pega informações do símbolo
            info = mt5.symbol_info(symbol)
            if not info:
                logger.info(f"  ❌ Símbolo não encontrado")
                continue

            logger.info(f"  Select: {info.select}")
            logger.info(f"  Trade Mode: {info.trade_mode}")
            logger.info(f"  Ask: {info.ask}")
            logger.info(f"  Bid: {info.bid}")
            logger.info(f"  Point: {info.point}")

            if info.trade_mode not in [1, 4]:
                logger.info(f"  ❌ Trade mode inválido: {info.trade_mode}")
                continue

            # Pega dados históricos
            rates = utils.safe_copy_rates(symbol, timeframe=mt5.TIMEFRAME_D1, count=60)
            if rates is None:
                logger.info(f"  ❌ Sem dados históricos")
                continue

            logger.info(f"  📊 Dados históricos: {len(rates)} candles")

            if len(rates) < 20:
                logger.info(f"  ❌ Poucos dados: {len(rates)} candles")
                continue

            # Calcula indicadores
            avg_fin_volume = (rates["tick_volume"] * rates["close"]).mean()
            logger.info(f"  Volume financeiro médio: {avg_fin_volume:,.0f}")

            atr = utils.get_atr(rates, 14)
            current_price = rates["close"].iloc[-1]
            atr_pct = (atr / current_price) * 100
            logger.info(f"  ATR %: {atr_pct:.2f}%")

            ibov_corr = utils.get_ibov_correlation(symbol)
            logger.info(f"  Correlação IBOV: {ibov_corr:.3f}")

            spread_pct = (
                (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
            )
            logger.info(f"  Spread %: {spread_pct:.3f}%")

            fund = utils.fundamental_fetcher.get_fundamentals(symbol)
            mcap = fund.get("market_cap", 0)
            logger.info(f"  Market Cap: R$ {mcap:,.0f}")

            # Calcula score
            score = utils.calculate_asset_score(
                volume=avg_fin_volume,
                atr_pct=atr_pct,
                corr=ibov_corr,
                mcap=mcap,
                spread_pct=spread_pct,
            )

            logger.info(f"  🏆 Score final: {score:.1f}")

            # Verifica se passa nos filtros
            filters = {
                "volume": avg_fin_volume >= 1_000_000,
                "atr_min": atr_pct >= 0.5,
                "atr_max": atr_pct <= 10.0,
                "spread": spread_pct <= 0.35,
                "score": score >= 50,
            }

            logger.info(f"  ✅ Filtros: {filters}")

            if all(filters.values()):
                logger.info(f"  🎉 {symbol} APROVADO!")
            else:
                logger.info(f"  ❌ {symbol} REPROVADO")

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    debug_universe_builder()
