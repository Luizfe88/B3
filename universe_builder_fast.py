#!/usr/bin/env python3
"""
Universe Builder otimizado - analisa apenas os ativos pré-definidos
"""
import sys

sys.path.append(".")

import utils
import MetaTrader5 as mt5
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_universe_fast():
    """
    Universe Builder otimizado - analisa apenas ativos pré-definidos
    """
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return None

        logger.info("✅ MT5 conectado")

        # Primeiro adiciona os ativos ao Market Watch
        logger.info("📌 Adicionando ativos ELITE ao Market Watch...")
        utils.auto_add_stocks_to_market_watch(utils.ELITE_SYMBOLS, log_name="ELITE")

        logger.info("📌 Adicionando ativos OPORTUNIDADE ao Market Watch...")
        utils.auto_add_stocks_to_market_watch(
            utils.OPORTUNIDADE_SYMBOLS, log_name="OPORTUNIDADE"
        )

        # Combina todos os símbolos para análise
        all_symbols = list(set(utils.ELITE_SYMBOLS + utils.OPORTUNIDADE_SYMBOLS))
        logger.info(f"🎯 Analisando {len(all_symbols)} ativos pré-definidos...")

        # Coleta dados de cada símbolo
        results = []

        for symbol in all_symbols:
            try:
                # Verifica se é futuro
                if utils.is_future(symbol):
                    continue

                # Pega informações do símbolo
                info = mt5.symbol_info(symbol)
                if not info or info.trade_mode not in [1, 4]:
                    continue

                # Pega dados históricos para cálculos
                rates = utils.safe_copy_rates(
                    symbol, timeframe=mt5.TIMEFRAME_D1, count=60
                )
                if rates is None or len(rates) < 20:
                    continue

                # Calcula indicadores
                avg_fin_volume = (rates["tick_volume"] * rates["close"]).mean()
                atr = utils.get_atr(rates, 14)
                current_price = rates["close"][-1]
                atr_pct = (atr / current_price) * 100
                ibov_corr = utils.get_ibov_correlation(symbol)
                spread_pct = (
                    (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
                )

                # Market cap mock (pode ser substituído por dados reais)
                fund = utils.fundamental_fetcher.get_fundamentals(symbol)
                mcap = fund.get("market_cap", 0)

                # Calcula score
                score = utils.calculate_asset_score(
                    volume=avg_fin_volume,
                    atr_pct=atr_pct,
                    corr=ibov_corr,
                    mcap=mcap,
                    spread_pct=spread_pct,
                )

                results.append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "volume": volume_fin,
                        "atr_pct": atr_pct,
                        "ibov_corr": ibov_corr,
                        "spread_pct": spread_pct,
                        "market_cap": mcap,
                    }
                )

                logger.info(
                    f"📊 {symbol}: Score {score:.1f} | ATR {atr_pct:.2f}% | Vol {volume_fin:,.0f}"
                )

            except Exception as e:
                logger.debug(f"❌ Erro ao analisar {symbol}: {e}")
                continue

        # Classifica por score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Separa em categorias
        elite = [r["symbol"] for r in results if r["score"] >= 85]
        oportunidade = [r["symbol"] for r in results if 65 <= r["score"] < 85]
        total = [r["symbol"] for r in results if r["score"] >= 50]

        universe = {
            "ELITE": elite,
            "OPORTUNIDADE": oportunidade,
            "TOTAL": total,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_analyzed": len(all_symbols),
                "valid_stocks": len(results),
                "elite_count": len(elite),
                "oportunidade_count": len(oportunidade),
                "total_count": len(total),
            },
        }

        # Salva em JSON
        utils.atomic_save_json("elite_symbols_latest.json", universe)

        logger.info(f"🏆 Universe Builder concluído:")
        logger.info(f"  📈 ELITE: {len(elite)} ativos")
        logger.info(f"  🔍 OPORTUNIDADE: {len(oportunidade)} ativos")
        logger.info(f"  📊 TOTAL: {len(total)} ativos")

        if elite:
            logger.info(f"  🥇 Top 5 ELITE: {', '.join(elite[:5])}")

        return universe

    except Exception as e:
        logger.error(f"❌ Erro no Universe Builder: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    universe = build_universe_fast()
    if universe:
        logger.info("✅ Universe Builder executado com sucesso!")
        sys.exit(0)
    else:
        logger.error("❌ Universe Builder falhou!")
        sys.exit(1)
