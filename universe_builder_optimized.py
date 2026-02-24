#!/usr/bin/env python3
"""
Universe Builder otimizado com thresholds ajustados - apenas símbolos pré-definidos
"""
import utils
import MetaTrader5 as mt5
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_universe_optimized():
    """
    Universe Builder otimizado com thresholds realistas para o mercado B3 atual
    """
    try:
        logger.info("🔌 Conectando MT5...")
        if not mt5.initialize():
            logger.error(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

        logger.info("✅ MT5 conectado")

        # Usa apenas os símbolos pré-definidos e aplica filtro de ações
        all_symbols_raw = list(set(utils.ELITE_SYMBOLS + utils.OPORTUNIDADE_SYMBOLS))
        all_symbols = [s for s in all_symbols_raw if utils.is_stock(s)]
        logger.info(
            f"🎯 Analisando {len(all_symbols)} ativos pré-definidos (filtrados para apenas ações)..."
        )
        if len(all_symbols_raw) != len(all_symbols):
            logger.info(
                f"ℹ️ {len(all_symbols_raw) - len(all_symbols)} ativos foram filtrados (ETFs/FIIs/futuros)"
            )

        universe = {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

        # Thresholds ajustados baseado no debug
        THRESHOLDS = {
            "min_fin_volume": 500_000,  # Reduzido de 15M para 500K
            "min_atr_pct": 0.80,  # Mantém
            "max_atr_pct": 7.0,  # Mantém
            "max_ibov_corr": 0.90,  # Aumentado de 0.82 para 0.90
            "min_market_cap": 300_000_000,  # Reduzido de 1B para 300M
            "max_spread_pct": 0.35,  # Mantém
        }

        logger.info(f"📊 Thresholds ajustados:")
        for key, value in THRESHOLDS.items():
            logger.info(
                f"  {key}: {value:,}"
                if isinstance(value, (int, float))
                else f"  {key}: {value}"
            )

        analyzed = 0
        valid = 0

        for symbol in all_symbols:
            try:
                name = symbol.upper().replace(".SA", "")

                # Verificações básicas
                if utils.is_future(name):
                    continue

                info = mt5.symbol_info(name)
                if not info or not info.select:
                    # Tenta selecionar o símbolo
                    if not mt5.symbol_select(name, True):
                        logger.debug(f"❌ Falha ao selecionar {name}")
                        continue
                    info = mt5.symbol_info(name)
                    if not info:
                        continue

                # Dados históricos
                rates = utils.safe_copy_rates(name, mt5.TIMEFRAME_D1, 60)
                if rates is None or len(rates) < 20:
                    continue

                analyzed += 1

                # Calcula indicadores
                avg_fin_volume = (rates["tick_volume"] * rates["close"]).mean()
                atr = utils.get_atr(rates, 14)
                current_price = rates["close"].iloc[-1]
                atr_pct = (atr / current_price) * 100
                ibov_corr = utils.get_ibov_correlation(name)
                spread_pct = (
                    (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
                )

                # Fundamentals
                fund = utils.fundamental_fetcher.get_fundamentals(name)
                mcap = fund.get("market_cap", 0)

                # Aplica filtros
                filters_passed = (
                    avg_fin_volume >= THRESHOLDS["min_fin_volume"]
                    and THRESHOLDS["min_atr_pct"]
                    <= atr_pct
                    <= THRESHOLDS["max_atr_pct"]
                    and ibov_corr <= THRESHOLDS["max_ibov_corr"]
                    and mcap >= THRESHOLDS["min_market_cap"]
                    and spread_pct <= THRESHOLDS["max_spread_pct"]
                )

                if not filters_passed:
                    continue

                valid += 1

                # Calcula score
                score = utils.calculate_asset_score(
                    volume=avg_fin_volume,
                    atr_pct=atr_pct,
                    corr=ibov_corr,
                    mcap=mcap,
                    spread_pct=spread_pct,
                )

                # Classifica
                if score >= 85:
                    universe["ELITE"].append((name, score))
                    logger.info(f"🏆 ELITE: {name} - Score: {score:.1f}")
                elif score >= 65:
                    universe["OPORTUNIDADE"].append((name, score))
                    logger.info(f"🎯 OPORTUNIDADE: {name} - Score: {score:.1f}")
                else:
                    universe["TOTAL"].append((name, score))
                    logger.debug(f"📊 TOTAL: {name} - Score: {score:.1f}")

            except Exception as e:
                logger.debug(f"❌ Erro ao analisar {symbol}: {e}")
                continue

        # Ordena por score
        for category in universe:
            universe[category].sort(key=lambda x: x[1], reverse=True)

        # Adiciona metadata
        universe["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_analyzed": analyzed,
            "valid_stocks": valid,
            "elite_count": len(universe["ELITE"]),
            "oportunidade_count": len(universe["OPORTUNIDADE"]),
            "total_count": len(universe["TOTAL"]),
            "thresholds": THRESHOLDS,
        }

        logger.info(f"\n🎯 Resultados finais:")
        logger.info(f"📊 Analisados: {analyzed}")
        logger.info(f"✅ Válidos: {valid}")
        logger.info(f"🏆 ELITE: {len(universe['ELITE'])} ativos")
        logger.info(f"🎯 OPORTUNIDADE: {len(universe['OPORTUNIDADE'])} ativos")
        logger.info(f"📈 TOTAL: {len(universe['TOTAL'])} ativos")

        # Salva JSON
        if universe["ELITE"] or universe["OPORTUNIDADE"]:
            logger.info("💾 Salvando elite_symbols_latest.json...")
            success = utils.atomic_save_json("elite_symbols_latest.json", universe)
            if success:
                logger.info("✅ JSON salvo com sucesso!")
            else:
                logger.error("❌ Falha ao salvar JSON")

        return universe

    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        return {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    build_universe_optimized()
