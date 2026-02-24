#!/usr/bin/env python3
"""
Resumo final do estado do Universe Builder
"""
import utils
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def show_current_status():
    """Mostra o estado atual do Universe Builder"""

    logger.info("📊 Estado Atual do Universe Builder B3")
    logger.info("=" * 50)

    # 1. Verifica o JSON atual
    try:
        with open("elite_symbols_latest.json", "r") as f:
            data = json.load(f)

        logger.info(f"✅ elite_symbols_latest.json encontrado!")
        logger.info(
            f"  📅 Criado em: {data.get('metadata', {}).get('timestamp', 'N/A')}"
        )
        logger.info(f"  🏆 ELITE: {len(data.get('ELITE', []))} ativos")
        logger.info(f"  🎯 OPORTUNIDADE: {len(data.get('OPORTUNIDADE', []))} ativos")
        logger.info(f"  📈 TOTAL: {len(data.get('TOTAL', []))} ativos")

        # Mostra exemplos
        if data.get("ELITE"):
            logger.info(f"\n🏆 Exemplos ELITE:")
            for item in data["ELITE"][:3]:
                symbol, score = item if isinstance(item, list) else (item, "N/A")
                logger.info(f"  - {symbol}: {score}")

        if data.get("OPORTUNIDADE"):
            logger.info(f"\n🎯 Exemplos OPORTUNIDADE:")
            for item in data["OPORTUNIDADE"][:5]:
                symbol, score = item if isinstance(item, list) else (item, "N/A")
                logger.info(f"  - {symbol}: {score}")

    except FileNotFoundError:
        logger.warning("⚠️ elite_symbols_latest.json não encontrado")
    except Exception as e:
        logger.error(f"❌ Erro ao ler JSON: {e}")

    # 2. Verifica funções implementadas
    logger.info(f"\n🔧 Funções Implementadas:")
    functions = [
        "build_b3_universe",
        "calculate_asset_score",
        "safe_copy_rates",
        "get_atr",
        "get_ibov_correlation",
        "FundamentalFetcher",
        "atomic_save_json",
        "load_elite_symbols_from_json",
        "ensure_market_watch_symbols",
        "auto_add_stocks_to_market_watch",
    ]

    for func in functions:
        if hasattr(utils, func):
            logger.info(f"  ✅ {func}")
        else:
            logger.error(f"  ❌ {func} - NÃO ENCONTRADO")

    # 3. Verifica listas de símbolos
    logger.info(f"\n📋 Listas de Símbolos:")
    if hasattr(utils, "ELITE_SYMBOLS"):
        logger.info(f"  ✅ ELITE_SYMBOLS: {len(utils.ELITE_SYMBOLS)} ativos")
    else:
        logger.error(f"  ❌ ELITE_SYMBOLS - NÃO ENCONTRADO")

    if hasattr(utils, "OPORTUNIDADE_SYMBOLS"):
        logger.info(
            f"  ✅ OPORTUNIDADE_SYMBOLS: {len(utils.OPORTUNIDADE_SYMBOLS)} ativos"
        )
    else:
        logger.error(f"  ❌ OPORTUNIDADE_SYMBOLS - NÃO ENCONTRADO")

    # 4. Testa carregamento do JSON
    logger.info(f"\n📂 Testando carregamento:")
    try:
        loaded = utils.load_elite_symbols_from_json()
        if loaded:
            logger.info(f"  ✅ load_elite_symbols_from_json() funcionando")
            logger.info(f"    ELITE: {len(loaded.get('ELITE', []))}")
            logger.info(f"    OPORTUNIDADE: {len(loaded.get('OPORTUNIDADE', []))}")
        else:
            logger.warning(f"  ⚠️ load_elite_symbols_from_json() retornou vazio")
    except Exception as e:
        logger.error(f"  ❌ load_elite_symbols_from_json() falhou: {e}")

    logger.info(f"\n🎉 Verificação concluída!")


if __name__ == "__main__":
    show_current_status()
