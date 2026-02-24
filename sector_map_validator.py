#!/usr/bin/env python3
"""
Validador diário do Sector Map - garante que apenas ações válidas sejam monitoradas
"""
import utils
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_sector_map():
    """
    Valida o sector map e retorna apenas ações válidas
    """
    try:
        # Carrega o sector map do config
        from config import SECTOR_MAP

        logger.info(f"📊 Validando {len(SECTOR_MAP)} símbolos do sector map...")

        valid_stocks = {}
        invalid_stocks = {}

        for symbol, sector in SECTOR_MAP.items():
            if utils.is_stock(symbol):
                valid_stocks[symbol] = sector
            else:
                invalid_stocks[symbol] = sector
                logger.info(f"❌ {symbol} ({sector}) - Não é ação válida")

        logger.info(f"✅ {len(valid_stocks)} ações válidas encontradas")
        logger.info(f"❌ {len(invalid_stocks)} ativos inválidos removidos")

        if invalid_stocks:
            logger.info(f"📝 Ativos inválidos: {list(invalid_stocks.keys())}")

        # Salva lista válida para uso futuro
        result = {
            "valid_stocks": valid_stocks,
            "invalid_stocks": invalid_stocks,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_original": len(SECTOR_MAP),
                "valid_stocks": len(valid_stocks),
                "invalid_stocks": len(invalid_stocks),
            },
        }

        utils.atomic_save_json("sector_map_valid.json", result)
        logger.info("💾 Lista válida salva em sector_map_valid.json")

        return valid_stocks

    except ImportError:
        logger.error("❌ Não foi possível importar SECTOR_MAP do config")
        return {}
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        return {}


def get_daily_sector_stocks():
    """
    Retorna lista diária de ações válidas por setor
    """
    valid_stocks = validate_sector_map()

    # Organiza por setor
    sector_groups = {}
    for symbol, sector in valid_stocks.items():
        if sector not in sector_groups:
            sector_groups[sector] = []
        sector_groups[sector].append(symbol)

    logger.info(f"📈 Setores identificados: {list(sector_groups.keys())}")
    for sector, symbols in sector_groups.items():
        logger.info(f"  {sector}: {len(symbols)} ações")

    return sector_groups


if __name__ == "__main__":
    logger.info("🔄 Iniciando validação diária do sector map...")
    sector_groups = get_daily_sector_stocks()
    logger.info("✅ Validação concluída!")
