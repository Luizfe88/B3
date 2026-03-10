#!/usr/bin/env python3
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("natural_selection")

def generate_elite_whitelist(calib_file="calibrations.json", output_file="whitelist_elite.json"):
    """
    Analisa calibrations.json e gera uma whitelist com base nos vereditos de alta performance.
    """
    if not os.path.exists(calib_file):
        logger.error(f"❌ Arquivo {calib_file} não encontrado.")
        return

    try:
        with open(calib_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        symbols_config = data.get("symbols", {})
        elite_symbols = []

        # Ativos que queremos priorizar
        TARGET_VERDICTS = ["SNIPER_ELITE", "TREND_HUNTER", "INTERMEDIATE_HIGH"]

        for symbol, params in symbols_config.items():
            verdict = params.get("verdict", "UNKNOWN")
            if verdict in TARGET_VERDICTS:
                elite_symbols.append(symbol)
                logger.info(f"🏆 {symbol} selecionado como ELITE (Status: {verdict})")

        if elite_symbols:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(elite_symbols, f, indent=4)
            logger.info(f"✅ Whitelist ELITE gerada com {len(elite_symbols)} ativos em {output_file}")
        else:
            logger.warning("⚠️ Nenhum ativo ELITE encontrado para gerar a whitelist.")
            
    except Exception as e:
        logger.error(f"❌ Erro ao gerar whitelist: {e}")

if __name__ == "__main__":
    generate_elite_whitelist()
