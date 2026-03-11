#!/usr/bin/env python3
import json
import os
import logging
import config
from datetime import datetime

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
        verdict_counts = {}

        # Ativos que queremos priorizar
        TARGET_VERDICTS = ["SNIPER_ELITE", "TREND_HUNTER", "INTERMEDIATE_HIGH"]
        
        # Ativos a serem excluídos (Blacklist Global)
        EXCLUDE_SYMBOLS = config.FORBIDDEN_SYMBOLS

        # Dicionário para organizar por veredito
        categorized_elite = {v: [] for v in TARGET_VERDICTS}
        
        for symbol, params in symbols_config.items():
            if symbol in EXCLUDE_SYMBOLS:
                logger.info(f"🚫 {symbol} ignorado (Blacklist)")
                continue

            verdict = params.get("verdict", "UNKNOWN")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            if verdict in TARGET_VERDICTS:
                # Busca categoria do SECTOR_MAP se disponível
                # No bot, Blue Chips podem ser identificados por uma lista ou padrão
                is_blue = symbol in getattr(config, "ELITE_BLUE_CHIPS", [])
                category = "BLUE CHIP" if is_blue else "OPORTUNIDADE"
                
                symbol_data = {
                    "symbol": symbol,
                    "category": category,
                    "verdict": verdict,
                    "timeframe": params.get("timeframe", "N/A")
                }
                
                elite_symbols.append(symbol)
                categorized_elite[verdict].append(symbol_data)
                logger.info(f"🏆 {symbol} selecionado como ELITE (Status: {verdict})")

        if elite_symbols:
            # Salva lista simples para o Bot
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(elite_symbols, f, indent=4)
            
            # Salva versão categorizada para relatório
            report_file = "whitelist_elite_categorized.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "summary": verdict_counts,
                    "elite_by_verdict": categorized_elite
                }, f, indent=4)
                
            # ✅ NOVO: Salva um JSON individual para cada ativo elite (opcional para o bot)
            assets_dir = "elite_assets"
            os.makedirs(assets_dir, exist_ok=True)
            for verdict, assets in categorized_elite.items():
                for asset in assets:
                    symbol = asset["symbol"]
                    # Busca os parâmetros completos da calibração para esse ativo
                    full_params = symbols_config.get(symbol, {})
                    asset_file = os.path.join(assets_dir, f"{symbol}.json")
                    with open(asset_file, 'w', encoding='utf-8') as f:
                        json.dump(full_params, f, indent=4)

            logger.info(f"✅ Whitelist ELITE gerada com {len(elite_symbols)} ativos em {output_file}")
            logger.info(f"📊 Relatório detalhado salvo em {report_file}")
            logger.info(f"📂 Arquivos individuais salvos na pasta '{assets_dir}/'")
        else:
            logger.warning("⚠️ Nenhum ativo ELITE encontrado para gerar a whitelist.")
            logger.info(f"📊 Resumo de vereditos encontrados: {verdict_counts}")
            logger.info("💡 Certifique-se de que o otimizador foi executado e gerou vereditos de alta performance.")
            
    except Exception as e:
        logger.error(f"❌ Erro ao gerar whitelist: {e}")

if __name__ == "__main__":
    generate_elite_whitelist()
