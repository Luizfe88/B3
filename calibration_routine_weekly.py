"""
XP3 v5 - Weekly Calibration Routine
Orquestra a calibração semanal de hiperparâmetros e Kelly Dinâmico.
Executa o otimizador com todas as ações do Market Watch.
"""

import os
import sys
import logging
from datetime import datetime

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("weekly_routine")

def run_weekly_calibration():
    logger.info("🚀 Iniciando Rotina de Calibração Semanal...")
    
    # 1. Configura ambiente para escanear tudo do Market Watch
    os.environ["XP3_LOAD_ALL_MT5"] = "1"
    os.environ["XP3_IGNORE_FUTURES"] = "0" # Calibra ativos e futuros se visíveis
    os.environ["XP3_SANDBOX"] = "0" # Modo Produção (mais janelas WFA)
    
    # 2. Importa o otimizador e inicia
    try:
        from otimizador_semanal import run_optimizer
        
        start_time = datetime.now()
        run_optimizer()
        end_time = datetime.now()
        
        duration = end_time - start_time
        logger.info(f"✅ Calibração concluída com sucesso! Duração: {duration}")
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Erro crítico na rotina semanal: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    run_weekly_calibration()
