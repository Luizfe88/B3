
import os
import sys
import logging
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Configuração de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_agents.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Forçar encoding UTF-8 no stdout para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger("MainBot")

# Imports do sistema
from core.execution import ExecutionEngine, OrderParams, OrderSide
from core.position_manager import PositionManager
from agents.fund_manager import FundManager
import config
import utils
import MetaTrader5 as mt5

def main():
    logger.info("🚀 Iniciando TradingAgents-B3 Framework...")
    
    # 1. Inicialização de Infraestrutura
    execution = ExecutionEngine()
    if not execution.connect():
        logger.critical("❌ Falha crítica: Não foi possível conectar ao MT5. Abortando.")
        return

    position_manager = PositionManager(execution)
    fund_manager = FundManager()
    
    # 2. Loop Principal
    logger.info("✅ Sistema online. Iniciando loop de mercado.")
    
    try:
        while True:
            # Verifica horário de mercado
            if not utils.is_market_open():
                logger.info("💤 Mercado fechado. Aguardando...")
                time.sleep(60)
                continue
                
            # Verifica conexões
            if not execution.connect():
                logger.warning("⚠️ MT5 desconectado. Tentando reconectar...")
                time.sleep(5)
                continue

            # Obtém lista de ativos (Universe Builder)
            # Por enquanto, usa lista estática ou do config
            symbols = config.MONITORED_SYMBOLS
            
            for symbol in symbols:
                try:
                    # 1. Coleta dados de mercado (Market Data)
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"⚠️ Não foi possível selecionar {symbol} no MT5. Pulando.")
                        continue
                    
                    # Candles (últimos 100 M15)
                    candles = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
                    if candles is None or candles.empty:
                        logger.warning(f"⚠️ Dados insuficientes (candles) para {symbol}. Pulando.")
                        continue
                        
                    # Ticks (últimos 1000 ticks)
                    try:
                        ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(hours=1), 1000, mt5.COPY_TICKS_ALL)
                    except Exception:
                        ticks = []

                    # Preço atual
                    tick = mt5.symbol_info_tick(symbol)
                    current_price = tick.last if tick else candles['close'].iloc[-1]
                    
                    market_data = {
                        "price": current_price,
                        "ticks": ticks if ticks is not None else [],
                        "candles": candles
                    }
                    
                    # 2. Decisão do Fund Manager (Agentes)
                    decision = fund_manager.decide(symbol, market_data)
                    
                    # 3. Execução
                    if decision["action"] == "BUY":
                        # Valida se já tem posição
                        open_positions = position_manager.get_open_positions()
                        if any(p['symbol'] == symbol for p in open_positions):
                            logger.info(f"⏭️ Posição já existente em {symbol}. Ignorando.")
                            continue
                            
                        # Cria ordem
                        order = OrderParams(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            volume=100 * decision["size"], # Ajustar lote mínimo
                            price=0.0, # Market order
                            sl=0.0, # Calcular SL
                            tp=0.0  # Calcular TP
                        )
                        execution.send_order(order)
                        
                    elif decision["action"] == "SELL":
                        # Implementar lógica de short
                        pass
                        
                except Exception as e:
                    logger.error(f"❌ Erro no loop para {symbol}: {e}")
            
            # Gerenciamento de posições abertas
            position_manager.update_stops()
            
            # Sleep para evitar sobrecarga (Timeframe M15/H1 sugerido)
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("🛑 Parada manual solicitada.")
    finally:
        execution.shutdown()
        logger.info("👋 Bot finalizado.")

if __name__ == "__main__":
    main()
