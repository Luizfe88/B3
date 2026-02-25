
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
            # Verifica se deve fechar posições no final do dia
            now = datetime.now().time()
            close_time_str = config.CLOSE_ALL_BY # "17:45"
            close_time = datetime.strptime(close_time_str, "%H:%M").time()
            
            if now >= close_time:
                 # Se ainda tiver posições abertas, fecha tudo
                 open_pos = position_manager.get_open_positions()
                 if open_pos:
                     logger.info("⏰ Horário de fechamento diário atingido. Zerando carteira...")
                     position_manager.close_all(reason="End of Day")
                 else:
                     logger.info("💤 Mercado fechado ou horário limite atingido. Aguardando...")
                 
                 time.sleep(60)
                 continue

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
                    
                    # Dados Globais de Risco
                    account_info = mt5.account_info()
                    equity = account_info.equity if account_info else 1000.0
                    
                    market_data = {
                        "price": current_price,
                        "ticks": ticks if ticks is not None else [],
                        "candles": candles,
                        "equity": equity,
                        "total_exposure": position_manager.get_total_exposure(),
                        "recent_entries_count": position_manager.count_recent_entries(minutes=60),
                        "ibov_trend": utils.get_market_regime()
                    }
                    
                    # 2. Decisão do Fund Manager (Agentes)
                    decision = fund_manager.decide(symbol, market_data)
                    
                    # 3. Execução
                    # Verifica account info para gestão de risco e lote
                    account_info = mt5.account_info()
                    if account_info:
                        equity = account_info.equity
                    else:
                        equity = 1000.0 # Fallback
                    
                    if decision["action"] == "BUY":
                        # Valida se já tem posição
                        open_positions = position_manager.get_open_positions()
                        for p in open_positions:
                            if p['symbol'] == symbol:
                                if p['type'] == 'SELL':
                                    logger.info(f"🔄 Invertendo mão em {symbol} (SELL -> BUY)")
                                    execution.close_position(p['ticket'], symbol)
                                else:
                                    logger.info(f"⏭️ Posição de COMPRA já existente em {symbol}. Mantendo.")
                                    continue # Já comprado, não faz nada (poderia aumentar posição)

                        # Cálculo de Lote:
                        # 1. Base: Config do capital (configurável)
                        # 2. Ajuste: size_multiplier do agente
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        size_multiplier = decision.get("size", 0.0)
                        
                        target_exposure = equity * base_allocation_pct * size_multiplier
                        raw_qty = target_exposure / current_price if current_price > 0 else 0
                        
                        final_volume = utils.normalize_volume(symbol, raw_qty)
                        
                        if final_volume <= 0:
                             logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                             continue
                        
                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(symbol, mt5.TIMEFRAME_M15, df=candles)
                        sl, tp = utils.calculate_dynamic_sl_tp(symbol, "BUY", current_price, ind)

                        # Cria ordem
                        order = OrderParams(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            volume=final_volume,
                            price=0.0, # Market order
                            sl=sl, # SL calculado
                            tp=tp  # TP calculado
                        )
                        execution.send_order(order)
                        
                    elif decision["action"] == "SELL":
                        # Valida se já tem posição
                        open_positions = position_manager.get_open_positions()
                        for p in open_positions:
                            if p['symbol'] == symbol:
                                if p['type'] == 'BUY':
                                    logger.info(f"🔄 Invertendo mão em {symbol} (BUY -> SELL)")
                                    execution.close_position(p['ticket'], symbol)
                                else:
                                    logger.info(f"⏭️ Posição de VENDA já existente em {symbol}. Mantendo.")
                                    continue

                        # Cálculo de Lote (Mesma lógica)
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        size_multiplier = decision.get("size", 0.0)
                        
                        target_exposure = equity * base_allocation_pct * size_multiplier
                        raw_qty = target_exposure / current_price if current_price > 0 else 0
                        
                        final_volume = utils.normalize_volume(symbol, raw_qty)
                        
                        if final_volume <= 0:
                             logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                             continue
                        
                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(symbol, mt5.TIMEFRAME_M15, df=candles)
                        sl, tp = utils.calculate_dynamic_sl_tp(symbol, "SELL", current_price, ind)
                            
                        # Cria ordem
                        order = OrderParams(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            volume=final_volume,
                            price=0.0, # Market order
                            sl=sl, # SL calculado
                            tp=tp  # TP calculado
                        )
                        execution.send_order(order)
                    
                    elif decision["action"] == "HOLD":
                        logger.info(f"⏸️ {symbol}: HOLD - Motivo: {decision.get('reason', 'N/A')}")
                        
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
