import MetaTrader5 as mt5
import logging
import time
from core.execution import ExecutionEngine, OrderParams, OrderSide
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestStability")

def test_robustness():
    engine = ExecutionEngine()
    if not engine.connect():
        logger.error("Falha ao conectar")
        return

    symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3"]
    
    logger.info("--- Testando Seleção de Símbolos ---")
    for sym in symbols:
        success = engine.safe_symbol_select(sym)
        logger.info(f"Seleção {sym}: {'✅ OK' if success else '❌ FALHA'}")

    logger.info("--- Testando Validação de Stops Level ---")
    for sym in symbols:
        tick = mt5.symbol_info_tick(sym)
        if not tick: continue
        
        # Simula SL/TP muito próximos (0.01 centavo)
        bad_sl = tick.bid - 0.01
        bad_tp = tick.bid + 0.01
        
        ind = {"atr": 0.1, "adx": 25}
        # A função calculate_dynamic_sl_tp agora chama validate_stops_level internamente
        sl, tp = utils.calculate_dynamic_sl_tp(sym, "BUY", tick.ask, ind)
        
        info = mt5.symbol_info(sym)
        stops_level = getattr(info, "trade_stops_level", 0)
        point = getattr(info, "point", 0.01)
        min_dist = stops_level * point
        
        actual_dist_sl = abs(tick.ask - sl)
        actual_dist_tp = abs(tick.ask - tp)
        
        logger.info(f"Asset: {sym} | stops_level: {stops_level} pts ({min_dist:.4f})")
        logger.info(f"  Price: {tick.ask} | Calculated SL: {sl} (Dist: {actual_dist_sl:.4f})")
        logger.info(f"  Price: {tick.ask} | Calculated TP: {tp} (Dist: {actual_dist_tp:.4f})")
        
        if actual_dist_sl < min_dist and stops_level > 0:
            logger.error(f"❌ SL ainda viola stops_level em {sym}!")
        elif actual_dist_tp < min_dist and stops_level > 0:
            logger.error(f"❌ TP ainda viola stops_level em {sym}!")
        else:
            logger.info(f"✅ Stops respeitados em {sym}")

if __name__ == "__main__":
    test_robustness()
