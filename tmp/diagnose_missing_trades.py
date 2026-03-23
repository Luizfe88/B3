
import sqlite3
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import os
import sys

# Add current dir to path to import config and database
sys.path.append(os.getcwd())

import config
import database

def diagnose():
    print("=== Diagnostic: Missing Trades ===")
    
    # 1. Check MT5 History
    import utils
    if not utils.safe_mt5_initialize():
        print("Failed to initialize MT5")
        return

    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = datetime.now() + timedelta(minutes=1)
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals:
        print(f"MT5 Deals today: {len(deals)}")
        symbols_in_mt5 = set(deal.symbol for deal in deals)
        print(f"Symbols in MT5 today: {symbols_in_mt5}")
        
        for symbol in symbols_in_mt5:
            deals_symbol = [d for d in deals if d.symbol == symbol]
            for deal in deals_symbol:
                is_monitored = symbol in config.MONITORED_SYMBOLS
                print(f"  - {symbol}: Magic={deal.magic}, Monitored={is_monitored}")
    else:
        print("No MT5 deals today.")

    # 2. Check Database
    conn = sqlite3.connect(database.DB_PATH)
    db_symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM trades", conn)['symbol'].tolist()
    conn.close()
    print(f"Symbols in DB (all time): {db_symbols}")

    # 3. Check for discrepancies
    if deals:
        for deal in deals:
            if deal.symbol not in config.MONITORED_SYMBOLS:
                print(f"EXPLAINED: Symbol {deal.symbol} is NOT in MONITORED_SYMBOLS and will be filtered out.")
            
    mt5.shutdown()

if __name__ == "__main__":
    diagnose()
