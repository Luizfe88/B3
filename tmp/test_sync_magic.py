
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

import database

def test_sync():
    print("=== Testing Sync with Magic 123456 ===")
    
    # Trigger sync
    database.sync_trades_from_mt5()
    
    # Check DB
    conn = sqlite3.connect(database.DB_PATH)
    # Check trades from today
    today = datetime.now().date().isoformat()
    df = pd.read_sql_query("SELECT timestamp, symbol, side, ticket, exit_price FROM trades WHERE date(timestamp) = date(?)", conn, params=(today,))
    conn.close()
    
    print(f"Trades found in DB for today: {len(df)}")
    if len(df) > 0:
        print(df)
    else:
        print("No trades found in DB for today. (Check if MT5 has deals with magic 123456 today)")

if __name__ == "__main__":
    test_sync()
