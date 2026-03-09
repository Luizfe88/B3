
import database
from datetime import datetime
import pandas as pd
import sqlite3

def verify_sync():
    print("--- Verifying Database Sync ---")
    database.sync_trades_from_mt5()
    
    conn = sqlite3.connect(database.DB_PATH)
    today = datetime.now().date().isoformat()
    df_today = pd.read_sql_query("SELECT * FROM trades WHERE date(timestamp) = date(?)", conn, params=(today,))
    conn.close()
    
    print(f"Total trades found for today in DB: {len(df_today)}")
    if len(df_today) > 0:
        wins = df_today[df_today["pnl_money"] > 0]
        print(f"Wins found today: {len(wins)}")
        winrate = (len(wins) / len(df_today[df_today['exit_time'].notnull()])) * 100 if len(df_today[df_today['exit_time'].notnull()]) > 0 else 0
        print(f"Calculated Winrate (closed trades): {winrate:.1f}%")
        print("\nDetails of today's trades:")
        print(df_today[["timestamp", "symbol", "side", "pnl_money", "exit_time"]])
    else:
        print("No trades found for today in DB after sync.")

if __name__ == "__main__":
    verify_sync()
