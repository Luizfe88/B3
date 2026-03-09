
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5

DB_PATH = "xp3_trades.db"

def inspect_db():
    print("--- SQLite Database Inspection ---")
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} does not exist.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
        print("Latest 20 trades in DB:")
        print(df[["timestamp", "symbol", "side", "pnl_money", "exit_time"]])
        
        today = datetime.now().date().isoformat()
        df_today = pd.read_sql_query("SELECT COUNT(*) as count, SUM(pnl_money) as total_pnl FROM trades WHERE date(timestamp) = date(?)", conn, params=(today,))
        print(f"\nSummary for today ({today}) in DB:")
        print(df_today)
    finally:
        conn.close()


def inspect_mt5_history():
    print("\n--- MT5 History Inspection ---")
    import config
    
    init_params = {
        "path": getattr(config, "MT5_TERMINAL_PATH", None),
        "login": int(config.MT5_ACCOUNT) if config.MT5_ACCOUNT else 0,
        "password": str(config.MT5_PASSWORD or ""),
        "server": str(config.MT5_SERVER or ""),
        "timeout": 10000
    }
    init_params = {k: v for k, v in init_params.items() if v is not None}
    
    if not mt5.initialize(**init_params):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return
    
    acc_info = mt5.account_info()
    if acc_info:
        print(f"Connected to Account: {acc_info.login} | Server: {acc_info.server}")

    from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    to_date = datetime.now() + timedelta(minutes=1)
    
    deals = mt5.history_deals_get(from_date, to_date)
    if deals:
        print(f"Found {len(deals)} deals today in MT5 history:")
        for deal in deals:
            # entry: 0=IN, 1=OUT, 2=IN/OUT
            entry_type = "IN" if deal.entry == 0 else ("OUT" if deal.entry == 1 else "IN/OUT")
            print(f"Deal {deal.ticket}: {deal.symbol} {entry_type} | Profit: {deal.profit:.2f} | Time: {datetime.fromtimestamp(deal.time)}")
    else:
        print("No deals found in MT5 history for today.")
    
    mt5.shutdown()

if __name__ == "__main__":
    import os
    inspect_db()
    inspect_mt5_history()
