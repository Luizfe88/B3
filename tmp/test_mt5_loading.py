import MetaTrader5 as mt5
import pandas as pd
from otimizador_semanal import load_data_with_retry, ensure_mt5_connection
import os
import logging

logging.basicConfig(level=logging.INFO)

symbols = ["PETR4", "ABEV3", "VALE3", "WIN$N"]

if ensure_mt5_connection():
    print("MT5 Connected.")
    for sym in symbols:
        print(f"\n--- Testing {sym} ---")
        df = load_data_with_retry(sym, 100, timeframe="M15")
        if df is not None and not df.empty:
            print(f"✅ {sym}: Loaded {len(df)} rows.")
            print(df.tail(2))
        else:
            print(f"❌ {sym}: Failed to load data.")
else:
    print("Failed to connect to MT5.")
