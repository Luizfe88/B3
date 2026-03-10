
import numpy as np
import pandas as pd
import sys
import os

# Mock dependencies if needed, or point to the right path
sys.path.append(r"c:\Users\luizf\Documents\xp3v5")

try:
    from optimizer_optuna import backtest_params_on_df
    print("[SUCCESS] Module imported")
except ImportError as e:
    print(f"[ERROR] Error importing module: {e}")
    sys.exit(1)

def verify_math():
    # Create a dummy dataframe with 200 bars (minimum for backtest_params_on_df)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    # Preco caindo constantemente: Longs vao perder
    df = pd.DataFrame({
        "open": np.linspace(100, 90, n),
        "high": np.linspace(101, 91, n),
        "low": np.linspace(99, 89, n),
        "close": np.linspace(100, 90, n),
        "tick_volume": np.random.randint(100, 1000, n)
    }, index=dates)

    params = {
        "ema_short": 5,
        "ema_long": 20,
        "rsi_low": 30,
        "rsi_high": 70,
        "adx_threshold": 10,
        "sl_atr_multiplier": 1.5,
        "tp_mult": 3.0,
        "base_slippage": 0.001,
        "enable_shorts": 0
    }

    print("\nRunning backtest with dummy (losing) data...")
    try:
        metrics = backtest_params_on_df("TEST_SYM", params, df)
        
        wr = metrics.get("win_rate", -1)
        pf = metrics.get("profit_factor", -1)
        trades = metrics.get("total_trades", 0)

        print(f"Results: Trades={trades} | WR={wr:.2f} | PF={pf:.2f}")

        if trades > 0:
            if wr == 0 and pf > 0:
                print("[FAILURE] Inconsistency still exists! WR=0 but PF>0")
            elif wr == 0 and pf == 0:
                print("[SUCCESS] Consistent! WR=0 and PF=0 for losing trades")
            else:
                print(f"[SUCCESS] Consistent metrics: WR={wr:.2f}, PF={pf:.2f}")
        else:
            print("[INFO] No trades executed in this run.")

    except Exception as e:
        print(f"[ERROR] Execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_math()
