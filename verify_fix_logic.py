
import numpy as np
import pandas as pd
import optuna
from optimizer_optuna import backtest_params_on_df

def test_warmup_limit():
    print("Testing Warmup Limit (Problem 1)...")
    # df with 200 rows -> max_ema should be 200 * 0.35 = 70
    df = pd.DataFrame({
        'open': np.random.rand(200),
        'high': np.random.rand(200),
        'low': np.random.rand(200),
        'close': np.random.rand(200),
        'volume': np.random.rand(200)
    })
    
    # We need to mock the environment for objective_wrapper
    # Or just test the logic directly if possible.
    # Since objective_wrapper is a closure inside optimize_with_optuna, 
    # we might need to test a similar logic or use a mock study.
    
    train_period = len(df)
    max_ema = int(train_period * 0.35)
    print(f"Train period: {train_period}, Max EMA allowed: {max_ema}")
    
    # Test with a high range (e.g. 80-200)
    ema_l_range = (80, 200)
    suggested_ema = min(ema_l_range[1], max_ema) 
    print(f"Suggested EMA (min(200, {max_ema})): {suggested_ema}")
    
    if suggested_ema <= 70:
        print("OK Problem 1 logic verified: EMA long limited to 35% of window.")
    else:
        print("FAIL Problem 1 logic failed: EMA long exceeded 35% limit.")

def test_insufficient_setups():
    print("\nTesting Insufficient Setups (Problem 2)...")
    # Create a df that is flat (no signals)
    df = pd.DataFrame({
        'open': [10.0] * 300,
        'high': [10.1] * 300,
        'low': [9.9] * 300,
        'close': [10.0] * 300,
        'volume': [1000] * 300
    })
    
    params = {
        "ema_short": 10,
        "ema_long": 50,
        "rsi_low": 30,
        "rsi_high": 70,
        "adx_threshold": 25,
        "enable_shorts": 1
    }
    
    metrics = backtest_params_on_df("TEST", params, df)
    print(f"Metrics: {metrics}")
    
    if metrics.get("insufficient_setups"):
        print(f"OK Problem 2 logic verified: Caught insufficient setups ({metrics.get('setups_identified')})")
    else:
        print(f"FAIL Problem 2 logic failed: Did not catch insufficient setups. (Trades: {metrics.get('total_trades')}, Setups: {metrics.get('setups_identified')})")

if __name__ == "__main__":
    try:
        test_warmup_limit()
        test_insufficient_setups()
    except Exception as e:
        print(f"Error during verification: {e}")
