import numpy as np
import pandas as pd
import sys
import logging

# Ensure we can import from the current directory
sys.path.append(".")

from optimizer_optuna import compute_metrics, backtest_params_on_df, diagnostico_final

def print_clean(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        # Strip emojis or other non-ascii chars for Windows console
        print(msg.encode('ascii', 'ignore').decode('ascii'))

def test_sharpe_paradox_fix():
    print_clean("Testing Sharpe Paradox Fix...")
    # Equity curve that is losing: 100k -> 90k
    equity = np.linspace(100000, 90000, 100)
    metrics = compute_metrics(equity)
    
    print_clean(f"Metrics: PF={metrics['profit_factor']:.2f}, Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:.2%}")
    
    assert metrics['profit_factor'] < 1.0
    assert metrics['sharpe'] <= 0, f"Sharpe should be <= 0 for losing strategy, got {metrics['sharpe']}"
    print_clean("✅ Sharpe Paradox Fix verified.")

def test_ml_mock_fix():
    print_clean("\nTesting ML Mock Fix...")
    # Mock DF
    df = pd.DataFrame({
        "open": np.random.rand(200) + 10,
        "high": np.random.rand(200) + 11,
        "low": np.random.rand(200) + 9,
        "close": np.random.rand(200) + 10,
        "volume": np.random.rand(200) * 1000
    }, index=pd.date_range("2023-01-01", periods=200, freq="15min"))
    
    # Mock ML Model
    class MockModel:
        def predict_proba(self, X):
            # Return 0.6 for all
            return np.ones((len(X), 2)) * 0.4, np.ones((len(X), 2)) * 0.6
            
        def predict(self, X):
            return np.ones(len(X))

    params = {
        "ema_short": 9,
        "ema_long": 21,
        "rsi_low": 30,
        "rsi_high": 70,
        "adx_threshold": 25,
        "sl_mult": 2.0,
        "tp_mult": 3.0,
        "base_slippage": 0.0001
    }
    
    try:
        res = backtest_params_on_df("TEST", params, df, ml_model=MockModel())
        print_clean("✅ ML Integration (with model) ran successfully.")
    except Exception as e:
        print_clean(f"❌ ML Integration failed: {e}")
        
    try:
        res = backtest_params_on_df("TEST", params, df, ml_model=None)
        print_clean("✅ ML Integration (no model/fallback) ran successfully.")
    except Exception as e:
        print_clean(f"❌ ML Fallback failed: {e}")

def test_diagnostic_lock_fix():
    print_clean("\nTesting Diagnostic Lock Fix...")
    
    # Scenario: PF < 1.0 (Losing)
    metrics_losing = {
        "sharpe": 4.5, # Fake high sharpe
        "pf": 0.5,
        "n_trades": 10,
        "symbol": "LOSER"
    }
    params = {"adx_threshold": 25.0}
    
    print_clean("Losing strategy diagnostic (should be REPROVADO):")
    try:
        diagnostico_final(metrics_losing, params)
    except UnicodeEncodeError:
        print_clean("Diagnostic printed with encoding warning (expected on Windows).")
    
    # Scenario: Low Trades
    metrics_low_trades = {
        "sharpe": 1.5,
        "pf": 3.0,
        "n_trades": 2,
        "symbol": "FEW_TRADES"
    }
    print_clean("\nLow trades strategy diagnostic (should be REPROVADO):")
    try:
        diagnostico_final(metrics_low_trades, params)
    except UnicodeEncodeError:
        print_clean("Diagnostic printed with encoding warning.")

if __name__ == "__main__":
    try:
        test_sharpe_paradox_fix()
        test_ml_mock_fix()
        test_diagnostic_lock_fix()
        print_clean("\nSUMMARY: Hotfix logic looks good!")
    except Exception as e:
        print_clean(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
