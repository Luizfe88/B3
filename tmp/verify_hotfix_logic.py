
import sys
import os
import logging

# Adicionar o diretório atual ao sys.path para importar os módulos locais
sys.path.append(os.getcwd())

import config_risk
from core.position_manager import PositionManager

# Mock Execution Engine
class MockExecution:
    def get_account_info(self):
        class Info:
            equity = 500.0
        return Info()
    def get_positions(self):
        return []
    def close_position(self, ticket, symbol):
        print(f"DEBUG: Closing {symbol} (Ticket: {ticket})")
        return True

def test_hotfix_sizing():
    print("Testing Hotfix Sizing and Conviction...")
    exec_engine = MockExecution()
    pm = PositionManager(exec_engine)
    
    # 1. Test Conviction Filter (Below 75%)
    vol, can, reason = pm.validate_and_size_order("PETR4", "BUY", 200, 0.70)
    print(f"PETR4 @ 70% Conviction: Vol={vol}, Can={can}, Reason={reason}")
    assert can == False
    assert "Conviccao insuficiente" in reason

    # 2. Test Fixed Lot (Equity 500 < 1000)
    vol, can, reason = pm.validate_and_size_order("VALE3", "BUY", 500, 0.80)
    print(f"VALE3 @ 80% Conviction (Equity 500): Vol={vol}, Can={can}, Reason={reason}")
    assert can == True
    assert vol == 100.0
    assert "Fixed Lot" in reason

    # 3. Test Futures Exclusion (Fixed Lot should NOT apply to WIN)
    vol, can, reason = pm.validate_and_size_order("WINJ24", "BUY", 1, 0.80)
    print(f"WINJ24 @ 80% Conviction: Vol={vol}, Can={can}, Reason={reason}")
    assert can == True
    # For futures, it should use base_volume * dynamic_multiplier
    # 80% conviction -> dynamic_multiplier = 0.50. Base 1 * 0.5 = 0.5
    assert vol == 0.5
    assert "Fixed Lot" not in reason

    print("✅ All Hotfix Sizing tests passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        test_hotfix_sizing()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
