import sys
from unittest.mock import MagicMock

# Mock do config e logging antes de importar o componente
class MockConfig:
    MAX_PYRAMIDS_PER_SYMBOL = 3

sys.modules['config'] = MockConfig
sys.modules['utils'] = MagicMock()

# Importamos a classe para teste
from core.position_manager import PositionManager

def test_smart_sizing_logic():
    # Setup
    execution_mock = MagicMock()
    pm = PositionManager(execution_mock)
    
    # 1. Teste Sizing Dinâmico (Sem posições abertas)
    pm.get_open_positions = MagicMock(return_value=[])
    
    # Faixa 60-70 (25%)
    vol, can, msg = pm.validate_and_size_order("PETR4", "BUY", 1000, 0.65)
    print(f"Test 65% Conviction: Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 250
    assert can == True

    # Faixa 70-85 (50%)
    vol, can, msg = pm.validate_and_size_order("PETR4", "BUY", 1000, 0.75)
    print(f"Test 75% Conviction: Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 500
    assert can == True

    # Faixa 85+ (100%)
    vol, can, msg = pm.validate_and_size_order("PETR4", "BUY", 1000, 0.90)
    print(f"Test 90% Conviction: Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 1000
    assert can == True

    # 2. Teste Anti-Martingale (P&L Negativo)
    pm.get_open_positions = MagicMock(return_value=[
        {"symbol": "VALE3", "profit": -150.0}
    ])
    vol, can, msg = pm.validate_and_size_order("VALE3", "BUY", 1000, 0.90)
    print(f"Test Anti-Martingale (Loss): Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 0
    assert can == False
    assert "ANTI-MARTINGALE" in msg.upper()

    # 3. Teste Smart Pyramiding (P&L Positivo)
    pm.get_open_positions = MagicMock(return_value=[
        {"symbol": "VALE3", "profit": 200.0}
    ])
    vol, can, msg = pm.validate_and_size_order("VALE3", "BUY", 1000, 0.90)
    print(f"Test Pyramiding (Profit): Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 1000
    assert can == True
    assert "PYRAMIDING" in msg.upper()

    # 4. Teste Pyramiding Cap (Limite atingido)
    pm.get_open_positions = MagicMock(return_value=[
        {"symbol": "VALE3", "profit": 100.0},
        {"symbol": "VALE3", "profit": 50.0},
        {"symbol": "VALE3", "profit": 30.0}
    ])
    vol, can, msg = pm.validate_and_size_order("VALE3", "BUY", 1000, 0.90)
    print(f"Test Pyramiding Cap: Vol={vol}, Can={can}, Msg={msg}")
    assert vol == 0
    assert can == False
    assert "Cap de piramides" in msg

    print("\nDONE: Todos os testes de Smart Sizing e Pyramiding passaram!")

if __name__ == "__main__":
    try:
        test_smart_sizing_logic()
    except Exception as e:
        print(f"FAIL: Teste falhou - {type(e).__name__}")
        sys.exit(1)
    sys.exit(0)
