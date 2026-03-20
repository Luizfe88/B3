import sys
import os
import logging
from unittest.mock import MagicMock

# Setup path to include current directory
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock components
sys.modules['MetaTrader5'] = MagicMock()
sys.modules['validation.permutation_test'] = MagicMock()

def test_macro_exception():
    import config
    import database
    from agents.risk_team import RiskGuardian
    
    # 1. Setup
    guardian = RiskGuardian("RiskSeeker", 0.8)
    symbol = "VALE3"
    
    # Caso A: Alta confiança (96%) + Bons Fundamentos (0.8) EM PÂNICO
    proposal_a = {
        "action": "BUY",
        "probability": 0.96,
        "fundamental_score": 0.8,
        "size_multiplier": 1.0
    }
    
    # Caso B: Confiança Comum (75%) EM PÂNICO (Deve ser bloqueado)
    proposal_b = {
        "action": "BUY",
        "probability": 0.75,
        "fundamental_score": 0.6,
        "size_multiplier": 1.0
    }
    
    market_context = {
        "equity": 100000.0,
        "total_exposure": 0.0,
        "ibov_trend": "bearish_extreme"
    }
    
    # 2. Execução Caso A
    print(f"\n--- Testando Exceção Macro (ML >= 95%) ---")
    result_a = guardian.validate_trade(symbol, proposal_a, market_context)
    print(f"Caso A (96%): Approved={result_a.get('approved')}, Reason={result_a.get('reason')}")
    assert result_a["approved"] == True, "Deveria aprovar 96% mesmo em Bearish Extreme"
    
    # 3. Execução Caso B
    print(f"\n--- Testando Bloqueio Macro Normal (ML < 95%) ---")
    result_b = guardian.validate_trade(symbol, proposal_b, market_context)
    print(f"Caso B (75%): Approved={result_b.get('approved')}, Reason={result_b.get('reason')}")
    assert result_b["approved"] == False, "Deveria bloquear 75% em Bearish Extreme"
    assert result_b["reason"] == "Market Panic Mode"

if __name__ == "__main__":
    test_macro_exception()
    print("\n✅ Sucesso: A exceção de regime macro está funcionando!")
