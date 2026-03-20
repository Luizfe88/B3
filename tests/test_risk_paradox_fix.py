import sys
import os
import logging
from unittest.mock import MagicMock

# Setup path to include current directory
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock components that are not needed for this specific test
sys.modules['MetaTrader5'] = MagicMock()
sys.modules['validation.permutation_test'] = MagicMock()

def test_risk_paradox():
    import config
    import database
    from agents.risk_team import RiskGuardian
    
    # 1. Setup
    guardian = RiskGuardian("RiskSeeker", 0.8)
    symbol = "PETR4"
    
    # Simula proposta de alta confiança (90%)
    # p=0.9, b=1.25 -> f* = (0.9*(1.25+1) - 1) / 1.25 = (0.9*2.25 - 1) / 1.25 = (2.025 - 1) / 1.25 = 1.025 / 1.25 = 82%
    # Meio-Kelly (0.5) -> 41%
    # Hard Cap 10%
    
    proposal = {
        "action": "BUY",
        "probability": 0.90,
        "size_multiplier": 1.0
    }
    
    market_context = {
        "equity": 100000.0,
        "total_exposure": 0.0,
        "ibov_trend": "neutral"
    }
    
    # Mock database to return insufficient trades for recalibration (use fallback)
    database.get_symbol_statistics = MagicMock(return_value={"total_trades": 0})
    
    # 2. Execução
    print(f"\n--- Testando Paradoxo de Risco (Kelly 10% vs Hard Limit) ---")
    result = guardian.validate_trade(symbol, proposal, market_context)
    
    # 3. Verificação
    print(f"Resultado: Approved={result.get('approved')}, Reason={result.get('reason')}")
    
    if result.get("approved"):
        adj = result.get("adjusted_proposal", {})
        final_multiplier = adj.get("size_multiplier")
        effective_risk = config.MAX_CAPITAL_ALLOCATION_PCT * final_multiplier
        print(f"Size Multiplier Final: {final_multiplier:.2f}x")
        print(f"Risco Efetivo Final: {effective_risk:.2%}")
        
        # Deve estar em 10% (0.10)
        assert abs(effective_risk - 0.10) < 0.001, f"Risco deveria ser 10%, mas é {effective_risk:.2%}"
        assert result["approved"] == True, "Deveria estar aprovado no novo limite de 10%"
    else:
        assert False, f"Teste falhou: Proposta foi rejeitada por {result.get('reason')}"

if __name__ == "__main__":
    test_risk_paradox()
    print("\n✅ Sucesso: O paradoxo de risco foi resolvido!")
