
import sys
import os
from datetime import datetime
import unittest
from unittest.mock import MagicMock, patch

# Adiciona o diretório atual ao sys.path para importar os módulos
sys.path.append(os.getcwd())

import config
from agents.fund_manager import FundManager

class TestFridayRestrictions(unittest.TestCase):
    def setUp(self):
        self.fund_manager = FundManager()
        # Mocking analyst team to avoid full analysis
        self.fund_manager.analysts = MagicMock()
        self.fund_manager.researchers = MagicMock()
        self.fund_manager.traders = MagicMock()
        self.fund_manager.risk_manager = MagicMock()

    def test_friday_after_3pm_entry_blocked(self):
        # 6 de Março de 2026 é uma Sexta-feira (weekday == 4)
        friday_after_3pm = datetime(2026, 3, 6, 15, 10, 0)
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = friday_after_3pm
            mock_datetime.strptime.side_effect = datetime.strptime
            
            # Setup dummy data
            symbol = "PETR4"
            market_data = {"price": 30.0}
            
            decision = self.fund_manager.decide(symbol, market_data)
            
            print(f"Decision at {friday_after_3pm}: {decision['action']} - Reason: {decision['reason']}")
            
            self.assertEqual(decision['action'], "HOLD")
            self.assertEqual(decision['reason'], "Entry Cutoff Time Reached")

    def test_friday_before_3pm_entry_allowed_logic(self):
        # Friday before 3pm
        friday_before_3pm = datetime(2026, 3, 6, 14, 50, 0)
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = friday_before_3pm
            mock_datetime.strptime.side_effect = datetime.strptime
            
            # Mock analysts to return something so it doesn't fail early
            self.fund_manager.analysts.analyze_all.return_value = {"orderflow": {"pressure": "neutral", "score": 0.5}}
            self.fund_manager.researchers.debate.return_value = {"confidence": 0.7, "consensus": "BULL"}
            self.fund_manager.traders.collect_proposals.return_value = []
            self.fund_manager.risk_manager.assess_risk.return_value = {"approved": False}
            
            symbol = "PETR4"
            market_data = {"price": 30.0}
            
            decision = self.fund_manager.decide(symbol, market_data)
            
            print(f"Decision at {friday_before_3pm}: {decision['action']} - Reason: {decision['reason']}")
            
            # It shouldn't be "Entry Cutoff Time Reached"
            self.assertNotEqual(decision['reason'], "Entry Cutoff Time Reached")

if __name__ == "__main__":
    # Testar se o config foi atualizado corretamente no teste
    print(f"Config FRIDAY_NO_ENTRY_AFTER: {config.FRIDAY_NO_ENTRY_AFTER}")
    unittest.main()
