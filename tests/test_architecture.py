
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import pandas as pd
from datetime import datetime

# Adiciona diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock MetaTrader5 BEFORE importing bot modules
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Configura Mock do MT5
mt5.initialize.return_value = True
mt5.symbol_info_tick.return_value = MagicMock(bid=10.0, ask=10.01)
mt5.symbol_info.return_value = MagicMock(point=0.01, trade_tick_size=0.01)
mt5.positions_get.return_value = []
# Configura constante de retorno
mt5.TRADE_RETCODE_DONE = 10009
mt5.order_send.return_value = MagicMock(retcode=10009, order=12345, price=10.0, volume=100, comment="Executed")

from agents.fund_manager import FundManager
from agents.analyst_team import AnalystTeam
from agents.researcher_team import ResearcherTeam
from agents.trader_agents import TraderTeam
from agents.risk_team import RiskTeam
from core.execution import ExecutionEngine, OrderParams, OrderSide

class TestTradingAgentsArchitecture(unittest.TestCase):
    
    def setUp(self):
        self.fund_manager = FundManager()
        
    def test_analyst_team_structure(self):
        """Verifica se todos os analistas estão presentes"""
        self.assertIsNotNone(self.fund_manager.analysts.fundamental)
        self.assertIsNotNone(self.fund_manager.analysts.sentiment)
        self.assertIsNotNone(self.fund_manager.analysts.technical)
        self.assertIsNotNone(self.fund_manager.analysts.orderflow)
        
    def test_full_decision_flow_bullish(self):
        """Simula um cenário de COMPRA completo"""
        symbol = "PETR4"
        
        # Mock dos dados de mercado
        market_data = {
            "price": 30.0,
            "candles": pd.DataFrame({
                "close": [29.0, 29.5, 30.0],
                "high": [29.2, 29.8, 30.2],
                "low": [28.8, 29.3, 29.9],
                "volume": [1000, 1500, 2000]
            }),
            "ticks": [{"flags": 0, "price": 30.0, "volume": 100}]
        }
        
        # Mock do resultado dos analistas para forçar BULLISH
        with patch.object(self.fund_manager.analysts.technical.predictor, 'predict', return_value={'probability': 0.8, 'signal': 'BUY'}):
            
            # Executa decisão
            decision = self.fund_manager.decide(symbol, market_data)
            
            # Verificações
            print("\n🔍 Decisão Final:", decision)
            
            # Deve haver uma ação definida (BUY ou HOLD)
            self.assertIn(decision['action'], ['BUY', 'HOLD'])
            
            # Verifica se o relatório contém todos os componentes
            reports = decision['reports']
            self.assertIn('fundamental', reports)
            self.assertIn('technical', reports)
            self.assertIn('orderflow', reports)
            
            # Verifica Debate
            debate = decision['debate']
            self.assertIn('consensus', debate)
            print(f"🗣️ Consenso do Debate: {debate['consensus']}")

    def test_orderflow_veto(self):
        """Verifica se o OrderFlow consegue vetar uma operação"""
        symbol = "VALE3"
        market_data = {}
        
        # Mock para OrderFlow BEARISH forte
        mock_of_report = {
            "type": "order_flow",
            "score": 0.1,
            "pressure": "bearish",
            "imbalance": -0.8
        }
        
        with patch.object(self.fund_manager.analysts.orderflow, 'analyze', return_value=mock_of_report):
            decision = self.fund_manager.decide(symbol, market_data)
            
            self.assertEqual(decision['action'], 'HOLD')
            self.assertIn("OrderFlow Veto", decision['reason'])
            print("\n🚫 Veto de OrderFlow confirmado!")

    def test_execution_engine(self):
        """Testa o envio de ordens simulado"""
        engine = ExecutionEngine()
        engine.connect()
        
        order = OrderParams(
            symbol="ITUB4",
            side=OrderSide.BUY,
            volume=100,
            price=30.0
        )
        
        result = engine.send_order(order)
        self.assertEqual(result['status'], 'filled')
        self.assertEqual(result['ticket'], 12345)
        print("\n✅ Execução de ordem simulada com sucesso")

if __name__ == '__main__':
    unittest.main()
