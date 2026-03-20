import sys
import os
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

# Adiciona o diretório atual ao path para importar os módulos
sys.path.append(os.getcwd())

from adaptive_intelligence import AdaptiveIntelligence, MarketMetrics

class TestAdaptiveMetrics(unittest.TestCase):
    def setUp(self):
        self.ai = AdaptiveIntelligence()
        # Limpa o histórico para o teste
        self.ai.metrics_history.clear()

    def test_collect_metrics_no_trades(self):
        """Testa se a coleta de métricas retorna valores neutros quando não há trades."""
        with patch.object(self.ai, '_get_recent_trades', return_value=[]):
            with patch.object(self.ai, '_calculate_market_volatility', return_value=0.02):
                with patch.object(self.ai, '_detect_volume_anomalies', return_value=1.0):
                    with patch.object(self.ai, '_calculate_market_correlation', return_value=0.5):
                        with patch.object(self.ai, '_calculate_trend_strength', return_value=0.5):
                            metrics = self.ai._collect_current_metrics()
                            
                            self.assertIsNotNone(metrics)
                            self.assertEqual(metrics.winrate_24h, 0.5)
                            self.assertEqual(metrics.sharpe_4h, 1.0)
                            print(f"OK Coleta sem trades - Winrate: {metrics.winrate_24h}, Sharpe: {metrics.sharpe_4h}")

    def test_performance_report_empty(self):
        """Testa o relatório de performance com histórico vazio."""
        report = self.ai.get_performance_report()
        metrics = report['performance_metrics']
        
        self.assertEqual(metrics['avg_winrate_24h'], 0.5)
        self.assertEqual(metrics['avg_sharpe_4h'], 1.0)
        print(f"OK Relatório vazio - Avg Winrate: {metrics['avg_winrate_24h']}, Avg Sharpe: {metrics['avg_sharpe_4h']}")

    def test_analyze_and_adjust_neutral(self):
        """Testa se valores neutros evitam recomendações drásticas de proteção."""
        # Popula o histórico com 10 snapshots neutros
        for i in range(10):
            m = MarketMetrics(
                timestamp=datetime.now() - timedelta(minutes=15*i),
                winrate_1h=0.5,
                winrate_4h=0.5,
                winrate_24h=0.5,
                sharpe_1h=1.0,
                sharpe_4h=1.0,
                sharpe_24h=1.0,
                avg_trade_duration=0.0,
                avg_sl_distance=1.5,
                avg_tp_distance=3.0,
                market_volatility=0.02,
                volume_anomaly=1.0,
                correlation_strength=0.5,
                trend_strength=0.5
            )
            self.ai.metrics_history.append(m)
        
        import pandas as pd
        df = pd.DataFrame([vars(m) for m in self.ai.metrics_history])
        recs = self.ai._generate_recommendations(df)
        
        # Não deve haver recomendações de aumento de confiança ou redução de kelly
        # (A menos que a volatilidade seja alta, mas aqui é 0.02)
        self.assertNotIn("increase_confidence_threshold", recs)
        self.assertNotIn("reduce_kelly_multiplier", recs)
        print(f"OK Recomendações neutras - Recomendações geradas: {list(recs.keys())}")

if __name__ == "__main__":
    unittest.main()
