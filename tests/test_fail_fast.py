
import logging
import unittest
from opportunity_ranker import OpportunityRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFailFast")

class TestFailFast(unittest.TestCase):
    def setUp(self):
        self.ranker = OpportunityRanker()

    def test_rejection_invalid_analyst(self):
        """Verifica se o ranker rejeita ativos com valid=False em qualquer analista"""
        opportunities = [
            ("VALID_STK", {
                "action": "BUY",
                "analysis": {
                    "technical": {"score": 0.75, "valid": True},
                    "fundamental": {"score": 0.60, "valid": True},
                    "sentiment": {"score": 0.70, "valid": True},
                    "orderflow": {"score": 0.65, "valid": True}
                },
                "debate_info": {"diff": 1.5}
            }),
            ("INVALID_STK", {
                "action": "SELL",
                "analysis": {
                    "technical": {"score": 0.75, "valid": True},
                    "fundamental": {"score": 0.0, "valid": False}, # INVALID
                    "sentiment": {"score": 0.70, "valid": True},
                    "orderflow": {"score": 0.65, "valid": True}
                },
                "debate_info": {"diff": 1.5}
            })
        ]
        
        ranked = self.ranker.rank_opportunities(opportunities)
        symbols = [item[0] for item in ranked]
        
        self.assertIn("VALID_STK", symbols)
        self.assertNotIn("INVALID_STK", symbols)
        logger.info("✅ Teste rejeição por valid=False passou.")

    def test_rejection_neutral_ml(self):
        """Verifica se o ranker rejeita ativos com score ML exatamente 0.50"""
        opportunities = [
            ("NEUTRAL_ML", {
                "action": "BUY",
                "analysis": {
                    "technical": {"score": 0.50, "valid": True}, # NEUTRAL ML
                    "fundamental": {"score": 0.60, "valid": True},
                    "sentiment": {"score": 0.70, "valid": True},
                    "orderflow": {"score": 0.65, "valid": True}
                },
                "debate_info": {"diff": 1.5}
            })
        ]
        
        ranked = self.ranker.rank_opportunities(opportunities)
        self.assertEqual(len(ranked), 0)
        logger.info("✅ Teste rejeição por ML=0.50 passou.")

if __name__ == "__main__":
    unittest.main()
