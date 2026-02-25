
import logging
from typing import Dict, Any, List
from .analyst_team import AnalystTeam
from .researcher_team import ResearcherTeam
from .trader_agents import TraderTeam
from .risk_team import RiskTeam

logger = logging.getLogger("FundManager")

class FundManager:
    """
    Agente decisor final.
    Orquestra o fluxo de todos os outros agentes.
    """
    def __init__(self):
        self.analysts = AnalystTeam()
        self.researchers = ResearcherTeam()
        self.traders = TraderTeam()
        self.risk_manager = RiskTeam()
        
    def decide(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa o pipeline completo de decisão (ReAct).
        Thought -> Action -> Observation
        """
        logger.info(f"🏦 [FundManager] Avaliando ativo {symbol}...")
        
        # 1. Analyst Team (Incluindo OrderFlow)
        reports = self.analysts.analyze_all(symbol, market_data)
        
        # Check OrderFlow Veto
        of_report = reports.get("orderflow", {})
        if of_report.get("pressure") == "bearish" and of_report.get("score") < 0.3:
             logger.warning(f"⚠️ [FundManager] Veto por Fluxo de Ordens negativo em {symbol}")
             return {
                "symbol": symbol,
                "action": "HOLD",
                "reason": "OrderFlow Veto (High Selling Pressure)",
                "size": 0.0,
                "reports": reports
            }
        
        # 2. Researcher Team (Debate)
        debate_result = self.researchers.debate(symbol, reports)
        
        # 3. Trader Agents (Proposals)
        proposals = self.traders.collect_proposals(symbol, reports, debate_result)
        
        # 4. Risk Management (Validation)
        risk_assessment = self.risk_manager.assess_risk(symbol, proposals, market_data)
        
        # 5. Final Decision
        final_decision = {
            "symbol": symbol,
            "action": "HOLD",
            "reason": "Risk rejected or no consensus",
            "size": 0.0,
            "reports": reports,
            "debate": debate_result
        }
        
        if risk_assessment.get("approved"):
            prop = risk_assessment.get("final_proposal")
            if prop:
                final_decision["action"] = prop.get("action", "HOLD")
                final_decision["size"] = prop.get("size_multiplier", 0.0)
                final_decision["reason"] = f"Approved by Risk Team. Based on {debate_result.get('consensus')} consensus."
                
        logger.info(f"✅ [FundManager] Decisão Final para {symbol}: {final_decision['action']} (Size: {final_decision['size']:.2%})")
        return final_decision
