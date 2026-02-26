
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
        # 0. Check de Horário de Entrada (Entry Cutoff)
        from datetime import datetime
        import config
        
        now = datetime.now()
        current_time = now.time()
        
        # Check de Inicio do Dia (10:30)
        start_time_str = getattr(config, 'TRADING_START', "10:30")
        start_time = datetime.strptime(start_time_str, "%H:%M").time()
        
        if current_time < start_time:
             # logger.info(f"zzz [FundManager] Aguardando abertura ({start_time_str}).") # Reduzir log spam
             return {
                "symbol": symbol,
                "action": "HOLD",
                "reason": "Market Not Yet Open",
                "size": 0.0
            }
        
        # Define horário de bloqueio de entrada (Sexta vs Outros dias)
        if now.weekday() == 4: # Sexta-feira
             no_entry_str = config.FRIDAY_NO_ENTRY_AFTER
        else:
             no_entry_str = config.NO_ENTRY_AFTER
             
        no_entry_time = datetime.strptime(no_entry_str, "%H:%M").time()
        
        if current_time >= no_entry_time:
             logger.info(f"🛑 [FundManager] Horário limite de entradas atingido ({no_entry_str}). Bloqueando novas posições.")
             return {
                "symbol": symbol,
                "action": "HOLD",
                "reason": "Entry Cutoff Time Reached",
                "size": 0.0
            }

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

        # 2.1 Check de Almoço (Golden Opportunity Only)
        lunch_start_str = getattr(config, 'TRADING_LUNCH_BREAK_START', "11:45")
        lunch_end_str = getattr(config, 'TRADING_LUNCH_BREAK_END', "13:15")
        try:
             lunch_start = datetime.strptime(lunch_start_str, "%H:%M").time()
             lunch_end = datetime.strptime(lunch_end_str, "%H:%M").time()
             
             if lunch_start <= current_time <= lunch_end:
                 confidence = debate_result.get('confidence', 0.0)
                 consensus = debate_result.get('consensus', 'NEUTRAL')
                 
                 # Critério Ouro: Confiança > 0.8 (80%) E Consenso Definido (BULL/BEAR)
                 if confidence < 0.8 or consensus == "NEUTRAL":
                      logger.info(f"🥪 [FundManager] Horário de Almoço. Oportunidade comum descartada (Conf: {confidence:.2f}).")
                      return {
                         "symbol": symbol,
                         "action": "HOLD",
                         "reason": "Lunch Break Filter (Not Golden Opportunity)",
                         "size": 0.0,
                         "debate": debate_result
                     }
                 else:
                      logger.info(f"💎 [FundManager] Oportunidade de OURO no almoço! (Conf: {confidence:.2f} | {consensus})")
        except Exception as e:
             logger.error(f"⚠️ Erro ao verificar horário de almoço: {e}")
        
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
