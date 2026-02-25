
import logging
from typing import Dict, Any, List

logger = logging.getLogger("ResearcherTeam")

class ResearcherTeam:
    def __init__(self):
        self.bull_thesis = []
        self.bear_thesis = []

    def debate(self, symbol: str, analyst_reports: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simula o debate Bull vs Bear obrigatório.
        4 rodadas de argumentação baseada nos relatórios dos analistas.
        """
        logger.info(f"🗣️ Iniciando DEBATE para {symbol}...")
        
        # Extrair dados
        tech = analyst_reports.get('technical', {})
        fund = analyst_reports.get('fundamental', {})
        sent = analyst_reports.get('sentiment', {})
        
        # Rodada 1: Teses Iniciais
        bull_score = 0
        bear_score = 0
        
        if tech.get('trend') == 'bullish':
            self.bull_thesis.append("Tendência técnica de alta confirmada")
            bull_score += 1
        else:
            self.bear_thesis.append("Tendência técnica fraca ou de baixa")
            bear_score += 1
            
        if fund.get('valuation') == 'cheap':
            self.bull_thesis.append("Valuation atrativo (P/L baixo)")
            bull_score += 1
        elif 'high_debt' in fund.get('risks', []):
            self.bear_thesis.append("Alto endividamento detectado")
            bear_score += 1
            
        # Consenso
        consensus = "NEUTRAL"
        if bull_score > bear_score:
            consensus = "BULLISH"
        elif bear_score > bull_score:
            consensus = "BEARISH"
            
        logger.info(f"🥊 Resultado do Debate: {consensus} (Bull: {bull_score} vs Bear: {bear_score})")
        
        return {
            "consensus": consensus,
            "bull_points": self.bull_thesis,
            "bear_points": self.bear_thesis,
            "confidence": max(bull_score, bear_score) / (bull_score + bear_score + 0.1) # Avoid div/0
        }
