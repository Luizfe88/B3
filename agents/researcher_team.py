
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
        of = analyst_reports.get('orderflow', {})
        
        # Reset
        self.bull_thesis = []
        self.bear_thesis = []
        bull_score = 0
        bear_score = 0
        
        # 1. Análise Técnica
        if tech.get('trend') == 'bullish':
            self.bull_thesis.append("Tendência técnica de alta")
            bull_score += 1
        elif tech.get('trend') == 'bearish':
            self.bear_thesis.append("Tendência técnica de baixa")
            bear_score += 1
            
        # 2. Análise Fundamentalista
        if fund.get('valuation') == 'cheap':
            self.bull_thesis.append("Valuation descontado")
            bull_score += 1
        elif fund.get('valuation') == 'expensive':
            self.bear_thesis.append("Valuation esticado")
            bear_score += 1
            
        # 3. Sentimento
        if sent.get('sentiment') == 'optimistic':
            self.bull_thesis.append("Sentimento positivo")
            bull_score += 0.5
        elif sent.get('sentiment') == 'pessimistic':
            self.bear_thesis.append("Sentimento negativo")
            bear_score += 0.5
            
        # 4. Fluxo
        if of.get('pressure') == 'bullish':
            self.bull_thesis.append("Fluxo comprador forte")
            bull_score += 1
        elif of.get('pressure') == 'bearish':
            self.bear_thesis.append("Fluxo vendedor forte")
            bear_score += 1
            
        # Consenso
        diff = bull_score - bear_score
        
        if diff >= 1.5:
            consensus = "BULLISH"
        elif diff <= -1.5:
            consensus = "BEARISH"
        else:
            consensus = "NEUTRAL"
            
        logger.info(f"🥊 Resultado do Debate: {consensus} (Bull: {bull_score} vs Bear: {bear_score})")
        
        return {
            "consensus": consensus,
            "bull_points": self.bull_thesis,
            "bear_points": self.bear_thesis,
            "confidence": max(bull_score, bear_score) / (bull_score + bear_score + 0.1) # Avoid div/0
        }
