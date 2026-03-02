
import logging
from typing import Dict, Any, List

logger = logging.getLogger("TraderAgents")

class Trader:
    def __init__(self, name: str, risk_profile: str):
        self.name = name
        self.risk_profile = risk_profile
        
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cada trader propõe uma ação baseada em seu perfil e nos dados
        """
        raise NotImplementedError

class RiskyTrader(Trader):
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🔥 [{self.name}] Analisando oportunidade agressiva...")
        
        # Lógica agressiva: Aceita drawdown, busca momentum
        tech = analysis.get('technical', {})
        
        # Compra
        if tech.get('trend') == 'bullish' and debate_result.get('consensus') == 'BULLISH':
            logger.info(f"   ↳ Proposta: COMPRA (Agressiva)")
            return {"action": "BUY", "size_multiplier": 1.2, "stop_loss_pct": 0.05} # Stop mais largo
            
        # Venda (Short)
        if tech.get('trend') == 'bearish' and debate_result.get('consensus') == 'BEARISH':
            logger.info(f"   ↳ Proposta: VENDA (Agressiva)")
            return {"action": "SELL", "size_multiplier": 1.2, "stop_loss_pct": 0.05}
            
        return {"action": "HOLD", "reason": "Sem momentum suficiente"}

class NeutralTrader(Trader):
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"⚖️ [{self.name}] Buscando equilíbrio...")
        
        # Lógica balanceada: Requer confirmação fundamental e técnica
        tech = analysis.get('technical', {})
        fund = analysis.get('fundamental', {})
        
        # Compra
        if tech.get('trend') == 'bullish' and fund.get('valuation') != 'expensive' and debate_result.get('consensus') == 'BULLISH':
            logger.info(f"   ↳ Proposta: COMPRA (Balanceada)")
            return {"action": "BUY", "size_multiplier": 1.0, "stop_loss_pct": 0.03}
            
        # Venda (Short)
        if tech.get('trend') == 'bearish' and fund.get('valuation') != 'cheap' and debate_result.get('consensus') == 'BEARISH':
             logger.info(f"   ↳ Proposta: VENDA (Balanceada)")
             return {"action": "SELL", "size_multiplier": 1.0, "stop_loss_pct": 0.03}
            
        return {"action": "HOLD", "reason": "Falta confluência"}

class SafeTrader(Trader):
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🛡️ [{self.name}] Priorizando proteção de capital...")
        
        # Lógica conservadora: Requer tudo alinhado e risco baixo
        tech = analysis.get('technical', {})
        fund = analysis.get('fundamental', {})
        sent = analysis.get('sentiment', {})
        
        # Compra — FIX: aceita 'fair' além de 'cheap' (antes 'fair' bloqueava o BUY)
        if (tech.get('trend') == 'bullish' and 
            fund.get('valuation') in ('cheap', 'fair') and 
            sent.get('sentiment') == 'optimistic'):
            logger.info(f"   ↳ Proposta: COMPRA (Conservadora) — val={fund.get('valuation')}")
            return {"action": "BUY", "size_multiplier": 0.8, "stop_loss_pct": 0.015}
            
        # Venda (Short)
        if (tech.get('trend') == 'bearish' and 
            fund.get('valuation') in ('expensive', 'neutral') and 
            sent.get('sentiment') == 'pessimistic'):
            logger.info(f"   ↳ Proposta: VENDA (Conservadora) — val={fund.get('valuation')}")
            return {"action": "SELL", "size_multiplier": 0.8, "stop_loss_pct": 0.015}

        return {"action": "HOLD", "reason": "Risco inaceitável"}

class TraderTeam:
    def __init__(self):
        self.risky = RiskyTrader("Risky", "aggressive")
        self.neutral = NeutralTrader("Neutral", "balanced")
        self.safe = SafeTrader("Safe", "conservative")
        
    def collect_proposals(self, symbol: str, analysis: Dict[str, Any], debate: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            self.risky.propose_action(symbol, analysis, debate),
            self.neutral.propose_action(symbol, analysis, debate),
            self.safe.propose_action(symbol, analysis, debate)
        ]
