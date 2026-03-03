
import logging
from typing import Dict, Any, List

logger = logging.getLogger("TraderAgents")

class Trader:
    def __init__(self, name: str, risk_profile: str):
        self.name = name
        self.risk_profile = risk_profile
        
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RiskyTrader(Trader):
    """Busca momentum: age em consenso BULLISH/BEARISH com size maior."""
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🔥 [{self.name}] Avaliando momentum para {symbol}...")
        consensus = debate_result.get('consensus', 'NEUTRAL')

        if consensus == 'BULLISH':
            logger.info(f"   ↳ Proposta: COMPRA (Agressiva) — consenso={consensus}")
            return {"action": "BUY", "size_multiplier": 1.0, "stop_loss_pct": 0.04}

        if consensus == 'BEARISH':
            logger.info(f"   ↳ Proposta: VENDA (Agressiva) — consenso={consensus}")
            return {"action": "SELL", "size_multiplier": 1.0, "stop_loss_pct": 0.04}

        return {"action": "HOLD", "reason": "Dados insuficientes ou consenso neutro"}


class NeutralTrader(Trader):
    """
    Balanceado: age em BULLISH/BEARISH + caminho extra por macro+orderflow.
    NÃO exige trend técnico explicitamente — basta o debate ter decidido.
    """
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"⚖️ [{self.name}] Buscando equilíbrio para {symbol}...")

        consensus = debate_result.get('consensus', 'NEUTRAL')
        fund      = analysis.get('fundamental', {})
        sent      = analysis.get('sentiment', {})
        of        = analysis.get('orderflow', {})
        tech      = analysis.get('technical', {})

        # ── Caminho 1: Consenso BULLISH + fundamental não explosivo ──
        if consensus == 'BULLISH' and fund.get('valuation') != 'expensive':
            logger.info(f"   ↳ Proposta: COMPRA (Balanceada) — consenso BULLISH val={fund.get('valuation')}")
            return {"action": "BUY", "size_multiplier": 0.9, "stop_loss_pct": 0.03}

        # ── Caminho 2: Macro+Flow Trade (sem depender do ML) ──
        # Se IBOV bullish + comprador forte + fundamental ok → compra
        ibov_bullish  = sent.get('sentiment') == 'optimistic'
        of_bullish    = of.get('pressure') == 'bullish'
        fund_ok       = fund.get('valuation') in ('cheap', 'fair', 'neutral')
        ml_not_strong = float(tech.get('score', 0.5)) < 0.72  # ML não está fortemente bearish

        if ibov_bullish and of_bullish and fund_ok and ml_not_strong:
            logger.info(f"   ↳ Proposta: COMPRA (Macro+Flow) — IBOV_bull + OF_bull + fund={fund.get('valuation')}")
            return {"action": "BUY", "size_multiplier": 0.7, "stop_loss_pct": 0.025}

        # ── Caminho SELL: Consenso BEARISH ──
        if consensus == 'BEARISH' and fund.get('valuation') not in ('cheap',):
            logger.info(f"   ↳ Proposta: VENDA (Balanceada) — consenso BEARISH")
            return {"action": "SELL", "size_multiplier": 0.9, "stop_loss_pct": 0.03}

        return {"action": "HOLD", "reason": "Dados insuficientes ou cenário neutro"}


class SafeTrader(Trader):
    """
    Conservador: age com múltipla confirmação.
    Aceita valuation 'fair' (não só 'cheap') para BUY.
    Também tem caminho por macro+sentiment.
    """
    def propose_action(self, symbol: str, analysis: Dict[str, Any], debate_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"🛡️ [{self.name}] Priorizando proteção de capital em {symbol}...")

        tech = analysis.get('technical', {})
        fund = analysis.get('fundamental', {})
        sent = analysis.get('sentiment', {})
        of   = analysis.get('orderflow', {})
        consensus = debate_result.get('consensus', 'NEUTRAL')

        # ── Caminho 1: Consenso BULLISH + sentiment + fundo ──
        if (consensus == 'BULLISH'
                and sent.get('sentiment') == 'optimistic'
                and fund.get('valuation') in ('cheap', 'fair')):
            logger.info(f"   ↳ Proposta: COMPRA (Conservadora) — val={fund.get('valuation')}")
            return {"action": "BUY", "size_multiplier": 0.7, "stop_loss_pct": 0.02}

        # ── Caminho 2: IBOV bull + sentiment forte + sem pressão vendedora ──
        if (sent.get('sentiment') == 'optimistic'
                and of.get('pressure') != 'bearish'
                and fund.get('valuation') != 'expensive'
                and consensus != 'BEARISH'):
            logger.info(f"   ↳ Proposta: COMPRA (Macro-Conservative) — IBOV_bull + sent_opt")
            return {"action": "BUY", "size_multiplier": 0.6, "stop_loss_pct": 0.02}

        # ── SELL: tudo alinhado bearish ──
        if (tech.get('trend') == 'bearish'
                and fund.get('valuation') in ('expensive', 'neutral')
                and sent.get('sentiment') == 'pessimistic'):
            logger.info(f"   ↳ Proposta: VENDA (Conservadora) — val={fund.get('valuation')}")
            return {"action": "SELL", "size_multiplier": 0.7, "stop_loss_pct": 0.015}

        return {"action": "HOLD", "reason": "Dados insuficientes ou cenário neutro"}


class TraderTeam:
    def __init__(self):
        self.risky   = RiskyTrader("Risky", "aggressive")
        self.neutral = NeutralTrader("Neutral", "balanced")
        self.safe    = SafeTrader("Safe", "conservative")

    def collect_proposals(self, symbol: str, analysis: Dict[str, Any], debate: Dict[str, Any]) -> List[Dict[str, Any]]:
        proposals = [
            self.risky.propose_action(symbol, analysis, debate),
            self.neutral.propose_action(symbol, analysis, debate),
            self.safe.propose_action(symbol, analysis, debate),
        ]
        actions = [p['action'] for p in proposals]
        logger.info(f"📋 [{symbol}] Propostas coletadas: {actions}")
        return proposals
