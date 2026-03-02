
import logging
from typing import Dict, Any, List

logger = logging.getLogger("ResearcherTeam")

class ResearcherTeam:
    def __init__(self):
        self.bull_thesis = []
        self.bear_thesis = []

    def debate(self, symbol: str, analyst_reports: Dict[str, Any]) -> Dict[str, Any]:
        """
        Debate Bull vs Bear — pesos calibrados para eliminar dominância do ML bearish.

        Problemas corrigidos:
        - P1: Técnico (ML) peso fixo 1.5 → agora escalado pela confiança:
            ML 61% (weak edge) → contribuição 0.6; ML 85% → contribuição 1.3
        - P2: Fundamental 'expensive' por falta de dados → agora retorna 'neutral'
            (corrigido no FundamentalAnalyst — bars=0 → neutral)
        - P3: OrderFlow bullish sendo ignorado → peso 1.2 agora (era 1.0)
        - P4: Sentimento → peso 1.0 (era 0.75)
        - P5: Threshold reduzido 1.0 → 0.8 para facilitar consenso
        - Override: se sentiment + OF fortemente bullish, mesmo com ML fraco,
            pode desempatar para BULLISH
        """
        logger.info(f"🗣️ Iniciando DEBATE para {symbol}...")
        
        # Extrair dados
        tech = analyst_reports.get('technical', {})
        fund = analyst_reports.get('fundamental', {})
        sent = analyst_reports.get('sentiment', {})
        of   = analyst_reports.get('orderflow', {})
        
        # Reset
        self.bull_thesis = []
        self.bear_thesis = []
        bull_score = 0.0
        bear_score = 0.0
        
        # ── 1. TÉCNICO: escalado por confiança do ML ──────────────────────────
        # Antes: ML 61% de SELL → bear+1.5 (dominava tudo)
        # Agora: 61% → 0.6 contribuição; 85% → 1.2 contribuição; máx 1.3
        # Fórmula: min(1.3, max(0.0, (confidence - 0.38) / 0.62 * 1.3))
        tech_confidence = float(tech.get('score', 0.5))
        tech_trend      = tech.get('trend', 'neutral')

        def _tech_weight(conf: float) -> float:
            """Contribuição entre 0 e 1.3 conforme confiança do ML"""
            return min(1.3, max(0.0, (conf - 0.38) / 0.62 * 1.3))

        tech_contrib = _tech_weight(tech_confidence)

        if tech_trend == 'bullish':
            self.bull_thesis.append(f"Tendência técnica alta (conf:{tech_confidence:.0%})")
            bull_score += tech_contrib
        elif tech_trend == 'bearish':
            self.bear_thesis.append(f"Tendência técnica baixa (conf:{tech_confidence:.0%})")
            bear_score += tech_contrib

        # ── 2. FUNDAMENTAL: peso 1.0 (bars=0 agora retorna 'neutral' → sem penalidade) ──
        fund_val   = fund.get('valuation', 'neutral')
        fund_score = float(fund.get('score', 0.5))
        if fund_val == 'cheap':
            self.bull_thesis.append("Métricas MT5 favoráveis (liquidez/volatilidade)")
            bull_score += 1.0
        elif fund_val == 'fair':
            self.bull_thesis.append("Métricas MT5 neutras-positivas")
            bull_score += 0.5
        elif fund_val == 'expensive':
            self.bear_thesis.append("Métricas MT5 desfavoráveis (baixa liq/vol)")
            bear_score += 0.7   # reduzido de 1.0 → 0.7 (antes era excessivo)
            
        # ── 3. SENTIMENTO: peso 1.0 (era 0.75) ────────────────────────────────
        sent_val   = sent.get('sentiment', 'neutral')
        sent_score = float(sent.get('score', 0.5))
        if sent_val == 'optimistic':
            self.bull_thesis.append(f"Sentimento IBOV positivo (score:{sent_score:.2f})")
            bull_score += 1.0   # aumentado de 0.75 → 1.0
        elif sent_val == 'pessimistic':
            self.bear_thesis.append(f"Sentimento IBOV negativo (score:{sent_score:.2f})")
            bear_score += 1.0
            
        # ── 4. ORDER FLOW: peso 1.2 (era 1.0) ────────────────────────────────
        of_pressure  = of.get('pressure', 'neutral')
        of_imbalance = float(of.get('imbalance', 0.0))
        # Escala por magnitude do imbalance: imbalance 0.15 → +1.0; 0.4 → +1.2
        of_weight = min(1.5, 1.0 + abs(of_imbalance))
        if of_pressure == 'bullish':
            self.bull_thesis.append(f"Fluxo comprador (imbalance:{of_imbalance:+.2f})")
            bull_score += of_weight
        elif of_pressure == 'bearish':
            self.bear_thesis.append(f"Fluxo vendedor (imbalance:{of_imbalance:+.2f})")
            bear_score += of_weight
            
        # ── OVERRIDE: macro+flow vs ML fraco ─────────────────────────────────
        # Se IBOV bullish + OF bullish + ML fraco (< 0.68) → adiciona bull bônus
        # Isso captura: mercado subindo, compradores ativos, mas ML desatualizado
        if (sent_val == 'optimistic' and of_pressure == 'bullish'
                and tech_trend == 'bearish' and tech_confidence < 0.68):
            self.bull_thesis.append("Override: macro+flow vs ML fraco bearish")
            bull_score += 0.8   # bônus para contrabalançar ML biased
            
        # ── CONSENSO: threshold 0.8 (era 1.0) ────────────────────────────────
        diff = bull_score - bear_score
        
        if diff >= 0.8:
            consensus = "BULLISH"
        elif diff <= -0.8:
            consensus = "BEARISH"
        else:
            consensus = "NEUTRAL"
        
        # Confiança normalizada
        total = bull_score + bear_score
        confidence = abs(diff) / total if total > 0 else 0.0
            
        logger.info(
            f"🥊 Debate {symbol}: {consensus} "
            f"(Bull:{bull_score:.2f} vs Bear:{bear_score:.2f} | "
            f"diff:{diff:+.2f} | conf:{confidence:.2f} | tech_w:{tech_contrib:.2f})"
        )
        
        return {
            "consensus": consensus,
            "bull_points": self.bull_thesis,
            "bear_points": self.bear_thesis,
            "confidence": confidence,
            "bull_score": bull_score,
            "bear_score": bear_score,
        }
