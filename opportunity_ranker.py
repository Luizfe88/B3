"""
Opportunity Ranker v1.0 - XP3 PRO QUANT-REFORM
Responsável por ordenar as oportunidades de trade detectadas
baseado em um Score de Convicção unificado, pesando:
40% Debate (Consensus Diff)
30% ML Probability
20% Order Flow Score
10% Sentiment / IBOV Trend
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("opportunity_ranker")

class OpportunityRanker:
    
    def __init__(self):
        # Pesos sugeridos para avaliação de convicção
        self.weights = {
            'debate': 0.40,
            'ml': 0.30,
            'flow': 0.20,
            'sentiment': 0.10
        }

    def _extract_scores(self, decision: Dict) -> Dict[str, float]:
        """Extrai sub-scores normalizados das análises do FundManager (0.0 a 1.0)"""
        
        # 1. Debate Diff (quão unânime é o research team)
        debate_info = decision.get("debate_info", {})
        diff = abs(debate_info.get("diff", 0.0))
        # Normalizando o diff (normalmente varia de 0.0 a ~3.5). Limitando a 2.5 
        debate_score = min(1.0, diff / 2.5) 

        # 2. ML Probability
        analysis = decision.get("analysis", {})
        tech = analysis.get("technical", {})
        # Usamos a score de probabilidade pura se diponível, ou uma média conservadora
        ml_prob = tech.get("score", 0.50)  

        # 3. Order Flow Imbalance
        flow = analysis.get("orderflow", {})
        imbalance = flow.get("imbalance", 0.0)
        # Normaliza o imbalance (-1.0 a 1.0) para score de força absoluta (0.0 a 1.0)
        flow_score = min(1.0, abs(imbalance))

        # 4. Sentiment
        sentiment = analysis.get("sentiment", {})
        sent_pct = sentiment.get("score", 0.50)
        # Transforma o score (0.20 a 0.80) em força de convicção (distância do meio)
        # Quanto mais longe do neutro, mais convicção
        dist_neutral = abs(sent_pct - 0.50) * 2  # range 0.0 a 1.0
        sentiment_score = min(1.0, dist_neutral)

        return {
            'debate': debate_score,
            'ml': ml_prob,
            'flow': flow_score,
            'sentiment': sentiment_score
        }

    def rank_opportunities(self, opportunities: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """
        Ordena a lista de oportunidades do loop principal.
        
        Args:
            opportunities: Lista de tuplas (symbol, decision_dict) dos trades aprovados.
            
        Returns:
            Lista sorteada com os maiores scores globais.
        """
        if not opportunities:
            return []

        ranked_list = []
        for symbol, decision in opportunities:
            scores = self._extract_scores(decision)
            
            # Conviction Score (Ponderada)
            final_conviction = (
                (scores['debate'] * self.weights['debate']) +
                (scores['ml'] * self.weights['ml']) +
                (scores['flow'] * self.weights['flow']) +
                (scores['sentiment'] * self.weights['sentiment'])
            )
            
            ranked_list.append({
                'symbol': symbol,
                'decision': decision,
                'conviction': final_conviction,
                'breakdown': scores
            })

        # Ordena ordem decrescente de convicção
        ranked_list.sort(key=lambda x: x['conviction'], reverse=True)

        # Log de transparência do Ranking Resultante
        logger.info(f"🏆 Rank das {len(ranked_list)} Top Opportunities:")
        for idx, item in enumerate(ranked_list):
            sym = item['symbol']
            action = item['decision']['action']
            conv = item['conviction']
            bd = item['breakdown']
            logger.info(
                f"   #{idx+1} {sym} -> {action} | "
                f"Conviction: {conv*100:.1f} (Debate:{bd['debate']:.2f} ML:{bd['ml']:.2f} Flow:{bd['flow']:.2f} Sent:{bd['sentiment']:.2f})"
            )

        # Retorna apenas a tupla original re-ordenada
        return [(item['symbol'], item['decision']) for item in ranked_list]

# Singleton
opportunity_ranker = OpportunityRanker()
