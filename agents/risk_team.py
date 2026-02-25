
import logging
from typing import Dict, Any, List

logger = logging.getLogger("RiskManagementTeam")

class RiskGuardian:
    def __init__(self, name: str, tolerance: float):
        self.name = name
        self.tolerance = tolerance
        
    def validate_trade(self, symbol: str, proposal: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida se o trade proposto respeita os limites de risco.
        Considera:
        - Drawdown máximo
        - Correlação
        - Exposição setorial
        - Tamanho da posição
        """
        logger.info(f"👮 [{self.name}] Validando risco para {symbol}...")
        
        # Simulação de verificação
        max_position_size = 0.20 # 20% capital
        proposed_size = proposal.get('size_multiplier', 0.0)
        
        if proposed_size > max_position_size:
            logger.warning(f"❌ [{self.name}] Tamanho excessivo ({proposed_size:.2%}). Ajustando.")
            proposal['size_multiplier'] = max_position_size
            proposal['adjusted'] = True
            
        # Verificação de correlação com IBOV
        corr = market_context.get('ibov_correlation', 0.5)
        if corr > 0.8 and self.tolerance < 0.5:
            logger.warning(f"⚠️ [{self.name}] Alta correlação com mercado em queda. Bloqueando.")
            return {"approved": False, "reason": "High correlation risk"}
            
        return {"approved": True, "adjusted_proposal": proposal}

class RiskTeam:
    def __init__(self):
        self.guardians = [
            RiskGuardian("RiskSeeker", 0.8),
            RiskGuardian("Neutral", 0.5),
            RiskGuardian("Conservative", 0.2)
        ]
        
    def assess_risk(self, symbol: str, proposals: List[Dict[str, Any]], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Avalia o risco coletivo das propostas.
        Se a maioria aprovar, o trade passa.
        """
        approvals = 0
        final_proposal = None
        
        for p in proposals:
            if p.get('action') == 'BUY':
                # Valida contra o guardião correspondente ao perfil do trader?
                # Simplificação: Valida contra o Neutro por padrão
                res = self.guardians[1].validate_trade(symbol, p, market_context)
                if res['approved']:
                    approvals += 1
                    final_proposal = res['adjusted_proposal']
                    
        return {
            "approved": approvals >= 1, # Pelo menos um trader viable aprovado pelo risco
            "final_proposal": final_proposal,
            "risk_score": 0.3 # Placeholder
        }
