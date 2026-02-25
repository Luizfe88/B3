
import logging
from typing import Dict, Any, List

logger = logging.getLogger("RiskManagementTeam")

import config

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
        max_position_size = 1.5 # 150% do lote base (flexibilidade)
        proposed_size = proposal.get('size_multiplier', 0.0)
        
        if proposed_size > max_position_size:
            logger.warning(f"❌ [{self.name}] Tamanho excessivo ({proposed_size:.2%}). Ajustando.")
            proposal['size_multiplier'] = max_position_size
            proposal['adjusted'] = True
            
        # 1. Limite Global de Exposição Financeira
        total_exposure = market_context.get('total_exposure', 0.0)
        equity = market_context.get('equity', 1000.0)
        max_exposure = equity * config.MAX_TOTAL_EXPOSURE_PCT
        
        # Estima exposição da nova ordem
        current_price = market_context.get('price', 0.0)
        new_exposure = (equity * config.MAX_CAPITAL_ALLOCATION_PCT * proposed_size)
        
        if (total_exposure + new_exposure) > max_exposure:
             logger.warning(f"❌ [{self.name}] Limite Global de Exposição atingido! "
                            f"(Atual: R${total_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_exposure:.2f} | Equity: R${equity:.2f})")
             return {"approved": False, "reason": f"Exposure Limit (Eq: {equity:.0f})"}

        # 2. Throttle (Limite de novas posições por hora)
        recent_entries = market_context.get('recent_entries_count', 0)
        if recent_entries >= config.MAX_NEW_POSITIONS_PER_HOUR:
             logger.warning(f"❌ [{self.name}] Throttle ativado! ({recent_entries} novas posições na última hora)")
             return {"approved": False, "reason": "Entry Throttle Active"}

        # 3. Limite de Exposição Setorial (25% do Capital)
        sector = config.SECTOR_MAP.get(symbol, "OUTROS")
        current_sector_exposure = market_context.get(f'sector_exposure_{sector}', 0.0)
        max_sector_exposure = equity * config.MAX_SECTOR_ALLOCATION_PCT
        
        if (current_sector_exposure + new_exposure) > max_sector_exposure:
             logger.warning(f"❌ [{self.name}] Limite de Setor ({sector}) atingido! "
                            f"(Atual: R${current_sector_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_sector_exposure:.2f})")
             return {"approved": False, "reason": f"Sector Limit ({sector})"}

        # 4. Market Regime Guard (Filtro de Pânico)
        if config.MARKET_REGIME_FILTER:
            ibov_trend = market_context.get('ibov_trend', 'neutral')
            if ibov_trend == 'bearish_extreme' and proposal.get('action') == 'BUY':
                 logger.warning(f"⚠️ [{self.name}] Market Regime Guard: Bloqueando COMPRA em pânico.")
                 return {"approved": False, "reason": "Market Panic Mode"}

        # 4. Verificação de correlação com IBOV
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
            if p.get('action') in ['BUY', 'SELL']:
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
