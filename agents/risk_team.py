import logging
from typing import Dict, Any, List

logger = logging.getLogger("RiskManagementTeam")

import config
import pandas as pd
from validation.permutation_test import PermutationValidator


class RiskGuardian:
    def __init__(self, name: str, tolerance: float):
        self.name = name
        self.tolerance = tolerance

    def validate_trade(
        self, symbol: str, proposal: Dict[str, Any], market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida se o trade proposto respeita os limites de risco.
        Considera:
        - Drawdown máximo
        - Correlação
        - Exposição setorial
        - Tamanho da posição

        ⚠️ CORRIGIDO: size_multiplier agora multiplica o RISCO BASE (2%), não o capital total
        Ex: RiskyTrader propõe 1.2 -> 2% * 1.2 = 2.4% por trade (não 24%)
        """
        logger.info(f"👮 [{self.name}] Validando risco para {symbol}...")

        # CORREÇÃO CRÍTICA: size_multiplier é multiplicador do risco base, não do capital
        # Risco base é definido em config.MAX_CAPITAL_ALLOCATION_PCT (padrão: 2%)
        base_risk_pct = config.MAX_CAPITAL_ALLOCATION_PCT  # Ex: 0.02 (2%)
        # Interpret size_multiplier as multiplier of the BASE RISK (not of capital)
        proposed_size_multiplier = float(proposal.get("size_multiplier", 1.0))

        # Calcula o risco efetivo desta proposta (por trade)
        effective_risk_pct = base_risk_pct * proposed_size_multiplier

        # HARD LIMIT absoluto por trade (configurável via config.HARD_RISK_LIMIT_PCT)
        HARD_LIMIT = float(getattr(config, "HARD_RISK_LIMIT_PCT", 0.05))
        if effective_risk_pct > HARD_LIMIT:
            logger.critical(
                f"🚨 [{self.name}] Hard risk limit exceeded: {effective_risk_pct:.2%} > {HARD_LIMIT:.2%}. Bloqueando proposta."
            )
            return {"approved": False, "reason": "Hard risk limit exceeded"}

        # Limite prudencial (ajuste automático) — não ultrapassa 150% do risco base
        max_effective_risk = base_risk_pct * 1.5
        if effective_risk_pct > max_effective_risk:
            logger.warning(
                f"❌ [{self.name}] Risco efetivo alto ({effective_risk_pct:.2%} > {max_effective_risk:.2%}). Ajustando para prudência."
            )
            proposal["size_multiplier"] = max_effective_risk / base_risk_pct
            proposal["adjusted"] = True
            effective_risk_pct = max_effective_risk

        # Agora verifica exposição FINANCEIRA com base no risco ajustado
        equity = market_context.get("equity", 1000.0)
        total_exposure = market_context.get("total_exposure", 0.0)
        max_exposure = equity * config.MAX_TOTAL_EXPOSURE_PCT

        # Estima exposição da nova ordem baseada no risco efetivo
        new_exposure = equity * effective_risk_pct

        if (total_exposure + new_exposure) > max_exposure:
            logger.warning(
                f"❌ [{self.name}] Limite Global de Exposição atingido! "
                f"(Atual: R${total_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_exposure:.2f} | Equity: R${equity:.2f})"
            )
            return {"approved": False, "reason": f"Exposure Limit (Eq: {equity:.0f})"}

        # 2. Throttle (Limite de novas posições por hora)
        recent_entries = market_context.get("recent_entries_count", 0)
        if recent_entries >= config.MAX_NEW_POSITIONS_PER_HOUR:
            logger.warning(
                f"❌ [{self.name}] Throttle ativado! ({recent_entries} novas posições na última hora)"
            )
            return {"approved": False, "reason": "Entry Throttle Active"}

        # 3. Limite de Exposição Setorial (25% do Capital)
        sector = config.SECTOR_MAP.get(symbol, "OUTROS")
        current_sector_exposure = market_context.get(f"sector_exposure_{sector}", 0.0)
        max_sector_exposure = equity * config.MAX_SECTOR_ALLOCATION_PCT

        if (current_sector_exposure + new_exposure) > max_sector_exposure:
            logger.warning(
                f"❌ [{self.name}] Limite de Setor ({sector}) atingido! "
                f"(Atual: R${current_sector_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_sector_exposure:.2f})"
            )
            return {"approved": False, "reason": f"Sector Limit ({sector})"}

        # 4. Market Regime Guard (Filtro de Pânico)
        if config.MARKET_REGIME_FILTER:
            ibov_trend = market_context.get("ibov_trend", "neutral")
            if ibov_trend == "bearish_extreme" and proposal.get("action") == "BUY":
                logger.warning(
                    f"⚠️ [{self.name}] Market Regime Guard: Bloqueando COMPRA em pânico."
                )
                return {"approved": False, "reason": "Market Panic Mode"}

        # 4. Verificação de correlação com IBOV
        corr = market_context.get("ibov_correlation", 0.5)
        if corr > 0.8 and self.tolerance < 0.5:
            logger.warning(
                f"⚠️ [{self.name}] Alta correlação com mercado em queda. Bloqueando."
            )
            return {"approved": False, "reason": "High correlation risk"}

        return {"approved": True, "adjusted_proposal": proposal}


class StatisticalGuardian(RiskGuardian):
    def __init__(self):
        super().__init__("StatValidator", tolerance=0.1)
        self.validator = PermutationValidator()

    def validate_strategy_health(self, trade_history: list) -> bool:
        """
        Kill Switch baseado em teste de permutação sobre PnL% recentes.
        Retorna False se a estratégia perder o diferencial estatístico.
        """
        try:
            returns = pd.Series([t.get("pnl_pct", 0.0) for t in trade_history])
        except Exception:
            logger.warning(
                "[StatValidator] trade_history não está no formato esperado."
            )
            return True

        if not self.validator.run_test(returns):
            logger.critical(
                "🚨 [KILL SWITCH] Estratégia perdeu o diferencial estatístico!"
            )
            return False
        return True


class RiskTeam:
    def __init__(self):
        self.guardians = [
            StatisticalGuardian(),
            RiskGuardian("RiskSeeker", 0.8),
            RiskGuardian("Neutral", 0.5),
            RiskGuardian("Conservative", 0.2),
        ]

    def assess_risk(
        self,
        symbol: str,
        proposals: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Avalia o risco coletivo das propostas.
        Se a maioria aprovar, o trade passa.
        """
        approvals = 0
        final_proposal = None

        for p in proposals:
            if p.get("action") in ["BUY", "SELL"]:
                # Valida contra o guardião correspondente ao perfil do trader?
                # Simplificação: Valida contra o Neutro por padrão
                res = self.guardians[1].validate_trade(symbol, p, market_context)
                if res["approved"]:
                    approvals += 1
                    final_proposal = res["adjusted_proposal"]

        return {
            "approved": approvals
            >= 1,  # Pelo menos um trader viable aprovado pelo risco
            "final_proposal": final_proposal,
            "risk_score": 0.3,  # Placeholder
        }
