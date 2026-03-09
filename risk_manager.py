"""
Risk Manager v6.0 - XP3 PRO QUANT-REFORM
Centraliza toda a gestão de risco e cálculo de posição
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

import config
import utils
import database
from adaptive_integration import get_adaptive_kelly_multiplier

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Resultado do cálculo de Kelly"""

    fraction: float
    position_size: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expected_value: float
    ruin_probability: float


class RiskManager:
    """Gerenciador de risco com Kelly Fracionado Inteligente"""

    def __init__(self):
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.symbol_losses = {}
        self.last_reset_date = datetime.now().date()

    def reset_daily_limits(self):
        """Reseta limites diários"""
        self.daily_pnl = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()
        logger.info("🔄 Limites diários resetados")

    def is_daily_loss_exceeded(self) -> bool:
        """Verifica se perda diária foi excedida"""
        if datetime.now().date() != self.last_reset_date:
            self.reset_daily_limits()

        return abs(self.daily_pnl) >= config.MAX_DAILY_LOSS_MONEY

    def update_daily_pnl(self, pnl: float):
        """Atualiza PnL diário"""
        self.daily_pnl += pnl
        self.daily_trades.append({"timestamp": datetime.now(), "pnl": pnl})

    def get_symbol_loss_count(self, symbol: str) -> int:
        """Retorna número de losses consecutivos no símbolo"""
        return self.symbol_losses.get(symbol, 0)

    def update_symbol_loss(self, symbol: str, is_loss: bool):
        """Atualiza contador de losses do símbolo"""
        if is_loss:
            self.symbol_losses[symbol] = self.symbol_losses.get(symbol, 0) + 1
        else:
            self.symbol_losses[symbol] = 0

    def is_anti_chop_active(self, symbol: str) -> bool:
        """Verifica se filtro anti-chop está ativo para o símbolo"""
        # No modo agressivo, o anti-chop só é ativado após o limite de perdas do dia
        if config.is_aggressive_mode():
            consecutive_losses = self.get_symbol_loss_count(symbol)
            max_losses = (
                config.get_max_losses_per_symbol()
            )  # Já retorna 4 no modo agressivo
            if consecutive_losses >= max_losses:
                logger.warning(
                    f"🚫 AGGRESSIVE: Anti-Chop ativado para {symbol} após {consecutive_losses} perdas"
                )
                return True
            return False

        # Modo Padrão
        consecutive_losses = self.get_symbol_loss_count(symbol)
        if consecutive_losses >= config.ANTI_CHOP_CONSECUTIVE_LOSSES:
            logger.warning(
                f"🚫 Anti-Chop ativo para {symbol}: {consecutive_losses} losses"
            )
            return True

        return False

    def calculate_kelly_fraction(
        self,
        symbol: str,
        side: str,
        win_rate_7d: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
    ) -> KellyResult:
        """
        Calcula fração de Kelly com ajustes inteligentes

        Args:
            symbol: Símbolo do ativo
            side: BUY ou SELL
            win_rate_7d: Winrate dos últimos 7 dias
            avg_win: Média de ganhos (%)
            avg_loss: Média de perdas (%)

        Returns:
            KellyResult com fração ajustada e metadados
        """
        # Base: fração de Kelly clássica
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            kelly_fraction = (
                win_rate_7d * avg_win - (1 - win_rate_7d) * avg_loss
            ) / avg_loss

        # Aplica fração base (Regra #3)
        adjusted_fraction = kelly_fraction * config.KELLY_BASE

        # Aplica multiplicador adaptativo
        adaptive_multiplier = get_adaptive_kelly_multiplier()
        adjusted_fraction *= adaptive_multiplier

        # Ajusta baseado no modo agressivo
        if config.is_aggressive_mode():
            adjusted_fraction *= 1.29  # 0.45 / 0.35 = ~1.29

        # Bônus por winrate alto
        if win_rate_7d > 0.62:
            adjusted_fraction += config.KELLY_WINRATE_BOOST

        # Penalidade por streak de perdas
        symbol_losses = self.get_symbol_loss_count(symbol)
        if symbol_losses >= 2:
            adjusted_fraction += config.KELLY_LOSS_STREAK_PENALTY

        logger.info(
            f"📊 Kelly ajustes: Base={config.KELLY_BASE:.2f}, Adaptive={adaptive_multiplier:.2f}, Final={adjusted_fraction:.3f}"
        )

        # Limita fração (proteção contra Kelly > 1.0)
        max_fraction = 0.25  # Máximo 25% do capital por trade
        final_fraction = max(0.0, min(adjusted_fraction, max_fraction))

        # Calcula probabilidade de ruína (simulação Monte Carlo simplificada)
        ruin_prob = self._calculate_ruin_probability(
            win_rate_7d, avg_win, avg_loss, final_fraction
        )

        # Calcula valor esperado
        expected_value = win_rate_7d * avg_win - (1 - win_rate_7d) * avg_loss

        result = KellyResult(
            fraction=final_fraction,
            position_size=0.0,  # Será calculado posteriormente
            win_rate=win_rate_7d,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expected_value=expected_value,
            ruin_probability=ruin_prob,
        )

        logger.info(
            f"📊 Kelly para {symbol} {side}: {final_fraction:.3f} "
            f"(WR: {win_rate_7d:.1%}, EV: {expected_value:.2%}, "
            f"Ruin: {ruin_prob:.2%})"
        )

        return result

    def _calculate_ruin_probability(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float,
        simulations: int = 10000,
    ) -> float:
        """Calcula probabilidade de ruína via Monte Carlo"""
        initial_capital = 100000  # Capital simulado
        ruin_threshold = initial_capital * 0.2  # Ruína = perder 80%

        ruin_count = 0

        for _ in range(simulations):
            capital = initial_capital

            for _ in range(252):  # 1 ano de trading (dias úteis)
                if np.random.random() < win_rate:
                    capital += capital * kelly_fraction * avg_win
                else:
                    capital -= capital * kelly_fraction * avg_loss

                if capital <= ruin_threshold:
                    ruin_count += 1
                    break

        ruin_probability = ruin_count / simulations

        # Alerta se probabilidade for alta
        if ruin_probability > config.MAX_RUIN_PROBABILITY:
            logger.warning(f"⚠️ Probabilidade de ruína alta: {ruin_probability:.2%}")

        return ruin_probability

    def calculate_position_size(
        self,
        symbol: str,
        side: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """
        Calcula tamanho da posição baseado em Risco Financeiro Fixo + Ajuste de Kelly.
        Garante que a perda máxima no Stop Loss não ultrapasse 1-2% do capital da conta (Priority 4).

        Args:
            symbol: Ticker do ativo
            side: BUY ou SELL
            account_balance: Saldo total da conta/patrimônio
            entry_price: Preço de entrada projetado
            stop_loss_price: Preço de stop-loss estabelecido
            
        Returns:
            Tamanho da posição em unidades (lot_size)
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.error(f"❌ Preços inválidos para {symbol}: Entrada={entry_price}, SL={stop_loss_price}")
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.error(f"❌ Stop loss inválido (igual à entrada) para {symbol}")
            return 0.0

        # Risco base: MAX_CAPITAL_ALLOCATION_PCT define o MÁXIMO a perder num trade
        # ex: 0.02 = 2% do capital em risco
        max_risk_pct = getattr(config, 'MAX_CAPITAL_ALLOCATION_PCT', 0.02)
        base_risk_amount = account_balance * max_risk_pct

        # Obtém estatísticas e ajusta via Kelly para dosar a agressividade (0.0 a 1.0)
        # Kelly muito ruim = opera pequeno; Kelly ótimo = usa até o risco máximo base
        stats = self._get_recent_stats(symbol, days=7)
        kelly_result = self.calculate_kelly_fraction(
            symbol,
            side,
            win_rate_7d=stats["win_rate"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
        )
        
        # O Kelly_fraction original era calculado como % de ALOCAÇÃO TOTAL (até 25%),
        # o que é extremamente perigoso para B3. Vamos escalar para ser um multiplicador [0.2, 1.0]
        # do nosso RISK_AMOUNT (que já está capeado em 2%).
        # Se max_fraction do kelly era 0.25, normalizamos: fraction / 0.25 -> range [0, 1]
        kelly_confidence_multiplier = min(1.0, max(0.2, kelly_result.fraction / 0.25))

        # Calcula o risco final em Reais
        final_risk_amount = base_risk_amount * kelly_confidence_multiplier

        # Calcula número bruto de ações para arriscar 'final_risk_amount'
        units = final_risk_amount / risk_per_unit

        # Arredonda para lotes padrão da B3 (100 ações, ou fracionário se necessário)
        # Por regra do sistema xp3, operamos em lotes de 100
        lot_size = int(units // 100) * 100

        # --- MANDATORY SAFETY CAP: 10% of equity (Priority 1) ---
        # Kelly may hallucinate, but the bankroll must be protected.
        max_position_value = account_balance * 0.10
        current_value = lot_size * entry_price
        
        if current_value > max_position_value:
            new_lot_size = int((max_position_value / entry_price) // 100) * 100
            logger.warning(
                f"🛡️ [RISK CAP] Posição em {symbol} reduzida de {lot_size} para {new_lot_size} "
                f"para respeitar o teto de 10% da conta (R${max_position_value:.2f})."
            )
            lot_size = new_lot_size
        # -------------------------------------------------------

        # Segurança final extra: A posição total (financeiro total investido)
        # não deve ultrapassar 4x o capital para daytrade alavancado normal.
        exposure_limit_money = account_balance * 4.0
        max_lot_exposure = int(exposure_limit_money / entry_price)

        final_lot_size = max(0, min(lot_size, max_lot_exposure))

        logger.info(
            f"📏 Sizing dinâmico {symbol} {side}: {final_lot_size} ações "
            f"(Distância SL: R${risk_per_unit:.2f}, Risco Trade: R${final_risk_amount:.2f} max 2%, "
            f"Kelly Conf:{kelly_confidence_multiplier:.0%})"
        )

        return float(final_lot_size)

    def _get_recent_stats(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """Obtém estatísticas recentes do banco de dados"""
        try:
            # Busca trades dos últimos dias
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            trades = []
            if hasattr(database, "get_trades_by_symbol_and_date"):
                trades = database.get_trades_by_symbol_and_date(
                    symbol, start_date, end_date
                )
            else:
                logger.debug("⚠️ database.get_trades_by_symbol_and_date não disponível. Usando fallback.")

            if not trades:
                # Retorna valores padrão conservadores
                return {"win_rate": 0.55, "avg_win": 0.015, "avg_loss": 0.012}

            # Calcula estatísticas
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]

            win_rate = len(wins) / len(trades) if trades else 0.55
            avg_win = np.mean(wins) if wins else 0.015
            avg_loss = abs(np.mean(losses)) if losses else 0.012

            return {
                "win_rate": max(0.1, min(0.9, win_rate)),  # Limita entre 10% e 90%
                "avg_win": max(0.005, avg_win),  # Mínimo 0.5%
                "avg_loss": max(0.005, avg_loss),  # Mínimo 0.5%
            }

        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas para {symbol}: {e}")
            return {"win_rate": 0.55, "avg_win": 0.015, "avg_loss": 0.012}

    def is_safety_mode_active(self) -> bool:
        """Verifica se modo de segurança está ativo"""
        try:
            # Verifica drawdown do IBOV
            ibov_drawdown = utils.get_ibov_intraday_drawdown()
            if ibov_drawdown and ibov_drawdown >= config.SAFETY_MARKET_DRAWDOWN_PCT:
                logger.warning(f"🚨 Safety Mode: Drawdown IBOV {ibov_drawdown:.1f}%")
                return True

            # Verifica VIX Brasil
            vix_br = utils.get_vix_brasil()
            if vix_br and vix_br >= config.SAFETY_VIX_BR_TRIGGER:
                logger.warning(f"🚨 Safety Mode: VIX Brasil {vix_br:.1f}")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Erro ao verificar safety mode: {e}")
            return False


# Instância global
risk_manager = RiskManager()
