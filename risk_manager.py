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
        Calcula tamanho da posição baseado em Kelly

        Returns:
            Tamanho da posição em unidades (lot_size)
        """
        # Obtém estatísticas dos últimos 7 dias
        stats = self._get_recent_stats(symbol, days=7)

        # Calcula fração de Kelly
        kelly_result = self.calculate_kelly_fraction(
            symbol,
            side,
            win_rate_7d=stats["win_rate"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
        )

        # Calcula risco em dinheiro
        risk_amount = account_balance * kelly_result.fraction

        # Calcula risco por unidade (baseado no stop loss)
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit == 0:
            logger.error(f"❌ Stop loss inválido para {symbol}")
            return 0.0

        # Calcula número de unidades
        units = risk_amount / risk_per_unit

        # Arredonda para lotes padrão da B3 (100 ações)
        lot_size = max(100, int(units / 100) * 100)

        # Limita pelo capital disponível
        max_units = int(
            account_balance * 0.25 / entry_price
        )  # Máximo 25% em uma posição
        final_lot_size = min(lot_size, max_units)

        logger.info(
            f"📏 Posição {symbol} {side}: {final_lot_size} ações "
            f"(Kelly: {kelly_result.fraction:.2%}, Risco: R$ {risk_amount:.2f})"
        )

        return float(final_lot_size)

    def _get_recent_stats(self, symbol: str, days: int = 7) -> Dict[str, float]:
        """Obtém estatísticas recentes do banco de dados"""
        try:
            # Busca trades dos últimos dias
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            trades = database.get_trades_by_symbol_and_date(
                symbol, start_date, end_date
            )

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
