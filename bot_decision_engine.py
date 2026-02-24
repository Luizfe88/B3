"""
Bot Decision Flow v6.0 - XP3 PRO QUANT-REFORM
Novo fluxo de decisão com apenas 4 filtros obrigatórios
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import config
import utils
from risk_manager import risk_manager
from ml_signals_v6 import ml_signal_generator
from rejection_logger import rejection_logger
from adaptive_integration import (
    get_adaptive_ml_threshold,
    get_adaptive_spread_multiplier,
)

logger = logging.getLogger(__name__)


class BotDecisionEngine:
    """Motor de decisão otimizado para XP3 PRO"""

    def __init__(self):
        self.decision_cache = {}
        self.cache_timeout = 60  # segundos

    def should_enter_position(
        self, symbol: str, side: str, ind_data: Dict, account_balance: float
    ) -> Tuple[bool, str, Dict]:
        """
        Novo fluxo de decisão com 4 filtros obrigatórios

        Returns:
            (should_enter, reason, info)
        """

        start_time = datetime.now()
        info = {"decision_time": start_time}

        # 1. Verifica Panic Mode (mantém do sistema anterior)
        is_panic = self._check_panic_mode(ind_data)
        info["panic_mode"] = is_panic

        if is_panic:
            logger.info(f"🚨 PANIC MODE ativado para {symbol} - bypassando filtros")
            return True, "Panic Mode activated", info

        # 2. Filtro 1: News Blackout
        if self._is_news_blackout():
            rejection_logger.log_rejection(
                symbol,
                side,
                score=0.6,  # Score alto para ser registrado
                rejection_reason="News blackout active",
                entry_price=ind_data.get("close", 0),
                indicators=ind_data,
            )
            return False, "News blackout active", info

        # 3. Filtro 2: Spread Check
        if not self._check_spread(symbol):
            rejection_logger.log_rejection(
                symbol,
                side,
                score=0.6,
                rejection_reason="Spread too high",
                entry_price=ind_data.get("close", 0),
                indicators=ind_data,
            )
            return False, "Spread too high", info

        # 4. Filtro 3: Daily Loss Money
        if risk_manager.is_daily_loss_exceeded():
            rejection_logger.log_rejection(
                symbol,
                side,
                score=0.6,
                rejection_reason="Daily loss limit exceeded",
                entry_price=ind_data.get("close", 0),
                indicators=ind_data,
            )
            return False, "Daily loss limit exceeded", info

        # 5. Filtro 4: Anti-Chop (só após 2+ losses consecutivos)
        if risk_manager.is_anti_chop_active(symbol):
            rejection_logger.log_rejection(
                symbol,
                side,
                score=0.6,
                rejection_reason="Anti-chop filter active",
                entry_price=ind_data.get("close", 0),
                indicators=ind_data,
            )
            return False, "Anti-chop filter active", info

        # 6. Sinal de ML (com threshold adaptativo)
        adaptive_threshold = get_adaptive_ml_threshold()
        ml_enter, ml_confidence, ml_info = (
            ml_signal_generator.get_signal_with_threshold(
                symbol, side, custom_threshold=adaptive_threshold
            )
        )

        info["ml_confidence"] = ml_confidence
        info["ml_threshold"] = adaptive_threshold
        info["ml_threshold_adaptive"] = True

        if not ml_enter:
            rejection_logger.log_rejection(
                symbol,
                side,
                score=ml_confidence,
                rejection_reason=f"ML confidence too low ({ml_confidence:.1%} < {adaptive_threshold:.1%}) [ADAPTIVE]",
                entry_price=ind_data.get("close", 0),
                indicators=ind_data,
            )
            return (
                False,
                f"ML confidence too low ({ml_confidence:.1%}) [ADAPTIVE]",
                info,
            )

        # 7. Verificação final de risco
        if risk_manager.is_safety_mode_active():
            # Em safety mode, aumenta threshold (usa threshold adaptativo + bônus)
            safety_threshold = get_adaptive_ml_threshold() + 0.08  # +8% no safety mode
            if ml_confidence < safety_threshold:
                rejection_logger.log_rejection(
                    symbol,
                    side,
                    score=ml_confidence,
                    rejection_reason=f"Safety mode: confidence too low ({ml_confidence:.1%} < {safety_threshold:.1%})",
                    entry_price=ind_data.get("close", 0),
                    indicators=ind_data,
                )
                return False, f"Safety mode: confidence too low", info

        # Tempo de decisão
        decision_time = (datetime.now() - start_time).total_seconds()
        info["decision_latency"] = decision_time

        logger.info(
            f"✅ {symbol} {side} APROVADO - Confidence: {ml_confidence:.1%} "
            f"(Latência: {decision_time:.2f}s)"
        )

        return True, "All filters passed", info

    def _check_panic_mode(self, ind_data: Dict) -> bool:
        """Verifica condições de Panic Mode"""
        volume_ratio = ind_data.get("volume_ratio", 1.0)
        adx = ind_data.get("adx", 0)

        return volume_ratio > 2.0 and adx > 40

    def _is_news_blackout(self) -> bool:
        """Verifica se está em período de blackout de notícias"""
        try:
            # Verifica se há notícias nos próximos 15 minutos
            upcoming_news = utils.get_upcoming_news(minutes=15)
            return len(upcoming_news) > 0
        except:
            return False

    def _check_spread(self, symbol: str) -> bool:
        """Verifica spread atual vs média"""
        try:
            ok_spread, cur_spread, avg_spread = utils.check_spread(
                symbol, 15, lookback_bars=20
            )

            # Em modo agressivo, tolerância maior (com ajuste adaptativo)
            if config.is_aggressive_mode():
                base_multiplier = 3.0  # Mais tolerante
            else:
                base_multiplier = 2.5

            # Aplica multiplicador adaptativo
            adaptive_multiplier = get_adaptive_spread_multiplier()
            max_multiplier = base_multiplier * adaptive_multiplier

            return ok_spread and cur_spread <= avg_spread * max_multiplier

        except:
            return False

    def calculate_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        account_balance: float,
    ) -> float:
        """Calcula tamanho da posição com Kelly Fracionado"""

        # Usa o Risk Manager para calcular posição
        lot_size = risk_manager.calculate_position_size(
            symbol, side, account_balance, entry_price, stop_loss_price
        )

        # Em Panic Mode, reduz tamanho
        if self._check_panic_mode({"volume_ratio": 2.5, "adx": 45}):
            lot_size *= 0.6  # Reduz 40%
            logger.info(f"🚨 Panic Mode: Reduzindo posição para {lot_size:.0f} ações")

        return lot_size

    def get_filtered_symbols(self, all_symbols: List[str]) -> List[str]:
        """Filtra símbolos elegíveis (removido filtros antigos)"""

        filtered = []

        for symbol in all_symbols:
            try:
                # Verificações básicas apenas
                if not utils.is_symbol_active(symbol):
                    continue

                # Verifica se já atingiu limite de losses
                if (
                    risk_manager.get_symbol_loss_count(symbol)
                    >= config.get_max_losses_per_symbol()
                ):
                    continue

                filtered.append(symbol)

            except Exception as e:
                logger.error(f"❌ Erro ao filtrar {symbol}: {e}")
                continue

        return filtered

    def log_decision_metrics(self):
        """Loga métricas de decisão"""
        try:
            summary = rejection_logger.get_summary()

            logger.info(f"📊 Métricas de Decisão:")
            logger.info(f"  - Sinais rejeitados (score >58%): {summary['total']}")
            logger.info(f"  - PnL médio what-if: {summary['avg_pnl']:+.2%}")
            logger.info(f"  - Win rate what-if: {summary['win_rate']:.1%}")

            # Alerta se estamos rejeitando trades bons
            if summary["win_rate"] > 0.6 and summary["total"] > 10:
                logger.warning(f"⚠️ Alta taxa de rejeição de trades vencedores!")

        except Exception as e:
            logger.error(f"❌ Erro ao logar métricas: {e}")


# Instância global
decision_engine = BotDecisionEngine()
