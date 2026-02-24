"""
Adaptive Integration v1.0 - XP3 PRO AUTO-TUNING INTEGRATION
Integração entre inteligência adaptativa e sistema existente
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import config
import database
import utils
from adaptive_intelligence import adaptive_intelligence

logger = logging.getLogger(__name__)


class AdaptiveIntegration:
    """Integração entre Adaptive Intelligence e sistema XP3 PRO"""

    def __init__(self):
        self.integration_thread = None
        self.running = False
        self.last_sync = datetime.now()
        self.sync_interval = 300  # 5 minutos
        self.performance_cache = {}

    def start_integration(self):
        """Inicia a integração com o sistema"""
        if self.running:
            return

        self.running = True

        # Inicia Adaptive Intelligence
        adaptive_intelligence.start_monitoring()

        # Inicia thread de integração
        self.integration_thread = threading.Thread(
            target=self._integration_loop, daemon=True
        )
        self.integration_thread.start()

        logger.info("🔗 Adaptive Integration iniciada")

    def stop_integration(self):
        """Para a integração"""
        self.running = False

        # Para Adaptive Intelligence
        adaptive_intelligence.stop_monitoring()

        if self.integration_thread:
            self.integration_thread.join(timeout=10)

        logger.info("🔗 Adaptive Integration finalizada")

    def _integration_loop(self):
        """Loop principal de integração"""
        while self.running:
            try:
                # Sincroniza parâmetros com sistema
                self._sync_parameters()

                # Atualiza performance cache
                self._update_performance_cache()

                # Verifica necessidade de ajustes emergenciais
                self._check_emergency_adjustments()

                self.last_sync = datetime.now()

            except Exception as e:
                logger.error(f"Erro no loop de integração: {e}")

            time.sleep(self.sync_interval)

    def _sync_parameters(self):
        """Sincroniza parâmetros adaptativos com o sistema"""
        try:
            current_params = adaptive_intelligence.get_current_parameters()

            # Atualiza configurações dinâmicas
            config.ADAPTIVE_ML_THRESHOLD = current_params.ml_confidence_threshold
            config.ADAPTIVE_KELLY_MULTIPLIER = current_params.kelly_fraction_multiplier
            config.ADAPTIVE_SPREAD_MULTIPLIER = current_params.spread_filter_multiplier

            # Atualiza no banco de dados
            database.save_adaptive_config(
                {
                    "timestamp": datetime.now(),
                    "ml_threshold": current_params.ml_confidence_threshold,
                    "kelly_multiplier": current_params.kelly_fraction_multiplier,
                    "spread_multiplier": current_params.spread_filter_multiplier,
                    "max_losses_per_symbol": current_params.max_losses_per_symbol,
                }
            )

            logger.debug(
                f"🔄 Parâmetros sincronizados: ML={current_params.ml_confidence_threshold:.3f}, Kelly={current_params.kelly_fraction_multiplier:.2f}"
            )

        except Exception as e:
            logger.error(f"Erro ao sincronizar parâmetros: {e}")

    def _update_performance_cache(self):
        """Atualiza cache de performance"""
        try:
            report = adaptive_intelligence.get_performance_report()
            self.performance_cache = {
                "last_update": datetime.now(),
                "winrate_24h": report.get("performance_metrics", {}).get(
                    "avg_winrate_24h", 0.5
                ),
                "sharpe_4h": report.get("performance_metrics", {}).get(
                    "avg_sharpe_4h", 0.0
                ),
                "volatility": report.get("performance_metrics", {}).get(
                    "avg_volatility", 0.02
                ),
                "market_state": report.get("market_state", {}),
                "total_adjustments": report.get("total_adjustments", 0),
            }

        except Exception as e:
            logger.error(f"Erro ao atualizar cache de performance: {e}")

    def _check_emergency_adjustments(self):
        """Verifica e aplica ajustes emergenciais"""
        try:
            # Verifica performance crítica
            if (
                self.performance_cache.get("winrate_24h", 0.5) < 0.4
            ):  # Winrate muito baixo
                logger.warning(
                    f"🚨 Performance crítica detectada: Winrate={self.performance_cache['winrate_24h']:.1%}"
                )
                self._apply_emergency_adjustment("low_performance")

            # Verifica volatilidade extrema
            if (
                self.performance_cache.get("volatility", 0.02) > 0.05
            ):  # Volatilidade muito alta
                logger.warning(
                    f"🚨 Volatilidade extrema detectada: {self.performance_cache['volatility']:.3f}"
                )
                self._apply_emergency_adjustment("high_volatility")

            # Verifica drawdown diário
            daily_pnl = self._calculate_daily_pnl()
            max_daily_loss = config.MAX_DAILY_LOSS_MONEY * 0.8  # 80% do limite
            if daily_pnl < -max_daily_loss:
                logger.warning(f"🚨 Drawdown diário crítico: R${daily_pnl:,.2f}")
                self._apply_emergency_adjustment("high_drawdown")

        except Exception as e:
            logger.error(f"Erro ao verificar ajustes emergenciais: {e}")

    def _apply_emergency_adjustment(self, situation: str):
        """Aplica ajuste emergencial"""
        try:
            current_params = adaptive_intelligence.get_current_parameters()

            if situation == "low_performance":
                # Reduz exposição e aumenta filtros
                current_params.ml_confidence_threshold = min(
                    0.75, current_params.ml_confidence_threshold + 0.05
                )
                current_params.kelly_fraction_multiplier *= 0.5
                current_params.max_losses_per_symbol = max(
                    1, current_params.max_losses_per_symbol - 1
                )

            elif situation == "high_volatility":
                # Aumenta SL e reduz posições
                current_params.kelly_fraction_multiplier *= 0.7
                current_params.spread_filter_multiplier *= 1.5

            elif situation == "high_drawdown":
                # Modo conservador extremo
                current_params.ml_confidence_threshold = 0.70
                current_params.kelly_fraction_multiplier = 0.2
                current_params.max_losses_per_symbol = 1

            logger.info(f"🚨 Ajuste emergencial aplicado: {situation}")

        except Exception as e:
            logger.error(f"Erro ao aplicar ajuste emergencial: {e}")

    def _calculate_daily_pnl(self) -> float:
        """Calcula P&L diário"""
        try:
            today_start = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            daily_trades = database.get_trades_since(today_start)
            return sum(float(t.get("net_pnl", 0)) for t in daily_trades)

        except Exception as e:
            logger.error(f"Erro ao calcular P&L diário: {e}")
            return 0.0

    def get_adaptive_parameters(self) -> Dict:
        """Retorna parâmetros adaptativos atuais"""
        try:
            current_params = adaptive_intelligence.get_current_parameters()

            return {
                "ml_confidence_threshold": current_params.ml_confidence_threshold,
                "kelly_fraction_multiplier": current_params.kelly_fraction_multiplier,
                "spread_filter_multiplier": current_params.spread_filter_multiplier,
                "max_losses_per_symbol": current_params.max_losses_per_symbol,
                "panic_volume_threshold": current_params.panic_volume_threshold,
                "panic_adx_threshold": current_params.panic_adx_threshold,
                "last_update": self.last_sync.isoformat(),
                "performance_metrics": self.performance_cache,
            }

        except Exception as e:
            logger.error(f"Erro ao obter parâmetros adaptativos: {e}")
            return {}

    def get_adaptive_status(self) -> Dict:
        """Retorna status completo do sistema adaptativo"""
        try:
            ai_report = adaptive_intelligence.get_performance_report()
            integration_status = {
                "running": self.running,
                "last_sync": self.last_sync.isoformat(),
                "sync_interval": self.sync_interval,
                "adaptive_intelligence": ai_report,
                "current_parameters": self.get_adaptive_parameters(),
            }

            return integration_status

        except Exception as e:
            logger.error(f"Erro ao obter status adaptativo: {e}")
            return {"error": str(e)}

    def force_analysis(self) -> Dict:
        """Força análise e ajuste imediato"""
        try:
            logger.info("🔍 Forçando análise adaptativa...")

            # Força coleta de métricas
            metrics = adaptive_intelligence._collect_current_metrics()
            if metrics:
                adaptive_intelligence.metrics_history.append(metrics)

            # Força análise e ajuste
            adaptive_intelligence._analyze_and_adjust()

            # Sincroniza parâmetros
            self._sync_parameters()

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "parameters_updated": True,
                "current_params": self.get_adaptive_parameters(),
            }

        except Exception as e:
            logger.error(f"Erro ao forçar análise: {e}")
            return {"status": "error", "error": str(e)}


# Instância global
adaptive_integration = AdaptiveIntegration()


# Funções auxiliares para integração com bot.py
def get_adaptive_ml_threshold() -> float:
    """Retorna threshold ML adaptativo"""
    try:
        return adaptive_integration.get_adaptive_parameters().get(
            "ml_confidence_threshold", config.ML_CONFIDENCE_BASE
        )
    except:
        return config.ML_CONFIDENCE_BASE


def get_adaptive_kelly_multiplier() -> float:
    """Retorna multiplicador Kelly adaptativo"""
    try:
        return adaptive_integration.get_adaptive_parameters().get(
            "kelly_fraction_multiplier", 1.0
        )
    except:
        return 1.0


def get_adaptive_spread_multiplier() -> float:
    """Retorna multiplicador de spread adaptativo"""
    try:
        return adaptive_integration.get_adaptive_parameters().get(
            "spread_filter_multiplier", 1.0
        )
    except:
        return 1.0


def start_adaptive_system():
    """Inicia sistema adaptativo completo"""
    adaptive_integration.start_integration()


def stop_adaptive_system():
    """Para sistema adaptativo"""
    adaptive_integration.stop_integration()


def get_adaptive_status() -> Dict:
    """Retorna status completo do sistema adaptativo"""
    return adaptive_integration.get_adaptive_status()
