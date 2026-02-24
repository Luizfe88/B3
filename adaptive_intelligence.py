"""
Adaptive Intelligence v1.0 - XP3 PRO AUTO-TUNING
Sistema de auto-ajuste inteligente baseado em análise contínua de métricas
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json
import threading
import time

import config
import database
import utils

logger = logging.getLogger(__name__)


@dataclass
class MarketMetrics:
    """Métricas de mercado em tempo real"""

    timestamp: datetime
    winrate_1h: float
    winrate_4h: float
    winrate_24h: float
    sharpe_1h: float
    sharpe_4h: float
    sharpe_24h: float
    avg_trade_duration: float
    avg_sl_distance: float
    avg_tp_distance: float
    market_volatility: float
    volume_anomaly: float
    correlation_strength: float
    trend_strength: float


@dataclass
class AdaptiveParameters:
    """Parâmetros que podem ser ajustados dinamicamente"""

    ml_confidence_threshold: float
    kelly_fraction_multiplier: float
    max_losses_per_symbol: int
    spread_filter_multiplier: float
    anti_chop_threshold: int
    panic_volume_threshold: float
    panic_adx_threshold: float


class AdaptiveIntelligence:
    """Motor de inteligência adaptativa para auto-ajuste"""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)  # Últimas 1000 medições
        self.parameter_history = deque(maxlen=100)  # Últimos 100 ajustes
        self.current_params = self._load_default_params()
        self.adjustment_thread = None
        self.running = False
        self.last_adjustment = datetime.now()
        self.performance_window = deque(maxlen=48)  # Últimas 48 horas

    def _load_default_params(self) -> AdaptiveParameters:
        """Carrega parâmetros padrão do config"""
        cfg = config.get_config()
        return AdaptiveParameters(
            ml_confidence_threshold=cfg["ml_model"]["confidence_base"],
            kelly_fraction_multiplier=1.0,
            max_losses_per_symbol=cfg["risk_limits"]["max_losses_per_symbol_default"],
            spread_filter_multiplier=cfg["entry_filters"]["spread_filter"][
                "max_spread_multiplier"
            ],
            anti_chop_threshold=cfg["entry_filters"]["anti_chop_filter"][
                "consecutive_losses_to_activate"
            ],
            panic_volume_threshold=2.0,
            panic_adx_threshold=40.0,
        )

    def start_monitoring(self):
        """Inicia o monitoramento contínuo"""
        if self.running:
            return

        self.running = True
        self.adjustment_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.adjustment_thread.start()
        logger.info("🧠 Adaptive Intelligence iniciado")

    def stop_monitoring(self):
        """Para o monitoramento"""
        self.running = False
        if self.adjustment_thread:
            self.adjustment_thread.join(timeout=5)
        logger.info("🧠 Adaptive Intelligence finalizado")

    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Coleta métricas a cada 15 minutos
                metrics = self._collect_current_metrics()
                if metrics:
                    self.metrics_history.append(metrics)

                    # Analisa e ajusta a cada hora
                    if (datetime.now() - self.last_adjustment).total_seconds() >= 3600:
                        self._analyze_and_adjust()
                        self.last_adjustment = datetime.now()

            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")

            time.sleep(900)  # Espera 15 minutos

    def _collect_current_metrics(self) -> Optional[MarketMetrics]:
        """Coleta métricas atuais do mercado"""
        try:
            # Análise de trades recentes
            recent_trades = self._get_recent_trades(hours=24)
            if len(recent_trades) < 5:
                return None

            # Calcula métricas de performance
            winrate_1h = self._calculate_winrate(recent_trades, hours=1)
            winrate_4h = self._calculate_winrate(recent_trades, hours=4)
            winrate_24h = self._calculate_winrate(recent_trades, hours=24)

            sharpe_1h = self._calculate_sharpe(recent_trades, hours=1)
            sharpe_4h = self._calculate_sharpe(recent_trades, hours=4)
            sharpe_24h = self._calculate_sharpe(recent_trades, hours=24)

            # Análise de trade duration
            avg_duration = np.mean(
                [t.get("duration_minutes", 0) for t in recent_trades[-50:]]
            )

            # Análise de SL/TP
            sl_distances = []
            tp_distances = []
            for trade in recent_trades[-50:]:
                if "entry_price" in trade and "sl" in trade and "tp" in trade:
                    entry = trade["entry_price"]
                    sl_distances.append(abs(entry - trade["sl"]) / entry * 100)
                    tp_distances.append(abs(trade["tp"] - entry) / entry * 100)

            avg_sl = np.mean(sl_distances) if sl_distances else 1.0
            avg_tp = np.mean(tp_distances) if tp_distances else 2.0

            # Análise de volatilidade de mercado
            volatility = self._calculate_market_volatility()

            # Análise de volume
            volume_anomaly = self._detect_volume_anomalies()

            # Análise de correlação
            correlation = self._calculate_market_correlation()

            # Força da tendência
            trend_strength = self._calculate_trend_strength()

            return MarketMetrics(
                timestamp=datetime.now(),
                winrate_1h=winrate_1h,
                winrate_4h=winrate_4h,
                winrate_24h=winrate_24h,
                sharpe_1h=sharpe_1h,
                sharpe_4h=sharpe_4h,
                sharpe_24h=sharpe_24h,
                avg_trade_duration=avg_duration,
                avg_sl_distance=avg_sl,
                avg_tp_distance=avg_tp,
                market_volatility=volatility,
                volume_anomaly=volume_anomaly,
                correlation_strength=correlation,
                trend_strength=trend_strength,
            )

        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {e}")
            return None

    def _analyze_and_adjust(self):
        """Analisa métricas e ajusta parâmetros"""
        try:
            if len(self.metrics_history) < 10:
                logger.info("📊 Métricas insuficientes para análise")
                return

            # Converte para DataFrame para análise
            df = pd.DataFrame([vars(m) for m in self.metrics_history])

            # Detecta tendências e padrões
            recommendations = self._generate_recommendations(df)

            # Aplica ajustes
            if recommendations:
                self._apply_adjustments(recommendations)

        except Exception as e:
            logger.error(f"Erro na análise e ajuste: {e}")

    def _generate_recommendations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Gera recomendações baseadas nas métricas"""
        recommendations = {}

        # Análise de tendência do winrate
        recent_wr = df["winrate_4h"].tail(12).mean()  # Últimas 3 horas
        historical_wr = df["winrate_24h"].mean()

        if recent_wr < historical_wr * 0.8:  # Winrate caiu 20%
            recommendations["increase_confidence_threshold"] = 0.05
            recommendations["reduce_kelly_multiplier"] = 0.8
            logger.info(f"📉 Winrate em queda: {recent_wr:.1%} vs {historical_wr:.1%}")

        elif recent_wr > historical_wr * 1.2:  # Winrate aumentou 20%
            recommendations["decrease_confidence_threshold"] = -0.03
            recommendations["increase_kelly_multiplier"] = 1.2
            logger.info(f"📈 Winrate em alta: {recent_wr:.1%} vs {historical_wr:.1%}")

        # Análise de volatilidade
        current_vol = df["market_volatility"].iloc[-1]
        avg_vol = df["market_volatility"].mean()

        if current_vol > avg_vol * 1.5:  # Alta volatilidade
            recommendations["increase_sl_distance"] = 1.3
            recommendations["reduce_kelly_multiplier"] = 0.7
            recommendations["increase_spread_filter"] = 1.5
            logger.info(f"⚡ Alta volatilidade detectada: {current_vol:.2f}")

        elif current_vol < avg_vol * 0.7:  # Baixa volatilidade
            recommendations["decrease_sl_distance"] = 0.8
            recommendations["increase_kelly_multiplier"] = 1.1
            logger.info(f"😴 Baixa volatilidade detectada: {current_vol:.2f}")

        # Análise de Sharpe
        current_sharpe = df["sharpe_4h"].iloc[-1]
        if current_sharpe < 0.5:
            recommendations["reduce_position_sizes"] = 0.6
            recommendations["increase_confidence_threshold"] = 0.04
            logger.info(f"⚠️ Sharpe baixo: {current_sharpe:.2f}")

        elif current_sharpe > 2.0:
            recommendations["increase_position_sizes"] = 1.2
            recommendations["decrease_confidence_threshold"] = -0.02
            logger.info(f"🎯 Sharpe excelente: {current_sharpe:.2f}")

        # Análise de duração média dos trades
        avg_duration = df["avg_trade_duration"].iloc[-1]
        if avg_duration > 120:  # Trades muito longos
            recommendations["reduce_tp_distance"] = 0.8
            recommendations["increase_sl_distance"] = 1.1
            logger.info(f"⏱️ Trades longos detectados: {avg_duration:.0f}min")

        # Análise de anomalias de volume
        volume_anomaly = df["volume_anomaly"].iloc[-1]
        if volume_anomaly > 2.0:  # Volume anormalmente alto
            recommendations["increase_confidence_threshold"] = 0.06
            recommendations["reduce_kelly_multiplier"] = 0.8
            recommendations["clamp_sl_max_percent"] = (
                3.0  # Clamp SL to max 3% for stock trades
            )
            logger.info(f"📊 Anomalia de volume: {volume_anomaly:.1f}x")
            logger.info(f"🔒 Clamping SL to max 3% due to volume anomaly")

        return recommendations

    def _apply_adjustments(self, recommendations: Dict[str, float]):
        """Aplica os ajustes recomendados"""
        try:
            old_params = self.current_params
            new_params = AdaptiveParameters(
                ml_confidence_threshold=old_params.ml_confidence_threshold,
                kelly_fraction_multiplier=old_params.kelly_fraction_multiplier,
                max_losses_per_symbol=old_params.max_losses_per_symbol,
                spread_filter_multiplier=old_params.spread_filter_multiplier,
                anti_chop_threshold=old_params.anti_chop_threshold,
                panic_volume_threshold=old_params.panic_volume_threshold,
                panic_adx_threshold=old_params.panic_adx_threshold,
            )

            # Aplica ajustes
            if "increase_confidence_threshold" in recommendations:
                new_params.ml_confidence_threshold += recommendations[
                    "increase_confidence_threshold"
                ]
            if "decrease_confidence_threshold" in recommendations:
                new_params.ml_confidence_threshold += recommendations[
                    "decrease_confidence_threshold"
                ]

            if "increase_kelly_multiplier" in recommendations:
                new_params.kelly_fraction_multiplier *= recommendations[
                    "increase_kelly_multiplier"
                ]
            if "reduce_kelly_multiplier" in recommendations:
                new_params.kelly_fraction_multiplier *= recommendations[
                    "reduce_kelly_multiplier"
                ]

            if "reduce_position_sizes" in recommendations:
                new_params.kelly_fraction_multiplier *= recommendations[
                    "reduce_position_sizes"
                ]
            if "increase_position_sizes" in recommendations:
                new_params.kelly_fraction_multiplier *= recommendations[
                    "increase_position_sizes"
                ]

            if "increase_spread_filter" in recommendations:
                new_params.spread_filter_multiplier *= recommendations[
                    "increase_spread_filter"
                ]

            # Limita valores extremos
            new_params.ml_confidence_threshold = max(
                0.50, min(0.80, new_params.ml_confidence_threshold)
            )
            new_params.kelly_fraction_multiplier = max(
                0.3, min(2.0, new_params.kelly_fraction_multiplier)
            )
            new_params.spread_filter_multiplier = max(
                1.0, min(5.0, new_params.spread_filter_multiplier)
            )

            # Registra ajuste
            self.current_params = new_params
            self.parameter_history.append(
                {
                    "timestamp": datetime.now(),
                    "old_params": vars(old_params),
                    "new_params": vars(new_params),
                    "recommendations": recommendations,
                }
            )

            logger.info(f"🎯 Parâmetros ajustados:")
            logger.info(
                f"   Confidence: {old_params.ml_confidence_threshold:.3f} → {new_params.ml_confidence_threshold:.3f}"
            )
            logger.info(
                f"   Kelly Mult: {old_params.kelly_fraction_multiplier:.2f} → {new_params.kelly_fraction_multiplier:.2f}"
            )
            logger.info(
                f"   Spread Mult: {old_params.spread_filter_multiplier:.1f} → {new_params.spread_filter_multiplier:.1f}"
            )

            # Salva ajuste no banco
            self._save_adjustment_to_db(old_params, new_params, recommendations)

        except Exception as e:
            logger.error(f"Erro ao aplicar ajustes: {e}")

    def get_current_parameters(self) -> AdaptiveParameters:
        """Retorna parâmetros atuais ajustados"""
        return self.current_params

    def get_adjusted_ml_threshold(self) -> float:
        """Retorna threshold ML ajustado"""
        base_threshold = config.ML_CONFIDENCE_BASE
        return (
            base_threshold
            * self.current_params.ml_confidence_threshold
            / config.ML_CONFIDENCE_BASE
        )

    def get_adjusted_kelly_fraction(self, base_fraction: float) -> float:
        """Retorna fração Kelly ajustada"""
        return base_fraction * self.current_params.kelly_fraction_multiplier

    def get_adjusted_spread_multiplier(self) -> float:
        """Retorna multiplicador de spread ajustado"""
        return self.current_params.spread_filter_multiplier

    def _get_recent_trades(self, hours: int = 24) -> List[Dict]:
        """Obtém trades recentes do banco de dados"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            trades = database.get_trades_since(start_time)
            return trades if trades else []
        except Exception as e:
            logger.error(f"Erro ao obter trades recentes: {e}")
            return []

    def _calculate_winrate(self, trades: List[Dict], hours: int = 4) -> float:
        """Calcula winrate para período específico"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_trades = [
                t for t in trades if t.get("exit_time", datetime.min) >= cutoff_time
            ]

            if not recent_trades:
                return 0.5

            winners = sum(1 for t in recent_trades if float(t.get("net_pnl", 0)) > 0)
            return winners / len(recent_trades) if recent_trades else 0.5

        except Exception as e:
            logger.error(f"Erro ao calcular winrate: {e}")
            return 0.5

    def _calculate_sharpe(self, trades: List[Dict], hours: int = 4) -> float:
        """Calcula Sharpe ratio para período específico"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_trades = [
                t for t in trades if t.get("exit_time", datetime.min) >= cutoff_time
            ]

            if len(recent_trades) < 3:
                return 0.0

            pnls = [float(t.get("net_pnl", 0)) for t in recent_trades]
            returns = np.array(pnls)

            if returns.std() == 0:
                return 0.0

            return returns.mean() / returns.std() * np.sqrt(len(returns))

        except Exception as e:
            logger.error(f"Erro ao calcular Sharpe: {e}")
            return 0.0

    def _calculate_market_volatility(self) -> float:
        """Calcula volatilidade atual do mercado"""
        try:
            # Usa o IBOV como proxy de volatilidade de mercado
            ibov_data = utils.get_ibov_minute_data(period=60)  # Última hora
            if len(ibov_data) < 10:
                return 0.02  # Valor padrão

            returns = np.diff(np.log(ibov_data))
            return np.std(returns) * np.sqrt(252 * 390)  # Anualizado

        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade: {e}")
            return 0.02

    def _detect_volume_anomalies(self) -> float:
        """Detecta anomalias de volume no mercado"""
        try:
            # Analisa volume médio vs volume atual para principais ativos
            symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4"]  # Ativos líquidos
            anomalies = []

            for symbol in symbols:
                try:
                    current_vol = utils.get_current_volume_ratio(symbol)
                    anomalies.append(current_vol)
                except:
                    continue

            return max(anomalies) if anomalies else 1.0

        except Exception as e:
            logger.error(f"Erro ao detectar anomalias de volume: {e}")
            return 1.0

    def _calculate_market_correlation(self) -> float:
        """Calcula força de correlação entre ativos"""
        try:
            # Analisa correlação entre principais ativos
            symbols = ["PETR4", "VALE3", "ITUB4", "BBDC4"]
            correlations = []

            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i + 1 :]:
                    try:
                        corr = utils.get_correlation(sym1, sym2, period=60)
                        correlations.append(abs(corr))
                    except:
                        continue

            return np.mean(correlations) if correlations else 0.5

        except Exception as e:
            logger.error(f"Erro ao calcular correlação: {e}")
            return 0.5

    def _calculate_trend_strength(self) -> float:
        """Calcula força da tendência de mercado"""
        try:
            # Analisa ADX do IBOV como proxy de tendência
            ibov_adx = utils.get_ibov_adx(period=14)
            return ibov_adx / 100.0  # Normaliza para 0-1

        except Exception as e:
            logger.error(f"Erro ao calcular força da tendência: {e}")
            return 0.5

    def _save_adjustment_to_db(
        self,
        old_params: AdaptiveParameters,
        new_params: AdaptiveParameters,
        recommendations: Dict[str, float],
    ):
        """Salva ajuste no banco de dados"""
        try:
            adjustment_data = {
                "timestamp": datetime.now(),
                "old_parameters": vars(old_params),
                "new_parameters": vars(new_params),
                "recommendations": recommendations,
                "market_conditions": {
                    "winrate_1h": (
                        self.metrics_history[-1].winrate_1h
                        if self.metrics_history
                        else 0.5
                    ),
                    "sharpe_4h": (
                        self.metrics_history[-1].sharpe_4h
                        if self.metrics_history
                        else 0.0
                    ),
                    "volatility": (
                        self.metrics_history[-1].market_volatility
                        if self.metrics_history
                        else 0.02
                    ),
                },
            }

            database.save_adaptive_adjustment(adjustment_data)

        except Exception as e:
            logger.error(f"Erro ao salvar ajuste no banco: {e}")

    def get_performance_report(self) -> Dict:
        """Gera relatório de performance do sistema adaptativo"""
        try:
            if not self.metrics_history:
                return {"status": "Sem dados suficientes"}

            df = pd.DataFrame([vars(m) for m in self.metrics_history])

            return {
                "status": "Ativo",
                "total_adjustments": len(self.parameter_history),
                "current_parameters": vars(self.current_params),
                "performance_metrics": {
                    "avg_winrate_24h": df["winrate_24h"].mean(),
                    "avg_sharpe_4h": df["sharpe_4h"].mean(),
                    "avg_volatility": df["market_volatility"].mean(),
                    "winrate_trend": (
                        "Subindo"
                        if df["winrate_4h"].iloc[-10:].mean()
                        > df["winrate_4h"].iloc[-20:-10].mean()
                        else "Descendo"
                    ),
                },
                "last_adjustment": self.last_adjustment.isoformat(),
                "market_state": {
                    "volatility_level": (
                        "Alto"
                        if df["market_volatility"].iloc[-1]
                        > df["market_volatility"].mean() * 1.2
                        else "Normal"
                    ),
                    "correlation_level": (
                        "Alto"
                        if df["correlation_strength"].iloc[-1] > 0.7
                        else "Normal"
                    ),
                    "trend_strength": df["trend_strength"].iloc[-1],
                },
            }

        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return {"status": "Erro", "error": str(e)}


# Instância global
adaptive_intelligence = AdaptiveIntelligence()
