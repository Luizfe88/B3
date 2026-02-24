"""
Rejection Logger v6.0 - XP3 PRO QUANT-REFORM
Registra sinais rejeitados e monitora "what if" performance
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import time
import numpy as np

import config
import utils

logger = logging.getLogger(__name__)


@dataclass
class RejectedSignal:
    """Sinal rejeitado para análise what-if"""

    symbol: str
    side: str
    timestamp: datetime
    score: float
    rejection_reason: str
    entry_price: float
    indicators: Dict[str, float]

    # Resultado what-if (preenchido posteriormente)
    exit_price: Optional[float] = None
    pnl_whatif: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    monitoring_complete: bool = False


class RejectionLogger:
    """Logger inteligente para sinais rejeitados"""

    def __init__(self, monitoring_hours: int = 4):
        self.rejected_signals: List[RejectedSignal] = []
        self.monitoring_hours = monitoring_hours
        self.monitoring_thread = None
        self.stop_monitoring = False

    def log_rejection(
        self,
        symbol: str,
        side: str,
        score: float,
        rejection_reason: str,
        entry_price: float,
        indicators: Dict[str, float],
    ) -> None:
        """Registra sinal rejeitado"""

        # Só registra se score > 0.58 (regra #6)
        if score < 0.58:
            return

        rejected_signal = RejectedSignal(
            symbol=symbol,
            side=side,
            timestamp=datetime.now(),
            score=score,
            rejection_reason=rejection_reason,
            entry_price=entry_price,
            indicators=indicators.copy(),
        )

        self.rejected_signals.append(rejected_signal)

        logger.info(
            f"📊 What-If: {symbol} {side} rejeitado (Score: {score:.1%}) - {rejection_reason}"
        )

        # Inicia monitoramento se não estiver rodando
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.start_monitoring()

    def start_monitoring(self):
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(
            target=self._monitor_rejected_signals, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("🔄 Monitoramento What-If iniciado")

    def stop_monitoring_thread(self):
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

    def _monitor_rejected_signals(self):
        while not self.stop_monitoring:
            try:
                now = datetime.now()
                for signal in self.rejected_signals:
                    if signal.monitoring_complete:
                        continue
                    time_elapsed = now - signal.timestamp
                    if time_elapsed < timedelta(hours=self.monitoring_hours):
                        continue
                    self._calculate_whatif_result(signal)
                    signal.monitoring_complete = True
                cutoff_time = now - timedelta(hours=24)
                self.rejected_signals = [
                    s for s in self.rejected_signals if s.timestamp > cutoff_time
                ]
                time.sleep(60)
            except Exception as e:
                logger.error(f"❌ Erro no monitoramento What-If: {e}")
                time.sleep(300)

    def _calculate_whatif_result(self, signal: RejectedSignal):
        try:
            start_time = signal.timestamp
            end_time = start_time + timedelta(hours=self.monitoring_hours)
            exit_price = self._simulate_trade_exit(
                signal.symbol,
                signal.side,
                signal.entry_price,
                start_time,
                end_time,
                signal.indicators,
            )
            if exit_price:
                signal.exit_price = exit_price
                signal.exit_timestamp = end_time
                if signal.side == "BUY":
                    pnl_pct = (exit_price - signal.entry_price) / signal.entry_price
                else:
                    pnl_pct = (signal.entry_price - exit_price) / signal.entry_price
                signal.pnl_whatif = pnl_pct
                self._log_whatif_result(signal)
        except Exception as e:
            logger.error(f"❌ Erro ao calcular what-if para {signal.symbol}: {e}")

    def _simulate_trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        start_time: datetime,
        end_time: datetime,
        indicators: Dict[str, float],
    ) -> Optional[float]:
        try:
            df = utils.get_rates_range(symbol, start_time, end_time, timeframe=15)
            if df is None or len(df) < 2:
                return None
            atr = indicators.get("atr", entry_price * 0.02)
            if side == "BUY":
                tp_price = entry_price + (atr * 2.0)
                sl_price = entry_price - (atr * 1.5)
            else:
                tp_price = entry_price - (atr * 2.0)
                sl_price = entry_price + (atr * 1.5)
            for i in range(1, len(df)):
                high = df.iloc[i]["high"]
                low = df.iloc[i]["low"]
                close = df.iloc[i]["close"]
                if side == "BUY":
                    if high >= tp_price:
                        return tp_price
                    if low <= sl_price:
                        return sl_price
                else:
                    if low <= tp_price:
                        return tp_price
                    if high >= sl_price:
                        return sl_price
            return df.iloc[-1]["close"]
        except Exception as e:
            logger.error(f"❌ Erro ao simular saída para {symbol}: {e}")
            return None

    def _log_whatif_result(self, signal: RejectedSignal):
        if signal.pnl_whatif is None:
            return
        status = "✅ GANHO" if signal.pnl_whatif > 0 else "❌ PERDA"
        logger.info(
            f"📈 What-If Result: {signal.symbol} {signal.side} - {status} "
            f"({signal.pnl_whatif:+.2%}) | Entry: {signal.entry_price:.2f} "
            f"| Exit: {signal.exit_price:.2f} | Score: {signal.score:.1%}"
        )
        if signal.pnl_whatif < -0.02:
            logger.warning(
                f"⚠️ What-If ALERT: Rejeição custou {abs(signal.pnl_whatif):.2%}"
            )

    def get_summary(self) -> Dict[str, any]:
        completed_signals = [s for s in self.rejected_signals if s.monitoring_complete]
        if not completed_signals:
            return {"total": 0, "avg_pnl": 0, "win_rate": 0}
        pnls = [s.pnl_whatif for s in completed_signals if s.pnl_whatif is not None]
        return {
            "total": len(completed_signals),
            "avg_pnl": np.mean(pnls) if pnls else 0,
            "win_rate": len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls) if pnls else 0,
        }

    def export_report(self, filename: str = "whatif_report.json"):
        try:
            data = {
                "generated_at": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "signals": [
                    asdict(s) for s in self.rejected_signals if s.monitoring_complete
                ],
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"📊 Relatório What-If exportado: {filename}")
        except Exception as e:
            logger.error(f"❌ Erro ao exportar relatório: {e}")


# Função global para compatibilidade com código existente
def log_trade_rejection(
    symbol: str,
    side: str,
    rejection_reason: str,
    indicators: Dict[str, float] = None,
    **kwargs,
):
    """
    Função global para registrar rejeições (mantida para compatibilidade)
    Usa o logger global com score padrão de 0.6
    """
    try:
        # Criar logger global se não existir
        if not hasattr(log_trade_rejection, "_logger"):
            log_trade_rejection._logger = RejectionLogger()

        logger_instance = log_trade_rejection._logger

        # Parâmetros padrão
        score = 0.6  # Score padrão para compatibilidade
        entry_price = kwargs.get("entry_price", 0.0)
        indicators = indicators or {}

        # Se tiver kwargs adicionais, adicionar aos indicadores
        for key, value in kwargs.items():
            if key not in ["entry_price"]:
                indicators[key] = value

        logger_instance.log_rejection(
            symbol=symbol,
            side=side,
            score=score,
            rejection_reason=rejection_reason,
            entry_price=entry_price,
            indicators=indicators,
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Erro ao logar rejeição: {e}")


# Instância global
rejection_logger = RejectionLogger()
