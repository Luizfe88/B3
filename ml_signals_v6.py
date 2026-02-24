"""
ML Signals v6.0 - XP3 PRO QUANT-REFORM
Modelo de ML treinado com dados reais dos últimos 90 dias
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

import config
import utils
import database

logger = logging.getLogger(__name__)


@dataclass
class MLFeatures:
    """Features para o modelo de ML"""

    # Técnicos tradicionais
    rsi: float
    macd: float
    macd_signal: float
    adx: float
    volume_ratio: float
    price_vs_vwap: float
    atr_ratio: float

    # Features novas (Regra #4)
    time_since_last_trade: float  # minutos
    session_momentum: float  # força da sessão

    # Order flow
    delta_volume: float
    cumulative_delta: float

    # Fundamentals
    sector_rotation: float
    market_sentiment: float

    # Contexto
    ibov_correlation: float
    sector_correlation: float
    time_of_day: float  # hora normalizada [0,1]
    day_of_week: int


class MLSignalGenerator:
    """Gerador de sinais baseado em Machine Learning"""

    def __init__(self):
        self.models = {"buy": None, "sell": None}
        self.feature_names = []
        self.scaler = None
        self.last_training_date = None
        self.training_days = config.ML_TRAINING_DAYS

    def extract_features(
        self, symbol: str, timeframe: int = 15
    ) -> Optional[MLFeatures]:
        """Extrai features do ativo"""
        try:
            # Dados de mercado
            df = utils.safe_copy_rates(symbol, timeframe, 200)
            if df is None or len(df) < 50:
                return None

            # Features técnicas
            rsi = utils.calculate_rsi(df["close"], 14)[-1]
            macd_data = utils.calculate_macd(df["close"])
            macd = macd_data["macd"][-1]
            macd_signal = macd_data["signal"][-1]
            adx = utils.calculate_adx(df, 14)[-1]

            # Volume
            avg_volume = df["volume"].rolling(20).mean().iloc[-1]
            volume_ratio = df["volume"].iloc[-1] / avg_volume if avg_volume > 0 else 1.0

            # VWAP
            vwap = utils.calculate_vwap(df)
            price_vs_vwap = (df["close"].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]

            # ATR
            atr = utils.calculate_atr(df, 14)[-1]
            atr_ratio = atr / df["close"].iloc[-1]

            # Features novas
            time_since_last_trade = self._get_time_since_last_trade(symbol)
            session_momentum = self._calculate_session_momentum(symbol)

            # Order flow (se disponível)
            delta_volume = self._get_order_flow_delta(symbol)
            cumulative_delta = self._get_cumulative_delta(symbol)

            # Fundamentals
            sector_rotation = self._calculate_sector_rotation(symbol)
            market_sentiment = self._get_market_sentiment()

            # Contexto de mercado
            ibov_corr = utils.get_correlation_with_ibov(symbol, 20)
            sector_corr = utils.get_correlation_with_sector(symbol, 20)

            # Tempo
            now = datetime.now()
            time_of_day = (now.hour + now.minute / 60) / 24.0
            day_of_week = now.weekday()

            return MLFeatures(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                adx=adx,
                volume_ratio=volume_ratio,
                price_vs_vwap=price_vs_vwap,
                atr_ratio=atr_ratio,
                time_since_last_trade=time_since_last_trade,
                session_momentum=session_momentum,
                delta_volume=delta_volume,
                cumulative_delta=cumulative_delta,
                sector_rotation=sector_rotation,
                market_sentiment=market_sentiment,
                ibov_correlation=ibov_corr,
                sector_correlation=sector_corr,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
            )

        except Exception as e:
            logger.error(f"❌ Erro ao extrair features para {symbol}: {e}")
            return None

    def _get_time_since_last_trade(self, symbol: str) -> float:
        """Tempo desde o último trade no mesmo símbolo (minutos)"""
        try:
            last_trade = database.get_last_trade_by_symbol(symbol)
            if last_trade:
                time_diff = datetime.now() - last_trade.exit_time
                return time_diff.total_seconds() / 60.0
            return 999.0  # Muito tempo atrás
        except:
            return 999.0

    def _calculate_session_momentum(self, symbol: str) -> float:
        """Calcula momentum da sessão atual"""
        try:
            # Dados intraday da sessão
            df = utils.get_intraday_session_data(symbol)
            if df is None or len(df) < 10:
                return 0.0

            # Calcula momentum baseado em mudanças de preço
            returns = df["close"].pct_change().dropna()
            if len(returns) == 0:
                return 0.0

            # Momentum ponderado pelo tempo
            weights = np.linspace(0.5, 1.0, len(returns))
            momentum = np.average(returns, weights=weights)

            return momentum

        except:
            return 0.0

    def _get_order_flow_delta(self, symbol: str) -> float:
        """Delta de volume (compras - vendas)"""
        try:
            # Busca dados de order flow se disponível
            of_data = database.get_order_flow_data(symbol, minutes=30)
            if of_data:
                buy_volume = sum([x.volume for x in of_data if x.side == "buy"])
                sell_volume = sum([x.volume for x in of_data if x.side == "sell"])
                total_volume = buy_volume + sell_volume

                if total_volume > 0:
                    return (buy_volume - sell_volume) / total_volume

            return 0.0
        except:
            return 0.0

    def _get_cumulative_delta(self, symbol: str) -> float:
        """Delta cumulativo do dia"""
        try:
            of_data = database.get_order_flow_data(symbol, minutes=240)  # Dia todo
            if of_data:
                buy_volume = sum([x.volume for x in of_data if x.side == "buy"])
                sell_volume = sum([x.volume for x in of_data if x.side == "sell"])

                return buy_volume - sell_volume
            return 0.0
        except:
            return 0.0

    def _calculate_sector_rotation(self, symbol: str) -> float:
        """Força de rotação setorial"""
        try:
            sector = utils.get_sector_by_symbol(symbol)
            if sector:
                # Compara performance do setor vs mercado
                sector_return = utils.get_sector_return(sector, days=5)
                market_return = utils.get_ibov_return(days=5)

                if sector_return is not None and market_return is not None:
                    return sector_return - market_return

            return 0.0
        except:
            return 0.0

    def _get_market_sentiment(self) -> float:
        """Sentimento geral do mercado"""
        try:
            # Análise simples baseada em dados de mercado
            ibov_return = utils.get_ibov_return(days=1)
            vix_level = utils.get_vix_brasil()

            sentiment = 0.0

            if ibov_return is not None:
                sentiment += np.sign(ibov_return) * 0.5

            if vix_level is not None:
                # VIX alto = sentimento negativo
                if vix_level > 30:
                    sentiment -= 0.3
                elif vix_level < 20:
                    sentiment += 0.2

            return np.clip(sentiment, -1.0, 1.0)

        except:
            return 0.0

    def features_to_array(self, features: MLFeatures) -> np.ndarray:
        """Converte features para array numpy"""
        feature_list = [
            features.rsi,
            features.macd,
            features.macd_signal,
            features.adx,
            features.volume_ratio,
            features.price_vs_vwap,
            features.atr_ratio,
            features.time_since_last_trade,
            features.session_momentum,
            features.delta_volume,
            features.cumulative_delta,
            features.sector_rotation,
            features.market_sentiment,
            features.ibov_correlation,
            features.sector_correlation,
            features.time_of_day,
            features.day_of_week,
        ]

        return np.array(feature_list).reshape(1, -1)

    def should_retrain(self) -> bool:
        """Verifica se modelo precisa ser retrado"""
        if self.last_training_date is None:
            return True

        days_since_training = (datetime.now() - self.last_training_date).days
        return days_since_training >= 7  # Retreina semanalmente

    def train_models(self, force_retrain: bool = False):
        """Treina modelos com dados reais dos últimos 90 dias"""

        if not self.should_retrain() and not force_retrain:
            logger.info("📊 Modelos ainda estão atualizados")
            return

        logger.info(
            f"🧠 Iniciando treinamento com {self.training_days} dias de dados reais..."
        )

        try:
            # Busca dados de trades reais
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.training_days)

            trades = database.get_all_trades(start_date, end_date)

            if len(trades) < 100:
                logger.warning(f"⚠️ Poucos dados para treinamento: {len(trades)} trades")
                return

            # Prepara dados de treinamento
            X_buy, y_buy, X_sell, y_sell = self._prepare_training_data(trades)

            if len(X_buy) < 50 or len(X_sell) < 50:
                logger.warning("⚠️ Dados insuficientes para treinar modelos")
                return

            # Treina modelos
            self.models["buy"] = self._train_model(X_buy, y_buy, "buy")
            self.models["sell"] = self._train_model(X_sell, y_sell, "sell")

            self.last_training_date = datetime.now()

            # Salva modelos
            self._save_models()

            logger.info("✅ Modelos treinados com sucesso")

        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")

    def _prepare_training_data(
        self, trades: List
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados de treinamento"""

        X_buy, y_buy = [], []
        X_sell, y_sell = [], []

        for trade in trades:
            # Extrai features no momento da entrada
            features = self.extract_features(trade.symbol, timeframe=15)
            if features is None:
                continue

            feature_array = self.features_to_array(features).flatten()

            # Resultado do trade (1 = sucesso, 0 = falha)
            success = 1 if trade.pnl > 0 else 0

            if trade.side == "BUY":
                X_buy.append(feature_array)
                y_buy.append(success)
            else:
                X_sell.append(feature_array)
                y_sell.append(success)

        return (np.array(X_buy), np.array(y_buy), np.array(X_sell), np.array(y_sell))

    def _train_model(self, X: np.ndarray, y: np.ndarray, side: str):
        """Treina modelo individual"""

        # Usa ensemble de modelos
        models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
        ]

        best_model = None
        best_score = 0

        # Validação temporal
        tscv = TimeSeriesSplit(n_splits=5)

        for model in models:
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Métrica principal: F1-score
                f1 = f1_score(y_val, y_pred)
                scores.append(f1)

            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_model = model

        logger.info(f"📈 Modelo {side} treinado - F1: {best_score:.3f}")

        # Retreina no conjunto completo
        best_model.fit(X, y)

        return best_model

    def predict_signal(self, symbol: str, side: str) -> Tuple[float, Dict[str, float]]:
        """
        Gera sinal de ML para o ativo

        Returns:
            (confidence, feature_values)
        """

        # Verifica se modelos estão treinados
        if self.models[side] is None:
            logger.warning(f"⚠️ Modelo {side} não treinado")
            return 0.5, {}

        # Extrai features
        features = self.extract_features(symbol, timeframe=15)
        if features is None:
            return 0.5, {}

        # Converte para array
        X = self.features_to_array(features)

        # Predição
        try:
            confidence = self.models[side].predict_proba(X)[0][
                1
            ]  # Probabilidade de sucesso

            # Converte features para dict
            feature_dict = {
                "rsi": features.rsi,
                "macd": features.macd,
                "adx": features.adx,
                "volume_ratio": features.volume_ratio,
                "time_since_last_trade": features.time_since_last_trade,
                "session_momentum": features.session_momentum,
                "market_sentiment": features.market_sentiment,
            }

            return confidence, feature_dict

        except Exception as e:
            logger.error(f"❌ Erro na predição para {symbol}: {e}")
            return 0.5, {}

    def get_signal_with_threshold(
        self, symbol: str, side: str, base_threshold: float = None
    ) -> Tuple[bool, float, Dict]:
        """
        Gera sinal com threshold dinâmico

        Returns:
            (should_enter, confidence, info)
        """

        confidence, features = self.predict_signal(symbol, side)

        # Threshold base
        if base_threshold is None:
            base_threshold = config.ML_CONFIDENCE_BASE

        # Ajusta threshold baseado no modo
        if config.is_aggressive_mode():
            threshold = base_threshold - 0.08  # Regra #5
        else:
            threshold = base_threshold

        # Limita threshold máximo
        max_threshold = config.ML_DYNAMIC_THRESHOLD_MAX
        threshold = min(threshold, max_threshold)

        # Decisão
        should_enter = confidence >= threshold

        info = {
            "confidence": confidence,
            "threshold": threshold,
            "features": features,
            "model_trained": self.models[side] is not None,
        }

        return should_enter, confidence, info

    def _save_models(self):
        """Salva modelos treinados"""
        try:
            joblib.dump(self.models, "ml_models.pkl")
            logger.info("💾 Modelos salvos com sucesso")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelos: {e}")

    def load_models(self):
        """Carrega modelos salvos"""
        try:
            self.models = joblib.load("ml_models.pkl")
            logger.info("📂 Modelos carregados com sucesso")
        except:
            logger.warning("⚠️ Nenhum modelo salvo encontrado")


# Instância global
ml_signal_generator = MLSignalGenerator()
