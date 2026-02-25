import logging
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional
import ta

# Tenta importar módulos auxiliares da raiz
try:
    import sys
    sys.path.append(os.getcwd())
    import utils
    from fundamentals import fundamental_fetcher
    from news_filter import get_news_sentiment
except ImportError:
    utils = None
    fundamental_fetcher = None
    get_news_sentiment = None

logger = logging.getLogger("MLPredictor")

class MLPredictor:
    """
    Realiza inferência usando modelos treinados (Genéricos).
    Substitui a lógica antiga por MLSignalPredictor (16 features).
    """
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.rf_model = None
        self.scaler = None
        self.load_models()
        
    def load_models(self):
        """Carrega modelo Random Forest e Scaler genéricos."""
        try:
            rf_path = os.path.join(self.models_dir, "rf_signal.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                logger.info(f"✅ Modelo ML carregado: {rf_path}")
            else:
                logger.warning(f"⚠️ Modelo não encontrado em {rf_path}")

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler carregado: {scaler_path}")
            else:
                logger.warning(f"⚠️ Scaler não encontrado em {scaler_path}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")

    def compute_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula indicadores técnicos necessários para as features."""
        if df.empty or len(df) < 30:
            return {}
            
        # Garante colunas minúsculas
        df.columns = [c.lower() for c in df.columns]
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['real_volume'] if 'real_volume' in df.columns else df['tick_volume']
        
        # RSI 14
        rsi = ta.momentum.rsi(close, window=14).iloc[-1]
        
        # ADX 14
        adx = ta.trend.adx(high, low, close, window=14).iloc[-1]
        
        # ATR %
        atr = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100
        
        # Volume Ratio (Last vs Avg 20)
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        
        # Momentum (10 periods)
        momentum = close.pct_change(10).iloc[-1]
        
        # EMA Diff (9 vs 21)
        ema9 = ta.trend.ema_indicator(close, window=9).iloc[-1]
        ema21 = ta.trend.ema_indicator(close, window=21).iloc[-1]
        ema_diff = (ema9 - ema21) / close.iloc[-1]
        
        # MACD
        macd = ta.trend.macd(close).iloc[-1]
        
        # Price vs VWAP (Aproximação: SMA 20 como proxy se não tiver vwap real)
        vwap_proxy = close.rolling(20).mean().iloc[-1]
        price_vs_vwap = (close.iloc[-1] - vwap_proxy) / close.iloc[-1]
        
        return {
            "rsi": rsi,
            "adx": adx,
            "atr_pct": atr_pct,
            "volume_ratio": vol_ratio,
            "momentum": momentum,
            "ema_diff": ema_diff,
            "macd": macd,
            "price_vs_vwap": price_vs_vwap
        }

    def extract_features(self, symbol: str, indicators: Dict[str, float]) -> np.ndarray:
        """
        Extrai as 16 features esperadas pelo modelo rf_signal.pkl
        """
        # Dados externos (Mock ou Reais)
        pe_ratio = 0.0
        roe = 0.0
        market_cap = 0.0
        sentiment = 0.0
        
        # Tenta buscar dados reais se os módulos estiverem disponíveis
        if fundamental_fetcher:
            try:
                fund = fundamental_fetcher.get_fundamentals(symbol)
                pe_ratio = fund.get("pe_ratio", 0.0)
                roe = fund.get("roe", 0.0)
                market_cap = fund.get("market_cap", 0.0) / 1e9 # Bilhões
            except: pass
            
        if get_news_sentiment:
            try:
                sentiment = get_news_sentiment(symbol)
            except: pass

        # Dados de Fluxo e Volatilidade (Requer utils)
        imbalance = 0.0
        cvd = 0.0
        vix_br = 0.0
        book_imbalance = 0.0
        
        if utils:
            try:
                order_flow = utils.get_order_flow(symbol, bars=10)
                imbalance = order_flow.get("imbalance", 0.0)
                cvd = order_flow.get("cvd", 0.0) / 10000
                vix_br = utils.get_vix_br() / 50.0
                book_imbalance = utils.get_book_imbalance(symbol) or 0.0
            except: pass

        # Vetor de 16 features (Ordem CRÍTICA - deve bater com ml_signals.py)
        features = np.array([
            indicators.get("rsi", 50.0),          # 0
            indicators.get("adx", 20.0),          # 1
            indicators.get("atr_pct", 2.0),       # 2
            indicators.get("volume_ratio", 1.0),  # 3
            indicators.get("momentum", 0.0),      # 4
            indicators.get("ema_diff", 0.0),      # 5
            indicators.get("macd", 0.0),          # 6
            indicators.get("price_vs_vwap", 0.0), # 7
            pe_ratio,                             # 8
            roe,                                  # 9
            market_cap,                           # 10
            sentiment,                            # 11
            imbalance,                            # 12
            cvd,                                  # 13
            vix_br,                               # 14
            book_imbalance                        # 15
        ], dtype=np.float32)
        
        # Substitui NaNs e Infinitos
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def predict(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera predição de compra/venda usando o modelo carregado.
        """
        if self.rf_model is None:
            # Fallback se modelo não carregar
            logger.warning(f"⚠️ Modelo RF não disponível para predição de {symbol}.")
            return {"probability": 0.5, "signal": "NEUTRAL", "reason": "model_missing"}
            
        # Calcula indicadores
        indicators = self.compute_indicators(current_data)
        if not indicators:
             return {"probability": 0.5, "signal": "NEUTRAL", "reason": "insufficient_data"}
             
        # Extrai features
        features = self.extract_features(symbol, indicators)
        
        # Aplica Scaler
        if self.scaler:
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            except Exception as e:
                logger.error(f"Erro no scaler: {e}")
                features_scaled = features.reshape(1, -1)
        else:
            features_scaled = features.reshape(1, -1)
            
        # Predição
        try:
            prob = self.rf_model.predict_proba(features_scaled)[0]
            # Probabilidade de classes: 0=BUY, 1=SELL, 2=HOLD (Assumindo ordem do treinamento)
            # VERIFICAR: No treino do ml_signals.py:
            # y = 0 (BUY), 1 (SELL), 2 (HOLD) ??? 
            # NÃO! Olhando train_rf_model em ml_signals.py:
            # y = np.array([0] * 500 + [1] * 500 + [2] * 500) -> 0=Buy, 1=Sell, 2=Hold
            # Mas espera, predict_proba retorna vetor de probs para cada classe.
            
            # Vamos assumir classes [0, 1, 2] -> [BUY, SELL, HOLD]
            # prob[0] -> Buy
            # prob[1] -> Sell
            # prob[2] -> Hold
            
            prob_buy = prob[0]
            prob_sell = prob[1]
            prob_hold = prob[2] if len(prob) > 2 else 0.0
            
            if prob_buy > 0.6 and prob_buy > prob_sell and prob_buy > prob_hold:
                signal = "BUY"
                confidence = prob_buy
            elif prob_sell > 0.6 and prob_sell > prob_buy and prob_sell > prob_hold:
                signal = "SELL"
                confidence = prob_sell
            else:
                signal = "NEUTRAL"
                confidence = prob_hold
                
            return {
                "probability": confidence,
                "signal": signal,
                "details": {
                    "buy_prob": prob_buy,
                    "sell_prob": prob_sell,
                    "hold_prob": prob_hold
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            return {"probability": 0.5, "signal": "NEUTRAL", "reason": "inference_error"}
