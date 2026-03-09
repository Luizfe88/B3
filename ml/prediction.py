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
    from ml_optimizer import ml_optimizer
except ImportError:
    utils = None
    fundamental_fetcher = None
    get_news_sentiment = None
    ml_optimizer = None

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
        FIX: Fallback técnico real quando RF indisponível (em vez de sempre NEUTRAL 0.5).
        """
        if self.rf_model is None:
            logger.warning(f"⚠️ Modelo RF não disponível para {symbol} — usando fallback técnico.")
            return self._technical_fallback(current_data)
            
        # Calcula indicadores
        indicators = self.compute_indicators(current_data)
        if not indicators:
             return self._technical_fallback(current_data)
             
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
            # prob[0]=BUY, prob[1]=SELL, prob[2]=HOLD
            prob_buy  = prob[0]
            prob_sell = prob[1]
            prob_hold = prob[2] if len(prob) > 2 else 0.0
            
            # FIX: Maioria simples (sem threshold fixo de 0.6 que nunca era atingido)
            # Exige apenas que a classe seja a maior E tenha >38% (descarta empates)
            # ✅ Pilar 1: Filtragem de Ruído via KNN (Cenário Histórico Próximo)
            knn_adj = 0.0
            if ml_optimizer:
                # Converte as features atuais para o dicionário que o extract_features de ml_optimizer espera
                # Para simplificar, passamos o dicionário de indicators + dados externos
                full_features = indicators.copy()
                full_features.update({
                    "pe_ratio": pe_ratio, "roe": roe, "market_cap": market_cap,
                    "sentiment": sentiment, "imbalance": imbalance, "cvd": cvd,
                    "vix": vix_br, "asset_type": "STOCK" # default
                })
                knn_pnl = ml_optimizer.knn_predict_expected_return(full_features, k=7)
                
                # Ajuste: se o KNN histórico diz que ganhamos >0.5%, damos bônus de 5% na confiança
                # Se diz que perdemos, penalizamos
                if signal == "BUY":
                    knn_adj = np.tanh(knn_pnl * 20) * 0.1 # max +/- 10%
                elif signal == "SELL":
                    knn_adj = np.tanh(-knn_pnl * 20) * 0.1

            confidence = float(np.clip(confidence + knn_adj, 0.0, 0.99))

            logger.info(
                f"   ↳ ML RF: {signal} | P(base)={prob_buy if signal=='BUY' else prob_sell:.2f} | "
                f"KNN Adj: {knn_adj:+.2f} | Final: {confidence:.2f}"
            )
            return {
                "probability": confidence,
                "signal": signal,
                "knn_expected_pnl": knn_pnl if ml_optimizer else 0.0,
                "details": {
                    "buy_prob": prob_buy,
                    "sell_prob": prob_sell,
                    "hold_prob": prob_hold,
                    "knn_adj": knn_adj
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência: {e}")
            return self._technical_fallback(current_data)

    def _technical_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback técnico baseado em EMA crossover + RSI + ADX.
        Evita que o bot fique preso em NEUTRAL 0.5 quando o RF não está disponível.
        """
        try:
            if df is None or df.empty or len(df) < 30:
                return {"probability": 0.5, "signal": "NEUTRAL", "reason": "insufficient_data"}
            
            indicators = self.compute_indicators(df)
            if not indicators:
                return {"probability": 0.5, "signal": "NEUTRAL", "reason": "no_indicators"}
            
            ema_diff   = indicators.get("ema_diff", 0.0)     # EMA9 vs EMA21
            rsi        = indicators.get("rsi", 50.0)
            adx        = indicators.get("adx", 0.0)
            momentum   = indicators.get("momentum", 0.0)
            vol_ratio  = indicators.get("volume_ratio", 1.0)
            
            score = 0.0  # -1 (forte sell) a +1 (forte buy)
            
            # EMA crossover — sinal primário
            if ema_diff > 0.002:    score += 0.40
            elif ema_diff < -0.002: score -= 0.40
            
            # RSI — confirma ou filtra sinal
            if rsi < 35:            score += 0.25   # sobrevenda → bull
            elif rsi > 65:          score -= 0.25   # sobrecompra → bear
            elif 45 <= rsi <= 55:   pass            # neutro
            
            # Momentum
            if momentum > 0.01:     score += 0.20
            elif momentum < -0.01:  score -= 0.20
            
            # ADX confirma tendência
            adx_factor = min(adx / 40.0, 1.0)  # 0=sem tendência, 1=tendência forte
            score *= (0.5 + 0.5 * adx_factor)  # Diminui score se ADX baixo
            
            # Volume confirma (sinal fraco sem volume)
            if vol_ratio < 0.7:     score *= 0.7
            
            score = max(-1.0, min(1.0, score))
            
            # Converte score para sinal e probabilidade
            THRESHOLD = 0.25  # Score mínimo para agir
            if score >= THRESHOLD:
                signal = "BUY"
                probability = 0.50 + (score * 0.30)  # mapeia 0.25→0.575, 1.0→0.80
            elif score <= -THRESHOLD:
                signal = "SELL"
                probability = 0.50 + (abs(score) * 0.30)
            else:
                signal = "NEUTRAL"
                probability = 0.50
            
            logger.info(
                f"   ↳ ML Fallback Técnico: {signal} | score={score:.3f} "
                f"(EMA:{ema_diff:.4f} RSI:{rsi:.0f} ADX:{adx:.0f} mom:{momentum:.3f})"
            )
            return {"probability": probability, "signal": signal, "reason": "technical_fallback"}
            
        except Exception as e:
            logger.error(f"Erro no fallback técnico: {e}")
            return {"probability": 0.5, "signal": "NEUTRAL", "reason": "fallback_error"}
