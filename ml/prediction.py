
import logging
import joblib
import pandas as pd
from typing import Dict, Any
from .feature_store import FeatureStore

logger = logging.getLogger("MLPredictor")

class MLPredictor:
    """
    Realiza inferência usando modelos treinados.
    """
    def __init__(self):
        self.feature_store = FeatureStore()
        self.models = {}
        
    def load_model(self, symbol: str):
        if symbol in self.models:
            return
            
        try:
            path = f"models/{symbol}_rf_model.pkl"
            self.models[symbol] = joblib.load(path)
            logger.info(f"✅ Modelo carregado para {symbol}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Modelo não encontrado para {symbol}. Usando heurística.")
            self.models[symbol] = None

    def predict(self, symbol: str, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera predição de compra/venda.
        """
        self.load_model(symbol)
        model = self.models.get(symbol)
        
        # Gera features
        features_df = self.feature_store.compute_features(current_data)
        
        if features_df.empty:
            return {"probability": 0.5, "signal": "NEUTRAL"}
            
        # Pega a última linha (estado atual)
        latest = features_df.iloc[[-1]]
        
        # Seleciona colunas usadas no treino
        cols = ['ema_9', 'ema_21', 'sma_200', 'rsi', 'adx', 'atr', 'bb_width', 'obv']
        X = latest[cols]
        
        if model:
            prob = model.predict_proba(X)[0][1] # Probabilidade da classe 1 (Alta)
            signal = "BUY" if prob > 0.6 else "SELL" if prob < 0.4 else "NEUTRAL"
            return {"probability": prob, "signal": signal}
        else:
            # Fallback Heurístico (se não houver modelo treinado)
            rsi = latest['rsi'].values[0]
            if rsi < 30: return {"probability": 0.7, "signal": "BUY"}
            if rsi > 70: return {"probability": 0.3, "signal": "SELL"}
            return {"probability": 0.5, "signal": "NEUTRAL"}
