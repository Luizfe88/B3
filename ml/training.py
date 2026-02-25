
import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from .feature_store import FeatureStore

logger = logging.getLogger("MLTrainer")

class MLTrainer:
    """
    Treinador de modelos ML com dados reais.
    """
    def __init__(self):
        self.feature_store = FeatureStore()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
    def train_model(self, symbol: str, start_date: datetime, end_date: datetime):
        """
        Executa o pipeline de treinamento para um símbolo.
        """
        logger.info(f"📚 Treinando modelo para {symbol}...")
        
        # 1. Dados
        df = self.feature_store.fetch_data(symbol, start_date, end_date)
        df = self.feature_store.compute_features(df)
        
        if df.empty:
            logger.error(f"Dados insuficientes para {symbol}")
            return
            
        # 2. Features vs Target
        features = ['ema_9', 'ema_21', 'sma_200', 'rsi', 'adx', 'atr', 'bb_width', 'obv'] # Simplified
        X = df[features]
        y = df['target']
        
        # 3. TimeSeries Split
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            logger.info(f"CV Accuracy: {accuracy_score(y_test, preds):.2f}")
            
        # 4. Treino Final (Full Data)
        self.model.fit(X, y)
        
        # 5. Salvar Modelo
        joblib.dump(self.model, f"models/{symbol}_rf_model.pkl")
        logger.info(f"✅ Modelo salvo em models/{symbol}_rf_model.pkl")

if __name__ == "__main__":
    trainer = MLTrainer()
    # Exemplo: Treinar PETR4 de 2018 a 2025
    trainer.train_model("PETR4", datetime(2018, 1, 1), datetime(2025, 12, 31))
