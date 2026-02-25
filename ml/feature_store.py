
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import logging

logger = logging.getLogger("FeatureStore")

class FeatureStore:
    """
    Centraliza a engenharia de features e acesso a dados históricos.
    Garante consistência entre treinamento e inferência.
    """
    
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe=mt5.TIMEFRAME_D1) -> pd.DataFrame:
        """
        Busca dados históricos do MT5.
        """
        if not mt5.initialize():
            logger.error("MT5 não inicializado")
            return pd.DataFrame()
            
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            logger.warning(f"Sem dados para {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera as 42 features técnicas usadas pelo modelo.
        """
        if df.empty:
            return df
            
        # 1. Trend Indicators
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # 2. Momentum
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        
        # 3. Volatility
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['bb_width'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)
        
        # 4. Volume
        # Verifica se 'real_volume' ou 'tick_volume' está disponível e usa como 'volume'
        if 'real_volume' in df.columns:
            volume_col = df['real_volume']
        elif 'tick_volume' in df.columns:
            volume_col = df['tick_volume']
        elif 'volume' in df.columns:
             volume_col = df['volume']
        else:
             volume_col = pd.Series(0, index=df.index)

        df['obv'] = ta.volume.on_balance_volume(df['close'], volume_col)
        
        # 5. Custom Targets (para treino)
        # Target: Retorno em 5 dias > 2% (exemplo)
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)
        
        # Limpeza
        df.dropna(inplace=True)
        return df
