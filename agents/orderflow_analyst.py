import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None # Fallback para ambientes de teste sem MT5

logger = logging.getLogger("OrderFlowAnalyst")

class OrderFlowAnalyst:
    """
    Analista especializado em Fluxo de Ordens (Tape Reading).
    Analisa agressão, desbalanceamento do livro e VWAP.
    """
    def __init__(self):
        pass

    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa o fluxo de ordens para o símbolo.
        Espera receber 'ticks' ou 'depth' em data.
        """
        logger.info(f"🌊 [OrderFlow] Analisando fluxo para {symbol}...")
        
        ticks = data.get('ticks', [])
        
        # Verificação de dados mínimos (Relaxado para 10 ticks para evitar mass-rejection)
        if ticks is None or len(ticks) < 10:
             logger.warning(f"⚠️ [{symbol}] Dados de fluxo insuficientes ({len(ticks) if ticks is not None else 0} ticks) → VALID=FALSE")
             return {
                "type": "order_flow",
                "score": 0.0,
                "valid": False,
                "pressure": "neutral",
                "imbalance": 0.0,
                "reason": "insufficient_tick_data"
            }

        # Conversão para DataFrame
        if isinstance(ticks, list):
            df = pd.DataFrame(ticks)
        elif isinstance(ticks, np.ndarray):
            df = pd.DataFrame(ticks)
        else:
            df = ticks

        if df.empty:
             logger.warning(f"⚠️ [{symbol}] DataFrame de fluxo vazio → VALID=FALSE")
             return {
                "type": "order_flow",
                "score": 0.0,
                "valid": False,
                "pressure": "neutral",
                "imbalance": 0.0,
                "reason": "empty_tick_data"
            }

        # Cálculo de Agressão Real
        # Tenta usar flags se disponível
        if 'flags' in df.columns:
            # Constantes MT5 (caso mt5 não esteja importado ou conectado)
            FLAG_BUY = 32  # mt5.TICK_FLAG_BUY
            FLAG_SELL = 64 # mt5.TICK_FLAG_SELL
            
            # Vetorizado para performance
            flags = df['flags'].values
            volumes = df['volume_real'].values if 'volume_real' in df.columns else df['volume'].values
            
            buy_mask = (flags & FLAG_BUY) == FLAG_BUY
            sell_mask = (flags & FLAG_SELL) == FLAG_SELL
            
            buy_vol = np.sum(volumes[buy_mask])
            sell_vol = np.sum(volumes[sell_mask])
        else:
            # Fallback: Tick Direction (baseado na variação de preço)
            # Se preço subiu = compra, desceu = venda
            prices = df['last'].values
            deltas = np.diff(prices, prepend=prices[0])
            volumes = df['volume_real'].values if 'volume_real' in df.columns else df['volume'].values
            
            buy_vol = np.sum(volumes[deltas > 0])
            sell_vol = np.sum(volumes[deltas < 0])
            # Ticks sem variação (0) são ignorados ou distribuídos

        total_vol = buy_vol + sell_vol
        
        if total_vol > 0:
            buy_pressure = buy_vol / total_vol
            sell_pressure = sell_vol / total_vol
        else:
            buy_pressure = 0.5
            sell_pressure = 0.5
        
        imbalance = buy_pressure - sell_pressure # -1.0 (Full Sell) a +1.0 (Full Buy)
        
        # Score ajustado: 0.5 base + (imbalance / 2)
        # Ex: Imbalance +0.4 (Forte compra) -> Score 0.7
        # Ex: Imbalance -0.4 (Forte venda) -> Score 0.3
        score = 0.5 + (imbalance * 0.5)
        
        # Determina tendência baseada no imbalance
        if imbalance > 0.15:
            trend = "bullish"
        elif imbalance < -0.15:
            trend = "bearish"
        else:
            trend = "neutral"
        
        logger.info(f"   ↳ Pressure: {trend} | Imbalance: {imbalance:.2f} | Score: {score:.2f} | Vol: {total_vol:.0f}")
        
        return {
            "type": "order_flow",
            "score": score,
            "valid": True,
            "pressure": trend,
            "imbalance": imbalance,
            "metrics": {
                "buy_aggression": float(buy_pressure),
                "sell_aggression": float(sell_pressure),
                "total_volume": float(total_vol)
            }
        }
