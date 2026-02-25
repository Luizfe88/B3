
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("AnalystTeam")

class Analyst:
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class FundamentalAnalyst(Analyst):
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Tenta obter dados reais do fetcher global (utils.py)
        import utils
        
        logger.info(f"🔎 [Fundamental] Analisando balanços de {symbol}...")
        
        # Valores padrão
        valuation = "neutral"
        score = 0.5
        risks = []
        drivers = []
        
        try:
            fund = utils.fundamental_fetcher.get_fundamentals(symbol)
            mcap = fund.get("market_cap", 0)
            sector = fund.get("sector", "Outros")
            
            # Lógica simples baseada em Market Cap (Blue Chips vs Small Caps)
            # Em um sistema real, usaria P/L, ROE, Dívida Líquida/EBITDA
            if mcap > 100_000_000_000: # > 100B (Blue Chip)
                score = 0.7
                valuation = "fair"
                drivers.append("high_liquidity")
            elif mcap > 20_000_000_000: # > 20B
                score = 0.6
                valuation = "neutral"
            else:
                score = 0.4
                valuation = "undervalued" # ou risky
                risks.append("low_liquidity")
                
            # Ajuste por setor (Exemplo)
            if sector == "Bancos":
                score += 0.1 # Bancos costumam ser sólidos
                drivers.append("sector_resilience")
            elif sector == "Varejo":
                score -= 0.1 # Varejo sofre com juros
                risks.append("macro_headwinds")
                
            score = max(0.1, min(0.9, score))
            
        except Exception as e:
            logger.warning(f"Erro na análise fundamentalista: {e}")
        
        result = {
            "type": "fundamental",
            "score": score,
            "valuation": valuation,
            "risks": risks,
            "drivers": drivers,
            "details": fund if 'fund' in locals() else {}
        }
        logger.info(f"   ↳ Valuation: {valuation} | Score: {score:.2f}")
        return result

class SentimentAnalyst(Analyst):
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simula análise de redes sociais/notícias com alguma variação randômica
        # para não parecer estático, mas idealmente conectaria a uma API
        import random
        
        logger.info(f"🐦 [Sentiment] Escaneando Twitter/News para {symbol}...")
        
        # Gera um score base levemente otimista (mercado tende a subir no longo prazo)
        # Variação aleatória para simular fluxo de notícias
        base_score = 0.55 
        noise = random.uniform(-0.1, 0.1)
        score = base_score + noise
        
        sentiment = "neutral"
        if score > 0.6:
            sentiment = "optimistic"
        elif score < 0.4:
            sentiment = "pessimistic"
        
        result = {
            "type": "sentiment",
            "score": score,
            "sentiment": sentiment,
            "sources": ["twitter_br", "valor_economico"]
        }
        logger.info(f"   ↳ Sentiment: {sentiment} | Score: {score:.2f}")
        return result

class TechnicalAnalyst(Analyst):
    def __init__(self):
        from ml.prediction import MLPredictor
        self.predictor = MLPredictor()
        
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Usa indicadores técnicos existentes do bot
        logger.info(f"📈 [Technical] Verificando gráficos de {symbol}...")
        
        # Obter DataFrame de candles (passado em 'data' ou fetch direto)
        # Assumindo que 'data' contém 'candles' (pd.DataFrame)
        df = data.get('candles')
        
        if df is None or df.empty:
             return {
                "type": "technical",
                "score": 0.5,
                "trend": "neutral",
                "reason": "no_data"
            }
            
        prediction = self.predictor.predict(symbol, df)
        
        trend = "neutral"
        if prediction['signal'] == "BUY":
            trend = "bullish"
        elif prediction['signal'] == "SELL":
            trend = "bearish"
        
        result = {
            "type": "technical",
            "score": prediction['probability'],
            "trend": trend,
            "signals": ["ml_rf_ensemble"],
            "raw_prediction": prediction
        }
        logger.info(f"   ↳ Trend: {result['trend']} | ML Prob: {result['score']:.2%}")
        return result

from .orderflow_analyst import OrderFlowAnalyst

class AnalystTeam:
    def __init__(self):
        self.fundamental = FundamentalAnalyst()
        self.sentiment = SentimentAnalyst()
        self.technical = TechnicalAnalyst()
        self.orderflow = OrderFlowAnalyst()
    
    def analyze_all(self, symbol: str, market_data: Any) -> Dict[str, Any]:
        logger.info(f"🚀 Iniciando rodada de análise completa para {symbol}")
        
        f_report = self.fundamental.analyze(symbol, market_data)
        s_report = self.sentiment.analyze(symbol, market_data)
        t_report = self.technical.analyze(symbol, market_data)
        of_report = self.orderflow.analyze(symbol, market_data)
        
        return {
            "fundamental": f_report,
            "sentiment": s_report,
            "technical": t_report,
            "orderflow": of_report,
            "timestamp": pd.Timestamp.now()
        }
