
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("AnalystTeam")

class Analyst:
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class FundamentalAnalyst(Analyst):
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX: usa métricas MT5 reais (avg_tick_volume, atr_pct, bars)
        em vez de market_cap que MT5 nunca retorna (sempre 0 → score sempre 0.40).
        FIX2: quando MT5 retorna bars=0 (sem conexão), retorna NEUTRO em vez de
        penalizar 3x gerando score 0.25 'expensive' (bear falso por conectividade).
        """
        import utils
        
        logger.info(f"🔎 [Fundamental] Analisando métricas de liquidez/vol de {symbol}...")
        
        valuation = "neutral"
        score = 0.50  # base neutra
        risks = []
        drivers = []
        fund = {}
        
        try:
            fund = utils.fundamental_fetcher.get_fundamentals(symbol)
            avg_vol = fund.get("mt5_avg_tick_volume", 0.0)
            atr_pct = fund.get("mt5_atr_pct", 0.0)
            bars    = fund.get("mt5_bars", 0)
            
            # --- CRITICO: Sem dados MT5 → retorna NEUTRO, nunca penaliza ---
            # Falha de conectividade/símbolo não selecionado NÃO é sinal bearish!
            # Antes: bars=0 → penalizava 3x → score 0.25 → 'expensive' → bear+1.0 (falso!)
            if bars <= 0:
                logger.info(f"   ↳ Sem dados MT5 para {symbol} (bars=0) → Neutro por default")
                return {
                    "type": "fundamental",
                    "score": 0.50,
                    "valuation": "neutral",
                    "risks": ["no_mt5_data"],
                    "drivers": [],
                    "details": fund
                }
            
            # --- Liquidez (tick volume médio diário — escala B3 dia: 10-10000) ---
            # Blue chips (PETR4, VALE3, ITUB4): avg_vol >> 2000
            # Mid caps: 300-2000 | Small caps: < 300
            if avg_vol > 2000:
                score += 0.15
                drivers.append("high_liquidity")
            elif avg_vol > 300:
                score += 0.07
                drivers.append("moderate_liquidity")
            elif avg_vol > 50:
                pass   # liquidez mínima aceitável → sem ajuste
            else:
                score -= 0.08   # penalidade reduzida (antes -0.10)
                risks.append("low_liquidity")
            
            # --- Volatilidade (ATR% = (H-L)/Close médio diário) ---
            # B3 típico: 0.010 a 0.030 (1-3% ao dia)
            if atr_pct > 0.030:    # > 3%
                score += 0.05
                drivers.append("high_volatility")
            elif atr_pct > 0.008:  # > 0.8% → normal para B3
                score += 0.08
                drivers.append("healthy_volatility")
            elif atr_pct > 0.002:  # > 0.2% → aceitável
                pass               # sem ajuste (antes penalizava isso — errado)
            else:
                score -= 0.08
                risks.append("low_volatility")
            
            # --- Histórico disponível ---
            if bars >= 200:
                score += 0.05
                drivers.append("solid_history")
            elif bars < 30:
                score -= 0.05  # penalidade leve (antes -0.10, agressivo demais)
                risks.append("short_history")
            
            # --- Contexto setorial via config ---
            import config as cfg
            sector = cfg.SECTOR_MAP.get(symbol, "OUTROS")
            if sector in ("FINANCEIRO", "PETROLEO", "MINERACAO", "UTILIDADE_PUBLICA"):
                score += 0.05
                drivers.append("resilient_sector")
            elif sector in ("VAREJO", "EDUCACAO"):
                score -= 0.03   # penalidade reduzida (antes -0.05)
                risks.append("rate_sensitive_sector")
            
            score = max(0.15, min(0.90, score))
            
        except Exception as e:
            logger.warning(f"Erro na análise fundamentalista: {e}")
            return {
                "type": "fundamental",
                "score": 0.50,
                "valuation": "neutral",
                "risks": ["analysis_error"],
                "drivers": [],
                "details": fund
            }
        
        # --- Mapeamento semântico do score ---
        # 'fair' agora a partir de 0.52 (antes 0.55 — restritivo demais)
        if score >= 0.65:
            valuation = "cheap"       # forte → bullish
        elif score >= 0.52:
            valuation = "fair"        # levemente positivo
        elif score >= 0.40:
            valuation = "neutral"
        else:
            valuation = "expensive"   # sinal negativo real (não por falta de dados)
        
        result = {
            "type": "fundamental",
            "score": score,
            "valuation": valuation,
            "risks": risks,
            "drivers": drivers,
            "details": fund
        }
        logger.info(f"   ↳ Valuation: {valuation} | Score: {score:.2f} | Drivers: {drivers} | Risks: {risks}")
        return result

class SentimentAnalyst(Analyst):
    def analyze(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIX: usa ibov_trend real do market_data em vez de random.uniform.
        Random é ruído puro — não contribui para sinal algum.
        """
        logger.info(f"🐦 [Sentiment] Avaliando sentimento de mercado para {symbol}...")
        
        # Mapa de tendência do IBOV → score de sentimento
        ibov_trend = data.get("ibov_trend", "neutral")
        trend_to_score = {
            "bullish":        0.72,
            "neutral":        0.52,
            "bearish":        0.32,
            "bearish_extreme": 0.18,
        }
        score = trend_to_score.get(ibov_trend, 0.52)
        
        # Pequeno ajuste se IBOV está em alta e o ativo tem alta correlação setorial
        # (heurística simples — melhora quando tivermos dados de correlação)
        
        sentiment = "neutral"
        if score >= 0.65:
            sentiment = "optimistic"
        elif score < 0.40:
            sentiment = "pessimistic"
        
        result = {
            "type": "sentiment",
            "score": score,
            "sentiment": sentiment,
            "sources": ["ibov_trend"],
            "ibov_trend_used": ibov_trend
        }
        logger.info(f"   ↳ Sentiment: {sentiment} | Score: {score:.2f} | IBOV: {ibov_trend}")
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
