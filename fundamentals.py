# fundamentals.py
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict
import MetaTrader5 as mt5
import pandas as pd
import config

logger = logging.getLogger("fundamentals")

# Limites configuráveis para filtro de tradeabilidade (MT5-only)
FUNDAMENTAL_LIMITS = {"min_avg_tick_volume": 1.0, "min_atr_pct": 0.005}


class FundamentalFetcher:
    """
    Coleta métricas essenciais via MT5 (sem dependência do Yahoo).
    Usa cache leve para evitar leituras repetidas quando possível.
    """

    def __init__(self, cache_file="fundamentals_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.cache_validity_minutes = 60

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar cache fundamentista: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Erro ao salvar cache fundamentista: {e}")

    def get_fundamentals(self, symbol: str) -> dict:
        now = datetime.now()
        cache_key = symbol
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            updated_str = entry.get("updated_at")
            if updated_str:
                try:
                    updated_at = datetime.strptime(updated_str, "%Y-%m-%d %H:%M:%S")
                    if (now - updated_at) <= timedelta(
                        minutes=self.cache_validity_minutes
                    ):
                        return entry.get("data", {})
                except Exception:
                    pass
            else:
                data = entry.get("data")
                if isinstance(data, dict) and data:
                    self.cache[cache_key] = {
                        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "data": data,
                    }
                    self._save_cache()
                    return data

        mt5_symbol = symbol.replace(".SA", "") if symbol.endswith(".SA") else symbol
        
        # Fallback dictionary for no data
        empty_data = {"mt5_bars": 0, "mt5_avg_tick_volume": 0.0, "mt5_atr_pct": 0.0}

        try:
            # Usa H1 invés de D1 pois o cache local na B3 atualiza H1/M15 muito mais rapido, e
            # a função safe garante retry e delay backoff pra carregar de verdade os dados sem retornar None
            df = utils.safe_copy_rates(mt5_symbol, mt5.TIMEFRAME_H1, 200)
            
            if df is None or df.empty or len(df) < 5:
                # Log isolado para rastreio
                logger.warning(f"⚠️ {mt5_symbol}: Sem dados MT5 (bars=0) apos safe_copy_rates")
                data = empty_data
            else:
                avg_vol = (
                    float(df["tick_volume"].mean())
                    if "tick_volume" in df.columns
                    else float(df["volume"].mean()) if "volume" in df.columns else 0.0
                )
                hl_range = (
                    (df["high"] - df["low"]).mean()
                    if {"high", "low"}.issubset(df.columns)
                    else 0.0
                )
                close_mean = df["close"].mean() if "close" in df.columns else 0.0
                atr_pct = float(hl_range / close_mean) if close_mean > 0 else 0.0
                data = {
                    "mt5_bars": int(len(df)),
                    "mt5_avg_tick_volume": avg_vol,
                    "mt5_atr_pct": atr_pct,
                }

            self.cache[cache_key] = {
                "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "data": data,
            }
            self._save_cache()
            return data
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas MT5 para {mt5_symbol}: {e}")
            return empty_data

    def check_tradeability(self, symbol: str) -> Tuple[bool, str]:
        data = self.get_fundamentals(symbol)
        avg_vol = data.get("mt5_avg_tick_volume", 0.0)
        atr_pct = data.get("mt5_atr_pct", 0.0)
        bars = data.get("mt5_bars", 0)
        if bars <= 0:
            return False, "Sem histórico no MT5"
        if avg_vol < FUNDAMENTAL_LIMITS["min_avg_tick_volume"]:
            return False, "Sem liquidez MT5"
        if atr_pct < FUNDAMENTAL_LIMITS["min_atr_pct"]:
            return False, "Volatilidade insuficiente MT5"
        return True, f"OK MT5 (vol:{avg_vol:.0f} atr%:{atr_pct:.3f})"

    def is_fundamentally_sound(self, symbol: str) -> bool:
        """
        Retorna True se o ativo passa nos filtros fundamentalistas.
        """
        tradeable, _ = self.check_tradeability(symbol)
        return tradeable

    def get_fundamental_score(self, symbol: str) -> float:
        data = self.get_fundamentals(symbol)
        avg_vol = data.get("mt5_avg_tick_volume", 0.0)
        atr_pct = data.get("mt5_atr_pct", 0.0)
        bars = data.get("mt5_bars", 0)
        s = 40.0
        if bars < 100:
            s -= 10
        if avg_vol > 1000:
            s += 25
        elif avg_vol > 200:
            s += 10
        else:
            s -= 10
        if atr_pct > 0.02:
            s += 25
        elif atr_pct > 0.01:
            s += 10
        else:
            s -= 10
        return max(0, min(100, s))


# Instância global
fundamental_fetcher = FundamentalFetcher()

if __name__ == "__main__":
    # Teste rápido
    logging.basicConfig(level=logging.INFO)

    for sym in ["PETR4", "VALE3", "ITUB4", "MGLU3"]:
        data = fundamental_fetcher.get_fundamentals(sym)
        tradeable, reason = fundamental_fetcher.check_tradeability(sym)
        score = fundamental_fetcher.get_fundamental_score(sym)

        print(f"\n{sym}:")
        print(
            f"  MT5 barras: {data.get('mt5_bars', 0)} | vol médio: {data.get('mt5_avg_tick_volume', 0.0):.0f} | ATR%: {data.get('mt5_atr_pct', 0.0):.3f}"
        )
        print(f"  Tradeable: {tradeable} ({reason})")
        print(f"  Score: {score:.0f}/100")
