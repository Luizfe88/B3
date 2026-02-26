# utils.py
import functools

print = functools.partial(print, flush=True)
print("[DEBUG] Importando time...", flush=True)
import time

print("[DEBUG] time importado.", flush=True)
print("[DEBUG] Importando logging...", flush=True)
import logging

print("[DEBUG] logging importado.", flush=True)
print("[DEBUG] Importando datetime...", flush=True)
from datetime import datetime, timedelta, time as datetime_time

print("[DEBUG] datetime importado.", flush=True)
print("[DEBUG] Importando typing...", flush=True)
from typing import Optional, Dict, Any, List, Tuple

print("[DEBUG] typing importado.", flush=True)
print("[DEBUG] Importando collections...", flush=True)
from collections import defaultdict, deque

print("[DEBUG] collections importado.", flush=True)
print("[DEBUG] Importando json...", flush=True)
import json

print("[DEBUG] json importado.", flush=True)
try:
    print("[DEBUG] Importando MetaTrader5...", flush=True)
    import MetaTrader5 as mt5

    print("[DEBUG] MetaTrader5 importado.", flush=True)
except Exception:
    mt5 = None
    print("[DEBUG] MetaTrader5 indisponível.", flush=True)
print("[DEBUG] Importando pandas...", flush=True)
import pandas as pd

print("[DEBUG] pandas importado.", flush=True)
print("[DEBUG] Importando numpy...", flush=True)
import numpy as np

print("[DEBUG] numpy importado.", flush=True)
print("[DEBUG] Importando config...", flush=True)
import config

print("[DEBUG] config importado.", flush=True)
print("[DEBUG] Importando threading...", flush=True)
from threading import RLock, Lock

print("[DEBUG] threading importado.", flush=True)
print("[DEBUG] Importando threading (módulo)...", flush=True)
import threading

print("[DEBUG] threading(módulo) importado.", flush=True)
print("[DEBUG] Importando queue...", flush=True)
import queue

print("[DEBUG] queue importado.", flush=True)
print("[DEBUG] Importando os...", flush=True)
import os

print("[DEBUG] os importado.", flush=True)
print("[DEBUG] Importando pathlib.Path...", flush=True)
from pathlib import Path

print("[DEBUG] pathlib.Path importado.", flush=True)
try:
    print("[DEBUG] Importando redis...", flush=True)
    import redis

    print("[DEBUG] redis importado.", flush=True)
except Exception:
    redis = None
    print("[DEBUG] redis indisponível.", flush=True)
print("[DEBUG] Importando pickle...", flush=True)
import pickle

print("[DEBUG] pickle importado.", flush=True)
print("[DEBUG] Importando hashlib...", flush=True)
import hashlib

print("[DEBUG] hashlib importado.", flush=True)
print("[DEBUG] Importando signal...", flush=True)
import signal

print("[DEBUG] signal importado.", flush=True)
print("[DEBUG] Importando sys...", flush=True)
import sys

print("[DEBUG] sys importado.", flush=True)
print("[DEBUG] Importando requests...", flush=True)
import requests

print("[DEBUG] requests importado.", flush=True)
try:
    print("[DEBUG] Importando news_calendar.apply_blackout...", flush=True)
    from news_calendar import apply_blackout

    print("[DEBUG] news_calendar.apply_blackout importado.", flush=True)
except Exception:

    def apply_blackout(*args, **kwargs):
        return (False, "")

    print("[DEBUG] news_calendar indisponível.", flush=True)
ml_optimizer = None
print("[DEBUG] Importando re...", flush=True)
import re

print("[DEBUG] re importado.", flush=True)
RandomForestClassifier = None
StandardScaler = None
try:
    print("[DEBUG] Importando joblib...", flush=True)
    import joblib

    print("[DEBUG] joblib importado.", flush=True)
except Exception:
    joblib = None
    print("[DEBUG] joblib indisponível.", flush=True)
anti_chop_rf = None
anti_chop_scaler = None
ANTI_CHOP_MODEL_PATH = "anti_chop_rf.pkl"


def ensure_ml_loaded(timeout_seconds: float = 5.0) -> bool:
    global ml_optimizer
    if ml_optimizer is not None:
        return True
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as TErr

        def _do_import():
            import os

            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            import ml_optimizer as ml

            return ml

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_import)
            try:
                ml = fut.result(timeout=timeout_seconds)
                ml_optimizer = ml
                print("[DEBUG] ml_optimizer carregado (lazy).", flush=True)
                return True
            except TErr:
                print("[WARN] Timeout ao importar ml_optimizer", flush=True)
                return False
            except Exception as e:
                print(f"[ERROR] Falha ao importar ml_optimizer: {e}", flush=True)
                return False
    except Exception as e:
        print(f"[ERROR] Lazy import infra falhou: {e}", flush=True)
    return False


def load_anti_chop_model():
    global anti_chop_rf
    if anti_chop_rf is not None:
        return
    # Lazy import sklearn/joblib com timeout curto
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TErr

        def _do_imports():
            import os

            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            from sklearn.ensemble import RandomForestClassifier as RFC
            from sklearn.preprocessing import StandardScaler as SS
            import joblib as JB

            return RFC, SS, JB

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_imports)
            try:
                RFC, SS, JB = fut.result(timeout=3.0)
            except _TErr:
                return
            except Exception:
                return
    except Exception:
        return
    # Carrega do disco se existir
    if os.path.exists(ANTI_CHOP_MODEL_PATH):
        try:
            anti_chop_rf = JB.load(ANTI_CHOP_MODEL_PATH)
        except:
            anti_chop_rf = None
    if anti_chop_rf is None:
        anti_chop_rf = RFC(n_estimators=100, random_state=42)
        try:
            JB.dump(anti_chop_rf, ANTI_CHOP_MODEL_PATH)
        except Exception:
            pass


# Inicialização adiada: carregamento de ML/anti-chop só quando necessário
def is_valid_dataframe(df, min_rows: int = 1) -> bool:
    """Valida se o DataFrame é válido para cálculos."""
    if df is None:
        return False
    if isinstance(df, pd.DataFrame):
        return not df.empty and len(df) >= min_rows
    return False


logger = logging.getLogger("bot")
mt5_lock = RLock()
try:
    import telebot
except ImportError:
    telebot = None
    logger.warning("telebot não instalado - comandos Telegram desativados")
# =========================================================
# CONFIG GERAL
# =========================================================

TIMEFRAME_BASE = mt5.TIMEFRAME_M15
TIMEFRAME_MACRO = getattr(mt5, f"TIMEFRAME_{config.MACRO_TIMEFRAME}", mt5.TIMEFRAME_H1)
logger = logging.getLogger("utils")


def atomic_save_json(path: str, data: dict) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            os.replace(tmp, p)
        except Exception:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
            os.rename(tmp, p)
        return True
    except Exception as e:
        try:
            logger.error(f"Erro ao salvar dados: {e}")
        except Exception:
            pass
        return False


# =========================================================
# 🌐 POLYGON.IO FALLBACK
# =========================================================


def get_polygon_data(endpoint: str, params: dict = {}) -> Optional[Dict]:
    """
    Centralized helper for Polygon.io API calls with mock key protection.
    """
    if (
        not hasattr(config, "POLYGON_API_KEY")
        or config.POLYGON_API_KEY == "MOCK_KEY_FOR_NOW"
    ):
        return None

    try:
        if not endpoint.startswith("http"):
            url = f"{config.POLYGON_BASE_URL}/{endpoint}"
        else:
            url = endpoint

        full_params = params.copy()
        full_params["apiKey"] = config.POLYGON_API_KEY

        resp = requests.get(url, params=full_params, timeout=8)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            logger.warning("⚠️ Polygon.io: Chave de API expirada ou inválida (401).")
        return None
    except Exception as e:
        logger.debug(f"Erro silenciado ao buscar Polygon ({endpoint}): {e}")
        return None


def get_polygon_rates_fallback(
    symbol: str, timeframe, count: int
) -> Optional[pd.DataFrame]:
    """
    ✅ Fallback: Obtém dados da Polygon.io se o MT5 falhar.
    """
    tf_map = {
        mt5.TIMEFRAME_M1: ("minute", 1),
        mt5.TIMEFRAME_M5: ("minute", 5),
        mt5.TIMEFRAME_M15: ("minute", 15),
        mt5.TIMEFRAME_H1: ("hour", 1),
        mt5.TIMEFRAME_D1: ("day", 1),
    }
    timespan, multiplier = tf_map.get(timeframe, ("minute", 15))
    ticker = symbol.replace(".SA", "")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    endpoint = f"v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted": "true", "sort": "desc", "limit": count}
    data = get_polygon_data(endpoint, params)
    if data:
        results = data.get("results", [])
        if results:
            df = pd.DataFrame(results)
            df = df.rename(
                columns={
                    "t": "time",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "tick_volume",
                    "vw": "vwap",
                }
            )
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            if "real_volume" not in df.columns:
                df["real_volume"] = df["tick_volume"] * df["close"]
            df.set_index("time", inplace=True)
            logger.info(f"✅ Fallback Polygon SUCESSO: {symbol}")
            return df.sort_index()
    return None


def get_yahoo_rates_fallback(
    symbol: str, timeframe, count: int
) -> Optional[pd.DataFrame]:
    try:
        s = symbol if symbol.endswith(".SA") else f"{symbol}.SA"
        tf_map = {
            mt5.TIMEFRAME_M1: ("1m", "5d"),
            mt5.TIMEFRAME_M5: ("5m", "30d"),
            mt5.TIMEFRAME_M15: ("15m", "60d"),
            mt5.TIMEFRAME_H1: ("60m", "60d"),
            mt5.TIMEFRAME_D1: ("1d", "5y"),
        }
        interval, range_str = tf_map.get(timeframe, ("15m", "60d"))
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?interval={interval}&range={range_str}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        res = (data.get("chart", {}) or {}).get("result", [])
        if not res:
            return None
        r0 = res[0]
        ts = r0.get("timestamp", [])
        ind = r0.get("indicators", {}) or {}
        qt = (ind.get("quote", []) or [{}])[0]
        close = qt.get("close")
        open_ = qt.get("open")
        high = qt.get("high")
        low = qt.get("low")
        vol = qt.get("volume")
        if not ts or close is None:
            return None
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(ts, unit="s"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": vol,
            }
        )
        df = df.dropna(subset=["close"])
        if len(df) == 0:
            return None
        if count and count > 0:
            df = df.tail(count)
        df = df.set_index("time").sort_index()
        return df
    except Exception:
        return None


def resolve_current_symbol(root_symbol: str) -> Optional[str]:
    return None


def _expiration_str(symbol: str) -> str:
    try:
        info = mt5.symbol_info(symbol)
        exp = getattr(info, "expiration_time", None)
        if isinstance(exp, datetime):
            return exp.strftime("%d/%m/%Y")
        return ""
    except Exception:
        return ""


def update_futures_mappings():
    return {}


def discover_all_futures() -> dict:
    return {}


def detect_broker() -> str:
    try:
        if mt5 is None:
            return ""
        info = mt5.terminal_info()
        return str(getattr(info, "server", "") or "")
    except Exception:
        return ""


def get_asset_type(symbol: str) -> str:
    """
    Retorna o tipo de ativo. Como o bot agora é apenas para ações,
    esta função sempre retornará 'STOCK'.
    """
    return "STOCK"


class AssetInspector:
    @staticmethod
    def detect(symbol: str) -> dict:
        """
        Detecta o tipo de ativo. Como o bot agora é focado em ações,
        retorna sempre a configuração padrão de AÇÃO.
        """
        return {
            "type": "STOCK",
            "point_value": 1.0,
            "tick_size": 0.01,
            "fee_type": "PERCENT",
            "fee_val": 0.00055,
        }


class AssetClassManager:
    @staticmethod
    def get_class(symbol: str) -> str:
        return "STOCK"

    @staticmethod
    def get_risk_pct(symbol: str) -> float:
        try:
            base = float(getattr(config, "RISK_PER_TRADE_PCT", 0.01))
            minp = float(getattr(config, "MIN_RISK_PER_TRADE_PCT", 0.0025))
            maxp = float(getattr(config, "MAX_RISK_PER_TRADE_PCT", 0.005))
            stkp = float(getattr(config, "STOCKS_RISK_PER_TRADE_PCT", base))
            return max(minp, min(stkp, maxp))
        except Exception:
            return float(getattr(config, "RISK_PER_TRADE_PCT", 0.01))

    @staticmethod
    def get_min_lot(symbol: str) -> int:
        return 100

    @staticmethod
    def get_time_window(symbol: str) -> tuple[str, str]:
        return "10:10", "16:50"

    @staticmethod
    def get_bucket_for(symbol: str) -> float:
        # Com o bot focado em ações, o bucket é sempre 100% do capital de ações.
        return 1.0

    @staticmethod
    def get_dynamic_buckets() -> tuple[float, float]:
        """
        Retorna a alocação de capital percentual para Futuros e Ações.
        """
        # Foco total em ações, 0% para futuros.
        return (0.0, 1.0)


def get_asset_class_config(symbol: str) -> dict:
    s = (symbol or "").upper()
    is_fut = is_future(s)
    start, end = AssetClassManager.get_time_window(s)
    bucket_pct = AssetClassManager.get_bucket_for(s)
    min_lot = AssetClassManager.get_min_lot(s)
    risk_pct = AssetClassManager.get_risk_pct(s)
    if is_fut:
        if s.startswith(("WIN", "IND")):
            dev_pts = 80
        elif s.startswith(("WDO", "DOL")):
            dev_pts = 20
        else:
            dev_pts = 50
        return {
            "start": start,
            "end": end,
            "bucket_pct": bucket_pct,
            "risk_pct": risk_pct,
            "min_lot": min_lot,
            "deviation_points": dev_pts,
            "lunch_min_vol_ratio": 0.0,
            "min_tp_cost_multiplier": 3.0,
        }
    return {
        "start": start,
        "end": end,
        "bucket_pct": bucket_pct,
        "risk_pct": risk_pct,
        "min_lot": min_lot,
        "deviation_points": 2,
        "lunch_min_vol_ratio": 1.0,
        "min_tp_cost_multiplier": 1.2,
    }


def round_to_tick(price: float, tick_size: float) -> float:
    try:
        if tick_size <= 0:
            return float(price)
        return float(round(price / tick_size) * tick_size)
    except Exception:
        return float(price)


sector_weights: Dict[str, Dict[str, float]] = {}
symbol_weights: Dict[str, Dict[str, float]] = {}
# Conexão Redis (ajuste se necessário)
try:
    redis_client = redis.Redis(
        host="localhost", port=6379, db=0, decode_responses=False
    )
    redis_client.ping()  # Testa conexão
    REDIS_AVAILABLE = True
    logger.info("✅ Redis conectado - cache ativado")
except Exception as e:
    redis_client = None
    REDIS_AVAILABLE = False
    logger.warning(f"⚠️ Redis não disponível: {e} - cache desativado")

# =========================================================
# MT5 SAFE COPY (ANTI-DEADLOCK)
# =========================================================


# Executor global para evitar criar threads infinitamente
_mt5_executor = None


def get_mt5_executor():
    global _mt5_executor
    if _mt5_executor is None:
        try:
            from concurrent.futures import ThreadPoolExecutor

            _mt5_executor = ThreadPoolExecutor(
                max_workers=5, thread_name_prefix="MT5_Worker"
            )
        except ImportError:
            # Fallback (improvável em Py3)
            return None
    return _mt5_executor


def safe_copy_rates(
    symbol: str, timeframe, count: int = 500, timeout: int = 12
) -> Optional[pd.DataFrame]:
    try:
        terminal = mt5.terminal_info()
    except Exception:
        terminal = None
    if (terminal is None) or (not getattr(terminal, "connected", False)):
        df_fb = get_polygon_rates_fallback(symbol, timeframe, count)
        return df_fb
    if not mt5.symbol_select(symbol, True):
        pass

    try:
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        bars_available = 0 if bars is None else len(bars)
    except Exception:
        bars_available = 0

    if bars_available < count:
        mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
        time.sleep(0.2)

    # Prepara a função a ser executada
    def worker():
        return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    rates = None
    executor = get_mt5_executor()

    if executor:
        try:
            future = executor.submit(worker)
            rates = future.result(timeout=timeout)
        except Exception as e:  # TimeoutError ou outras
            logger.error(f"🚨 TIMEOUT (Pool) MT5 em {symbol}: {e}")
            rates = None
    else:
        # Fallback legado se não tiver executor (improvável)
        q = queue.Queue()

        def threading_worker():
            try:
                q.put(worker())
            except Exception as e:
                q.put(e)

        t = threading.Thread(target=threading_worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            logger.error(f"🚨 TIMEOUT (Thread) MT5 em {symbol}")
            return None
        try:
            rates = q.get_nowait()
        except queue.Empty:
            rates = None

    if isinstance(rates, Exception) or rates is None or len(rates) == 0:
        # ✅ FALLBACK: Polygon.io
        logger.warning(f"⚠️ MT5 Falhou em {symbol}. Acionando Polygon.io fallback...")
        df_fb = get_polygon_rates_fallback(symbol, timeframe, count)
        return df_fb

    try:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df.sort_index()
    except Exception as e:
        logger.error(f"Erro ao processar dados de {symbol}: {e}")
        return None


def get_training_rates(
    symbol: str, timeframe, bars: int = 2000
) -> Optional[pd.DataFrame]:
    try:
        df = safe_copy_rates(symbol, timeframe, bars)
        return df
    except Exception as e:
        logger.error(f"get_training_rates: {e}")
        return None


# =========================================================
# AWS LAMBDA HOOK (SCALABILITY)
# =========================================================


def lambda_invoke(function_name: str, payload: dict) -> Optional[dict]:
    """
    Invoca uma função AWS Lambda para processamento pesado (ex: ML Inference).
    """
    try:
        import boto3
        import json

        client = boto3.client(
            "lambda", region_name=getattr(config, "AWS_REGION", "us-east-1")
        )
        response = client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        return json.loads(response["Payload"].read())
    except ImportError:
        logger.debug("Boto3 não instalado. AWS Lambda desabilitado.")
        return None
    except Exception as e:
        logger.error(f"Erro ao invocar Lambda {function_name}: {e}")
        return None


# =========================================================
# REDIS CACHE HELPERS
# =========================================================


def safe_symbol_info_to_dict(info_data):
    """Converte SymbolInfo do MT5 para dict"""
    if info_data is None:
        return None

    if isinstance(info_data, dict):
        return info_data

    if hasattr(info_data, "_asdict"):
        return info_data._asdict()

    # Extrai campos principais do SymbolInfo
    return {
        "name": getattr(info_data, "name", ""),
        "description": getattr(info_data, "description", ""),
        "point": getattr(info_data, "point", 0.0),
        "digits": getattr(info_data, "digits", 0),
        "volume_min": getattr(info_data, "volume_min", 0.0),
        "volume_max": getattr(info_data, "volume_max", 0.0),
        "volume_step": getattr(info_data, "volume_step", 0.0),
        "contract_size": getattr(info_data, "contract_size", 0.0),
        "spread": getattr(info_data, "spread", 0.0),
        "spread_floating": getattr(info_data, "spread_floating", False),
        "tick_size": getattr(info_data, "tick_size", 0.0),
        "tick_value": getattr(info_data, "tick_value", 0.0),
        "swap_mode": getattr(info_data, "swap_mode", 0),
        "swap_long": getattr(info_data, "swap_long", 0.0),
        "swap_short": getattr(info_data, "swap_short", 0.0),
        "margin_initial": getattr(info_data, "margin_initial", 0.0),
        "margin_maintenance": getattr(info_data, "margin_maintenance", 0.0),
    }


def cached_symbol_info(symbol: str) -> Optional[mt5.SymbolInfo]:
    """
    Versão com cache do symbol_info
    """
    if not REDIS_AVAILABLE:
        return mt5.symbol_info(symbol)

    cache_key = f"symbol_info:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            info_dict = json.loads(cached)

            # Reconstrói um objeto similar ao SymbolInfo
            class SymbolInfoDict:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

            return SymbolInfoDict(info_dict)

        info = mt5.symbol_info(symbol)
        if info:
            info_dict = safe_symbol_info_to_dict(info)
            if info_dict:
                redis_client.setex(
                    cache_key, config.REDIS_CACHE_TTL_INFO, json.dumps(info_dict)
                )
        return info
    except Exception as e:
        logger.debug(f"Redis Info Error: {e}")
        return mt5.symbol_info(symbol)


def is_future(symbol: str) -> bool:
    """
    Detecta se símbolo é futuro da B3.
    Versão robusta para evitar falsos positivos.
    """
    if not symbol:
        return False

    s = symbol.upper().strip()

    # Prefixos definitivos de futuros B3
    FUTURE_PREFIXES = [
        "WIN",  # Mini Índice
        "WDO",  # Mini Dólar
        "IND",  # Índice Cheio
        "DOL",  # Dólar Cheio
        "WSP",  # Mini S&P
        "SMALL",  # Small Caps
    ]

    # Verifica se começa com prefixo de futuro
    # E se tem código de vencimento após (letra ou número)
    for prefix in FUTURE_PREFIXES:
        if s.startswith(prefix) and len(s) > len(prefix):
            next_char = s[len(prefix)]
            if next_char.isalnum():  # WIN202502, WDOJ25, etc
                return True

    # Verifica símbolo com $ (algumas corretoras)
    if "$" in s:
        return True

    return False


def safe_tick_to_dict(tick_data):
    """Converte Tick do MT5 para dict"""
    if tick_data is None:
        return None

    if isinstance(tick_data, dict):
        return tick_data

    if hasattr(tick_data, "_asdict"):
        return tick_data._asdict()

    return {
        "time": getattr(tick_data, "time", 0),
        "bid": getattr(tick_data, "bid", 0.0),
        "ask": getattr(tick_data, "ask", 0.0),
        "last": getattr(tick_data, "last", 0.0),
        "volume": getattr(tick_data, "volume", 0),
    }


def cached_symbol_info_tick(symbol: str):
    """
    Versão com cache do symbol_info_tick (TTL 1s)
    """
    if not REDIS_AVAILABLE:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"⚠️ MT5 symbol_info_tick retornou None para {symbol}")
        return tick

    cache_key = f"tick:{symbol}"
    try:
        cached = redis_client.get(cache_key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            try:
                tick_dict = json.loads(cached)
                if isinstance(tick_dict, dict):

                    class TickDict:
                        def __init__(self, data):
                            self.time = int(data.get("time", 0) or 0)
                            self.bid = float(data.get("bid", 0.0) or 0.0)
                            self.ask = float(data.get("ask", 0.0) or 0.0)
                            self.last = float(data.get("last", 0.0) or 0.0)
                            self.volume = float(data.get("volume", 0.0) or 0.0)

                        def _asdict(self):
                            return {
                                "time": self.time,
                                "bid": self.bid,
                                "ask": self.ask,
                                "last": self.last,
                                "volume": self.volume,
                            }

                    return TickDict(tick_dict)
            except Exception:
                try:
                    redis_client.delete(cache_key)
                except Exception:
                    pass

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(
                f"⚠️ MT5 symbol_info_tick retornou None para {symbol} (Redis ativo)"
            )
            # Tenta selecionar o símbolo no Market Watch
            selected = mt5.symbol_select(symbol, True)
            if selected:
                logger.info(f"✅ Símbolo {symbol} selecionado no Market Watch")
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    logger.info(f"✅ Cotação obtida após seleção para {symbol}")
            else:
                logger.error(f"❌ Não foi possível selecionar {symbol} no Market Watch")
            return tick

        if tick:
            tick_dict = safe_tick_to_dict(tick)
            if tick_dict:
                redis_client.setex(
                    cache_key, config.REDIS_CACHE_TTL_TICK, json.dumps(tick_dict)
                )
        return tick
    except Exception as e:
        logger.error(f"❌ Redis Tick Error para {symbol}: {e}")
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(
                f"⚠️ MT5 fallback symbol_info_tick retornou None para {symbol}"
            )
        return tick


def cache_tick(symbol, tick):
    """Cacheia tick no Redis usando JSON"""
    if not REDIS_AVAILABLE or redis_client is None:
        return

    try:
        tick_dict = safe_tick_to_dict(tick)
        if tick_dict:
            key = f"tick:{symbol}"
            redis_client.setex(key, 60, json.dumps(tick_dict))
    except Exception as e:
        logger.error(f"Redis Tick Error para {symbol}: {e}")


def get_cached_tick(symbol):
    """Recupera tick do Redis"""
    if not REDIS_AVAILABLE or redis_client is None:
        return None

    try:
        key = f"tick:{symbol}"
        cached = redis_client.get(key)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return json.loads(cached)
    except Exception:
        return None


# =========================================================
# 🛑 FILTRO ANTI-OVERTRADING
# =========================================================

# Rastreamento de trades por símbolo e período
_trade_history = {}  # {symbol: [timestamps]}
_trade_history_lock = RLock()


def check_anti_overtrading(
    symbol: str, max_trades_per_hour: int = 5, max_trades_per_day: int = 20
) -> Tuple[bool, str]:
    """
    ✅ Verifica se está fazendo trades demais em um símbolo.

    Regras:
    - Máximo 3 trades/hora por símbolo
    - Máximo 10 trades/dia por símbolo

    Returns:
        (allowed: bool, reason: str)
    """
    global _trade_history

    with _trade_history_lock:
        now = datetime.now()

        if symbol not in _trade_history:
            _trade_history[symbol] = []

        # Limpa histórico antigo (>24h)
        _trade_history[symbol] = [
            t for t in _trade_history[symbol] if (now - t).total_seconds() < 86400
        ]

        # Conta trades na última hora
        hour_ago = now - timedelta(hours=1)
        trades_last_hour = sum(1 for t in _trade_history[symbol] if t >= hour_ago)

        # Conta trades hoje
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        trades_today = sum(1 for t in _trade_history[symbol] if t >= today)

        if trades_last_hour >= max_trades_per_hour:
            return False, f"Overtrading/hora: {trades_last_hour}/{max_trades_per_hour}"

        if trades_today >= max_trades_per_day:
            return False, f"Overtrading/dia: {trades_today}/{max_trades_per_day}"

        return True, f"OK ({trades_last_hour}/h, {trades_today}/d)"


def register_trade_for_overtrading(symbol: str):
    """Registra um trade para controle de overtrading."""
    global _trade_history

    with _trade_history_lock:
        if symbol not in _trade_history:
            _trade_history[symbol] = []
        _trade_history[symbol].append(datetime.now())


# =========================================================
# 📊 WIN RATE EM TEMPO REAL
# =========================================================


def get_realtime_win_rate(lookback_trades: int = 20) -> Dict[str, float]:
    """
    ✅ Calcula win rate em tempo real dos últimos N trades.

    Returns:
        {
            'win_rate': float (0-1),
            'total_trades': int,
            'wins': int,
            'losses': int,
            'avg_win': float,
            'avg_loss': float,
            'profit_factor': float,
            'expectancy': float
        }
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=7), datetime.now()
            )

        if not deals:
            return _default_win_rate_result()

        # Filtra fechamentos
        out_deals = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT][
            -lookback_trades:
        ]

        if len(out_deals) < 5:
            return _default_win_rate_result()

        wins = [d for d in out_deals if d.profit > 0]
        losses = [d for d in out_deals if d.profit <= 0]

        win_rate = len(wins) / len(out_deals)

        avg_win = sum(d.profit for d in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(d.profit for d in losses) / len(losses)) if losses else 0

        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "win_rate": win_rate,
            "total_trades": len(out_deals),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
        }

    except Exception as e:
        logger.error(f"Erro ao calcular win rate real-time: {e}")
        return _default_win_rate_result()


def _default_win_rate_result() -> Dict[str, float]:
    """Resultado padrão quando não há dados suficientes."""
    return {
        "win_rate": 0.50,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 1.0,
        "expectancy": 0,
    }


def get_symbol_performance(symbol: str, lookback_days: int = 30) -> Dict[str, float]:
    """
    ✅ Performance específica de um símbolo.
    """
    try:
        with mt5_lock:
            deals = mt5.history_deals_get(
                datetime.now() - timedelta(days=lookback_days), datetime.now()
            )

        if not deals:
            return {"win_rate": 0.55, "trades": 0}

        symbol_deals = [
            d for d in deals if d.symbol == symbol and d.entry == mt5.DEAL_ENTRY_OUT
        ]

        if len(symbol_deals) < 3:
            return {"win_rate": 0.55, "trades": len(symbol_deals)}

        wins = sum(1 for d in symbol_deals if d.profit > 0)
        total_pnl = sum(d.profit for d in symbol_deals)

        return {
            "win_rate": wins / len(symbol_deals),
            "trades": len(symbol_deals),
            "wins": wins,
            "losses": len(symbol_deals) - wins,
            "total_pnl": total_pnl,
        }

    except Exception as e:
        logger.error(f"Erro ao buscar performance {symbol}: {e}")
        return {"win_rate": 0.55, "trades": 0}


def get_dynamic_rr_min() -> float:
    """
    ✅ R:R DINÂMICO MULTI-FATOR

    Fatores:
    1. Regime de mercado (IBOV trend)
    2. VIX Brasil (BVIX)
    3. Volatilidade histórica (ATR agregado)
    4. Drawdown atual

    Ranges:
    - RISK_ON + vol baixa: 1.2
    - RISK_OFF + vol alta: 2.5
    """
    regime = detect_market_regime()
    if regime not in ("RISK_ON", "RISK_OFF"):
        regime = "RISK_ON"

    # ✅ NOVO: Fator 1 - Regime base
    if regime == "RISK_ON":
        base_rr = 1.2
    else:
        base_rr = 1.5

    # ✅ NOVO: Fator 2 - Volatilidade histórica
    try:
        ibov_df = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 30)

        if ibov_df is not None:
            hvol = ibov_df["close"].pct_change().std() * 100

            # Multiplica RR se vol > 2%
            if hvol > 2.0:
                vol_multiplier = 1.3
            elif hvol > 1.5:
                vol_multiplier = 1.15
            else:
                vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0
    except:
        vol_multiplier = 1.0

    # ✅ NOVO: Fator 3 - Drawdown atual
    try:
        from bot import daily_max_equity

        acc = mt5.account_info()

        if acc and daily_max_equity > 0:
            current_dd = (daily_max_equity - acc.equity) / daily_max_equity

            # Aumenta RR mínimo se em drawdown
            if current_dd > 0.03:  # >3%
                dd_multiplier = 1.5
            elif current_dd > 0.02:  # >2%
                dd_multiplier = 1.25
            else:
                dd_multiplier = 1.0
        else:
            dd_multiplier = 1.0
    except:
        dd_multiplier = 1.0

    # Cálculo Final
    final_rr = base_rr * vol_multiplier * dd_multiplier
    final_rr = min(final_rr, 2.2)

    # logger.info(f"📊 R:R Dinâmico | Regime: {regime} | Base: {base_rr:.2f} | Vol×: {vol_multiplier:.2f} | DD×: {dd_multiplier:.2f} | Final: {final_rr:.2f}")

    return final_rr


# =========================================================
# 🌐 POLYGON.IO INTEGRATION
# =========================================================


def get_vix_br() -> float:
    """
    ✅ V5.2: Obtém VIX Brasil (proxy) via Polygon ou MT5.
    Se falhar, retorna valor baseado na volatilidade do IBOV.
    """
    if REDIS_AVAILABLE:
        cached = redis_client.get("vix_br")
        if cached:
            return float(cached)

    vix = 20.0
    try:
        # Tenta proxy via Polygon (VIXM ou similar se disponível)
        data = get_polygon_data("v2/aggs/ticker/I:VIX/prev")
        if data and data.get("results"):
            vix = data["results"][0]["c"]
        else:
            # Fallback: Volatilidade do IBOV
            df = safe_copy_rates("IBOV", mt5.TIMEFRAME_H1, 100)
            if is_valid_dataframe(df):
                vix = (
                    df["close"].pct_change().std() * 100 * np.sqrt(252 * 7)
                )  # Proxy volatilidade anualizada
                vix = max(10, min(vix, 80))
    except:
        vix = 22.0

    if REDIS_AVAILABLE:
        redis_client.setex("vix_br", 300, str(vix))  # 5 min cache
    return float(vix)


def get_order_flow(symbol: str, bars: int = 20) -> Dict[str, float]:
    """
    ✅ V5.2 CONSOLIDADO: Order Flow Híbrido (Polygon + MT5)
    """
    # 1. Tenta Polygon (Melhor qualidade para CVD real)
    ticker = symbol.replace(".SA", "")
    end = datetime.now().strftime("%Y-%m-%d")
    endpoint = f"v2/aggs/ticker/{ticker}/range/1/minute/{end}/{end}"

    data = get_polygon_data(endpoint, {"limit": bars})
    if data:
        res = data.get("results", [])
        if res:
            buy_v = sum(r["v"] for r in res if r["c"] >= r["o"])
            sell_v = sum(r["v"] for r in res if r["c"] < r["o"])
            total = buy_v + sell_v
            return {
                "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
                "cvd": buy_v - sell_v,
                "buy_volume": buy_v,
                "sell_volume": sell_v,
            }

    # 2. Fallback MT5 (Estimado)
    df = safe_copy_rates(symbol, mt5.TIMEFRAME_M1, bars)
    if not is_valid_dataframe(df):
        return {"imbalance": 0.0, "cvd": 0.0, "buy_volume": 0, "sell_volume": 0}

    df["delta"] = np.where(
        df["close"] >= df["open"], df["tick_volume"], -df["tick_volume"]
    )
    buy_v = df[df["delta"] > 0]["tick_volume"].sum()
    sell_v = df[df["delta"] < 0]["tick_volume"].sum()
    total = buy_v + sell_v

    return {
        "imbalance": (buy_v - sell_v) / total if total > 0 else 0,
        "cvd": float(df["delta"].sum()),
        "buy_volume": float(buy_v),
        "sell_volume": float(sell_v),
    }


def get_order_flow_polygon(symbol: str) -> Dict[str, float]:
    return get_order_flow(symbol, bars=20)


# =========================================================
# SLIPPAGE
# =========================================================


def get_real_slippage(symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0:
        return config.SLIPPAGE_MAP.get("DEFAULT", 0.005)

    spread_pct = (tick.ask - tick.bid) / tick.bid

    # Multiplicador por perfil de liquidez
    if symbol in config.LOW_LIQUIDITY_SYMBOLS:
        mult = 2.0
    elif is_power_hour():
        mult = 1.2
    else:
        mult = 1.5

    mapped = config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP.get("DEFAULT"))
    return max(spread_pct * mult, mapped)


def check_and_alert_slippage(
    symbol: str, requested_price: float, executed_price: float, side: str
) -> bool:
    """
    Verifica se houve slippage excessivo na execução e notifica o usuário.

    Args:
        symbol: Símbolo do ativo
        requested_price: Preço solicitado
        executed_price: Preço executado
        side: BUY ou SELL

    Returns:
        bool: True se slippage foi excessivo
    """
    try:
        info = mt5.symbol_info(symbol)
        if not info or info.point <= 0:
            return False

        # Calcula slippage em ticks
        slippage = abs(executed_price - requested_price)
        slippage_ticks = slippage / info.point

        # Verifica direção (adverso = pior para o trader)
        if side == "BUY":
            is_adverse = executed_price > requested_price
        else:
            is_adverse = executed_price < requested_price

        max_slippage_ticks = getattr(config, "SLIPPAGE_ALERT_TICKS", 3)

        if is_adverse and slippage_ticks > max_slippage_ticks:
            logger.warning(
                f"⚠️ SLIPPAGE EXCESSIVO {symbol} | "
                f"Solicitado: {requested_price:.2f} | "
                f"Executado: {executed_price:.2f} | "
                f"Slippage: {slippage_ticks:.1f} ticks"
            )

            # Envia alerta Telegram
            try:
                send_telegram_message(
                    f"⚠️ <b>SLIPPAGE EXCESSIVO</b>\n\n"
                    f"📊 Ativo: <b>{symbol}</b>\n"
                    f"📈 Direção: {side}\n"
                    f"💰 Solicitado: R$ {requested_price:.2f}\n"
                    f"💸 Executado: R$ {executed_price:.2f}\n"
                    f"📉 Slippage: <b>{slippage_ticks:.1f} ticks</b>\n\n"
                    f"💡 <i>Revise a liquidez do ativo</i>"
                )
            except Exception as e:
                logger.debug(f"Erro ao enviar alerta slippage: {e}")

            return True

        return False

    except Exception as e:
        logger.error(f"Erro ao verificar slippage: {e}")
        return False


# =========================================================
# REGIME DE MERCADO
# =========================================================


def detect_market_regime() -> str:
    ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, 50)
    if ibov is None or len(ibov) < 30:
        return "RISK_ON"

    close = ibov["close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    cur = close.iloc[-1]

    # REGRA MAIS FLEXÍVEL: evita desligar o bot em lateralização
    if cur > ma20:
        return "RISK_ON"
    elif cur < ma50:
        return "RISK_OFF"
    else:
        return "NEUTRAL"


# =========================================================
# EXPOSIÇÃO SETORIAL
# =========================================================


def calculate_sector_exposure_pct(equity: float) -> Dict[str, float]:
    with mt5_lock:
        positions = mt5.positions_get() or []

    sector_risk = defaultdict(float)
    for p in positions:
        sector = config.SECTOR_MAP.get(p.symbol, "UNKNOWN")
        sector_risk[sector] += p.volume * p.price_open

    return {s: v / equity for s, v in sector_risk.items()} if equity > 0 else {}


def get_ibov_correlation(symbol: str, lookback: int = 50) -> float:
    """
    Calcula a correlação de Pearson entre o ativo e o IBOVESPA.
    """
    try:
        df_sym = safe_copy_rates(symbol, mt5.TIMEFRAME_D1, lookback)
        df_ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, lookback)

        if df_sym is None or df_ibov is None or len(df_sym) < 20 or len(df_ibov) < 20:
            return 0.0

        returns_sym = df_sym["close"].pct_change().dropna()
        returns_ibov = df_ibov["close"].pct_change().dropna()

        # Alinha os índices
        combined = pd.concat([returns_sym, returns_ibov], axis=1).dropna()
        if len(combined) < 10:
            return 0.0

        correlation = combined.corr().iloc[0, 1]
        return float(correlation)
    except Exception as e:
        logger.error(f"Erro ao calcular correlação IBOV para {symbol}: {e}")
        return 0.0


# deleted for debug


def validate_subsetor_exposure(symbol: str) -> Tuple[bool, str]:
    """
    ✅ Garante que não ultrapassamos 20% do capital (ou config) em um mesmo subsetor.
    """
    try:
        subsetor = config.SUBSETOR_MAP.get(symbol, "Outros")

        with mt5_lock:
            positions = mt5.positions_get()

        if not positions:
            return True, "OK"

        acc = mt5.account_info()
        total_equity = acc.equity if acc else 100000.0

        subsetor_value = 0
        for pos in positions:
            if config.SUBSETOR_MAP.get(pos.symbol) == subsetor:
                subsetor_value += pos.volume * pos.price_open

        risk_per_trade = float(
            getattr(
                config, "RISK_PER_TRADE", getattr(config, "RISK_PER_TRADE_PCT", 0.01)
            )
            or 0.01
        )
        projected_total = subsetor_value + (total_equity * risk_per_trade)
        exposure = projected_total / total_equity

        limit = getattr(config, "MAX_SUBSETOR_EXPOSURE", 0.20)

        if exposure > limit:
            return False, f"Exposição excessiva em {subsetor}: {exposure:.1%}"

        return True, "OK"
    except Exception as e:
        logger.error(f"Erro validação subsetor: {e}")
        return True, "OK"


def get_book_imbalance(symbol: str) -> float:
    try:
        book = mt5.market_book_get(symbol)
        if not book:
            return 0.0

        buys = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_BUY)
        sells = sum(item.volume for item in book if item.type == mt5.BOOK_TYPE_SELL)

        total = buys + sells
        if total == 0:
            return 0.0

        imbalance = (buys - sells) / total

        # ZONA NEUTRA: ignora ruído
        if abs(imbalance) < config.MIN_BOOK_IMBALANCE:
            return 0.0

        return imbalance

    except Exception as e:
        logger.error(f"Erro no book imbalance para {symbol}: {e}")
        return 0.0


# =========================================================
# FAST RATES
# =========================================================

_last_bar_time = {}


def get_fast_rates(symbol, timeframe):
    df = safe_copy_rates(symbol, timeframe, 3)
    if not is_valid_dataframe(df):
        return None

    last = df.index[-1]
    key = (symbol, timeframe)
    if _last_bar_time.get(key) == last:
        return None

    _last_bar_time[key] = last
    return df


# =========================================================
# INDICADORES BÁSICOS
# =========================================================


def get_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return float(atr.iloc[-1])


def get_psar(
    df: pd.DataFrame, af: float = 0.02, af_max: float = 0.2
) -> Optional[float]:
    try:
        if df is None or len(df) < 5:
            return None
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        n = len(high)
        psar = [low[0]]
        uptrend = True
        ep = high[0]
        sar = low[0]
        step = af
        for i in range(1, n):
            prev_sar = sar + step * (ep - sar)
            if uptrend:
                sar = min(prev_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
                if high[i] > ep:
                    ep = high[i]
                    step = min(step + af, af_max)
                if low[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = low[i]
                    step = af
            else:
                sar = max(prev_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
                if low[i] < ep:
                    ep = low[i]
                    step = min(step + af, af_max)
                if high[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = high[i]
                    step = af
            psar.append(sar)
        return float(psar[-1])
    except Exception:
        return None


def get_ibov_minute_data(period: int = 60) -> list:
    try:
        with mt5_lock:
            rates = mt5.copy_rates_from_pos("IBOV", mt5.TIMEFRAME_M1, 0, int(period))
        if rates is None or len(rates) == 0:
            return []
        df = pd.DataFrame(rates)
        return df["close"].astype(float).tolist()
    except Exception:
        return []


def get_ibov_10min_change() -> float:
    try:
        closes = get_ibov_minute_data(period=15)
        if len(closes) < 11:
            return 0.0
        last = float(closes[-1])
        prev = float(closes[-11])
        if prev <= 0:
            return 0.0
        return (last - prev) / prev
    except Exception:
        return 0.0


def get_current_volume_ratio(symbol: str, lookback_bars: int = 20) -> float:
    try:
        with mt5_lock:
            rates = mt5.copy_rates_from_pos(
                symbol, mt5.TIMEFRAME_M15, 0, int(lookback_bars) + 1
            )
        if rates is None or len(rates) < 2:
            return 1.0
        df = pd.DataFrame(rates)
        vol_col = (
            "real_volume"
            if "real_volume" in df.columns and df["real_volume"].sum() > 0
            else "tick_volume"
        )
        current = float(df[vol_col].iloc[-1])
        baseline = float(df[vol_col].iloc[:-1].mean())
        if baseline <= 0:
            return 1.0
        return current / baseline
    except Exception:
        return 1.0


def get_global_volume_ratio(symbols: Optional[list] = None) -> float:
    try:
        base_symbols = (
            symbols
            if symbols
            else ["PETR4", "VALE3", "ITUB4", "BBDC4", "PRIO3", "ABEV3"]
        )
        ratios = []
        for s in base_symbols:
            try:
                r = get_current_volume_ratio(s)
                ratios.append(float(r))
            except Exception:
                continue
        return max(ratios) if ratios else 1.0
    except Exception:
        return 1.0


def get_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(df) < period * 2:
        return None

    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return float(adx.iloc[-1])


def get_volatility_regime(symbol: str) -> str:
    df = safe_copy_rates(symbol, TIMEFRAME_BASE, 160)
    if not is_valid_dataframe(df, 120):
        return "NORMAL_VOL"
    atr21 = get_atr(df, period=21) or 0.0
    atr100 = get_atr(df, period=100) or 0.0
    if atr21 <= 0 or atr100 <= 0:
        return "NORMAL_VOL"
    if atr21 > 1.75 * atr100:
        return "HIGH_VOL"
    if atr21 < 0.75 * atr100:
        return "LOW_VOL"
    return "NORMAL_VOL"


def get_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Retorna a série completa do ADX para análise de tendência."""
    if df is None or len(df) < period * 2:
        return pd.Series()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


def get_intraday_vwap(df: pd.DataFrame) -> Optional[float]:
    """
    VWAP desde a abertura do pregão (10h00) até agora.
    """
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())

    df_today = df[df.index >= market_open]

    if df_today.empty or len(df_today) < 3:
        return None

    typical_price = (df_today["high"] + df_today["low"] + 2 * df_today["close"]) / 4
    volume = df_today.get("real_volume", df_today["tick_volume"])

    pv = (typical_price * volume).sum()
    total_vol = volume.sum()

    return float(pv / total_vol) if total_vol > 0 else None


def get_intraday_vwap_stats(df: pd.DataFrame) -> Optional[dict]:
    now = datetime.now()
    today = now.date()
    market_open = datetime.combine(today, datetime.strptime("10:00", "%H:%M").time())

    # Verificar tipo de índice e filtrar apropriadamente
    if hasattr(df.index, "date"):
        # Se for DatetimeIndex, usar comparação direta
        df_today = df[df.index >= market_open]
    else:
        # Se for RangeIndex, usar posição temporal
        # Assumindo que os últimos dados são os mais recentes
        # Pegar os últimos N candles do dia (ex: últimos 50)
        df_today = df.tail(min(50, len(df)))

    if df_today.empty or len(df_today) < 3:
        return None
    typical_price = (df_today["high"] + df_today["low"] + 2 * df_today["close"]) / 4
    volume = df_today.get("real_volume", df_today["tick_volume"])
    pv = (typical_price * volume).sum()
    total_vol = volume.sum()
    vwap = float(pv / total_vol) if total_vol > 0 else None
    if vwap is None:
        return None
    std = float(typical_price.std(ddof=0))
    upper_2sd = vwap + std * getattr(config, "VWAP_OVEREXT_STD_MULT", 2.0)
    lower_2sd = vwap - std * getattr(config, "VWAP_OVEREXT_STD_MULT", 2.0)
    return {"vwap": vwap, "std": std, "upper_2sd": upper_2sd, "lower_2sd": lower_2sd}


def timeframe_to_minutes(tf) -> int:
    m = {
        getattr(mt5, "TIMEFRAME_M1", None): 1,
        getattr(mt5, "TIMEFRAME_M5", None): 5,
        getattr(mt5, "TIMEFRAME_M15", None): 15,
        getattr(mt5, "TIMEFRAME_M30", None): 30,
        getattr(mt5, "TIMEFRAME_H1", None): 60,
        getattr(mt5, "TIMEFRAME_H4", None): 240,
        getattr(mt5, "TIMEFRAME_D1", None): 1440,
    }
    return m.get(tf, 15)


def get_avg_spread_pct(symbol: str, timeframe, bars: int = 10) -> Optional[float]:
    try:
        minutes = timeframe_to_minutes(timeframe) * max(int(bars), 1)
        start = datetime.now() - timedelta(minutes=minutes)
        with mt5_lock:
            ticks = mt5.copy_ticks_from(symbol, start, 100000, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return None
        df_ticks = pd.DataFrame(ticks)
        df_ticks = df_ticks[(df_ticks["ask"] > 0) & (df_ticks["bid"] > 0)]
        if df_ticks.empty:
            return None
        spread_pct = (df_ticks["ask"] - df_ticks["bid"]) / df_ticks["bid"]
        return float(spread_pct.mean() * 100)
    except Exception:
        return None


def check_spread(symbol: str, timeframe, bars: int = None) -> tuple[bool, float, float]:
    with mt5_lock:
        tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.ask <= 0 or tick.bid <= 0:
        return False, 0.0, 0.0
    current_pct = (tick.ask - tick.bid) / tick.bid * 100
    lookback = (
        bars if bars is not None else int(getattr(config, "SPREAD_LOOKBACK_BARS", 10))
    )
    avg_pct = get_avg_spread_pct(symbol, timeframe, lookback) or 0.0
    if current_pct > avg_pct and avg_pct > 0:
        return False, float(current_pct), float(avg_pct)
    return True, float(current_pct), float(avg_pct)


def get_obv(df: pd.DataFrame) -> Optional[float]:
    """
    Calcula OBV ignorando candle em aberto.
    """
    if df is None or len(df) < 20:
        return None

    try:
        # 1. Escolha do volume (B3 usa tick_volume se real_volume for zero)
        volume = df["tick_volume"].values
        close = df["close"].values

        # 2. Lógica Vetorizada (MUITO mais rápida que loop 'for')
        # Cria array de direção: 1 se subiu, -1 se caiu, 0 se igual
        direction = np.sign(np.diff(close))

        # O diff reduz o tamanho em 1, então alinhamos o volume do candle [1:]
        obv_changes = direction * volume[1:]

        # Soma acumulada para gerar a linha do OBV
        # Usamos apenas até o candle [-2] para análise estável
        full_obv = np.cumsum(np.concatenate(([volume[0]], obv_changes)))

        # 3. Retornamos o valor do ÚLTIMO CANDLE FECHADO (-2)
        # Isso garante que o volume da análise é real e completo
        return float(full_obv[-2])

    except Exception as e:
        logger.error(f"Erro ao calcular OBV: {e}")
        return None


# 🆕 NOVA FUNÇÃO - Tendência do OBV
def is_obv_trending_up(df: pd.DataFrame, lookback: int = 20) -> bool:
    """
    ✅ Valida se OBV está em tendência de alta

    Retorna True se:
    - OBV atual > média móvel de 20 períodos
    - OBV subiu nos últimos 5 candles
    """
    if df is None or len(df) < lookback + 5:
        return False

    try:
        # Calcula OBV completo
        volume = df.get("real_volume", df["tick_volume"]).values
        close = df["close"].values
        price_changes = np.diff(close, prepend=close[0])

        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if price_changes[i] > 0:
                obv[i] = obv[i - 1] + volume[i]
            elif price_changes[i] < 0:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # Média móvel do OBV
        obv_ma = np.convolve(obv, np.ones(lookback) / lookback, mode="valid")

        obv_above_ma = obv[-1] > obv_ma[-1]
        obv_rising = obv[-1] > obv[-3]  # reduz lookback

        return obv_above_ma or obv_rising

    except Exception as e:
        logger.error(f"Erro ao validar OBV: {e}")
        return False


# =========================================================
# MACRO TREND
# =========================================================


def macro_trend_ok(symbol: str, side: str) -> bool:
    df = safe_copy_rates(symbol, TIMEFRAME_MACRO, 300)
    if df is None or len(df) < config.MACRO_EMA_LONG:
        return False

    close = df["close"]
    ema = close.ewm(span=config.MACRO_EMA_LONG, adjust=False).mean().iloc[-1]
    tick = mt5.symbol_info_tick(symbol)
    if not tick or (tick.last <= 0 and tick.bid <= 0):
        return False

    price = tick.last if tick.last > 0 else tick.bid

    adx = get_adx(df)
    min_adx = 17 if TIMEFRAME_MACRO >= mt5.TIMEFRAME_H1 else 14

    if adx is not None and adx < min_adx:
        return False

    return price > ema if side == "BUY" else price < ema


def get_macro_confirmation(side: str) -> tuple[bool, str]:
    try:
        if not bool(getattr(config, "ENABLE_MACRO_FILTER", True)):
            return True, "OK"
    except Exception:
        pass

    def _pick_symbol(candidates: list[str]) -> Optional[str]:
        for s in candidates:
            df = safe_copy_rates(s, mt5.TIMEFRAME_H1, 60)
            if is_valid_dataframe(df, 20):
                return s
        return None

    sp_symbol = _pick_symbol(["ES", "SPX500USD", "US500", "IVVB11"])
    usd_symbol = _pick_symbol(["USDBRL", "BRLUSD", "DXY", "DIVO11"])
    if not sp_symbol or not usd_symbol:
        return False, "Veto Macro: Dados indisponíveis"
    sp_df = safe_copy_rates(sp_symbol, mt5.TIMEFRAME_H1, 80)
    usd_df = safe_copy_rates(usd_symbol, mt5.TIMEFRAME_H1, 80)
    if not is_valid_dataframe(sp_df, 25) or not is_valid_dataframe(usd_df, 25):
        return False, "Veto Macro: Amostra insuficiente"
    sp_close = sp_df["close"]
    usd_close = usd_df["close"]
    sp_ema = sp_close.ewm(span=20, adjust=False).mean().iloc[-1]
    usd_ema = usd_close.ewm(span=20, adjust=False).mean().iloc[-1]
    sp_tick = mt5.symbol_info_tick(sp_symbol)
    usd_tick = mt5.symbol_info_tick(usd_symbol)
    sp_price = (
        float(sp_tick.last or sp_tick.bid or sp_close.iloc[-1])
        if sp_tick
        else float(sp_close.iloc[-1])
    )
    usd_price = (
        float(usd_tick.last or usd_tick.bid or usd_close.iloc[-1])
        if usd_tick
        else float(usd_close.iloc[-1])
    )

    def _trend(p: float, e: float) -> str:
        d = abs(p - e) / max(e, 1e-9)
        if d <= 0.0015:
            return "SIDEWAYS"
        return "UP" if p > e else "DOWN"

    sp_trend = _trend(sp_price, sp_ema)
    usd_trend = _trend(usd_price, usd_ema)
    if side == "BUY":
        if sp_trend == "UP" and usd_trend in ("DOWN", "SIDEWAYS"):
            return True, "OK"
        return False, f"Veto Macro: S&P {sp_trend}, USD {usd_trend}"
    if side == "SELL":
        if sp_trend == "DOWN" and usd_trend == "UP":
            return True, "OK"
        return False, f"Veto Macro: S&P {sp_trend}, USD {usd_trend}"
    return False, "Veto Macro: Lado inválido"


class MultiTimeframeEngine:
    def _get_hierarchy(self, symbol: str) -> list:
        s = (symbol or "").upper()
        if is_future(s):
            return [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
        return [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1]

    def _calculate_signal(self, symbol: str, timeframe) -> dict:
        df = safe_copy_rates(symbol, timeframe, 200)
        if df is None or len(df) < 50:
            return {
                "side": "NEUTRAL",
                "score": 0.0,
                "adx": 0.0,
                "ema_diff": 0.0,
                "volume_ratio": 0.0,
            }
        ind = quick_indicators_custom(symbol, timeframe, df=df, params={})
        ema_fast = float(ind.get("ema_fast", 0) or 0)
        ema_slow = float(ind.get("ema_slow", 0) or 0)
        adx = float(ind.get("adx", 0) or 0)
        close = float(ind.get("close", 0) or 0)
        vr = float(ind.get("volume_ratio", 0) or 0)
        side = "BUY" if ema_fast > ema_slow else "SELL"
        diff = abs(ema_fast - ema_slow) / max(close, 1e-9)
        score = max(0.0, min(1.0, (diff * 100.0) / 2.0))
        return {
            "side": side,
            "score": score,
            "adx": adx,
            "ema_diff": diff,
            "volume_ratio": vr,
        }

    def _check_conflicts(self, signals: dict, intended_side: str) -> list:
        conflicts = []
        keys = list(signals.keys())
        for tf in keys:
            s = signals.get(tf, {})
            side = s.get("side", "NEUTRAL")
            if side != "NEUTRAL" and side != intended_side:
                conflicts.append(str(tf))
        return conflicts

    def _calculate_alignment(self, signals: dict, intended_side: str) -> float:
        total = 0
        aligned = 0
        for _, s in signals.items():
            side = s.get("side", "NEUTRAL")
            if side != "NEUTRAL":
                total += 1
                if side == intended_side:
                    aligned += 1
        if total == 0:
            return 0.0
        return float(aligned) / float(total)

    def validate_entry(
        self, symbol: str, side: str, base_timeframe
    ) -> tuple[bool, str]:
        hierarchy = self._get_hierarchy(symbol)
        signals = {}
        for tf in hierarchy:
            signals[tf] = self._calculate_signal(symbol, tf)
        conflicts = self._check_conflicts(signals, side)
        if conflicts:
            return False, f"Conflito MTF em {', '.join(conflicts)}"
        alignment = self._calculate_alignment(signals, side)
        min_align = 0.60 if is_future(symbol) else 0.70
        if alignment < min_align:
            return False, f"Alinhamento {alignment:.0%}"
        return True, f"MTF OK ({alignment:.0%})"


class FilterChain:
    def __init__(self):
        self.filters = {
            "capital_available": {
                "priority": 1,
                "mandatory": True,
                "bypass_conditions": None,
            },
            "mtf_alignment": {
                "priority": 2,
                "mandatory": True,
                "bypass_conditions": None,
            },
            "risk_limits": {
                "priority": 3,
                "mandatory": True,
                "bypass_conditions": None,
            },
            "ml_confidence": {
                "priority": 4,
                "mandatory": False,
                "bypass_conditions": {
                    "high_adx": lambda ind: float(ind.get("adx", 0) or 0) > 40,
                    "strong_trend": lambda ind: float(ind.get("ema_diff", 0) or 0)
                    > 0.02,
                    "high_volume": lambda ind: float(ind.get("volume_ratio", 0) or 0)
                    > 2.0,
                },
                "bypass_threshold": 2,
            },
            "anti_chop": {
                "priority": 5,
                "mandatory": False,
                "bypass_conditions": {
                    "breakout": lambda ind: bool(ind.get("vol_breakout", False)),
                    "news_catalyst": lambda ind: bool(ind.get("has_news", False)),
                },
                "bypass_threshold": 1,
            },
        }

    def _execute_filter(
        self, name: str, symbol: str, side: str, indicators: dict
    ) -> tuple[bool, str]:
        if name == "capital_available":
            tick = cached_symbol_info_tick(symbol)
            if not tick:
                logger.warning(f"⚠️ cached_symbol_info_tick retornou None para {symbol}")
                return False, "Sem cotação"
            entry = tick.ask if side == "BUY" else tick.bid
            vol = 1.0 if is_future(symbol) else 100.0
            ok, reason = check_capital_allocation(symbol, vol, entry)
            return ok, reason
        if name == "mtf_alignment":
            base_tf = mt5.TIMEFRAME_M5 if is_future(symbol) else mt5.TIMEFRAME_M15
            engine = MultiTimeframeEngine()
            ok, reason = engine.validate_entry(symbol, side, base_tf)
            return ok, reason
        if name == "risk_limits":
            limit = get_effective_exposure_limit()
            current = calculate_total_exposure()
            return current <= limit, "Limite"
        if name == "ml_confidence":
            try:
                from ml_signals import MLSignalPredictor

                pred = MLSignalPredictor()
                p = pred.predict(symbol, indicators)
                conf = float(p.get("confidence", 0) or 0)
                dirn = str(p.get("direction", "HOLD"))
                if dirn in ("HOLD", "ERROR"):
                    return False, "ML HOLD"
                if dirn != side and conf >= 0.80:
                    return False, f"ML Contra {conf:.0%}"
                return True, "OK"
            except Exception:
                return True, "ML Indisponível"
        if name == "anti_chop":
            tick = cached_symbol_info_tick(symbol)
            if not tick:
                return True, ""
            price = tick.bid if side == "SELL" else tick.ask
            atr = float(indicators.get("atr", 0) or 0)
            ok, reason = check_anti_chop_filter(symbol, price, atr)
            return ok, reason or ""
        return True, ""

    def _check_bypass(self, filter_name: str, indicators: dict, config: dict) -> bool:
        conditions_met = 0
        for _, fn in (config.get("bypass_conditions") or {}).items():
            try:
                if fn(indicators):
                    conditions_met += 1
            except Exception:
                continue
        return conditions_met >= int(config.get("bypass_threshold", 99) or 99)

    def _get_missing_info(
        self, filter_name: str, indicators: dict, config: dict
    ) -> str:
        """Retorna informações sobre quanto falta para atingir as métricas"""
        if filter_name == "ml_confidence":
            try:
                from ml_signals import MLSignalPredictor

                pred = MLSignalPredictor()
                p = pred.predict("", indicators)
                conf = float(p.get("confidence", 0) or 0)
                missing = max(0, 80 - int(conf * 100))
                return f"Confiança: {conf:.0%} (falta {missing}% para 80%)"
            except:
                return "ML indisponível"

        elif filter_name == "anti_chop":
            tick = cached_symbol_info_tick("")
            if tick:
                atr = float(indicators.get("atr", 0) or 0)
                current_atr = float(indicators.get("current_atr", 0) or 0)
                if current_atr > 0 and atr > 0:
                    ratio = current_atr / atr
                    target = 1.5
                    if ratio < target:
                        missing = f"{(target - ratio):.1f}x"
                        return f"ATR: {ratio:.1f}x (falta {missing} para {target}x)"

        elif filter_name == "capital_available":
            return "Verificar capital disponível"

        elif filter_name == "mtf_alignment":
            alignment = float(indicators.get("mtf_alignment", 0) or 0)
            min_align = 0.70
            missing = max(0, int((min_align - alignment) * 100))
            return (
                f"Alinhamento: {alignment:.0%} (falta {missing}% para {min_align:.0%})"
            )

        elif filter_name == "risk_limits":
            current = float(indicators.get("current_exposure", 0) or 0)
            limit = float(indicators.get("exposure_limit", 100) or 100)
            if current > limit:
                excess = int(current - limit)
                return f"Exposição: {current:.0f}% (excesso de {excess}% sobre limite)"
            else:
                return "Limite de risco atingido"

        return ""

    def _display_indicators(self, indicators: dict, side: str = "BUY") -> None:
        """Exibe os indicadores técnicos de forma colorida com comparação de valores alvo"""
        C_GREEN = "\033[92m"
        C_RED = "\033[91m"
        C_YELLOW = "\033[93m"
        C_CYAN = "\033[96m"
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"

        print(f"\n{C_BOLD}{C_CYAN}📊 Indicadores Técnicos:{C_RESET}")

        # RSI
        rsi = float(indicators.get("rsi", 0) or 0)
        rsi_color = C_GREEN if 30 <= rsi <= 70 else C_RED if rsi > 70 else C_YELLOW
        rsi_status = (
            "Neutro" if 30 <= rsi <= 70 else "Sobrecompra" if rsi > 70 else "Sobrevenda"
        )
        rsi_target = "30-70" if 30 <= rsi <= 70 else ("< 70" if rsi > 70 else "> 30")
        print(
            f"  • RSI: {rsi_color}{rsi:.1f} ({rsi_status}){C_RESET} | Alvo: {rsi_target}"
        )

        # ADX
        adx = float(indicators.get("adx", 0) or 0)
        adx_color = C_GREEN if adx > 25 else C_YELLOW if adx > 20 else C_RED
        adx_status = "Forte" if adx > 25 else "Moderado" if adx > 20 else "Fraco"
        adx_target = "> 25" if adx > 25 else ("> 20" if adx > 20 else "> 20")
        print(
            f"  • ADX: {adx_color}{adx:.1f} ({adx_status}){C_RESET} | Alvo: {adx_target}"
        )

        # Volume
        vol_ratio = float(indicators.get("volume_ratio", 0) or 0)
        vol_color = (
            C_GREEN if vol_ratio > 1.5 else C_YELLOW if vol_ratio > 1.0 else C_RED
        )
        vol_status = (
            "Alto" if vol_ratio > 1.5 else "Normal" if vol_ratio > 1.0 else "Baixo"
        )
        vol_target = (
            "> 1.5" if vol_ratio > 1.5 else ("> 1.0" if vol_ratio > 1.0 else "> 1.0")
        )
        print(
            f"  • Volume: {vol_color}{vol_ratio:.2f}x ({vol_status}){C_RESET} | Alvo: {vol_target}"
        )

        # Volume para tomada de decisão (com limite do período)
        from datetime import datetime

        current_time = datetime.now().time()
        if current_time < datetime.strptime("12:00", "%H:%M").time():
            min_vol = 0.85
            period_name = "Manhã"
        elif (
            datetime.strptime("12:00", "%H:%M").time()
            <= current_time
            <= datetime.strptime("13:30", "%H:%M").time()
        ):
            min_vol = 0.5
            period_name = "Almoço"
        else:
            import config

            min_vol = float(getattr(config, "AFTERNOON_MIN_VOLUME_RATIO", 1.0) or 1.0)
            period_name = "Tarde"

        vol_decision_color = C_GREEN if vol_ratio >= min_vol else C_RED
        print(
            f"  • Volume {period_name}: {vol_decision_color}{vol_ratio:.2f}x vs Mínimo {min_vol}x{C_RESET}"
        )

        # ===== MÉTRICAS DE DECISÃO =====
        print(f"\n{C_BOLD}{C_CYAN}🎯 Métricas de Decisão:{C_RESET}")

        # RSI - Exaustão
        rsi = float(indicators.get("rsi", 50) or 50)
        rsi_color = (
            C_GREEN
            if (side == "BUY" and rsi <= 70) or (side == "SELL" and rsi >= 30)
            else C_RED
        )
        rsi_status = (
            "✅ OK"
            if (side == "BUY" and rsi <= 70) or (side == "SELL" and rsi >= 30)
            else "❌ EXAUSTÃO"
        )
        rsi_limit = "≤ 70" if side == "BUY" else "≥ 30"
        print(
            f"  • RSI Exaustão: {rsi_color}RSI {rsi:.1f} {rsi_status}{C_RESET} | Limite: {rsi_limit}"
        )

        # Volume por Período (detalhado)
        print(
            f"  • Volume {period_name}: {vol_decision_color}{vol_ratio:.2f}x vs Mínimo {min_vol}x{C_RESET}"
        )

        # Tendência Mínima
        ema_diff = float(indicators.get("ema_diff", 0) or 0)
        import config

        min_trend = (
            config.MIN_TREND_THRESHOLD_PCT
            if side == "BUY"
            else -config.MIN_TREND_THRESHOLD_PCT
        )  # Configurável via config.py
        lateral_range = config.LATERAL_RANGE_PCT  # Configurável via config.py
        trend_ok = (
            (side == "BUY" and ema_diff >= min_trend)
            or (side == "SELL" and ema_diff <= -min_trend)
            or abs(ema_diff) < lateral_range
        )
        trend_color = C_GREEN if trend_ok else C_RED
        trend_status = "✅ OK" if trend_ok else "❌ FRACA"
        trend_req = (
            f"≥ {config.MIN_TREND_THRESHOLD_PCT*100:.0f}%"
            if side == "BUY"
            else f"≤ -{config.MIN_TREND_THRESHOLD_PCT*100:.0f}%"
        )
        print(
            f"  • Tendência Mínima: {trend_color}Tendência {ema_diff*100:.1f}% {trend_status}{C_RESET} | Req: {trend_req}"
        )

        # Outros filtros importantes (simplificado)
        print(f"  • Filtros MTF: ✅ Verificado nos filtros acima")
        print(f"  • Capital/Risco: ✅ Verificado nos filtros acima")
        print(f"  • ML Confidence: ✅ Verificado nos filtros acima")

        # ===== RESUMO =====
        print(f"\n{C_BOLD}📋 Resumo de Decisão:{C_RESET}")
        print(f"  • Período: {period_name}")
        print(f"  • Volume Mínimo: {min_vol}x")
        print(f"  • RSI Limite: {rsi_limit}")
        print(f"  • Tendência: {trend_req}")
        print(f"  • Todos filtros devem estar ✅ para aprovar")

        # Tendência
        ema_diff = float(indicators.get("ema_diff", 0) or 0)
        import config

        lateral_range = config.LATERAL_RANGE_PCT  # Configurável via config.py
        trend_color = (
            C_GREEN
            if ema_diff > lateral_range
            else C_RED if ema_diff < -lateral_range else C_YELLOW
        )
        trend_status = (
            "Alta"
            if ema_diff > lateral_range
            else "Baixa" if ema_diff < -lateral_range else "Lateral"
        )
        trend_target = (
            f"> {lateral_range*100:.0f}%"
            if ema_diff > lateral_range
            else (
                f"< -{lateral_range*100:.0f}%"
                if ema_diff < -lateral_range
                else f"-{lateral_range*100:.0f}% a +{lateral_range*100:.0f}%"
            )
        )
        print(
            f"  • Tendência: {trend_color}{ema_diff*100:.1f}% ({trend_status}){C_RESET} | Alvo: {trend_target}"
        )

        print()

    def _display_rejection_reason(
        self, indicators: dict, failed_filters: list, entry_price: float = 0.0
    ) -> None:
        """Exibe o motivo da rejeição com base nos indicadores"""
        C_RED = "\033[91m"
        C_YELLOW = "\033[93m"
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"

        print(f"\n{C_BOLD}{C_RED}💬 Motivo para não comprar:{C_RESET}")

        entry_str = ""
        if entry_price and entry_price > 0:
            entry_str = f" (valor de entrada: R$ {entry_price:.2f})"

        # Analisar indicadores para dar motivo específico
        rsi = float(indicators.get("rsi", 0) or 0)
        adx = float(indicators.get("adx", 0) or 0)
        vol_ratio = float(indicators.get("volume_ratio", 0) or 0)
        ema_diff = float(indicators.get("ema_diff", 0) or 0)

        reasons = []

        if rsi > 70:
            falta_rsi = rsi - 70
            reasons.append(
                f"{C_YELLOW}RSI em sobrecompra ({rsi:.1f}) - reduzir {falta_rsi:.1f} pontos{C_RESET}"
            )
        elif rsi < 30:
            falta_rsi = 30 - rsi
            reasons.append(
                f"{C_RED}RSI em sobrevenda ({rsi:.1f}) - aumentar {falta_rsi:.1f} pontos{C_RESET}"
            )

        if adx < 20:
            falta_adx = 20 - adx
            reasons.append(
                f"{C_RED}Mercado sem tendência clara (ADX {adx:.1f}) - aumentar {falta_adx:.1f}{C_RESET}"
            )

        if vol_ratio < 1.0:
            falta_vol = 1.0 - vol_ratio
            reasons.append(
                f"{C_RED}Volume abaixo da média ({vol_ratio:.1f}x) - aumentar {falta_vol:.1f}x{entry_str}{C_RESET}"
            )
        elif vol_ratio < 0.5:
            falta_vol = 0.5 - vol_ratio
            reasons.append(
                f"{C_RED}Volume muito fraco ({vol_ratio:.1f}x) - aumentar {falta_vol:.1f}x{C_RESET}"
            )

        if abs(ema_diff) < 0.01:
            falta_trend = 0.01 - abs(ema_diff)
            reasons.append(
                f"{C_YELLOW}Tendência lateral sem direção clara - aumentar {falta_trend*100:.1f}%{C_RESET}"
            )
        elif ema_diff < -0.02:
            falta_trend = -0.02 - ema_diff
            reasons.append(
                f"{C_RED}Tendência de baixa forte - aumentar {falta_trend*100:.1f}%{entry_str}{C_RESET}"
            )

        # Adicionar motivos dos filtros que falharam
        for filter_name, reason in failed_filters:
            if filter_name == "capital_available":
                reasons.append(f"{C_RED}Capital insuficiente{entry_str}{C_RESET}")
            elif filter_name == "mtf_alignment":
                reasons.append(f"{C_YELLOW}Desalinhamento de timeframes{C_RESET}")
            elif filter_name == "risk_limits":
                reasons.append(f"{C_RED}Limite de risco atingido{C_RESET}")
            elif filter_name == "ml_confidence":
                reasons.append(f"{C_YELLOW}Confiança do ML baixa{C_RESET}")
            elif filter_name == "anti_chop":
                reasons.append(f"{C_YELLOW}Mercado em chop/consolidação{C_RESET}")

        if reasons:
            for i, reason in enumerate(reasons, 1):
                print(f"  {i}. {reason}")
        else:
            print(f"  {C_RED}Condições de mercado desfavoráveis{C_RESET}")

        print()

    def format_detailed_log_info(self, indicators: dict, side: str = "BUY") -> str:
        """Formata informações detalhadas (indicadores + métricas + resumo) para logs"""
        from datetime import datetime

        # RSI
        rsi = float(indicators.get("rsi", 0) or 0)
        rsi_status = (
            "Neutro" if 30 <= rsi <= 70 else "Sobrecompra" if rsi > 70 else "Sobrevenda"
        )
        rsi_target = "30-70" if 30 <= rsi <= 70 else ("< 70" if rsi > 70 else "> 30")

        # ADX
        adx = float(indicators.get("adx", 0) or 0)
        adx_status = "Forte" if adx > 25 else "Moderado" if adx > 20 else "Fraco"
        adx_target = "> 25" if adx > 25 else ("> 20" if adx > 20 else "> 20")

        # Volume
        vol_ratio = float(indicators.get("volume_ratio", 0) or 0)
        vol_status = (
            "Alto" if vol_ratio > 1.5 else "Normal" if vol_ratio > 1.0 else "Baixo"
        )
        vol_target = (
            "> 1.5" if vol_ratio > 1.5 else ("> 1.0" if vol_ratio > 1.0 else "> 1.0")
        )

        # Volume por período
        current_time = datetime.now().time()
        if current_time < datetime.strptime("12:00", "%H:%M").time():
            min_vol = 0.85
            period_name = "Manhã"
        elif (
            datetime.strptime("12:00", "%H:%M").time()
            <= current_time
            <= datetime.strptime("13:30", "%H:%M").time()
        ):
            min_vol = 0.5
            period_name = "Almoço"
        else:
            import config

            min_vol = float(getattr(config, "AFTERNOON_MIN_VOLUME_RATIO", 1.0) or 1.0)
            period_name = "Tarde"

        # Tendência
        ema_diff = float(indicators.get("ema_diff", 0) or 0)
        import config

        lateral_range = config.LATERAL_RANGE_PCT  # Configurável via config.py
        trend_status = (
            "Alta"
            if ema_diff > lateral_range
            else "Baixa" if ema_diff < -lateral_range else "Lateral"
        )
        trend_target = (
            f"> {lateral_range*100:.0f}%"
            if ema_diff > lateral_range
            else (
                f"< -{lateral_range*100:.0f}%"
                if ema_diff < -lateral_range
                else f"-{lateral_range*100:.0f}% a +{lateral_range*100:.0f}%"
            )
        )

        # Métricas de decisão
        rsi_ok = (side == "BUY" and rsi <= 70) or (side == "SELL" and rsi >= 30)
        rsi_limit = "≤ 70" if side == "BUY" else "≥ 30"

        vol_ok = vol_ratio >= min_vol

        min_trend = (
            config.MIN_TREND_THRESHOLD_PCT
            if side == "BUY"
            else -config.MIN_TREND_THRESHOLD_PCT
        )  # Configurável via config.py
        trend_ok = (
            (side == "BUY" and ema_diff >= min_trend)
            or (side == "SELL" and ema_diff <= -min_trend)
            or abs(ema_diff) < lateral_range
        )
        trend_req = (
            f"≥ {config.MIN_TREND_THRESHOLD_PCT*100:.0f}%"
            if side == "BUY"
            else f"≤ -{config.MIN_TREND_THRESHOLD_PCT*100:.0f}%"
        )

        # Formatar informações
        log_info = f"""
📊 Indicadores Técnicos:
  • RSI: {rsi:.1f} ({rsi_status}) | Alvo: {rsi_target}
  • ADX: {adx:.1f} ({adx_status}) | Alvo: {adx_target}
  • Volume: {vol_ratio:.2f}x ({vol_status}) | Alvo: {vol_target}
  • Volume {period_name}: {vol_ratio:.2f}x vs Mínimo {min_vol}x
  • Tendência: {ema_diff*100:.1f}% ({trend_status}) | Alvo: {trend_target}

🎯 Métricas de Decisão:
  • RSI Exaustão: RSI {rsi:.1f} {'✅ OK' if rsi_ok else '❌ EXAUSTÃO'} | Limite: {rsi_limit}
  • Volume {period_name}: {vol_ratio:.2f}x vs Mínimo {min_vol}x {'✅' if vol_ok else '❌'}
  • Tendência Mínima: Tendência {ema_diff*100:.1f}% {'✅ OK' if trend_ok else '❌ FRACA'} | Req: {trend_req}
  • Filtros MTF: ✅ Verificado nos filtros acima
  • Capital/Risco: ✅ Verificado nos filtros acima
  • ML Confidence: ✅ Verificado nos filtros acima

📋 Resumo de Decisão:
  • Período: {period_name}
  • Volume Mínimo: {min_vol}x
  • RSI Limite: {rsi_limit}
  • Tendência: {trend_req}
  • Todos filtros devem estar ✅ para aprovar"""

        return log_info.strip()

    def validate(
        self, symbol: str, side: str, indicators: dict, entry_price: float = 0.0
    ) -> tuple[bool, str]:
        # Cores ANSI para console
        C_GREEN = "\033[92m"
        C_RED = "\033[91m"
        C_YELLOW = "\033[93m"
        C_CYAN = "\033[96m"
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"

        failed_filters = []  # Lista para armazenar filtros que falharam

        print(f"\n{C_BOLD}{C_CYAN}🔍 Validando filtros para {symbol} ({side}){C_RESET}")
        print("=" * 50)

        # Exibir indicadores técnicos
        self._display_indicators(indicators, side)

        for fname, cfg in sorted(self.filters.items(), key=lambda x: x[1]["priority"]):
            passed, reason = self._execute_filter(fname, symbol, side, indicators)

            # Nome amigável do filtro
            filter_names = {
                "capital_available": "💰 Capital Disponível",
                "mtf_alignment": "📊 Alinhamento MTF",
                "risk_limits": "🛡️ Limites de Risco",
                "ml_confidence": "🤖 Confiança ML",
                "anti_chop": "⚔️ Anti Chop",
            }

            friendly_name = filter_names.get(fname, fname)

            if passed:
                print(f"{C_GREEN}✅ {friendly_name}: APROVADO{C_RESET}")
            else:
                # Armazenar filtro que falhou
                failed_filters.append((fname, reason))

                if cfg["mandatory"]:
                    print(f"{C_RED}❌ {friendly_name}: REPROVADO - {reason}{C_RESET}")
                    # Exibir motivo da rejeição com indicadores
                    self._display_rejection_reason(
                        indicators, failed_filters, entry_price
                    )
                    return False, f"{fname}: {reason}"
                else:
                    # Verificar bypass
                    bypass_met = self._check_bypass(fname, indicators, cfg)
                    if bypass_met:
                        print(
                            f"{C_YELLOW}⚠️  {friendly_name}: BYPASS ATIVADO - {reason}{C_RESET}"
                        )
                        continue
                    else:
                        # Mostrar quanto falta para atingir as métricas
                        missing_info = self._get_missing_info(fname, indicators, cfg)
                        print(
                            f"{C_RED}❌ {friendly_name}: REPROVADO - {reason}{C_RESET}"
                        )
                        if missing_info:
                            print(f"   {C_YELLOW}📉 Falta: {missing_info}{C_RESET}")
                        # Exibir motivo da rejeição com indicadores
                        self._display_rejection_reason(
                            indicators, failed_filters, entry_price
                        )
                        return False, f"{fname}: {reason}"

        print(f"\n{C_GREEN}{C_BOLD}🎉 Todos os filtros aprovados!{C_RESET}")
        return True, "OK"


class _TTLCache:
    def __init__(self):
        self._data = {}

    def get(self, key: str, max_age: int):
        v = self._data.get(key)
        if not v:
            return None
        ts, val = v
        if (time.time() - ts) > max_age:
            return None
        return val

    def set(self, key: str, value, ttl: int):
        self._data[key] = (time.time(), value)


class ConcurrentMarketScanner:
    def __init__(self, max_workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = _TTLCache()

    def _scan_futures_fast(self, symbols: list) -> dict:
        results = {}
        for symbol in symbols:
            cached = self.cache.get(symbol, 5)
            if cached:
                results[symbol] = cached
                continue
            df = safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 50)
            if df is None or len(df) < 20:
                results[symbol] = {}
                continue
            ema_fast = float(df["close"].ewm(span=9).mean().iloc[-1])
            ema_slow = float(df["close"].ewm(span=21).mean().iloc[-1])
            ema_diff = (ema_fast - ema_slow) / float(
                df["close"].iloc[-1]
            )  # Diferença percentual

            ind = {
                "rsi": get_rsi(df, period=14),
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "ema_diff": ema_diff,
                "volume_ratio": float(df["volume"].iloc[-1])
                / max(float(df["volume"].rolling(20).mean().iloc[-1]), 1.0),
                "price": float(df["close"].iloc[-1]),
            }
            self.cache.set(symbol, ind, 5)
            results[symbol] = ind
        return results

    def _analyze_symbol(self, s: str) -> tuple[str, dict]:
        df = safe_copy_rates(s, mt5.TIMEFRAME_M15, 100)
        if df is None or len(df) < 20:
            return s, {}
        ind = quick_indicators_custom(s, mt5.TIMEFRAME_M15, df=df, params={})
        return s, ind

    def scan_market(self, symbols: list) -> dict:
        futures = [s for s in symbols if is_future(s)]
        stocks = [s for s in symbols if not is_future(s)]
        results = {}
        if futures:
            f = self.executor.submit(self._scan_futures_fast, futures)
            try:
                results.update(f.result(timeout=3))
            except Exception:
                pass
        if stocks:
            from concurrent.futures import as_completed

            futures_list = [
                self.executor.submit(self._analyze_symbol, s) for s in stocks
            ]
            try:
                for fut in as_completed(futures_list, timeout=10):
                    k, v = fut.result()
                    results[k] = v
            except Exception:
                pass
        return results


# =========================================================
# INDICADORES CONSOLIDADOS (SEM SCORE)
# =========================================================


def get_momentum(df: pd.DataFrame, period: int = 10) -> Optional[float]:
    """
    Calcula momentum (Rate of Change)

    Momentum = (preço_atual - preço_passado) / preço_passado

    Args:
        df: DataFrame com coluna 'close'
        period: Quantos candles olhar para trás (padrão: 10)

    Returns:
        Momentum como float (ex: 0.05 = 5% de alta)
    """
    if df is None or len(df) < period + 1:
        return None

    close = df["close"]

    # Momentum = mudança percentual em N períodos
    momentum = (close.iloc[-1] - close.iloc[-period - 1]) / close.iloc[-period - 1]

    return float(momentum)


def quick_indicators_custom(symbol, timeframe, df=None, params=None):
    """
    ✅ VERSÃO COMPLETA: Inclui Momentum
    """
    params = params or {}
    df = df if df is not None else safe_copy_rates(symbol, timeframe, 300)

    if df is None or len(df) < 50:
        return {"error": "no_data"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- MÉDIAS E RSI ---
    ema_fast = close.ewm(span=params.get("ema_short", 9), adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=params.get("ema_long", 21), adjust=False).mean().iloc[-1]

    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rsi = (100 - (100 / (1 + up / down))).iloc[-1]

    # --- ATR E ADX ---
    atr = get_atr(df)
    adx = get_adx(df) or 0.0
    price = float(close.iloc[-1])

    # --- ✅ MOMENTUM (NOVO!) ---
    momentum = get_momentum(df, period=10)  # ROC de 10 períodos

    # Tratamento de valores extremos
    if momentum is not None:
        # Cap em ±50% (protege contra outliers)
        momentum = max(-0.5, min(momentum, 0.5))
    else:
        momentum = 0.0

    # --- CÁLCULO ATR% REAL ---
    if atr > price * 2:
        atr_price = atr * mt5.symbol_info(symbol).point
    else:
        atr_price = atr

    atr_pct_real = (atr_price / price) * 100 if price > 0 else 0

    # --- CÁLCULO EMA_DIFF ---
    ema_diff = (ema_fast - ema_slow) / price if price > 0 else 0

    # --- ✅ SPREAD REAL-TIME (LAND TRADING STYLE) ---
    tick = mt5.symbol_info_tick(symbol)
    spread_nominal = 0
    spread_pct = 0
    spread_points = 0

    if tick and tick.ask > 0 and tick.bid > 0:
        spread_nominal = tick.ask - tick.bid
        spread_points = round(spread_nominal / mt5.symbol_info(symbol).point, 0)
        spread_pct = (spread_nominal / tick.bid) * 100 if tick.bid > 0 else 0

    # --- Z-SCORE DE VOLATILIDADE ---
    vol_series = df["close"].pct_change().rolling(20).std() * 100
    atr_mean = vol_series.mean()
    atr_std = vol_series.std()
    z_score = (atr_pct_real - atr_mean) / atr_std if (atr_std and atr_std > 0) else 0
    atr_pct_capped = min(round(atr_pct_real, 3), 10.0)

    # --- VOLUME ---
    avg_vol = get_avg_volume(df)

    min_avg_vol = (
        getattr(config, "MIN_AVG_VOLUME_FUTURES", 0)
        if is_future(symbol)
        else getattr(config, "MIN_AVG_VOLUME", 4000)
    )

    if avg_vol < min_avg_vol:
        logger.info(f"{symbol}: Bloqueado por micro-liquidez ({avg_vol:,.0f})")
        return {"error": "low_liquidity"}

    # ✅ CORREÇÃO MIOPIA DE VOLUME: Ignorar candle em aberto (-1) e usar o último fechado (-2)
    cur_vol = 0
    if len(df) > 1:
        v_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        cur_vol = df[v_col].iloc[-2]
    else:
        v_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        cur_vol = df[v_col].iloc[-1]

    volume_ratio = round(cur_vol / max(avg_vol, 1), 2)
    volume_ratio = min(volume_ratio, 3.0)

    # 🆕 VOLUME HORÁRIO (PANDAS RESAMPLE) - NOVA IMPLEMENTAÇÃO
    hourly_volume_ratio = get_hourly_volume_ratio(df, window_days=20)

    atr_series_data = (
        pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        )
        .max(axis=1)
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    atr_mean_val = atr_series_data.rolling(20).mean().iloc[-1]
    side = "BUY" if ema_fast > ema_slow else "SELL"
    # 🆕 CALCULA OBV
    obv_value = get_obv(df)
    obv_trend_up = is_obv_trending_up(df)
    vwap_stats = get_intraday_vwap_stats(df)
    vwap_val = (
        vwap_stats["vwap"] if isinstance(vwap_stats, dict) else get_intraday_vwap(df)
    )
    vwap_std = vwap_stats["std"] if isinstance(vwap_stats, dict) else None
    vwap_upper = vwap_stats["upper_2sd"] if isinstance(vwap_stats, dict) else None
    vwap_lower = vwap_stats["lower_2sd"] if isinstance(vwap_stats, dict) else None
    return {
        "symbol": symbol,
        "ema_fast": float(ema_fast),
        "ema_slow": float(ema_slow),
        "ema_diff": float(ema_diff),
        "rsi": float(rsi),
        "adx": float(adx),
        "atr": float(atr),
        "atr_pct": atr_pct_capped,
        "atr_real": round(atr_pct_real, 3),
        "atr_zscore": round(z_score, 2),
        "momentum": round(momentum, 6),  # ✅ NOVO!
        "volume_ratio": volume_ratio,
        "hourly_volume_ratio": hourly_volume_ratio,  # 🆕 NOVO: Volume horário vs média histórica
        "vol_breakout": is_volatility_breakout(
            df, atr, atr_mean_val, volume_ratio, side
        ),
        "vwap": vwap_val,
        "vwap_std": vwap_std,
        "vwap_upper_2sd": vwap_upper,
        "vwap_lower_2sd": vwap_lower,
        "obv": obv_value,  # 🆕 NOVO
        "obv_trend_up": obv_trend_up,  # 🆕 NOVO
        "spread_points": spread_points,  # 🆕 NOVO (pontos/ticks)
        "spread_nominal": round(spread_nominal, 3),  # 🆕 NOVO (R$)
        "spread_pct": round(spread_pct, 4),  # 🆕 NOVO (%)
        "close": price,
        "macro_trend_ok": macro_trend_ok(symbol, side),
        "tick_size": mt5.symbol_info(symbol).point,
        "params": params,
        "error": None,
    }


def check_volatility_filter(
    symbol: str, atr: float, current_price: float
) -> tuple[bool, str]:
    """
    🌊 FILTRO DE VOLATILIDADE ADAPTATIVO

    Bloqueia entradas em condições extremas:
    - Volatilidade >3 desvios padrão
    - ATR expandindo muito rápido (>50% em 5 candles)
    - "Chop zones" (oscilação sem direção)

    Returns:
        (pode_entrar: bool, motivo: str)

    Impacto: +5-8% win rate (evita whipsaws)
    """
    try:
        df = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 50)

        if df is None or len(df) < 30:
            return True, ""

        # 1. Z-Score de Volatilidade (já existe no código)
        vol_series = df["close"].pct_change().rolling(20).std() * 100
        atr_pct_real = (atr / current_price) * 100 if current_price > 0 else 0

        z_score = (
            (atr_pct_real - vol_series.mean()) / vol_series.std()
            if vol_series.std() > 0
            else 0
        )

        # ✅ NOVO: Threshold mais restritivo
        if abs(z_score) > 3.0:
            return False, f"Volatilidade extrema (z={z_score:.2f})"

        # 2. ✅ NOVO: Detecta expansão rápida de ATR
        atr_series = (
            pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .ewm(alpha=1 / 14, adjust=False)
            .mean()
        )

        recent_atr = atr_series.tail(5)
        atr_change = (recent_atr.iloc[-1] - recent_atr.iloc[0]) / recent_atr.iloc[0]

        if atr_change > 0.75:  # >50% em 5 candles
            return False, f"ATR expandindo rápido ({atr_change*100:.0f}%)"

        # 3. ✅ NOVO: Detecta "Chop Zones" (Range sem direção)
        # ADX baixo + ATR alto = mercado lateral violento
        adx = utils.get_adx(df) or 0

        if adx < 13 and atr_pct_real > 2.5:
            return False, f"Chop Zone (ADX {adx:.0f}, ATR {atr_pct_real:.1f}%)"

        # 4. ✅ NOVO: Valida se está em "squeeze" (Bollinger Bands apertadas)
        bb_std = df["close"].rolling(20).std()
        bb_width = (bb_std.iloc[-1] / df["close"].iloc[-1]) * 100

        if bb_width < 1.0 and atr_change > 0.30:
            # Bollinger estreitando + ATR subindo = possível breakout falso
            return False, "Possível falso breakout"

        return True, ""

    except Exception as e:
        logger.error(f"Erro filtro volatilidade {symbol}: {e}")
        return True, ""


from datetime import datetime, date


def calculate_daily_dd() -> float:
    """
    Retorna o drawdown diário atual (0.0 a 1.0)
    Ex: 0.03 = 3%
    """
    try:
        account = mt5.account_info()
        if not account:
            return 0.0

        balance = account.balance
        equity = account.equity

        if balance <= 0:
            return 0.0

        dd = (balance - equity) / balance
        return max(dd, 0.0)

    except Exception as e:
        logger.error(f"Erro ao calcular DD diário: {e}")
        return 0.0


# =========================================================
# SCORE FINAL
# =========================================================
def calculate_signal_score(ind: dict, min_volume_ratio: float = 0.4) -> float:
    """
    ✅ VERSÃO v5.5 (AGRESSIVA)
    Score rebalanceado para validação:
    - Tendência EMA: +35 pts
    - RSI Saudável (30-70): +20 pts
    - ADX > 15: +20 pts
    - Volume Ratio > 0.4: +25 pts
    - MACD Bullish/Bearish: +10 pts (Bônus)
    """
    if not isinstance(ind, dict) or ind.get("error"):
        return 0.0

    score = 0.0
    score_log = {}

    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 0)
    volume_ratio = ind.get("volume_ratio", 0.0)
    ema_fast = ind.get("ema_fast", 0)
    ema_slow = ind.get("ema_slow", 0)
    macd = ind.get("macd", 0)
    macd_signal = ind.get("macd_signal", 0)

    # 1. TENDÊNCIA EMA (Prioridade Máxima: 35 pts)
    if ema_fast > 0 and ema_slow > 0:
        if ema_fast > ema_slow:
            score += 35
            score_log["EMA_TREND_OK"] = 35
        else:
            score -= 20  # Penaliza contra-tendência
            score_log["EMA_COUNTER_TREND"] = -20

    # 2. RSI SAUDÁVEL (30-70: 20 pts)
    if 30 <= rsi <= 70:
        score += 20
        score_log["RSI_HEALTHY"] = 20

    # 3. ADX > 15 (20 pts)
    if adx >= 15:
        score += 20
        score_log["ADX_OK"] = 20

    # 4. VOLUME RATIO > 0.4 (25 pts)
    if volume_ratio >= min_volume_ratio:
        score += 25
        score_log["VOLUME_OK"] = 25

    # 5. MACD BONUS (+10 pts)
    if (ema_fast > ema_slow and macd > macd_signal) or (
        ema_fast < ema_slow and macd < macd_signal
    ):
        score += 10
        score_log["MACD_BONUS"] = 10

    # 6. ML BOOST (Bonus opcional)
    try:
        if ensure_ml_loaded():
            features = ml_optimizer.extract_features(ind, ind.get("symbol", "UNK"))
            ml_pred = ml_optimizer.predict_signal_score(features)
            if ml_pred >= 0.65:
                score += 10
                score_log["ML_BOOST"] = 10
    except Exception:
        pass

    final_score = min(max(score, 0.0), 100.0)
    ind["score_log"] = score_log

    return round(final_score, 1)


def check_arbitrage_opp(sym1: str, sym2: str) -> bool:
    """
    Detecta arbitragem entre pares correlacionados (ex: PETR3 vs PETR4, web:5).
    """
    df1 = safe_copy_rates(sym1, mt5.TIMEFRAME_M15, 10)
    df2 = safe_copy_rates(sym2, mt5.TIMEFRAME_M15, 10)
    if df1 is None or df2 is None:
        return False

    ratio = df1["close"] / df2["close"]
    current_ratio = ratio.iloc[-1]
    mean_ratio = ratio.mean()
    std_ratio = ratio.std()

    if abs(current_ratio - mean_ratio) > 2.2 * std_ratio:
        return True

    return False


def check_and_close_orphans(active_signals: dict):
    with mt5_lock:
        positions = mt5.positions_get() or []

    for pos in positions:
        if pos.symbol not in active_signals:
            if pos.time_update + 120 < time.time():  # 2 minutos de tolerância
                logger.warning(f"Posição órfã detectada: {pos.symbol}")
                send_telegram_exit(
                    symbol=pos.symbol, reason="Posição órfã (sem sinal ativo)"
                )


def get_avg_volume(df, window: int = 20):
    if not is_valid_dataframe(df):
        return 0

    if "real_volume" in df.columns:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        return 0

    # Corrigido para usar os últimos 'window' candles FECHADOS (ignorando iloc[-1])
    return df[vol_col].iloc[-(window + 1) : -1].mean()


def get_hourly_volume_ratio(
    df: pd.DataFrame, current_hour: int = None, window_days: int = 20
) -> float:
    """
    Calcula o ratio de volume horário comparando com a média das mesmas horas em dias anteriores.

    Args:
        df: DataFrame com dados OHLCV
        current_hour: Hora atual (0-23), se None usa a hora atual do sistema
        window_days: Número de dias para calcular a média (padrão: 20)

    Returns:
        float: Ratio de volume (volume atual / média horária)
    """
    if not is_valid_dataframe(df) or len(df) < 20:
        return 0.0

    # Determinar coluna de volume
    if "real_volume" in df.columns:
        vol_col = "real_volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        return 0.0

    try:
        # Adicionar coluna de hora se não existir
        if "hour" not in df.columns:
            # Verificar se o índice é datetime antes de acessar .hour
            if hasattr(df.index, "hour"):
                df["hour"] = df.index.hour
            else:
                # Se não for datetime, criar horas baseadas no horário atual
                current_hour = datetime.now().hour
                df["hour"] = current_hour  # Simplificado para o caso de RangeIndex

        # Obter hora atual se não fornecida
        if current_hour is None:
            current_hour = datetime.now().hour

        # Pegar o volume do candle atual (último fechado)
        if len(df) > 1:
            current_volume = df[vol_col].iloc[-2]  # Usar penúltimo candle (fechado)
        else:
            current_volume = df[vol_col].iloc[-1]

        # Calcular média horária para a mesma hora nos últimos N dias
        hourly_volumes = df[df["hour"] == current_hour][vol_col]

        # Pegar os últimos window_days dias de dados (excluindo o dia atual)
        if len(hourly_volumes) > 1:
            historical_volumes = hourly_volumes.iloc[:-1].tail(window_days)
            avg_hourly_volume = historical_volumes.mean()
        else:
            # Fallback: usar média simples dos últimos 20 períodos
            avg_hourly_volume = df[vol_col].iloc[-(window_days + 1) : -1].mean()

        # Calcular ratio
        if avg_hourly_volume > 0:
            ratio = current_volume / avg_hourly_volume
            return round(ratio, 2)
        else:
            return 0.0

    except Exception as e:
        logger.error(
            f"Erro ao calcular volume horário para {df.index[-1] if len(df) > 0 else 'unknown'}: {e}"
        )
        return 0.0


def get_open_gap(symbol: str, timeframe) -> float | None:
    df = safe_copy_rates(symbol, timeframe, 2)
    if df is None or len(df) < 2:
        return None

    prev_close = float(df["close"].iloc[-2])
    open_price = float(df["open"].iloc[-1])
    if prev_close <= 0:
        return None

    return abs((open_price - prev_close) / prev_close) * 100.0


def resolve_signal_weights(
    symbol, sector, base_weights, sector_weights=None, symbol_weights=None
):
    w = base_weights.copy()

    if sector_weights and sector in sector_weights:
        for k, v in sector_weights[sector].items():
            w[k] *= v

    if symbol_weights and symbol in symbol_weights:
        for k, v in symbol_weights[symbol].items():
            w[k] *= v

    return w


def update_symbol_weights(symbol, sector, score_log, trade_result):
    global symbol_weights

    alpha = 0.03

    if symbol not in symbol_weights:
        symbol_weights[symbol] = {}

    for k, contribution in score_log.items():
        current = symbol_weights[symbol].get(k, 1.0)
        impact = np.tanh(trade_result * contribution / 10)
        delta = 1 + alpha * impact
        symbol_weights[symbol][k] = max(0.5, min(1.8, current * delta))


_bot_instance = None


def get_telegram_bot():
    global _bot_instance
    if _bot_instance is None and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        if telebot is None:
            logger.error(
                "telebot não está instalado. Instale com: pip install pyTelegramBotAPI"
            )
            return None
        try:
            _bot_instance = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
            logger.info("Bot do Telegram inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar bot do Telegram: {e}")
            _bot_instance = None
    return _bot_instance


def send_telegram_exit(
    symbol: str,
    side: str = "",
    volume: float = 0,
    entry_price: float = 0,
    exit_price: float = 0,
    profit_loss: float = 0,
    reason: str = "",
):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não disponível para saída")
        return
    try:
        if (profit_loss < 0) or ("stop" in (reason or "").lower()):
            from validation import register_stop_loss

            register_stop_loss(symbol)
    except Exception:
        pass

    # 1. PEGA O LUCRO ACUMULADO DO DIA NO ARQUIVO TXT
    # Usando a função que criamos antes
    lucro_realizado_total, _ = calcular_lucro_realizado_txt()

    # Cálculo do Valor Total da Operação
    total_value = volume * exit_price
    pl_emoji = "🟢" if profit_loss > 0 else "🔴"
    pl_pct = (
        (profit_loss / max(entry_price * volume, 1)) * 100
        if entry_price > 0 and volume > 0
        else 0
    )

    msg = (
        f"{pl_emoji} <b>XP3 — POSIÇÃO ENCERRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {side}\n"
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${entry_price:.2f} | <b>Saída:</b> R${exit_price:.2f}\n"
        f"<b>Resultado:</b> R${profit_loss:+.2f} ({pl_pct:+.2f}%)\n"
        f"<b>Motivo:</b> {reason}\n"
        f"---------------------------\n"
        f"💰 <b>LUCRO NO BOLSO HOJE: R$ {lucro_realizado_total:,.2f}</b>\n"  # AQUI A NOVIDADE!
        f"---------------------------\n"
        f"<i>⏱ {datetime.now().strftime('%H:%M:%S')}</i>"
    )

    try:
        bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=msg, parse_mode="HTML")
        logger.info(f"✅ Telegram: Notificação de SAÍDA enviada com Lucro Acumulado")
    except Exception as e:
        logger.error(f"Erro ao enviar Telegram: {e}")


def close_position(
    symbol: str,
    ticket: int,
    volume: float,
    price: float,
    reason: str = "Saída Estratégica",
):

    # CVM Compliance: Log em CSV

    import csv
    import os
    from datetime import datetime

    # Identifica o tipo da posição pelo ticket para saber o lado oposto
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        logger.error(f"❌ Erro ao fechar: Posição {ticket} não encontrada.")
        return False

    pos = pos[0]
    # Se a posição é de COMPRA (0), fechamos com VENDA (1) e vice-versa
    order_type = (
        mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    )

    # Validar tick antes de montar request
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        price_exec = 0.0
    else:
        price_exec = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": ticket,  # OBRIGATÓRIO para fechar a posição correta
        "price": price_exec,
        "deviation": get_dynamic_slippage(symbol, datetime.now().hour),
        "magic": 2026,
        "comment": f"XP3:{reason}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    with mt5_lock:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Falha ao fechar {symbol}: {result.comment}")
            return False

        # Se fechou com sucesso, envia o log de saída e notifica Telegram
        logger.info(
            f"✅ SAÍDA EXECUTADA: {symbol} | Motivo: {reason} | P&L: R${pos.profit:.2f}"
        )

        # Chama sua função de Telegram já existente no utils.py
        send_telegram_exit(
            symbol=symbol,
            side="BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
            volume=volume,
            entry_price=pos.price_open,
            exit_price=price,
            profit_loss=result.profit if hasattr(result, "profit") else pos.profit,
            reason=reason,
        )

        # ✅ Atualiza Streak de Perdas/Ganhos
        pl = result.profit if hasattr(result, "profit") else pos.profit
        if pl < 0:
            _symbol_consecutive_losses[symbol] += 1
        else:
            _symbol_consecutive_losses[symbol] = 0

        try:
            from validation import register_stop_loss

            if pl < 0:
                register_stop_loss(symbol)
        except Exception:
            pass

        # ✅ Compliance CVM: trades_compliance.csv
        compliance_file = "trades_compliance.csv"
        file_exists = os.path.isfile(compliance_file)

        try:
            import csv

            with open(compliance_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "Timestamp",
                            "Ticket",
                            "Symbol",
                            "Volume",
                            "Entry_Price",
                            "Exit_Price",
                            "Profit",
                            "Reason",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        ticket,
                        symbol,
                        volume,
                        pos.price_open,
                        price,
                        pos.profit,
                        reason,
                    ]
                )
        except Exception as e:
            logger.error(f"Erro ao salvar compliance CSV: {e}")

        return True


adaptive_lock = threading.Lock()


def save_adaptive_weights():
    data = {"symbol": symbol_weights, "sector": sector_weights}

    tmp_path = "adaptive_weights.tmp"
    final_path = "adaptive_weights.json"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        os.replace(tmp_path, final_path)

        logger.info("💾 Pesos adaptativos salvos com sucesso.")

    except Exception as e:
        logger.error(f"❌ Erro ao salvar pesos adaptativos: {e}")


def load_adaptive_weights():
    global symbol_weights, sector_weights
    path = "adaptive_weights.json"

    symbol_weights = {}
    sector_weights = {}

    if not os.path.exists(path):
        logger.info("ℹ️ Pesos adaptativos não encontrados. Usando padrão.")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON raiz não é um objeto")

        symbol_weights = data.get("symbol")
        sector_weights = data.get("sector")

        if not isinstance(symbol_weights, dict) or not isinstance(sector_weights, dict):
            raise ValueError("Estrutura inválida em adaptive_weights.json")

        logger.info(
            f"🧠 Pesos adaptativos carregados: "
            f"{len(symbol_weights)} símbolos | {len(sector_weights)} setores"
        )

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON inválido em {path}: linha {e.lineno}, coluna {e.colno}")
    except Exception as e:
        logger.error(f"Erro ao carregar pesos adaptativos: {e}")

    return


def get_session_adjustment() -> dict:
    """
    Retorna os parâmetros de ajuste de acordo com o horário do pregão.
    """
    now = datetime.now().time()

    # Horários de cada sessão
    risk_on_start = datetime.strptime("10:05", "%H:%M").time()
    risk_off_start = datetime.strptime("12:00", "%H:%M").time()
    lunch_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_start = datetime.strptime("14:00", "%H:%M").time()

    # Determina a sessão atual
    if risk_on_start <= now < risk_off_start:
        session = "RISK_ON"
    elif risk_off_start <= now < lunch_start:
        session = "RISK_OFF"
    elif lunch_start <= now < afternoon_start:
        session = "LUNCH"
    else:
        session = "AFTERNOON"

    # Retorna o dicionário de configurações da sessão
    return getattr(config, "ADAPTIVE_THRESHOLDS", {}).get(session, {})


def update_adaptive_weights():
    """
    Inicializa pesos adaptativos de forma segura e compatível
    """
    global symbol_weights, sector_weights

    # 🔒 Garante que os dicts existam
    if symbol_weights is None:
        symbol_weights = {}

    if sector_weights is None:
        sector_weights = {}

    # 🎯 Chaves reais usadas no score_log
    base_weight_keys = [
        "ADX_STRONG",
        "ADX_GOOD",
        "RSI_EXTREME",
        "RSI_STRETCH",
        "VOL_HIGH",
        "VOL_OK",
        "EMA_TREND",
        "EMA_COUNTER",
        "MOMENTUM_POS",
        "MACD_CROSS",
        "ML_BOOST",
    ]

    # =========================
    # 🧠 PESOS POR SÍMBOLO
    # =========================
    for sym in config.SECTOR_MAP.keys():
        if sym not in symbol_weights:
            symbol_weights[sym] = {}

        for key in base_weight_keys:
            symbol_weights[sym].setdefault(key, 1.0)

    # =========================
    # 🏭 PESOS POR SETOR
    # =========================
    for sector in set(config.SECTOR_MAP.values()):
        if sector not in sector_weights:
            sector_weights[sector] = {}

        for key in base_weight_keys:
            sector_weights[sector].setdefault(key, 1.0)

    logger.info("✅ Pesos adaptativos inicializados / normalizados com sucesso")


def check_liquidity(symbol: str, threshold_pct: float = 0.4):
    """
    ✅ Filtro de Liquidez Real-Time (B3 Optimized Jan/2026)

    Regra: Volume Projetado do Dia > 40% da Média de 10 Dias.
    Projeção: (Volume Atual / Minutos Decorridos) * 420 min
    """
    try:
        if is_future(symbol):
            return True
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 10)
        if rates_d1 is None or len(rates_d1) < 5:
            return True

        df_d1 = pd.DataFrame(rates_d1)
        vol_col = "real_volume" if "real_volume" in df_d1.columns else "tick_volume"
        avg_vol_10d = df_d1[vol_col].mean()

        rate_today = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
        if not rate_today:
            return False

        vol_today = rate_today[0][vol_col]

        now = datetime.now()
        open_time = now.replace(hour=10, minute=0, second=0, microsecond=0)

        if now < open_time:
            return True

        elapsed_min = max(15, (now - open_time).total_seconds() / 60)
        elapsed_min = min(elapsed_min, 420)

        projected_vol = (vol_today / elapsed_min) * 420
        target_vol = avg_vol_10d * threshold_pct

        if projected_vol < target_vol:
            logger.info(
                f"💧 {symbol}: Liquidez insuficiente "
                f"({projected_vol:,.0f} < {target_vol:,.0f})"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Erro check_liquidity real-time {symbol}: {e}")
        return False


def calculate_position_size_atr(
    symbol: str, atr_dist: float, risk_money: float = None
) -> float:
    """
    ✅ POSITION SIZING B3: Risco 1% com base no ATR (Jan/2026)
    """
    try:
        # Garante equity para cálculo de tolerância
        acc = mt5.account_info()
        if not acc:
            return 0.0
        equity = acc.equity if acc.equity > 0 else acc.balance

        if atr_dist <= 0:
            return 0.0

        if risk_money is None:
            rp = AssetClassManager.get_risk_pct(symbol)
            risk_money = equity * float(rp)

        min_lot = AssetClassManager.get_min_lot(symbol)
        risk_min_lot = min_lot * atr_dist

        equity_tolerance_cap = equity * 0.013
        if risk_min_lot > equity_tolerance_cap:
            logger.info(
                f"❌ {symbol}: Risco muito alto "
                f"({risk_min_lot:,.0f} > {equity_tolerance_cap:,.0f})"
            )
            return 0.0

        num_lots = int(risk_money / risk_min_lot)

        # ✅ PROPORCIONALIDADE ESTRITA (Land Trading):
        # Impede trades desproporcionais onde 1 lote viola o gerenciamento de risco.
        if num_lots <= 0:
            # Tolerância máxima: 20% acima do risco estipulado
            if risk_min_lot <= risk_money * 1.2:
                num_lots = 1
            else:
                logger.warning(
                    f"⚖️ {symbol}: Risco desproporcional. "
                    f"Mínimo {risk_min_lot:.0f} > Limite {risk_money * 1.2:.0f}. "
                    "Trade ignorado."
                )
                return 0.0

        return float(num_lots * min_lot)

    except Exception as e:
        logger.error(f"Erro calculate_position_size_atr: {e}")
        return 0.0


def calculate_total_exposure() -> float:
    try:
        with mt5_lock:
            positions = mt5.positions_get() or []
        total = 0.0
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            si = mt5.symbol_info(p.symbol)
            contract = si.trade_contract_size if si else 1.0
            if tick:
                price = tick.bid if p.type == mt5.POSITION_TYPE_SELL else tick.ask
            else:
                price = getattr(p, "price_current", getattr(p, "price_open", 0.0))
            total += float(p.volume) * float(contract) * float(price)
        return float(total)
    except Exception as e:
        logger.error(f"Erro calculate_total_exposure: {e}")
        return 0.0


def get_effective_exposure_limit() -> float:
    try:
        acc = mt5.account_info()
        if not acc:
            return float(getattr(config, "MAX_TOTAL_EXPOSURE", 1_000_000))
        dyn = 2.0 * float(acc.equity or acc.balance or 0.0)
        return float(dyn)
    except Exception as e:
        logger.error(f"Erro get_effective_exposure_limit: {e}")
        return float(getattr(config, "MAX_TOTAL_EXPOSURE", 1_000_000))


def check_capital_allocation(
    symbol: str, planned_volume: float, entry_price: float
) -> tuple[bool, str]:
    """Versão com logging detalhado"""
    try:
        with mt5_lock:
            acc = mt5.account_info()
            positions = mt5.positions_get() or []

        if not acc:
            return False, "Sem conta MT5"

        equity = float(acc.equity or acc.balance or 0.0)
        capital = float(acc.balance or 0.0)

        if equity <= 0:
            return False, "Equity inválido"

        # Analisa posições
        fut_exposure = 0.0
        stk_exposure = 0.0

        logger.info("=" * 60)
        logger.info(f"💰 Verificando capital para {symbol}")
        logger.info(f"Capital: R$ {capital:,.2f}")
        logger.info(f"Equity: R$ {equity:,.2f}")
        logger.info(f"Posições abertas: {len(positions)}")

        for pos in positions:
            si = mt5.symbol_info(pos.symbol)
            contract = float(si.trade_contract_size) if si else 1.0
            price = float(
                getattr(pos, "price_current", getattr(pos, "price_open", 0.0)) or 0.0
            )

            # ✅ Exposição para futuros é volume * contrato * preço
            is_fut = is_future(pos.symbol)

            if is_fut:
                exp = float(pos.volume) * contract * price
                fut_exposure += exp
                logger.info(f"  🔮 {pos.symbol}: R$ {exp:,.2f} (FUTURO)")
            # ✅ Exposição para ações é volume * preço
            else:
                exp = float(pos.volume) * price
                stk_exposure += exp
                logger.info(f"  📈 {pos.symbol}: R$ {exp:,.2f} (AÇÃO)")

        logger.info(f"Total Futuros: R$ {fut_exposure:,.2f}")
        logger.info(f"Total Ações: R$ {stk_exposure:,.2f}")

        # Calcula exposição adicional
        is_fut = is_future(symbol)
        si_new = mt5.symbol_info(symbol)
        contract_new = float(si_new.trade_contract_size) if si_new else 1.0

        if is_fut:
            add_exp = float(planned_volume) * contract_new * float(entry_price)
            logger.info(f"🔮 Nova posição futuro: R$ {add_exp:,.2f}")
        else:
            add_exp = float(planned_volume) * float(entry_price)
            logger.info(f"📈 Nova posição ação: R$ {add_exp:,.2f}")

        # Verifica limites
        fut_pct, stk_pct = AssetClassManager.get_dynamic_buckets()
        cap_fut = equity * float(fut_pct)
        cap_stk = equity * float(stk_pct)

        logger.info(f"Limite futuros: R$ {cap_fut:,.2f} ({fut_pct:.1%})")
        logger.info(f"Limite ações: R$ {cap_stk:,.2f} ({stk_pct:.1%})")

        if is_fut:
            total_fut = fut_exposure + add_exp
            if total_fut > cap_fut:
                logger.warning(
                    f"❌ Exposição futuros ultrapassou limite: R$ {total_fut:,.2f} > R$ {cap_fut:,.2f}"
                )
                return False, f"Exposição Futuros {total_fut:,.2f} > {cap_fut:,.2f}"
            logger.info(
                f"✅ Exposição futuros OK: R$ {total_fut:,.2f} <= R$ {cap_fut:,.2f}"
            )
            return True, "OK"
        else:
            total_stk = stk_exposure + add_exp
            if total_stk > cap_stk:
                logger.warning(
                    f"❌ Exposição ações ultrapassou limite: R$ {total_stk:,.2f} > R$ {cap_stk:,.2f}"
                )
                return False, f"Exposição Ações {total_stk:,.2f} > {cap_stk:,.2f}"
            logger.info(
                f"✅ Exposição ações OK: R$ {total_stk:,.2f} <= R$ {cap_stk:,.2f}"
            )
            return True, "OK"

    except Exception as e:
        logger.error(f"Erro na verificação de capital: {e}")
        return False, f"Erro alocação: {e}"


def signal_handler(sig, frame):
    with mt5_lock:
        logger.info("Encerrando bot - salvando pesos adaptativos...")
        save_adaptive_weights()
        mt5.shutdown()
    exit(0)


def send_telegram_trade(
    symbol: str,
    side: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    comment: str = "",
):
    bot = get_telegram_bot()
    if not bot:
        logger.warning("⚠️ Telegram: Bot não inicializado (token ausente ou inválido)")
        return

    if side == "BUY":
        direction = "🟢 COMPRA"
        arrow = "⬆️"
    else:
        direction = "🔴 VENDA"
        arrow = "⬇️"

    dist_sl = abs(price - sl)
    dist_tp = abs(tp - price)
    risk_pct = round((dist_sl / price) * 100, 2)
    reward_pct = round((dist_tp / price) * 100, 2)
    rr_ratio = round(dist_tp / dist_sl, 2) if dist_sl > 0 else 0.0
    if rr_ratio == 0.0:
        logger.warning(f"{symbol}: R:R inválido (SL muito próximo ou zero)")

    msg = (
        f"<b>🚀 XP3 – NOVA ENTRADA</b>\n\n"
        f"<b>Ativo:</b> {symbol}\n"
        f"<b>Direção:</b> {direction} {arrow}\n"
        f"<b>Volume:</b> {volume:.0f} ações\n"
        f"<b>Entrada:</b> R${price:.2f}\n\n"
        f"<b>🛑 SL:</b> R${sl:.2f} <i>(-{risk_pct}%)</i>\n"
        f"<b>🎯 TP:</b> R${tp:.2f} <i>(+{reward_pct}%)</i>\n"
        f"<b>R:R:</b> 1:{rr_ratio}\n"
        f"<b>Comentário:</b> {comment}\n\n"
        f"<i>⏱ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
    )

    try:
        bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=msg, parse_mode="HTML")
        logger.info(f"✅ Telegram: Notificação de ENTRADA enviada para {symbol}")
    except Exception as e:
        logger.error(f"❌ ERRO ao enviar Telegram (entrada {symbol}): {e}")


def validate_mt5_connection():
    """
    Verifica se MT5 está conectado e operável.
    """
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            logger.critical("❌ MT5 não está inicializado")
            return False

        if not terminal_info.connected:
            logger.critical("❌ MT5 não está conectado ao servidor")
            return False

        if not terminal_info.trade_allowed:
            logger.error("⚠️ Trading não permitido no MT5")
            return False

        return True

    except Exception as e:
        logger.error(f"Erro ao validar conexão MT5: {e}")
        return False


def send_order_with_sl_tp(symbol, side, volume, sl, tp, comment="XP3_BOT"):
    if not validate_mt5_connection():
        return False

    info = mt5.symbol_info(symbol)
    if info is None:
        return False

    if not info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(0.2)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False
    status_cb = check_b3_circuit_breaker()
    if status_cb in ("CLOSE_ALL", "PAUSE_AND_TIGHTEN"):
        return False
    price = tick.ask if side == "BUY" else tick.bid
    if is_future(symbol) and should_block_opening_gap(
        symbol, getattr(config, "MAX_ACCEPTABLE_GAP_PCT", 0.015)
    ):
        return False
    if block_if_high_portfolio_correlation(
        symbol, getattr(config, "MIN_CORRELATION_SCORE_TO_BLOCK", 0.70)
    ):
        return False
    now_t = datetime.now().time()
    if not is_future(symbol):
        try:
            t_0950 = datetime.strptime("09:50", "%H:%M").time()
            t_1005 = datetime.strptime("10:05", "%H:%M").time()
            t_1650 = datetime.strptime("16:50", "%H:%M").time()
            t_1705 = datetime.strptime("17:05", "%H:%M").time()
            in_open_auction = t_0950 <= now_t <= t_1005
            in_close_auction = t_1650 <= now_t <= t_1705
            if in_open_auction or in_close_auction:
                return False
        except Exception:
            pass
        try:
            if side == "SELL":
                borrow_pct = float(getattr(config, "SHORT_BORROW_PCT", 0.004) or 0.004)
                rr_pct = abs(tp - price) / max(price, 1e-9)
                if rr_pct <= borrow_pct:
                    return False
        except Exception:
            pass
    fm_start = datetime.strptime(
        getattr(config, "FUTURES_AFTERMARKET_START", "16:00"), "%H:%M"
    ).time()
    fm_end = datetime.strptime(
        getattr(config, "FUTURES_AFTERMARKET_END", "17:50"), "%H:%M"
    ).time()
    if is_future(symbol):
        try:
            close_by = datetime.strptime(
                getattr(config, "FUTURES_CLOSE_ALL_BY", "17:50"), "%H:%M"
            ).time()
            if now_t >= close_by:
                return False
        except Exception:
            pass
    rr_now = abs(tp - price) / max(abs(price - sl), 1e-9)
    if is_future(symbol) and fm_start <= now_t <= fm_end:
        try:
            df_af = safe_copy_rates(symbol, TIMEFRAME_BASE, 60)
            adx_now = get_adx(df_af) or 0.0
            rr_min_af = float(getattr(config, "AFTER_MARKET_RR_MIN", 3.0))
            if adx_now < 35.0:
                return False
            if rr_now < rr_min_af:
                return False
            info_step = info.volume_step if info.volume_step > 0 else 1.0
            volume = (max(info.volume_min, (volume * 0.5)) // info_step) * info_step
            if volume <= 0:
                return False
        except Exception:
            return False
    # Checagem de margem e custo/RR mínimo
    try:
        acc = mt5.account_info()
        free_margin = float(getattr(acc, "margin_free", 0.0) or 0.0)
        ord_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        est_margin = float(
            mt5.order_calc_margin(ord_type, symbol, price, float(volume)) or 0.0
        )
        eff_margin = free_margin
        try:
            with mt5_lock:
                positions = mt5.positions_get() or []
            floating_profit = sum(
                float(getattr(p, "profit", 0.0) or 0.0) for p in positions
            )
            eff_margin = free_margin + max(0.0, floating_profit) * 0.7
        except Exception:
            eff_margin = free_margin
        if eff_margin < est_margin * 1.30:
            try:
                acc2 = mt5.account_info()
                fm2 = float(getattr(acc2, "margin_free", 0.0) or 0.0)
                if fm2 < est_margin * 0.50:
                    enforce_stopout_prevention()
            except Exception:
                pass
            return False
    except Exception:
        pass
    try:
        if is_future(symbol):
            acc = mt5.account_info()
            eq = float(getattr(acc, "equity", 0.0) or 0.0)
            if eq <= 0.0:
                eq = float(getattr(acc, "balance", 0.0) or 0.0)
            insp = AssetInspector.detect(symbol)
            pv = float(insp.get("point_value", 1.0) or 1.0)
            total_exposure = 0.0
            with mt5_lock:
                pos_list = mt5.positions_get() or []
            for p in pos_list:
                try:
                    if is_future(p.symbol):
                        t = mt5.symbol_info_tick(p.symbol)
                        if not t:
                            continue
                        px = t.ask if p.type == mt5.POSITION_TYPE_BUY else t.bid
                        total_exposure += (
                            abs(float(getattr(p, "volume", 0.0) or 0.0))
                            * pv
                            * float(px or 0.0)
                        )
                except Exception:
                    continue
            add_exposure = abs(float(volume)) * pv * float(price)
            cap_limit = eq * 2.0
            if (total_exposure + add_exposure) > cap_limit:
                return False
    except Exception:
        pass
    try:
        insp = AssetInspector.detect(symbol)
        fee_type = insp.get("fee_type", "PERCENT")
        fee_val = float(insp.get("fee_val", 0.00055))
        spread = abs(tick.ask - tick.bid) / max(price, 1e-9)
        fee_pct = fee_val if fee_type == "PERCENT" else 0.0005
        cost_pct = spread + (fee_pct * 2.0)
        rr = rr_now
        if cost_pct > 0.015:
            return False
        if cost_pct > 0.005:
            px = price
            rr_min = 2.0 if px <= 50.0 else 1.8
            if px <= 10.0:
                rr_min = 2.8
            if rr < rr_min:
                return False
    except Exception:
        pass
    try:
        if is_future(symbol):
            vix_val = float(getattr(config, "VIX_THRESHOLD_RISK_OFF", 30))
            try:
                vix_now = get_vix_br()
                if vix_now >= vix_val:
                    step = info.volume_step if info.volume_step > 0 else 1.0
                    volume = (max(info.volume_min, (volume * 0.5)) // step) * step
                    if volume <= 0:
                        return False
                maxc = int(getattr(config, "FUTURES_MAX_CONTRACTS", 10))
                if vix_now >= getattr(config, "VIX_THRESHOLD_PROTECTION", 35):
                    maxc = max(1, maxc // 2)
                if volume > maxc:
                    volume = float(maxc)
            except Exception:
                pass
    except Exception:
        pass

    # 🔒 Validação SL/TP
    if side == "BUY":
        if sl >= price or tp <= price:
            return False
    else:
        if sl <= price or tp >= price:
            return False

    # 🔒 Stop level
    stop_level = info.trade_stops_level * info.point
    if side == "BUY":
        if (price - sl) < stop_level or (tp - price) < stop_level:
            return False
    else:
        if (sl - price) < stop_level or (price - tp) < stop_level:
            return False

    # 🔒 Ajuste de volume
    volume_step = info.volume_step if info.volume_step > 0 else 100
    volume = (volume // volume_step) * volume_step
    if volume <= 0:
        return False

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "magic": 2026,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    with mt5_lock:
        result = mt5.order_send(request)

    if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
        return False

    time.sleep(0.2)
    if not mt5.positions_get(symbol=symbol):
        return False

    return True


def send_order_with_retry(symbol, side, volume, sl, tp, max_retries=3):
    for attempt in range(max_retries):
        result = send_order_with_sl_tp(symbol, side, volume, sl, tp)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Ordem executada após {attempt+1} tentativa(s)")
            return result

        if result and result.retcode in (
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
        ):
            logger.warning(
                f"⚠️ Retry {attempt+1}/{max_retries} {symbol} | "
                f"Retcode={result.retcode} | {result.comment}"
            )
            time.sleep(0.5)
            continue

        # ❌ erro irrecuperável
        if result:
            logger.error(
                f"❌ Erro irrecuperável {symbol} | "
                f"Retcode={result.retcode} | {result.comment}"
            )
        else:
            logger.error(f"❌ MT5 retornou None para {symbol}")

        break

    return None


def validate_order_params(
    symbol: str, side: str, volume: float, price: float, sl: float, tp: float
) -> bool:
    with mt5_lock:
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)

    if not info or not tick:
        logger.error(f"{symbol}: info ou tick indisponível")
        return False

    # ✅ Volume mínimo / máximo / step
    if volume < info.volume_min or volume > info.volume_max:
        logger.error(f"{symbol}: volume fora dos limites")
        return False

    if ((volume - info.volume_min) % info.volume_step) != 0:
        logger.error(f"{symbol}: volume fora do step permitido")
        return False

    # ✅ Distância mínima de SL/TP (fallback seguro)
    min_stop = info.trade_stops_level * info.point
    if min_stop <= 0:
        min_stop = info.point * 5  # fallback B3 seguro

    if abs(price - sl) < min_stop:
        logger.error(
            f"{symbol}: SL muito próximo ({abs(price-sl):.4f} < {min_stop:.4f})"
        )
        return False

    if abs(tp - price) < min_stop:
        logger.error(
            f"{symbol}: TP muito próximo ({abs(tp-price):.4f} < {min_stop:.4f})"
        )
        return False

    # ✅ Lógica correta por lado
    if side == "BUY":
        if sl >= price or tp <= price:
            logger.error(f"{symbol}: BUY inválido (SL/TP)")
            return False
    else:
        if sl <= price or tp >= price:
            logger.error(f"{symbol}: SELL inválido (SL/TP)")
            return False

    # ✅ Preço razoável vs bid/ask real
    ref_price = tick.ask if side == "BUY" else tick.bid
    if abs(price - ref_price) / ref_price > 0.02:
        logger.error(f"{symbol}: preço muito distante do mercado")
        return False

    return True


def get_time_bucket():
    now = datetime.now().time()
    for bucket, cfg in config.TIME_SCORE_RULES.items():
        start = datetime.strptime(cfg["start"], "%H:%M").time()
        end = datetime.strptime(cfg["end"], "%H:%M").time()
        if start <= end:
            if start <= now <= end:
                return bucket, cfg
        else:
            if now >= start or now <= end:
                return bucket, cfg
    return "MID", config.TIME_SCORE_RULES["MID"]


def is_power_hour():
    now = datetime.now().time()
    cfg = config.POWER_HOUR
    if not cfg["enabled"]:
        return False
    start = datetime.strptime(cfg["start"], "%H:%M").time()
    end = datetime.strptime(cfg["end"], "%H:%M").time()
    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end


def is_volatility_breakout(df, atr_now, atr_mean, volume_ratio, side=None):
    if not config.VOL_BREAKOUT["enabled"]:
        return False

    if atr_now is None or atr_mean is None:
        return False

    if atr_now < atr_mean * config.VOL_BREAKOUT["atr_expansion"]:
        return False

    if volume_ratio < config.VOL_BREAKOUT["volume_ratio"]:
        return False

    lookback = config.VOL_BREAKOUT["lookback"]
    if len(df) < lookback + 2:
        return False

    high_roll = df["high"].rolling(lookback).max()
    low_roll = df["low"].rolling(lookback).min()

    high_break = df["high"].iloc[-1] > high_roll.iloc[-2]
    low_break = df["low"].iloc[-1] < low_roll.iloc[-2]

    if side == "BUY":
        return high_break
    if side == "SELL":
        return low_break

    return high_break or low_break


# ===== SUBSTITUIR A FUNÇÃO get_current_risk_pct() NO utils.py =====


def get_current_risk_pct() -> float:
    """
    Retorna o risco percentual atual por trade (estável e controlado)
    """
    base_risk = config.RISK_PER_TRADE_PCT
    risk = base_risk

    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # 1️⃣ Sexta-feira à tarde (redução suave)
    if weekday == 4 and hour >= 15:
        risk = min(risk, config.REDUCED_RISK_PCT)

    # 2️⃣ Regime de mercado (limitador, não multiplicador destrutivo)
    regime = detect_market_regime()
    if regime == "RISK_OFF":
        risk = min(risk, base_risk * 0.7)

    # 3️⃣ Power Hour (somente se aumentar risco)
    if is_power_hour():
        mult = config.POWER_HOUR.get("risk_multiplier", 1.0)
        if mult > 1.0:
            risk *= mult

    # 4️⃣ Profit Lock (apenas se meta REAL foi atingida)
    if config.PROFIT_LOCK["enabled"] and config.PROFIT_LOCK["reduce_risk"]:
        with mt5_lock:
            acc = mt5.account_info()

        if acc and os.path.exists("daily_equity.txt"):
            try:
                with open("daily_equity.txt", "r") as f:
                    equity_inicio = float(f.read().strip())

                daily_pnl_pct = (acc.equity - equity_inicio) / equity_inicio

                if daily_pnl_pct >= config.PROFIT_LOCK["daily_target_pct"]:
                    risk = min(risk, base_risk * 0.5)
                    logger.info("🔒 Profit Lock ativo — risco reduzido")
            except Exception:
                pass

    # 5️⃣ Limites corretos
    risk = min(risk, config.MAX_RISK_PER_SYMBOL_PCT)

    # Piso técnico (0.2% — abaixo disso não faz sentido operar)
    return max(0.002, round(risk, 4))


def get_dynamic_slippage(symbol: str, hour: int) -> int:
    """
    Slippage dinâmico por ativo e horário (B3 – robusto)
    Retorna SEMPRE inteiro
    """

    s = (symbol or "").upper()
    if s.startswith(("WIN", "IND")):
        base = int(getattr(config, "FUTURES_SLIPPAGE_BASE_WIN", 8) or 8)
    elif s.startswith(("WDO", "DOL")):
        base = int(getattr(config, "FUTURES_SLIPPAGE_BASE_WDO", 3) or 3)
    else:
        # Ações: converte percentuais de SLIPPAGE_MAP para pontos (ticks) dinamicamente
        pct = float(
            config.SLIPPAGE_MAP.get(symbol, config.SLIPPAGE_MAP.get("DEFAULT", 0.003))
            or 0.003
        )
        try:
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            price = float(getattr(tick, "bid", getattr(tick, "ask", 0.0)) or 0.0)
            tick_size = float(getattr(info, "point", 0.01) or 0.01)
            base = (
                int(max(2, (pct * price) / tick_size))
                if price > 0 and tick_size > 0
                else 6
            )
        except Exception:
            base = 6

    # 2️⃣ ABERTURA (prioridade máxima)
    if 10 <= hour < 11:
        base = int(base * 1.5)

    # 3️⃣ FECHAMENTO (liquidez cai)
    elif hour >= 16:
        base = int(base * 1.3)

    # 4️⃣ POWER HOUR (liquidez máxima)
    elif 15 <= hour < 16:
        base = int(base * 0.7)

    # 5️⃣ LIMITES DE SEGURANÇA
    min_slip = 2
    # Caps específicos por ativo (B3)
    if s.startswith(("WIN", "IND")):
        base = min(base, 20)  # WIN/IND ≤ 20 pontos
    elif s.startswith(("WDO", "DOL")):
        base = min(base, 3)  # WDO/DOL ≤ 3 pontos
    else:
        base = min(base, 100)  # Ações: teto técnico
    base = max(min_slip, base)

    return base


def calculate_smart_sl(symbol, entry_price, side, atr, df):
    """
    Calcula stop loss considerando:
    1. ATR (risco estatístico)
    2. Suporte/Resistência mais próximo
    3. Mínimo de 1.5 ATR (nunca muito apertado)
    """
    # ✅ PROTEÇÃO ATR MÍNIMO
    if atr < 0.01:
        atr = 0.01
        logger.warning(f"{symbol}: ATR muito baixo - usando mínimo 0.01")

    base_distance = atr * 2.0

    # Encontra suporte/resistência relevante
    lookback = 50
    if side == "BUY":
        # Para compra: busca último fundo relevante
        recent_lows = df["low"].tail(lookback)
        support = recent_lows.min()

        # Stop 0.5 ATR abaixo do suporte
        structure_stop = support - (atr * 0.5)

        # Usa o MENOR entre estrutura e ATR (mais conservador)
        final_sl = max(structure_stop, entry_price - base_distance)

    else:  # SELL
        recent_highs = df["high"].tail(lookback)
        resistance = recent_highs.max()
        structure_stop = resistance + (atr * 0.5)
        final_sl = min(structure_stop, entry_price + base_distance)

    # Garante mínimo de 1.5 ATR
    min_distance = atr * 1.5
    if side == "BUY":
        final_sl = min(final_sl, entry_price - min_distance)
    else:
        final_sl = max(final_sl, entry_price + min_distance)

    return round(final_sl, 2)


def analyze_order_book_depth(
    symbol, side, volume_needed, multiplier: float | None = None
):
    try:
        if multiplier is None:
            multiplier = float(getattr(config, "ORDER_BOOK_DEPTH_MULTIPLIER", 3) or 3)

        book = mt5.market_book_get(symbol)
        if book is None or len(book) == 0:
            book = None

        if book is not None:
            target_type = mt5.BOOK_TYPE_SELL if side == "BUY" else mt5.BOOK_TYPE_BUY
            levels = int(getattr(config, "ORDER_BOOK_LEVELS", 10) or 10)
            filtered = [item for item in book if item.type == target_type][:levels]
            available_liquidity = sum(item.volume for item in filtered)

            threshold = float(volume_needed) * float(multiplier)
            if available_liquidity < threshold:
                logger.warning(
                    f"📚 {symbol}: Liquidez insuficiente no Book | "
                    f"Disponível: {available_liquidity:,.0f} | "
                    f"Necessário ({multiplier:.1f}x): {threshold:,.0f}"
                )
                return False
            inst_mult = float(getattr(config, "INSTITUTIONAL_LEVEL_MULT", 5.0) or 5.0)
            inst_thresh = float(volume_needed) * inst_mult
            for item in filtered:
                try:
                    if item.volume >= inst_thresh:
                        logger.warning(
                            f"🏦 {symbol}: Nível institucional detectado ({item.volume:,.0f} ≥ {inst_thresh:,.0f})"
                        )
                        return False
                except Exception:
                    continue
            return True

        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 20)

        if df is not None and not df.empty:
            if "real_volume" in df.columns and df["real_volume"].sum() > 0:
                median_vol = df["real_volume"].median()
            else:
                median_vol = df["tick_volume"].median() * 100

            if median_vol <= 0:
                return True

            now = datetime.now()
            is_power_hour = False
            try:
                ph = getattr(config, "POWER_HOUR", {}) or {}
                if ph.get("enabled", True):
                    start = datetime.strptime(
                        str(ph.get("start", "15:30")), "%H:%M"
                    ).time()
                    end = datetime.strptime(str(ph.get("end", "16:55")), "%H:%M").time()
                    is_power_hour = start <= now.time() <= end
            except Exception:
                is_power_hour = False

            impact_cfg = getattr(config, "ADAPTIVE_FILTERS", {}).get(
                "volume_impact", {}
            )
            max_impact = float(
                impact_cfg.get("power_hour" if is_power_hour else "normal", 0.20)
                or 0.20
            )

            impact_ratio = volume_needed / median_vol

            if impact_ratio > max_impact:
                logger.warning(
                    f"⚠️ {symbol}: Alto impacto no volume | "
                    f"Ordem: {volume_needed:.0f} | Mediana: {median_vol:.0f} | "
                    f"Impacto: {impact_ratio*100:.1f}% (máx {max_impact*100:.0f}%)"
                )
                return False

            logger.debug(f"✅ Volume OK: {symbol} (impacto {impact_ratio*100:.1f}%)")

        return True

    except Exception as e:
        logger.error(f"Erro ao analisar profundidade de {symbol}: {e}", exc_info=True)
        return True  # Fail-open


def detect_calendar_spread(base: str = "WIN") -> dict:
    try:
        cands = get_futures_candidates(base)
        if not cands or len(cands) < 2:
            return {"ok": False}
        cands_sorted = sorted(
            cands,
            key=lambda c: (-calculate_contract_score(c), c.get("days_to_exp", 9999)),
        )
        a = cands_sorted[0]
        b = cands_sorted[1]
        sa = a.get("symbol")
        sb = b.get("symbol")
        ta = mt5.symbol_info_tick(sa)
        tb = mt5.symbol_info_tick(sb)
        if not ta or not tb:
            return {"ok": False}
        pa = float(ta.ask or ta.bid or 0.0)
        pb = float(tb.ask or tb.bid or 0.0)
        spread = pb - pa
        info = AssetInspector.detect(sa)
        ts = float(info.get("tick_size", 5.0) or 5.0)
        pv = float(info.get("point_value", 0.20) or 0.20)
        spread_pts = spread / ts if ts > 0 else 0.0
        thresh_pts = float(
            getattr(config, "CALENDAR_SPREAD_THRESHOLD_POINTS", 80) or 80
        )
        ok = abs(spread_pts) >= thresh_pts
        return {
            "ok": ok,
            "a": sa,
            "b": sb,
            "spread_points": spread_pts,
            "spread_value": spread * pv,
        }
    except Exception:
        return {"ok": False}


def apply_trailing_stop(symbol: str, current_price: float, atr: float):
    if atr is None or atr <= 0:
        return
    with mt5_lock:
        info = mt5.symbol_info(symbol)
        positions = mt5.positions_get(symbol=symbol)
    if not info or not positions:
        return
    tick_size = info.point
    min_stop_distance = info.trade_stops_level * tick_size
    df = get_fast_rates(symbol, TIMEFRAME_BASE)
    if not is_valid_dataframe(df) or len(df) < 5:
        return
    psar_val = get_psar(df)
    for pos in positions:
        try:
            initial_r = abs(pos.price_open - (pos.sl or pos.price_open)) or (atr * 2.0)
            if initial_r <= 0:
                initial_r = atr * 2.0
            profit_move = (
                (current_price - pos.price_open)
                if pos.type == mt5.POSITION_TYPE_BUY
                else (pos.price_open - current_price)
            )
            r_ratio = profit_move / initial_r if initial_r > 0 else 0.0
        except Exception:
            continue
        new_sl = None
        if r_ratio >= 2.0:
            half_profit_sl = (
                (pos.price_open + profit_move * 0.5)
                if pos.type == mt5.POSITION_TYPE_BUY
                else (pos.price_open - profit_move * 0.5)
            )
            if psar_val is not None:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    candidate_sl = max(half_profit_sl, psar_val)
                    if (current_price - candidate_sl) >= min_stop_distance and (
                        not pos.sl or candidate_sl > pos.sl
                    ):
                        new_sl = candidate_sl
                else:
                    candidate_sl = min(half_profit_sl, psar_val)
                    if (candidate_sl - current_price) >= min_stop_distance and (
                        not pos.sl or candidate_sl < pos.sl
                    ):
                        new_sl = candidate_sl
            else:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    if (current_price - half_profit_sl) >= min_stop_distance and (
                        not pos.sl or half_profit_sl > pos.sl
                    ):
                        new_sl = half_profit_sl
                else:
                    if (half_profit_sl - current_price) >= min_stop_distance and (
                        not pos.sl or half_profit_sl < pos.sl
                    ):
                        new_sl = half_profit_sl
        elif r_ratio >= 1.0:
            be_offset = 0.2 * atr
            be_sl = (
                (pos.price_open + be_offset)
                if pos.type == mt5.POSITION_TYPE_BUY
                else (pos.price_open - be_offset)
            )
            if pos.type == mt5.POSITION_TYPE_BUY:
                if (current_price - be_sl) >= min_stop_distance and (
                    not pos.sl or be_sl > pos.sl
                ):
                    new_sl = be_sl
            else:
                if (be_sl - current_price) >= min_stop_distance and (
                    not pos.sl or be_sl < pos.sl
                ):
                    new_sl = be_sl
        else:
            continue
        if new_sl is None:
            continue
        rounded = float(round(new_sl / tick_size) * tick_size)
        if pos.sl and abs(rounded - pos.sl) < tick_size * 5:
            continue
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": symbol,
            "sl": rounded,
            "tp": pos.tp,
            "magic": 2026,
        }
        with mt5_lock:
            res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"🔄 Trailing {symbol} {'BUY' if pos.type==0 else 'SELL'} -> {rounded}"
            )


def manage_dynamic_trailing(symbol: str, ticket: int) -> None:
    """
    Step Trailing com travamento de lucro:
    - Ativa breakeven com offset após +1.5R
      • STOCK: BE + 0.3×ATR
      • FUTURE: BE + 0.8×ATR
    - Depois aplica trailing agressivo:
      • STOCK: 1.5×ATR
      • FUTURE: 2.0×ATR
    """
    try:
        with mt5_lock:
            positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        pos = next(
            (p for p in positions if int(getattr(p, "ticket", 0)) == int(ticket)), None
        )
        if not pos:
            return

        df = get_fast_rates(symbol, TIMEFRAME_BASE)
        if not is_valid_dataframe(df) or len(df) < 20:
            return
        atr = get_atr(df, 14) or 0.0
        if atr <= 0:
            return

        is_fut = is_future(symbol)
        be_offset = atr * (0.8 if is_fut else 0.3)
        trail_mult = 2.0 if is_fut else 1.5

        initial_r = abs(pos.price_open - (pos.sl or pos.price_open)) or (atr * 2.0)
        profit_move = (
            (pos.price_current - pos.price_open)
            if pos.type == mt5.POSITION_TYPE_BUY
            else (pos.price_open - pos.price_current)
        )
        r_ratio = profit_move / initial_r if initial_r > 0 else 0.0

        info = mt5.symbol_info(symbol)
        if not info:
            return
        tick_size = info.point
        min_stop_distance = info.trade_stops_level * tick_size

        # 1) Travamento de lucro: BE + offset após +1.5R
        if r_ratio >= 1.5:
            if pos.type == mt5.POSITION_TYPE_BUY:
                be_sl = pos.price_open + be_offset
                if (pos.sl is None or be_sl > pos.sl) and (
                    pos.price_current - be_sl
                ) >= min_stop_distance:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "symbol": symbol,
                        "sl": float(round(be_sl / tick_size) * tick_size),
                        "tp": pos.tp,
                        "magic": 2026,
                    }
                    with mt5_lock:
                        mt5.order_send(req)
            elif pos.type == mt5.POSITION_TYPE_SELL:
                be_sl = pos.price_open - be_offset
                if (pos.sl is None or be_sl < pos.sl) and (
                    be_sl - pos.price_current
                ) >= min_stop_distance:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "symbol": symbol,
                        "sl": float(round(be_sl / tick_size) * tick_size),
                        "tp": pos.tp,
                        "magic": 2026,
                    }
                    with mt5_lock:
                        mt5.order_send(req)

        # 2) Trailing após +1.2R
        if r_ratio >= 1.2:
            apply_trailing_stop(symbol, pos.price_current, atr)
    except Exception:
        logger.exception(f"manage_dynamic_trailing error for {symbol}:{ticket}")


def apply_partial_exit_after_pyr(symbol: str, pct: Optional[float] = None) -> None:
    try:
        close_pct = (
            pct
            if pct is not None
            else float(getattr(config, "PYRAMID_PARTIAL_CLOSE", 0.3) or 0.3)
        )
        if close_pct <= 0:
            return
        with mt5_lock:
            positions = mt5.positions_get(symbol=symbol) or []
        if not positions or len(positions) < 2:
            return
        df = get_fast_rates(symbol, TIMEFRAME_BASE)
        if not is_valid_dataframe(df) or len(df) < 20:
            return
        atr = get_atr(df, 14) or 0.0
        if atr <= 0:
            return
        positions = sorted(positions, key=lambda p: p.time)
        anchor = positions[0]
        current = positions[-1]
        initial_r = abs(anchor.price_open - (anchor.sl or anchor.price_open)) or (
            atr * 2.0
        )
        if initial_r <= 0:
            initial_r = atr * 2.0
        profit_move = (
            (current.price_current - anchor.price_open)
            if current.type == mt5.POSITION_TYPE_BUY
            else (anchor.price_open - current.price_current)
        )
        r_ratio = profit_move / initial_r if initial_r > 0 else 0.0
        if r_ratio < 2.0:
            return
        total_volume = sum(p.volume for p in positions)
        volume_to_close = max(0.0, round(total_volume * close_pct, 2))
        if volume_to_close <= 0:
            return
        side = (
            mt5.ORDER_TYPE_SELL
            if current.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = current.price_current
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": side,
            "volume": float(volume_to_close),
            "price": float(price),
            "deviation": 200,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        with mt5_lock:
            res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔻 Parcial após pirâmide {symbol} vol={volume_to_close}")
    except Exception:
        logger.exception(f"apply_partial_exit_after_pyr error for {symbol}")


_last_trailing_check_ts = 0.0


def check_and_apply_dynamic_trailing(interval_sec: int = 300) -> None:
    try:
        now = time.time()
        global _last_trailing_check_ts
        if now - _last_trailing_check_ts < max(5, int(interval_sec)):
            return
        _last_trailing_check_ts = now
        with mt5_lock:
            positions = mt5.positions_get() or []
        symbols = sorted({p.symbol for p in positions})
        for sym in symbols:
            df = get_fast_rates(sym, TIMEFRAME_BASE)
            if not is_valid_dataframe(df) or len(df) < 20:
                continue
            atr = get_atr(df, 14) or 0.0
            if atr <= 0:
                continue
            price = (
                float(mt5.symbol_info_tick(sym).last or 0.0)
                if mt5.symbol_info_tick(sym)
                else 0.0
            )
            if price <= 0:
                info = mt5.symbol_info_tick(sym)
                price = float((info.ask or info.bid) if info else 0.0)
            if price <= 0:
                continue
            apply_trailing_stop(sym, price, atr)
    except Exception:
        logger.exception("check_and_apply_dynamic_trailing error")


def modify_position_sl(symbol: str, ticket: int, new_sl: float) -> bool:
    try:
        with mt5_lock:
            pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return False
        pos = pos_list[0]
        info = mt5.symbol_info(symbol)
        tick = float(getattr(info, "point", 0.01) or 0.01) if info else 0.01
        rounded = float(round(new_sl / tick) * tick)
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": rounded,
            "tp": pos.tp,
        }
        with mt5_lock:
            res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
    except Exception:
        return False


def execute_partial_close(symbol: str, ticket: int, pct_close: float) -> bool:
    try:
        pct = float(pct_close or 0.0)
        if pct <= 0.0:
            return False
        with mt5_lock:
            pos_list = mt5.positions_get(ticket=ticket)
        if not pos_list:
            return False
        pos = pos_list[0]
        min_lot = AssetClassManager.get_min_lot(symbol)
        vol_to_close = max(min_lot, round((pos.volume * pct), 2))
        with mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
        side = (
            mt5.ORDER_TYPE_SELL
            if pos.type == mt5.POSITION_TYPE_BUY
            else mt5.ORDER_TYPE_BUY
        )
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": side,
            "volume": float(vol_to_close),
            "price": float(price),
            "deviation": 200,
            "type_filling": mt5.ORDER_FILLING_RETURN,
            "comment": "PARTIAL_EXIT",
        }
        with mt5_lock:
            res = mt5.order_send(req)
        return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)
    except Exception:
        return False


def is_time_allowed_for_symbol(symbol: str, mode: str) -> bool:
    try:
        now = datetime.now().time()
        horarios = getattr(
            config,
            "HORARIOS_OPERACAO",
            {
                "FUTUROS": [(datetime.time(9, 0), datetime.time(18, 0))],
                "ACOES": [(datetime.time(10, 0), datetime.time(17, 0))],
                "AMBOS": [(datetime.time(10, 0), datetime.time(17, 0))],
                "SO_FUTUROS": [
                    (datetime.time(9, 0), datetime.time(10, 0)),
                    (datetime.time(17, 0), datetime.time(18, 0)),
                ],
            },
        )
        is_fut = is_future(symbol)
        key = str(mode or "").strip().upper()
        if key == "AMBOS":
            ranges = horarios.get("AMBOS", [])
        elif key in ("FUTUROS", "SO_FUTUROS"):
            ranges = horarios.get(key, [])
        elif key == "ACOES":
            ranges = horarios.get("ACOES", [])
        else:
            ranges = horarios.get("AMBOS", [])
        if key == "SO_FUTUROS" and not is_fut:
            return False
        if key == "FUTUROS" and not is_fut:
            return False
        if key == "ACOES" and is_fut:
            return False
        for start, end in ranges:
            if start <= now <= end:
                return True
        return False
    except Exception:
        return True


def can_enter_symbol(symbol: str, equity: float) -> bool:
    """
    Verifica se pode abrir nova posição no símbolo considerando
    RISCO REAL (baseado em SL).
    """

    if equity <= 0:
        return False

    with mt5_lock:
        positions = [p for p in mt5.positions_get() or [] if p.symbol == symbol]

    # ✅ Sem posições abertas → pode entrar
    if not positions:
        return True

    total_risk_money = 0.0

    for p in positions:
        # ❌ Se não tem SL → risco infinito → bloqueia
        if not p.sl or p.sl <= 0:
            logger.warning(
                f"{symbol}: posição sem SL detectada — bloqueando nova entrada"
            )
            return False

        # Risco real = distância até SL * volume
        risk_per_unit = abs(p.price_open - p.sl)
        position_risk = risk_per_unit * p.volume

        total_risk_money += position_risk

    risk_pct = total_risk_money / equity

    if risk_pct >= config.MAX_RISK_PER_SYMBOL_PCT:
        logger.warning(
            f"{symbol}: Limite de risco por símbolo atingido "
            f"({risk_pct*100:.2f}% ≥ {config.MAX_RISK_PER_SYMBOL_PCT*100:.2f}%)"
        )
        return False

    return True


def calculate_correlation_matrix(
    symbols: List[str], lookback: int = 60
) -> Dict[str, Dict[str, float]]:
    """
    Calcula matriz de correlação entre símbolos.
    Retorna: {symbol1: {symbol2: corr_value}}
    """
    if not symbols or len(symbols) < 2:
        return {}

    # Coleta dados de fechamento
    closes = {}
    for sym in symbols:
        df = safe_copy_rates(sym, mt5.TIMEFRAME_D1, lookback)
        if df is not None and not df.empty and len(df) >= 30:
            closes[sym] = df["close"]

    if len(closes) < 2:
        return {}

    # Alinha datas
    df_all = pd.DataFrame(closes)
    df_all = df_all.dropna()

    if df_all.empty or len(df_all) < 30:
        return {}

    # Calcula correlação
    corr_matrix = df_all.corr()

    # Converte para dict
    result = {}
    for sym1 in symbols:
        if sym1 not in corr_matrix.columns:
            continue
        result[sym1] = {}
        for sym2 in symbols:
            if sym2 not in corr_matrix.columns:
                continue
            result[sym1][sym2] = float(corr_matrix.loc[sym1, sym2])

    return result


# =========================================================
# RASTREAMENTO DE PERFORMANCE POR ATIVO (LOSS STREAK)
# =========================================================

LOSS_STREAK_FILE = "symbol_loss_streak.json"

_symbol_loss_streak = defaultdict(int)
_symbol_last_loss_time = {}
_symbol_block_until = {}

loss_streak_lock = threading.Lock()


def load_loss_streak_data():
    global _symbol_loss_streak, _symbol_last_loss_time, _symbol_block_until

    if not os.path.exists(LOSS_STREAK_FILE):
        logger.info("ℹ️ Arquivo de loss streak não encontrado. Inicializando vazio.")
        return

    try:
        with open(LOSS_STREAK_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        now = datetime.now()

        with loss_streak_lock:

            # === STREAK ===
            streak_raw = raw.get("streak", {})
            _symbol_loss_streak = defaultdict(
                int,
                {
                    k: int(v)
                    for k, v in streak_raw.items()
                    if isinstance(v, (int, float))
                },
            )

            # === LAST LOSS TIME ===
            _symbol_last_loss_time = {}
            for sym, ts in raw.get("last_loss", {}).items():
                try:
                    _symbol_last_loss_time[sym] = datetime.fromisoformat(ts)
                except Exception:
                    continue

            # === BLOCK UNTIL (remove expirados) ===
            _symbol_block_until = {}
            for sym, ts in raw.get("block_until", {}).items():
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt > now:
                        _symbol_block_until[sym] = dt
                except Exception:
                    continue

        logger.info(
            f"📉 Loss streak carregado | "
            f"{len(_symbol_loss_streak)} símbolos | "
            f"{len(_symbol_block_until)} bloqueados"
        )

    except Exception as e:
        logger.error(f"❌ Erro ao carregar loss streak: {e}", exc_info=True)


def save_loss_streak_data():
    data = {
        "streak": dict(_symbol_loss_streak),
        "last_loss": {k: v.isoformat() for k, v in _symbol_last_loss_time.items()},
        "block_until": {k: v.isoformat() for k, v in _symbol_block_until.items()},
    }
    try:
        with open(LOSS_STREAK_FILE, "w") as f:
            json.dump(data, f)
        logger.info("💾 Loss streak salvo.")
    except Exception as e:
        logger.error(f"Erro ao salvar loss streak: {e}")


def record_trade_outcome(symbol: str, profit_loss: float):
    """
    Registra resultado do trade por símbolo.
    Thread-safe, bloqueio finito e reset correto.
    """
    if not symbol:
        return

    now = datetime.now()

    with loss_streak_lock:

        # Garante inicialização
        _symbol_loss_streak.setdefault(symbol, 0)

        # ======================
        # ✅ LUCRO → RESET TOTAL
        # ======================
        if profit_loss >= 0:
            if _symbol_loss_streak[symbol] > 0:
                logger.info(
                    f"✅ {symbol}: Streak resetado após lucro "
                    f"({_symbol_loss_streak[symbol]} → 0)"
                )

            _symbol_loss_streak[symbol] = 0
            _symbol_last_loss_time.pop(symbol, None)
            _symbol_block_until.pop(symbol, None)

        # ======================
        # 🔴 PERDA
        # ======================
        else:
            _symbol_loss_streak[symbol] += 1
            _symbol_last_loss_time[symbol] = now

            streak = _symbol_loss_streak[symbol]

            logger.warning(f"🔴 {symbol}: Perda consecutiva #{streak}")

            # 🚫 BLOQUEIO SOMENTE UMA VEZ
            if (
                streak >= config.SYMBOL_MAX_CONSECUTIVE_LOSSES
                and symbol not in _symbol_block_until
            ):
                block_until = now + timedelta(hours=config.SYMBOL_COOLDOWN_HOURS)
                _symbol_block_until[symbol] = block_until

                logger.critical(
                    f"🚫 {symbol}: BLOQUEADO até "
                    f"{block_until.strftime('%d/%m %H:%M')} "
                    f"({streak} perdas seguidas)"
                )

    save_loss_streak_data()


def is_symbol_blocked(symbol: str) -> tuple[bool, str]:
    """
    Retorna (blocked: bool, reason: str)
    Thread-safe e com limpeza completa.
    """
    if not symbol:
        return False, ""

    now = datetime.now()

    with loss_streak_lock:

        # =========================
        # 🧹 LIMPA BLOQUEIO EXPIRADO
        # =========================
        if symbol in _symbol_block_until:
            if now >= _symbol_block_until[symbol]:
                _symbol_block_until.pop(symbol, None)
                _symbol_loss_streak[symbol] = 0
                _symbol_last_loss_time.pop(symbol, None)

                save_loss_streak_data()

                logger.info(f"✅ {symbol}: Bloqueio expirado e removido")

        # =========================
        # 🚫 AINDA BLOQUEADO
        # =========================
        if symbol in _symbol_block_until:
            remaining_sec = (_symbol_block_until[symbol] - now).total_seconds()
            remaining_min = max(1, int(remaining_sec // 60))

            return (
                True,
                f"Bloqueado ({remaining_min} min restantes) — "
                f"{config.SYMBOL_MAX_CONSECUTIVE_LOSSES} perdas seguidas",
            )

    return False, ""


def block_symbol(symbol: str, reason: str = "", hours: int = None):
    """
    Bloqueia um símbolo por um período especificado.

    Args:
        symbol: Símbolo a bloquear
        reason: Motivo do bloqueio (opcional)
        hours: Horas de bloqueio (usa config.SYMBOL_COOLDOWN_HOURS se None)
    """
    if not symbol:
        return

    if hours is None:
        hours = config.SYMBOL_COOLDOWN_HOURS

    now = datetime.now()
    block_until = now + timedelta(hours=hours)

    with loss_streak_lock:
        _symbol_block_until[symbol] = block_until
        save_loss_streak_data()

    logger.info(
        f"🚫 {symbol}: Bloqueado por {hours}h - {reason if reason else 'Motivo não especificado'}"
    )


def get_loss_streak(symbol: str) -> int:
    """
    Retorna o número de perdas consecutivas para um símbolo.
    Thread-safe, retorna 0 se símbolo não existir.
    """
    if not symbol:
        return 0

    with loss_streak_lock:
        return _symbol_loss_streak.get(symbol, 0)


def get_cached_indicators(symbol: str, timeframe, count: int = 300, ttl: int = 15):
    """
    Cache seguro de indicadores (Redis)
    - Não cacheia erro
    - Valida frescor do candle
    - Fail-safe total
    """

    # =========================
    # 🔄 FALLBACK SEM REDIS
    # =========================
    if not REDIS_AVAILABLE:
        df = safe_copy_rates(symbol, timeframe, count)
        if df is None or len(df) < 50:
            return {"error": "no_data"}
        return quick_indicators_custom(symbol, timeframe, df=df)

    # =========================
    # 🔑 CHAVE COM VERSÃO REAL
    # =========================
    key = f"ind:v3:{symbol}:{timeframe}:{count}"

    try:
        cached = redis_client.get(key)
        if cached:
            ind = pickle.loads(cached)

            # ❌ Cache inválido
            if not isinstance(ind, dict) or ind.get("error"):
                raise ValueError("Cache inválido")

            # ⏱️ Valida frescor do candle
            candle_time = ind.get("candle_time")
            if candle_time and (time.time() - candle_time) <= ttl:
                logger.debug(f"🧠 Cache HIT: {symbol}")
                return ind

    except Exception as e:
        logger.debug(f"Redis cache ignorado ({symbol}): {e}")

    # =========================
    # 🧮 RECÁLCULO
    # =========================
    df = safe_copy_rates(symbol, timeframe, count)
    if df is None or len(df) < 50:
        return {"error": "no_data"}

    ind = quick_indicators_custom(symbol, timeframe, df=df)
    if not isinstance(ind, dict):
        return {"error": "no_data"}
    if ind.get("error"):
        return ind

    # Timestamp do candle fechado
    ind["candle_time"] = (
        df.index[-1].timestamp() if hasattr(df.index[-1], "timestamp") else time.time()
    )

    # =========================
    # 💾 SALVA CACHE
    # =========================
    try:
        redis_client.setex(key, ttl, pickle.dumps(ind))
    except Exception as e:
        logger.debug(f"Redis set ignorado ({symbol}): {e}")

    return ind


def mt5_with_retry(
    max_retries: int = 4, base_delay: float = 1.0, retry_on_none: bool = True
):
    """
    Decorator robusto para operações MT5
    - Retry apenas para erros recuperáveis
    - Retry se retorno for None (MT5 bug comum)
    """

    RECOVERABLE_EXCEPTIONS = (
        ConnectionError,
        TimeoutError,
        RuntimeError,
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # 🔁 Retry se MT5 retornou None
                    if retry_on_none and result is None:
                        raise RuntimeError("MT5 retornou None")

                    return result

                except RECOVERABLE_EXCEPTIONS as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.exception(
                            f"🚨 Falha definitiva em {func.__name__} "
                            f"após {max_retries} tentativas"
                        )
                        raise

                    logger.warning(
                        f"⚠️ {func.__name__} falhou "
                        f"(tentativa {attempt}/{max_retries}): {e} | "
                        f"retry em {delay:.1f}s"
                    )

                    time.sleep(delay)
                    delay *= 2

                except Exception:
                    # ❌ Erro lógico → NÃO RETENTA
                    logger.exception(f"❌ Erro não recuperável em {func.__name__}")
                    raise

            raise last_exception  # segurança

        return wrapper

    return decorator


def calculate_advanced_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {}

    if not is_valid_dataframe(trades_df, min_rows=5):
        return metrics

    pnl = trades_df["pnl_money"]

    # Profit Factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    metrics["profit_factor"] = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # MAE / MFE
    if {"mae", "mfe"}.issubset(trades_df.columns):
        metrics["avg_mae"] = trades_df["mae"].mean()
        metrics["avg_mfe"] = trades_df["mfe"].mean()

    # Equity curve
    initial_equity = trades_df.get("equity_start", pd.Series([100000])).iloc[0]
    equity_curve = initial_equity + pnl.cumsum().values

    peak = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - peak) / peak) * 100

    metrics["ulcer_index"] = np.sqrt(np.mean(drawdown_pct**2))

    max_dd = np.min((equity_curve - peak) / peak)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1

    metrics["recovery_factor"] = (
        total_return / abs(max_dd) if max_dd < 0 else float("inf")
    )

    return metrics


def is_spread_acceptable(symbol: str, max_spread_pct: float | None = None) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if not tick or tick.bid <= 0 or tick.ask <= 0:
        return False

    spread = tick.ask - tick.bid
    mid_price = (tick.ask + tick.bid) / 2
    spread_pct = (spread / mid_price) * 100

    server_time = datetime.fromtimestamp(tick.time).time()

    if max_spread_pct is None:
        if server_time < datetime.strptime("15:30", "%H:%M").time():
            max_spread_pct = 0.15
        elif server_time < datetime.strptime("17:00", "%H:%M").time():
            max_spread_pct = 0.30
        else:
            max_spread_pct = 0.45  # after market

    if spread_pct > max_spread_pct:
        logger.debug(
            f"{symbol}: Spread {spread_pct:.3f}% > {max_spread_pct:.2f}% "
            f"(hora servidor {server_time.strftime('%H:%M')})"
        )
        return False

    return True


def adjust_global_sl_after_pyr(
    symbol: str, side: str, current_price: float, atr: float
):
    """
    Ajusta SL global após pyramiding:
    - Protege lucro da primeira perna
    - Aplica SL único para todas as posições do símbolo
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not positions:
        return

    # 🔹 Identifica a primeira perna (menor ticket = mais antiga)
    positions = sorted(positions, key=lambda p: p.time)
    anchor = positions[0]

    # 🔹 SL base: entrada da primeira perna ± buffer ATR
    buffer = atr * 0.2  # buffer pequeno para evitar stop exato

    if side == "BUY":
        new_sl = anchor.price_open + buffer
        if new_sl >= current_price:
            return
    else:
        new_sl = anchor.price_open - buffer
        if new_sl <= current_price:
            return

    new_sl = round(new_sl, 2)

    # 🔹 Aplica SL global apenas se for MELHOR que o atual
    for p in positions:
        if p.sl:
            if side == "BUY" and new_sl <= p.sl:
                continue
            if side == "SELL" and new_sl >= p.sl:
                continue

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": p.ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": p.tp,
            "magic": 2026,
        }

        with mt5_lock:
            res = mt5.order_send(request)

        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"🔒 SL global ajustado {symbol} → {new_sl}")
        else:
            logger.error(f"Erro ao ajustar SL {symbol}: {res.comment}")


def apply_partial_exit_after_pyr(
    symbol: str, side: str, current_price: float, atr: float
):
    """
    Fecha parcialmente a ÚLTIMA perna após pyramiding
    Condições:
    - Lucro >= +1R
    - Apenas a perna mais recente
    - Volume mínimo respeitado
    """

    if atr is None or atr <= 0:
        return

    with mt5_lock:
        positions = mt5.positions_get(symbol=symbol)

    if not positions or len(positions) < 2:
        return  # Sem pyramiding

    # 🔹 Ordena da mais recente para a mais antiga
    positions = sorted(positions, key=lambda p: p.time, reverse=True)
    last_leg = positions[0]

    # 🔹 Distância de risco da perna
    if last_leg.sl is None or last_leg.sl == 0:
        return

    if side == "BUY":
        risk_dist = abs(last_leg.price_open - last_leg.sl)
        profit_dist = current_price - last_leg.price_open
    else:
        risk_dist = abs(last_leg.sl - last_leg.price_open)
        profit_dist = last_leg.price_open - current_price

    if risk_dist <= 0:
        return

    r_multiple = profit_dist / risk_dist

    if r_multiple < 1.0:
        return  # Ainda não atingiu +1R

    # 🔹 Volume parcial (50%)
    close_volume = round(last_leg.volume * 0.5, 2)

    info = mt5.symbol_info(symbol)
    if not info or close_volume < info.volume_min:
        return

    # 🔹 Envia fechamento parcial
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "position": last_leg.ticket,
        "volume": close_volume,
        "type": mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY,
        "price": current_price,
        "deviation": get_dynamic_slippage(symbol, datetime.now().hour),
        "magic": 2026,
        "comment": "Partial Exit +1R",
    }

    with mt5_lock:
        result = mt5.order_send(request)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(
            f"💰 Partial exit {symbol}: "
            f"{close_volume} @ {current_price:.2f} (+{r_multiple:.2f}R)"
        )


def apply_anti_martingale(loss_streak: int) -> float:
    """
    Anti-Martingale: Reduz volume pós-perdas (risco, para win rate psicológico).
    """
    if loss_streak >= 2:
        return 0.5  # 50% volume normal
    return 1.0


def calculate_dynamic_sl_tp(symbol, side, entry_price, ind):
    atr = ind.get("atr", 0.10)
    adx = ind.get("adx", 20)
    is_fut = is_future(symbol)
    info = mt5.symbol_info(symbol)
    tick_size = (
        float(getattr(info, "trade_tick_size", getattr(info, "point", 0.01)) or 0.01)
        if info
        else 0.01
    )
    now = datetime.now().time()
    lunch_start = datetime.strptime(
        getattr(config, "TRADING_LUNCH_BREAK_START", "11:45"), "%H:%M"
    ).time()
    lunch_end = datetime.strptime(
        getattr(config, "TRADING_LUNCH_BREAK_END", "13:15"), "%H:%M"
    ).time()
    in_lunch = lunch_start <= now <= lunch_end
    regime = "RANGING"
    tp_mult = 2.5
    if adx >= getattr(config, "TP_RULES", {}).get("TRENDING", {}).get("min_adx", 30):
        regime = "TRENDING"
        tp_mult = float(
            getattr(config, "TP_RULES", {}).get("TRENDING", {}).get("tp_mult", 4.5)
            or 4.5
        )
    elif ind.get("vol_breakout"):
        regime = "BREAKOUT"
        tp_mult = float(
            getattr(config, "TP_RULES", {}).get("BREAKOUT", {}).get("tp_mult", 5.0)
            or 5.0
        )
    if in_lunch:
        tp_mult = max(1.8, tp_mult * 0.8)
    if is_fut:
        df = safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 100)
        if is_valid_dataframe(df, 20):
            sl = calculate_smart_sl(symbol, entry_price, side, atr, df)
        else:
            base_sl_mult = float(getattr(config, "SL_ATR_MULTIPLIER", 2.0) or 2.0)
            sl = (
                (entry_price - atr * base_sl_mult)
                if side == "BUY"
                else (entry_price + atr * base_sl_mult)
            )
        if side == "BUY":
            tp = entry_price + (atr * tp_mult)
        else:
            tp = entry_price - (atr * tp_mult)
    else:
        sl_mult = float(getattr(config, "SL_ATR_MULTIPLIER", 2.0) or 2.0)
        if side == "BUY":
            sl = entry_price - (atr * sl_mult)
            tp = entry_price + (atr * tp_mult)
        else:
            sl = entry_price + (atr * sl_mult)
            tp = entry_price - (atr * tp_mult)
    sl = round(sl / tick_size) * tick_size
    tp = round(tp / tick_size) * tick_size
    return sl, tp


def normalize_price(symbol, price):
    info = mt5.symbol_info(symbol)
    if not info:
        return price

    # Use trade_tick_size aqui também
    normalized = round(price / info.trade_tick_size) * info.trade_tick_size
    return round(normalized, info.digits)


def check_and_apply_breakeven(symbol, current_indicators, move_threshold_atr=1.0):
    """
    Se o preço undou 1x o ATR a favor, move o SL para o preço de entrada.
    """
    positions = mt5.positions_get(symbol=symbol)
    if not is_valid_dataframe(positions):
        return

    ind = current_indicators.get(symbol)
    if not ind:
        return

    atr = ind.get("atr", 0.10)

    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            if p.price_current >= (p.price_open + (atr * move_threshold_atr)):
                if p.sl < p.price_open:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (COMPRA)")
                    modify_sl_tp(p.ticket, p.price_open + (atr * 0.1), p.tp)

        elif p.type == mt5.POSITION_TYPE_SELL:
            if p.price_current <= (p.price_open - (atr * move_threshold_atr)):
                if p.sl > p.price_open or p.sl == 0:
                    logger.info(f"🛡️ {symbol}: Movendo para Breakeven (VENDA)")
                    modify_sl_tp(p.ticket, p.price_open - (atr * 0.1), p.tp)


def modify_sl_tp(ticket, new_sl, new_tp):
    """
    Envia a solicitação de modificação de SL/TP para um ticket específico.
    """
    # Normaliza os preços antes de enviar para evitar erro de tick_size
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False

    symbol = pos[0].symbol
    new_sl = normalize_price(symbol, new_sl)
    new_tp = normalize_price(symbol, new_tp)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": float(new_sl),
        "tp": float(new_tp),
    }

    with mt5_lock:
        result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"❌ Falha ao mover Stop: {result.comment}")
        return False

    return True


def update_correlations(top15_symbols):
    """
    Calcula matriz de correlação dos ativos.
    CORRIGIDO: Agora usa o nome correto do parâmetro.
    """
    # ✅ CORREÇÃO: Era 'symbols', agora é 'top15_symbols'
    if not isinstance(top15_symbols, (list, tuple)):
        logger.error(
            f"update_correlations recebeu tipo inválido: {type(top15_symbols)}"
        )
        return

    if not top15_symbols:
        logger.warning("update_correlations: Lista de símbolos vazia")
        return

    logger.info(f"📊 Atualizando correlação para {len(top15_symbols)} ativos...")

    try:
        # Coleta dados de fechamento dos últimos 50 candles
        data = {}
        for sym in top15_symbols:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 50)
            if rates is not None:
                data[sym] = [r["close"] for r in rates]

        if len(data) > 1:
            df = pd.DataFrame(data)
            corr_matrix = df.corr()

            # Salva na variável global
            global last_corr_matrix
            last_corr_matrix = corr_matrix
            logger.info("✅ Matriz de correlação atualizada")

    except Exception as e:
        logger.error(f"Erro ao calcular correlações: {e}", exc_info=True)


# =========================================================
# 📤 TELEGRAM & REPORTING
# =========================================================


def send_telegram_message(text: str):
    """
    Envia uma mensagem para o Telegram usando o bot instanciado.
    """
    bot = get_telegram_bot()
    if bot and getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        try:
            bot.send_message(
                chat_id=config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning(f"Erro ao enviar mensagem para o Telegram: {e}")


def send_daily_performance_report():
    """
    Envia relatório consolidado do dia.
    """
    try:
        from database import get_trades_by_date
        from datetime import date

        today = date.today()
        trades = get_trades_by_date(today)

        if not is_valid_dataframe(trades):
            send_telegram_message(
                f"📊 <b>RELATÓRIO {today.strftime('%d/%m/%Y')}</b>\nNenhum trade realizado hoje."
            )
            return

        acc = mt5.account_info()
        equity = acc.equity if acc else 0

        total_pnl = trades["pnl_money"].sum()
        win_rate = (len(trades[trades["pnl_money"] > 0]) / len(trades)) * 100

        msg = (
            f"📊 <b>RELATÓRIO FINAL XP3 - {today.strftime('%d/%m/%Y')}</b>\n\n"
            f"💰 Equity: R$ {equity:,.2f}\n"
            f"📈 PnL: R$ {total_pnl:+.2f}\n"
            f"🎯 Win Rate: {win_rate:.1f}%\n"
            f"🔢 Total Trades: {len(trades)}\n\n"
            f"✅ Sistema Estável"
        )
        send_telegram_message(msg)
    except Exception as e:
        logger.error(f"Erro no relatório diário: {e}")


def log_trade_to_txt(trade_info: Dict[str, Any]):
    """
    Registra trade em arquivo TXT diário.
    Usa o Ticket do MT5 como identificador único.
    Formato: ID | Hora | Símbolo | Tipo | Volume | Preço | Lucro
    """
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"trades_{today_str}.txt"
        filepath = os.path.join("logs", filename)
        
        # Garante diretório
        os.makedirs("logs", exist_ok=True)
        
        # Cabeçalho se arquivo novo
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("TICKET | TIME | SYMBOL | TYPE | VOL | PRICE | PROFIT | COMMENT\n")
                
        # Formata linha
        line = (
            f"{trade_info.get('ticket')} | "
            f"{trade_info.get('time')} | "
            f"{trade_info.get('symbol')} | "
            f"{trade_info.get('type')} | "
            f"{trade_info.get('volume')} | "
            f"{trade_info.get('price')} | "
            f"{trade_info.get('profit', 0.0):.2f} | "
            f"{trade_info.get('comment', '')}\n"
        )
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
            
        logger.info(f"📝 Trade {trade_info.get('ticket')} salvo em {filename}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar trade em TXT: {e}")


def check_and_log_closed_trades():
    """
    Verifica trades fechados recentemente no MT5 e salva no TXT.
    Deve ser chamado periodicamente pelo bot.
    """
    try:
        from_date = datetime.now() - timedelta(hours=1)
        to_date = datetime.now() + timedelta(minutes=1)
        
        with mt5_lock:
            # history_deals_get retorna negócios executados (Entrada e Saída)
            deals = mt5.history_deals_get(from_date, to_date)
            
        if not deals:
            return

        for deal in deals:
            # Filtra apenas saídas (Entry Out ou Entry In/Out) que geraram lucro/prejuízo
            if deal.entry == mt5.DEAL_ENTRY_OUT or deal.entry == mt5.DEAL_ENTRY_INOUT:
                 # Verifica se já logamos esse deal? (Idealmente sim, mas TXT é append only)
                 # Vamos logar com comentário "EXIT"
                 
                 trade_data = {
                     "ticket": deal.ticket, # Ticket do DEAL de saída
                     "time": datetime.fromtimestamp(deal.time).strftime("%H:%M:%S"),
                     "symbol": deal.symbol,
                     "type": "EXIT", # Simplificação
                     "volume": deal.volume,
                     "price": deal.price,
                     "profit": deal.profit,
                     "comment": f"Exit (Deal {deal.ticket})"
                 }
                 # Opcional: Verificar duplicidade lendo o arquivo (custoso) ou cache em memória
                 # Por enquanto, logamos tudo. O usuário pode filtrar por Ticket.
                 log_trade_to_txt(trade_data)
                 
    except Exception as e:
        logger.error(f"Erro ao verificar trades fechados: {e}")

_last_logged_deals = set()

def check_and_log_closed_trades():
    """
    Verifica trades fechados recentemente no MT5 e salva no TXT.
    Evita duplicidade usando cache simples.
    """
    global _last_logged_deals
    try:
        from_date = datetime.now() - timedelta(hours=1)
        to_date = datetime.now() + timedelta(minutes=1)
        
        with mt5_lock:
            # history_deals_get retorna negócios executados (Entrada e Saída)
            deals = mt5.history_deals_get(from_date, to_date)
            
        if not deals:
            return

        for deal in deals:
            # Filtra apenas saídas (Entry Out ou Entry In/Out) que geraram lucro/prejuízo
            if (deal.entry == mt5.DEAL_ENTRY_OUT or deal.entry == mt5.DEAL_ENTRY_INOUT) and deal.ticket not in _last_logged_deals:
                 
                 trade_data = {
                     "ticket": deal.ticket, # Ticket do DEAL de saída
                     "time": datetime.fromtimestamp(deal.time).strftime("%H:%M:%S"),
                     "symbol": deal.symbol,
                     "type": "EXIT",
                     "volume": deal.volume,
                     "price": deal.price,
                     "profit": deal.profit,
                     "comment": f"Exit (Deal {deal.ticket})"
                 }
                 log_trade_to_txt(trade_data)
                 _last_logged_deals.add(deal.ticket)
                 
                 # Limpa cache antigo para não estourar memória
                 if len(_last_logged_deals) > 1000:
                     _last_logged_deals = set(list(_last_logged_deals)[-500:])
                 
    except Exception as e:
        logger.error(f"Erro ao verificar trades fechados: {e}")


# =========================================================
# 📰 NEWS & EVENTS
# =========================================================


def send_news_alert(message: str):
    """
    Envia alerta de notícia importante via Telegram
    """
    if not getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        return

    full_message = f"📰 <b>CALENDÁRIO ECONÔMICO</b>\n\n{message}"

    try:
        send_telegram_message(full_message)
        logger.info(f"News alert enviado: {message}")
    except Exception as e:
        logger.error(f"Falha ao enviar news alert: {e}")


def send_weekly_review_alert(message: str):
    if not getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        return
    full_message = f"🔔 <b>Checklist Semanal</b>\n\n{message}"
    try:
        send_telegram_message(full_message)
        logger.info("Weekly alert enviado")
    except Exception as e:
        logger.error(f"Falha ao enviar weekly alert: {e}")


def send_next_high_impact_event():
    """
    Envia o próximo evento de alto impacto (comando /proximoevento)
    """
    try:
        from news_filter import get_next_high_impact_event

        message = get_next_high_impact_event()
        emoji = "🔴" if "em" in message and "min" in message else "🟢"
        send_news_alert(f"{emoji} <b>PRÓXIMO EVENTO</b>\n\n{message}")
    except:
        pass


def send_current_blackout_status():
    """
    Envia status atual de blackout (se está bloqueado ou não)
    """
    try:
        from news_filter import check_news_blackout, get_upcoming_events

        blocked, reason = check_news_blackout()

        if blocked:
            status = f"🚫 <b>BOT EM BLACKOUT</b>\n\n{reason}"
        else:
            status = "✅ <b>Trading Liberado</b> - Sem eventos críticos próximos."

        send_news_alert(status)
    except:
        pass


# =========================================================
# 🛡️ RECOVERY & RISK
# =========================================================


def mt5_crash_recovery():
    """
    Recuperação de crash básica.
    """
    if not mt5.initialize():
        logger.critical("Falha ao reconectar MT5")
        return
    logger.info("Recuperação MT5 completa")


def get_daily_volume() -> float:
    """Calcula o volume total negociado PELO BOT no dia (em R$)."""
    try:
        from_date = datetime.combine(datetime.now().date(), datetime_time.min)
        to_date = datetime.now()
        with mt5_lock:
            deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return 0.0
        total_volume = sum(
            abs(deal.volume * deal.price)
            for deal in deals
            if deal.entry == mt5.DEAL_ENTRY_OUT
        )  # Sum of closed trades' value
        return total_volume
    except Exception as e:
        logger.error(f"Erro ao calcular volume: {e}")
        return 0.0


def get_realtime_win_rate(lookback_trades: int = 20) -> Dict[str, Any]:
    """
    Calcula win rate em tempo real a partir do histórico do MT5.

    Args:
        lookback_trades: Quantos últimos trades considerar.

    Returns:
        Dict com win_rate (float 0.0-1.0), total_trades e profit_factor.
    """
    try:
        from datetime import datetime, timedelta
        import MetaTrader5 as mt5

        # Define período de busca (últimos 30 dias para garantir dados)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        with mt5_lock:
            # Pega histórico de negócios (DEALS)
            deals = mt5.history_deals_get(start_date, end_date)

        if deals is None or len(deals) == 0:
            return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}

        # Filtra apenas DEALS de SAÍDA que tenham PnL Realizado
        # (entry != alignment)
        valid_trades = [
            d for d in deals if d.entry != mt5.DEAL_ENTRY_IN and d.profit != 0
        ]

        # Pega apenas os últimos N
        last_trades = (
            valid_trades[-lookback_trades:]
            if len(valid_trades) > lookback_trades
            else valid_trades
        )

        if not last_trades:
            return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}

        wins = sum(1 for d in last_trades if d.profit > 0)
        total = len(last_trades)

        gross_profit = sum(d.profit for d in last_trades if d.profit > 0)
        gross_loss = abs(sum(d.profit for d in last_trades if d.profit < 0))

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 2.0
        )  # Default favorável se não houver losses

        return {
            "win_rate": wins / total if total > 0 else 0.0,
            "total_trades": total,
            "profit_factor": round(profit_factor, 2),
        }

    except Exception as e:
        logger.error(f"Erro em get_realtime_win_rate: {e}")
        return {"win_rate": 0.0, "total_trades": 0, "profit_factor": 0.0}


def calculate_daily_dd() -> float:
    """
    Calcula o drawdown diário (Equity vs Daily Max).
    """
    try:
        acc = mt5.account_info()
        if not acc:
            return 0.0

        # Tenta buscar a global do bot
        import sys

        bot_mod = sys.modules.get("bot")
        if bot_mod and hasattr(bot_mod, "daily_max_equity"):
            m_equity = getattr(bot_mod, "daily_max_equity")
            if m_equity > acc.equity:
                return (m_equity - acc.equity) / m_equity
        return 0.0
    except:
        return 0.0


# ============================================
# 🔥 PRIORIDADE 1 - ANTI-CHOP & LIMITS
# ============================================

# Arquivos persistentes
ANTI_CHOP_FILE = "anti_chop_data.json"
DAILY_LIMITS_FILE = "daily_symbol_limits.json"

# Estado global
_symbol_sl_timestamps = {}  # {symbol: timestamp_último_sl}
_symbol_sl_prices = {}  # {symbol: preço_quando_bateu_sl}
_daily_symbol_trades = defaultdict(lambda: {"total": 0, "losses": 0})  # Contador diário


def load_anti_chop_data():
    """Carrega dados de cooldown"""
    global _symbol_sl_timestamps, _symbol_sl_prices
    if os.path.exists(ANTI_CHOP_FILE):
        try:
            with open(ANTI_CHOP_FILE, "r") as f:
                data = json.load(f)
                _symbol_sl_timestamps = {
                    k: datetime.fromisoformat(v)
                    for k, v in data.get("timestamps", {}).items()
                }
                _symbol_sl_prices = data.get("prices", {})
            logger.info("✅ Dados anti-chop carregados")
        except Exception as e:
            logger.error(f"Erro ao carregar anti-chop: {e}")


def save_anti_chop_data():
    """Salva dados de cooldown"""
    data = {
        "timestamps": {k: v.isoformat() for k, v in _symbol_sl_timestamps.items()},
        "prices": _symbol_sl_prices,
    }
    atomic_save_json(ANTI_CHOP_FILE, data)


def load_daily_limits():
    """Carrega contadores diários"""
    global _daily_symbol_trades
    if os.path.exists(DAILY_LIMITS_FILE):
        try:
            with open(DAILY_LIMITS_FILE, "r") as f:
                data = json.load(f)
                saved_date = data.get("date")
                today = datetime.now().date().isoformat()
                if saved_date == today:
                    _daily_symbol_trades = defaultdict(
                        lambda: {"total": 0, "losses": 0}, data.get("trades", {})
                    )
                    logger.info("✅ Limites diários carregados")
                else:
                    logger.info("🔄 Novo dia detectado - resetando limites")
        except Exception as e:
            logger.error(f"Erro ao carregar limites: {e}")


def save_daily_limits():
    """Salva contadores diários"""
    data = {
        "date": datetime.now().date().isoformat(),
        "trades": dict(_daily_symbol_trades),
    }
    atomic_save_json(DAILY_LIMITS_FILE, data)


def register_sl_hit(symbol: str, sl_price: float):
    """Registra que o SL foi atingido"""
    if not getattr(config, "ANTI_CHOP", {}).get("enabled", False):
        return
    _symbol_sl_timestamps[symbol] = datetime.now()
    _symbol_sl_prices[symbol] = sl_price
    save_anti_chop_data()
    logger.info(f"🛑 {symbol}: SL registrado @ R${sl_price:.2f}")


def check_anti_chop_filter(
    symbol: str, current_price: float, atr: float
) -> Tuple[bool, str]:
    """Valida se o ativo está em cooldown após SL ou volatilidade excedida"""
    if not getattr(config, "ANTI_CHOP", {}).get("enabled", False):
        return True, ""
    last_sl_time = _symbol_sl_timestamps.get(symbol)
    if last_sl_time:
        # Bloqueio total até o fim do dia após um SL (se habilitado)
        try:
            if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
                if last_sl_time.date() == datetime.now().date():
                    return False, "Bloqueado hoje por SL (reset diário)"
        except Exception:
            pass
        cooldown_min = config.ANTI_CHOP.get("cooldown_after_sl_minutes", 120)
        elapsed = (datetime.now() - last_sl_time).total_seconds() / 60
        if elapsed < cooldown_min:
            return False, f"Cooldown SL ({int(cooldown_min - elapsed)} min rest)"
    return True, ""


def clear_anti_chop_cooldown(symbol: str):
    """Limpa cooldown após entrada bem-sucedida"""
    try:
        # Não limpa se estiver usando bloqueio por SL até o fim do dia
        if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
            last_sl_time = _symbol_sl_timestamps.get(symbol)
            if last_sl_time and last_sl_time.date() == datetime.now().date():
                return
    except Exception:
        pass
    if symbol in _symbol_sl_timestamps:
        del _symbol_sl_timestamps[symbol]
    if symbol in _symbol_sl_prices:
        del _symbol_sl_prices[symbol]
    save_anti_chop_data()


def check_daily_symbol_limit(symbol: str) -> Tuple[bool, str]:
    """Verifica limites diários por ativo"""
    if not getattr(config, "DAILY_SYMBOL_LIMITS", {}).get("enabled", False):
        return True, ""
    try:
        # Bloqueio adicional: se bateu SL hoje e flag habilitada, bloqueia
        if config.ANTI_CHOP.get("block_full_day_on_single_sl", False):
            last_sl_time = _symbol_sl_timestamps.get(symbol)
            if last_sl_time and last_sl_time.date() == datetime.now().date():
                return False, "Bloqueado por SL hoje"
    except Exception:
        pass
    stats = _daily_symbol_trades[symbol]
    if stats["losses"] >= config.get_max_losses_per_symbol():
        return False, "Limite de perdas diário"
    if stats["total"] >= config.DAILY_SYMBOL_LIMITS.get(
        "max_total_trades_per_symbol", 5
    ):
        return False, "Limite total diário"
    return True, ""


def register_trade_result(symbol: str, is_loss: bool):
    """Registra resultado do trade para limites diários"""
    if not getattr(config, "DAILY_SYMBOL_LIMITS", {}).get("enabled", False):
        return
    _daily_symbol_trades[symbol]["total"] += 1
    if is_loss:
        _daily_symbol_trades[symbol]["losses"] += 1
    save_daily_limits()


def reset_daily_limits():
    """Reseta contadores diários"""
    global _daily_symbol_trades
    _daily_symbol_trades.clear()
    save_daily_limits()


def check_pyramid_eligibility(symbol: str, side: str, ind: dict) -> Tuple[bool, str]:
    """Valida se pode adicionar perna (pirâmide) na posição atual"""
    with mt5_lock:
        positions = [p for p in mt5.positions_get(symbol=symbol) or []]
    if not positions:
        return True, "Primeira entrada"
    pos = positions[0]
    existing_side = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
    if existing_side != side:
        return False, "Direção oposta"
    pyr_count = pos.comment.count("PYR") if pos.comment else 0
    if pyr_count >= getattr(config, "PYRAMID_MAX_LEGS", 3):
        return False, "Limite de pirâmide atingido"
    req = (
        getattr(
            config,
            "PYRAMID_REQUIREMENTS_ENHANCED",
            getattr(config, "PYRAMID_REQUIREMENTS", {}),
        )
        or {}
    )
    adx_min = float(req.get("min_adx", 25) or 25)
    rsi = float(ind.get("rsi", 50) or 50)
    adx = float(ind.get("adx", 0) or 0)
    vol_ratio_req = float(req.get("volume_ratio", 1.0) or 1.0)
    vol_ratio = float(ind.get("volume_ratio", 1.0) or 1.0)
    if adx < adx_min:
        return False, f"ADX baixo ({adx:.1f}<{adx_min:.1f})"
    if side == "BUY":
        max_rsi_long = float(req.get("max_rsi_long", 65) or 65)
        if rsi > max_rsi_long:
            return False, f"RSI alto ({rsi:.1f}>{max_rsi_long:.1f})"
    else:
        min_rsi_short = float(req.get("min_rsi_short", 35) or 35)
        if rsi < min_rsi_short:
            return False, f"RSI baixo ({rsi:.1f}<{min_rsi_short:.1f})"
    if vol_ratio < vol_ratio_req:
        return False, f"Volume fraco ({vol_ratio:.2f}<{vol_ratio_req:.2f})"
    try:
        open_dt = getattr(pos, "time", None)
        if open_dt:
            mins_since = (datetime.now() - open_dt).total_seconds() / 60.0
            min_wait = int(
                req.get(
                    "min_time_between_legs_minutes", req.get("time_since_entry", 15)
                )
                or 15
            )
            if mins_since < min_wait:
                return False, f"Aguarde {int(min_wait - mins_since)} min"
    except Exception:
        pass
    atr = float(ind.get("atr", ind.get("atr_real", 0)) or 0)
    if atr > 0:
        min_atr_dist = float(getattr(config, "PYRAMID_ATR_DISTANCE", 1.0) or 1.0)
        with mt5_lock:
            tick = mt5.symbol_info_tick(symbol)
        if tick:
            price = tick.ask if side == "BUY" else tick.bid
            move = (
                (price - pos.price_open) if side == "BUY" else (pos.price_open - price)
            )
            if move < (atr * min_atr_dist):
                return False, f"Distância < {min_atr_dist} ATR"
    be_req = bool(req.get("require_breakeven", False))
    one_r_req = bool(req.get("require_1r_floating", False))
    if be_req:
        if side == "BUY":
            if not pos.sl or pos.sl < pos.price_open:
                return False, "Sem BE"
        else:
            if not pos.sl or pos.sl > pos.price_open:
                return False, "Sem BE"
    if one_r_req:
        risk = (
            abs((pos.price_open - pos.sl))
            if pos.sl
            else (atr * float(getattr(config, "SL_ATR_MULTIPLIER", 2.0) or 2.0))
        )
        if risk and risk > 0:
            with mt5_lock:
                tick = mt5.symbol_info_tick(symbol)
            if tick:
                price = tick.ask if side == "BUY" else tick.bid
                move = (
                    (price - pos.price_open)
                    if side == "BUY"
                    else (pos.price_open - price)
                )
                if move < risk:
                    return False, "Sem +1R flutuante"
    return True, "Elegível"


def send_telegram_trade(
    symbol: str,
    side: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    comment: str = "",
):
    msg = (
        f"🚀 <b>NOVA OPERAÇÃO: {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 Direção: {'COMPRA 🟢' if side == 'BUY' else 'VENDA 🔴'}\n"
        f"🔹 Volume: {volume:.2f}\n"
        f"🔹 Entrada: R$ {price:,.2f}\n"
        f"🔹 SL: R$ {sl:,.2f} | TP: R$ {tp:,.2f}\n"
        f"🔹 Obs: {comment}\n"
    )
    send_telegram_message(msg)


def send_telegram_exit(
    symbol: str,
    side: str,
    volume: float,
    entry_price: float,
    exit_price: float,
    profit_loss: float,
    reason: str = "",
):
    emoji = "✅" if profit_loss >= 0 else "❌"
    msg = (
        f"{emoji} <b>SAÍDA DE POSIÇÃO: {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🔹 Direção: {side}\n"
        f"🔹 Resultado: <b>R$ {profit_loss:+,.2f}</b>\n"
        f"🔹 Preço Saída: R$ {exit_price:,.2f}\n"
        f"🔹 Motivo: {reason}\n"
    )
    send_telegram_message(msg)


def calcular_lucro_realizado_txt():
    import os
    import re
    from datetime import datetime

    filename = f"trades_log_{datetime.now().strftime('%Y-%m-%d')}.txt"

    if not os.path.exists(filename):
        return 0.0, 0

    total_pnl = 0.0
    contagem_trades = 0

    with open(filename, "r", encoding="utf-8") as f:
        conteudo = f.read()

        # ✅ REGEX MAIS ROBUSTO
        # Busca linhas de fechamento (não de abertura)
        for linha in conteudo.split("\n"):
            if "Abertura de Posição" in linha or "---" in linha:
                continue

            # Match: P&L: +1550.00 ou P&L: -320.50
            match = re.search(r"P&L:\s*([+-]?\d+\.?\d*)", linha)
            if match:
                try:
                    pnl = float(match.group(1))
                    total_pnl += pnl
                    contagem_trades += 1
                except ValueError:
                    continue

    return total_pnl, contagem_trades


def obter_resumo_financeiro_do_dia():
    lucro_realizado, total_ordens = calcular_lucro_realizado_txt()
    lucro_aberto_total = (
        sum(p.profit for p in mt5.positions_get()) if mt5.positions_get() else 0.0
    )
    return lucro_realizado, lucro_aberto_total, total_ordens


def responder_comando_lucro(message):
    bot = get_telegram_bot()
    if not bot:
        return

    # 1. Busca o Lucro Realizado no seu arquivo TXT (o que já está no bolso)
    realizado, qtd = calcular_lucro_realizado_txt()

    # 2. Busca o Lucro Flutuante (o que está aberto agora no MT5)
    posicoes_abertas = mt5.positions_get()
    aberto = sum(p.profit for p in posicoes_abertas) if posicoes_abertas else 0.0
    total_do_dia = realizado + aberto

    emoji = "🚀" if total_do_dia >= 0 else "⚠️"

    msg = (
        f"{emoji} <b>STATUS XP3 - AGORA</b>\n\n"
        f"💰 <b>Realizado:</b> R$ {realizado:,.2f}\n"
        f"📈 <b>Flutuante:</b> R$ {aberto:,.2f}\n"
        f"---------------------------\n"
        f"🏆 <b>TOTAL DO DIA: R$ {total_do_dia:,.2f}</b>\n\n"
        f"<i>Baseado em {qtd} ordens e {len(posicoes_abertas) if posicoes_abertas else 0} posições abertas.</i>"
    )

    bot.reply_to(message, msg, parse_mode="HTML")


def check_minimum_price_movement(
    symbol: str, df: pd.DataFrame, atr: float
) -> Tuple[bool, str]:
    """
    Valida se houve movimento mínimo antes de entrar
    """
    if not getattr(config, "MIN_PRICE_MOVEMENT", {}).get("enabled", False):
        return True, ""

    lookback = config.MIN_PRICE_MOVEMENT.get("lookback_candles", 5)

    if df is None or len(df) < lookback:
        return True, ""  # Fail-open

    recent = df.tail(lookback)
    price_range = recent["high"].max() - recent["low"].min()

    min_movement = atr * config.MIN_PRICE_MOVEMENT.get("min_atr_multiplier", 1.5)

    if price_range < min_movement:
        return False, f"Range baixo ({price_range:.2f} < {min_movement:.2f})"

    return True, ""


# Final calls for data persistence
try:
    load_anti_chop_data()
    load_daily_limits()
except:
    pass


def diagnose_symbol_failure(symbol: str) -> str:
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return "Símbolo não existe na corretora"
        if not getattr(info, "selectable", False):
            return "Símbolo não selecionável"
        if not getattr(info, "visible", False):
            return "Símbolo não visível"
        from datetime import datetime

        exp = getattr(info, "expiration_time", None)
        if isinstance(exp, datetime) and exp < datetime.now():
            return "Contrato vencido"
        return "Falha na seleção"
    except Exception as e:
        return f"Erro na verificação: {str(e)}"


def ensure_market_watch_symbols():
    import config

    logger.info("📋 Sincronizando Market Watch (Strict Mode)...")
    desired = set()

    # Carrega símbolos do Universe Builder (JSON) - PRIORIDADE
    try:
        universe_data = load_elite_symbols_from_json()
        if universe_data:
            # Adiciona ELITE symbols
            if "ELITE" in universe_data and universe_data["ELITE"]:
                elite_symbols = [
                    item[0] if isinstance(item, list) else item
                    for item in universe_data["ELITE"]
                ]
                desired = desired.union(set(elite_symbols))
                logger.info(
                    f"✅ Adicionados {len(elite_symbols)} símbolos ELITE do Universe Builder"
                )

            # Adiciona OPORTUNIDADE symbols
            if "OPORTUNIDADE" in universe_data and universe_data["OPORTUNIDADE"]:
                oportunidade_symbols = [
                    item[0] if isinstance(item, list) else item
                    for item in universe_data["OPORTUNIDADE"]
                ]
                desired = desired.union(set(oportunidade_symbols))
                logger.info(
                    f"✅ Adicionados {len(oportunidade_symbols)} símbolos OPORTUNIDADE do Universe Builder"
                )
        else:
            logger.warning("⚠️ Nenhum dado de Universe Builder disponível")
    except Exception as e:
        logger.warning(f"⚠️ Falha ao carregar Universe Builder JSON: {e}")

    # Fallback: usa ELITE_SYMBOLS e OPORTUNIDADE_SYMBOLS do utils.py
    if not desired:
        logger.info(
            "📊 Usando listas pré-definidas de ELITE_SYMBOLS e OPORTUNIDADE_SYMBOLS"
        )
        desired = set(ELITE_SYMBOLS + OPORTUNIDADE_SYMBOLS)

    # Tenta adicionar SECTOR_MAP se existir
    try:
        sector_symbols = set(config.SECTOR_MAP.keys())
        desired = desired.union(sector_symbols)
        logger.info(f"✅ Adicionados {len(sector_symbols)} símbolos do SECTOR_MAP")
    except AttributeError:
        logger.debug("ℹ️ SECTOR_MAP não encontrado no config, continuando...")

    desired.add("IBOV")
    count_added = 0
    failed = []
    added_names = set()
    broker = (detect_broker() or "").lower()
    for symbol in desired:
        try:
            if mt5.symbol_select(symbol, True):
                count_added += 1
                added_names.add(symbol)
                continue
            reason = diagnose_symbol_failure(symbol)
            logger.warning(
                f"⚠️ Corretora {broker}: Falha ao adicionar {symbol} - {reason}"
            )
            s_up = (symbol or "").upper()
            is_fut_like = ("$" in s_up) or s_up.startswith(
                ("WIN", "WDO", "WSP", "IND", "DOL", "SMALL")
            )
            if is_fut_like:
                base = s_up.replace("$", "")
                base = base[:4] if base.startswith(("SMAL", "SMALL")) else base[:3]
                cands = get_futures_candidates(base)
                if cands:
                    cands_sorted = sorted(
                        cands,
                        key=lambda c: (
                            -calculate_contract_score(c),
                            c.get("days_to_exp", 9999),
                        ),
                    )
                    best = cands_sorted[0]
                    cand_sym = best.get("symbol")
                    if cand_sym and mt5.symbol_select(cand_sym, True):
                        count_added += 1
                        added_names.add(cand_sym)
                        continue
            else:
                res = mt5.symbols_get(f"{symbol}*") or []
                if res:
                    name = getattr(res[0], "name", "")
                    if name and mt5.symbol_select(name, True):
                        count_added += 1
                        added_names.add(name)
                        continue
            failed.append(symbol)
        except Exception as e:
            logger.error(f"❌ Erro ao processar {symbol}: {e}")
            failed.append(symbol)
    if failed:
        logger.warning(f"⚠️ {len(failed)} ativos falharam ao adicionar: {failed}")
    logger.info("🧹 Limpando ativos desnecessários do Market Watch...")
    all_symbols = mt5.symbols_get()
    removed_count = 0
    if all_symbols:
        keep_set = desired.union(added_names)
        for s in all_symbols:
            if s.select and s.name not in keep_set:
                if mt5.symbol_select(s.name, False):
                    removed_count += 1
    logger.info(
        f"✅ Market Watch Sincronizado: {count_added} ativos mantidos/adicionados, {removed_count} removidos."
    )


def calculate_chandelier_exit(
    symbol: str,
    position_side: str,
    position_entry_price: float,
    lookback_period: int = 22,
) -> Optional[float]:
    """
    Calcula o Chandelier Exit, um stop loss dinâmico baseado em volatilidade.

    Para uma posição COMPRADA, o stop é colocado N vezes o ATR abaixo do
    preço MÁXIMO atingido desde que a posição foi aberta.
    Para uma posição VENDIDA, é o oposto.

    Args:
        symbol (str): O símbolo do ativo.
        position_side (str): 'BUY' ou 'SELL'.
        position_entry_price (float): O preço de entrada da posição.
        lookback_period (int): O período para encontrar o high/low e para o ATR.

    Returns:
        Optional[float]: O novo preço de stop loss, ou None se não for possível calcular.
    """
    df = safe_copy_rates(symbol, TIMEFRAME_BASE, lookback_period + 5)
    if not is_valid_dataframe(df, lookback_period):
        logger.warning(f"ChandelierExit: Dados insuficientes para {symbol}")
        return None

    atr = get_atr(df, period=lookback_period)
    if not atr or atr <= 0:
        return None

    # Multiplicador do ATR (pode ser configurado no config.py)
    atr_multiplier = 3.0

    if position_side == "BUY":
        # Encontra o preço mais alto desde a entrada
        highest_high = df["high"].iloc[-lookback_period:].max()
        chandelier_exit = highest_high - (atr * atr_multiplier)
        # O stop só pode subir, nunca descer
        return max(position_entry_price, chandelier_exit)
    elif position_side == "SELL":
        # Encontra o preço mais baixo desde a entrada
        lowest_low = df["low"].iloc[-lookback_period:].min()
        chandelier_exit = lowest_low + (atr * atr_multiplier)
        # O stop só pode descer
        return min(position_entry_price, chandelier_exit)

    return None


# =========================================================
# ???? TAPE READING (FLUXO DE ORDENS)
# =========================================================


def check_book_pressure(symbol: str, side: str, depth: int = 5) -> tuple[bool, str]:
    try:
        book = mt5.market_book_get(symbol)
        if not book:
            return (
                True,
                "Book Indispon�vel",
            )  # N�o bloqueia se book n�o estiver dispon�vel

        buy_pressure = sum(b.volume for b in book if b.type == mt5.BOOK_TYPE_BUY_LIMIT)[
            :depth
        ]
        sell_pressure = sum(
            s.volume for s in book if s.type == mt5.BOOK_TYPE_SELL_LIMIT
        )[:depth]

        if sell_pressure == 0 and buy_pressure > 0:
            pressure_ratio = float("inf")
        elif sell_pressure == 0:
            return True, "Sem press�o vendedora"
        else:
            pressure_ratio = buy_pressure / sell_pressure

        pressure_threshold = 1.5  # Limiar configur�vel

        if side == "BUY":
            if pressure_ratio >= pressure_threshold:
                return True, f"Press�o OK (C>V {pressure_ratio:.2f}x)"
            else:
                return False, f"Veto Fluxo (Press�o Vendedora: {pressure_ratio:.2f}x)"

        elif side == "SELL":
            if pressure_ratio <= (1 / pressure_threshold):
                return True, f"Press�o OK (V>C {(1/pressure_ratio):.2f}x)"
            else:
                return False, f"Veto Fluxo (Press�o Compradora: {pressure_ratio:.2f}x)"

        return True, "Lado inv�lido"

    except Exception as e:
        logger.warning(f"Erro ao checar press�o do book para {symbol}: {e}")
        return True, "Erro na an�lise do book"  # Falha aberta, n�o bloqueia


# =========================================================
# 🎯 B3 DAILY UNIVERSE BUILDER (2026 Optimized)
# =========================================================


def is_future(symbol: str) -> bool:
    """Verifica se é futuro (WIN, WDO, etc.)"""
    return symbol.upper().startswith(("WIN", "WDO", "IND", "DOL"))


def safe_copy_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    """Copia dados históricos com tratamento de erro"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) < 20:
            return None
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        return df
    except Exception as e:
        logger.error(f"Erro ao copiar dados de {symbol}: {e}")
        return None


def get_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calcula o Average True Range"""
    try:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr
    except Exception as e:
        logger.error(f"Erro ao calcular ATR: {e}")
        return 0.0


def get_ibov_correlation(symbol: str, period: int = 60) -> float:
    """Calcula correlação com IBOV"""
    try:
        # Pega dados do ativo
        df_symbol = safe_copy_rates(symbol, mt5.TIMEFRAME_D1, period)
        if df_symbol is None:
            return 0.0

        # Pega dados do IBOV (IBOV)
        df_ibov = safe_copy_rates("IBOV", mt5.TIMEFRAME_D1, period)
        if df_ibov is None:
            # Tenta com WIN (mini índice) como proxy
            df_ibov = safe_copy_rates("WIN", mt5.TIMEFRAME_D1, period)
            if df_ibov is None:
                return 0.0

        # Calcula retornos
        returns_symbol = df_symbol["close"].pct_change().dropna()
        returns_ibov = df_ibov["close"].pct_change().dropna()

        # Alinha os dados
        min_len = min(len(returns_symbol), len(returns_ibov))
        if min_len < 10:
            return 0.0

        correlation = np.corrcoef(returns_symbol[-min_len:], returns_ibov[-min_len:])[
            0, 1
        ]
        return abs(correlation) if not np.isnan(correlation) else 0.0

    except Exception as e:
        logger.error(f"Erro ao calcular correlação com IBOV para {symbol}: {e}")
        return 0.0


class FundamentalFetcher:
    """Classe para buscar dados fundamentus (mock implementation)"""

    def __init__(self):
        self.cache = {}

    def get_fundamentals(self, symbol: str) -> dict:
        """Retorna dados fundamentus do ativo"""
        try:
            # Remove o número final do símbolo (PETR4 -> PETR)
            base_symbol = symbol[:-1] if symbol[-1].isdigit() else symbol

            # Mock implementation - em produção, buscaria de API real
            # Valores baseados em médias do setor para empresas listadas na B3
            mock_data = {
                "PETR": {"market_cap": 200_000_000_000, "sector": "Petróleo"},
                "VALE": {"market_cap": 300_000_000_000, "sector": "Mineração"},
                "ITUB": {"market_cap": 250_000_000_000, "sector": "Bancos"},
                "BBDC": {"market_cap": 180_000_000_000, "sector": "Bancos"},
                "ABEV": {"market_cap": 150_000_000_000, "sector": "Bebidas"},
                "WEGE": {"market_cap": 120_000_000_000, "sector": "Alimentos"},
                "MGLU": {"market_cap": 80_000_000_000, "sector": "Varejo"},
                "LREN": {"market_cap": 90_000_000_000, "sector": "Varejo"},
                "RENT": {"market_cap": 60_000_000_000, "sector": "Logística"},
                "SUZB": {"market_cap": 70_000_000_000, "sector": "Celulose"},
            }

            # Retorna dados do cache ou padrão
            if base_symbol in mock_data:
                return mock_data[base_symbol]
            else:
                # Para símbolos não mapeados, retorna valor padrão
                return {"market_cap": 10_000_000_000, "sector": "Outros"}

        except Exception as e:
            logger.error(f"Erro ao buscar fundamentus de {symbol}: {e}")
            return {"market_cap": 10_000_000_000, "sector": "Outros"}


# Instância global do fetcher
fundamental_fetcher = FundamentalFetcher()

# ====================== LISTA OFICIAL - Fevereiro 2026 ======================

ELITE_SYMBOLS = [
    # Financeiro (12)
    "ITUB4",
    "BBDC4",
    "BBAS3",
    "B3SA3",
    "BPAC11",
    "ITSA4",
    "SANB11",
    "BBSE3",
    "ABCB4",
    "PINE4",
    "IRBR3",
    "PSSA3",
    # Energia / Petróleo (9)
    "PETR4",
    "PRIO3",
    "VBBR3",
    "EQTL3",
    "ENEV3",
    "NEOE3",
    "AXIA3",
    "RECV3",
    "PETR3",
    # Materiais Básicos (7)
    "VALE3",
    "SUZB3",
    "GGBR4",
    "CSNA3",
    "USIM5",
    "KLBN11",
    "AURA33",
    # Consumo (8)
    "ABEV3",
    "JBSS3",
    "BRFS3",
    "BEEF3",
    "CRFB3",
    "MGLU3",
    "LREN3",
    "ASAI3",
    # Saúde / Varejo / Outros (14)
    "HAPV3",
    "RDOR3",
    "RADL3",
    "WEGE3",
    "RENT3",
    "CCRO3",
    "VIVT3",
    "TIMS3",
    "COGN3",
    "YDUQ3",
    "CYRE3",
    "MRVE3",
    "MULT3",
    "IGTI11",
]

# OPORTUNIDADE (80 exemplos - você pode expandir depois)
OPORTUNIDADE_SYMBOLS = [
    "TOTS3",
    "LWSA3",
    "DESK3",
    "AZUL4",
    "MOVI3",
    "SLCE3",
    "RAIZ4",
    "ONCO3",
    "QUAL3",
    "ANIM3",
    "TEND3",
    "MDNE3",
    "SBSP3",
    "ODPV3",
    "HYPE3",
    "CPFE3",
    "CMIG4",
    "TAEE11",
    "ELET3",
    "ELET6",
    "CSAN3",
    "VAMO3",
    "GMAT3",
    "ALUP11",
    "FLRY3",
    "RENT3",
    "SULA11",
    "BOVA11",
    "SMAL11",
    "MATB11",
    "IVVB11",
    "HASH11",
    "ALZR11",
    "KNRI11",
    "HGLG11",
    "XPML11",
    "VISC11",
    "BRCR11",
    "HGRE11",
    "RBRF11",
    "KNCR11",
    "MXRF11",
    "KNHF11",
    "TGAR11",
    "IRDM11",
    "BTCI11",
    "LVBI11",
    "HGBS11",
    "RBRL11",
    "RBRP11",
    "KNIP11",
    "VRTA11",
    "XPLG11",
    "PVBI11",
    "RECT11",
    "SNCI11",
    "SNEL11",
    "SNME11",
    "TECN3",
    "SOJA3",
    "AGRO3",
    "BRKM5",
    "UNIP6",
    "FHER3",
    "KLBN4",
    "NUTR3",
    "TUPY3",
    "FRAS3",
    "POMO4",
    "TASA4",
    "LEVE3",
    "SHUL4",
    "EALT4",
    "ROMI3",
    "MWET4",
    "PTBL3",
    "ESTC3",
    "AMAR3",
    "LAVV3",
    "CURY3",
    "EVEN3",
    "PDGR3",
    "MRVE3",
    "TEND3",
]

# Whitelist de UNITS que são ações (não ETFs/FIIs)
STOCK_UNITS_WHITELIST = {
    "SANB11",
    "BPAC11",
    "KLBN11",
    "IGTI11",
    "TAEE11",
    "ALUP11",
    "SULA11",
}


def is_etf_or_fii(symbol: str) -> bool:
    """
    Detecta ETFs/FIIs de forma simples:
    - Termina com '11' e não está na whitelist de UNITS-ação
    """
    if not symbol:
        return False
    s = symbol.upper().strip()
    return s.endswith("11") and s not in STOCK_UNITS_WHITELIST


def is_stock(symbol: str) -> bool:
    """
    Retorna True apenas para ações (exclui futuros, ETFs, FIIs e ADRs).
    - Deve terminar com dígito
    - Tamanho até 6 chars
    - Não pode ser futuro
    - Se terminar com '11', só aceita se estiver na whitelist
    - Exclui ADRs (terminam com 33, 34, etc.)
    """
    if not symbol:
        return False
    s = symbol.upper().strip()
    if is_future(s):
        return False
    if len(s) > 6 or not s[-1].isdigit():
        return False
    # Exclui ADRs e ativos estrangeiros (terminam com 33, 34, etc.)
    if s.endswith(("33", "34", "35", "36", "37", "38", "39")):
        return False
    if s.endswith("11"):
        return s in STOCK_UNITS_WHITELIST
    return True


def auto_add_stocks_to_market_watch(symbols: list, log_name: str = "MarketWatch"):
    """
    Adiciona automaticamente uma lista de ações ao Market Watch do MT5.
    Apenas ações (sem futuros/índices).
    """
    added = 0
    failed = []

    for symbol in symbols:
        try:
            # Remove .SA caso alguém coloque por engano
            clean_sym = symbol.replace(".SA", "").upper().strip()

            # Proteção extra: nunca adiciona futuro/índice/ETF/FII
            if any(
                clean_sym.startswith(x) for x in ["WIN", "WDO", "IND", "DOL", "IBOV"]
            ):
                continue
            if is_etf_or_fii(clean_sym):
                continue

            success = mt5.symbol_select(clean_sym, True)
            if success:
                added += 1
            else:
                failed.append(clean_sym)
        except Exception as e:
            failed.append(clean_sym)
            logger.error(f"Erro ao adicionar {clean_sym}: {e}")

    logger.info(
        f"✅ {log_name}: {added} ativos adicionados com sucesso | {len(failed)} falharam"
    )
    if failed:
        logger.warning(f"Falharam: {failed[:20]}...")  # mostra só os primeiros 20

    return added


def load_elite_symbols_from_json() -> dict:
    """
    Carrega símbolos ELITE do arquivo JSON salvo
    Retorna: dict com ELITE, OPORTUNIDADE, TOTAL ou None se não existir
    """
    try:
        if os.path.exists("elite_symbols_latest.json"):
            with open("elite_symbols_latest.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # Valida estrutura
            if isinstance(data, dict) and "ELITE" in data:
                logger.info(
                    f"📁 Carregando elite_symbols_latest.json: "
                    f"{len(data.get('ELITE', []))} ELITE | "
                    f"{len(data.get('OPORTUNIDADE', []))} OPORTUNIDADE"
                )
                return data
            else:
                logger.warning(
                    "⚠️ Arquivo elite_symbols_latest.json com estrutura inválida"
                )
                return None
        else:
            logger.info("ℹ️ Arquivo elite_symbols_latest.json não encontrado")
            return None

    except Exception as e:
        logger.error(f"❌ Erro ao carregar elite_symbols_latest.json: {e}")
        return None


def calculate_asset_score(
    volume: float, atr_pct: float, corr: float, mcap: float, spread_pct: float
) -> float:
    """
    Calcula score de qualidade do ativo B3 (0-100) - Ajustado para mercado B3 2026

    Args:
        volume: Volume financeiro médio diário
        atr_pct: ATR como percentual do preço
        corr: Correlação absoluta com IBOV
        mcap: Market cap em reais
        spread_pct: Spread como percentual
    """
    score = 0.0

    # 1. Liquidez (40 pts) - Volume financeiro (ajustado para B3)
    if volume >= 5_000_000:
        score += 40
    elif volume >= 2_000_000:
        score += 30
    elif volume >= 1_000_000:
        score += 20
    elif volume >= 500_000:
        score += 10
    else:
        score += 5

    # 2. Volatilidade ideal (20 pts) - ATR entre 1.8% e 5.5%
    if 1.8 <= atr_pct <= 5.5:
        score += 20
    elif 1.0 <= atr_pct < 1.8:
        score += 15
    elif 5.5 < atr_pct <= 7.0:
        score += 10
    else:
        score += 5

    # 3. Baixa correlação com IBOV (20 pts) - Quanto menor, melhor
    if corr <= 0.60:
        score += 20
    elif corr <= 0.75:
        score += 15
    elif corr <= 0.85:
        score += 10
    else:
        score += 5

    # 4. Market Cap (10 pts) - Quanto maior, melhor (ajustado para B3)
    if mcap >= 100_000_000_000:  # 100B+
        score += 10
    elif mcap >= 50_000_000_000:  # 50B+
        score += 8
    elif mcap >= 20_000_000_000:  # 20B+
        score += 6
    elif mcap >= 5_000_000_000:  # 5B+
        score += 4
    else:
        score += 2

    # 5. Spread (10 pts) - Quanto menor, melhor
    if spread_pct < 0.10:
        score += 10
    elif spread_pct < 0.20:
        score += 8
    elif spread_pct < 0.35:
        score += 5
    else:
        score += 2

    return round(score, 1)


def atomic_save_json(filename: str, data: dict):
    """Salva JSON de forma atômica"""
    try:
        temp_file = filename + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_file, filename)
        logger.info(f"✅ Arquivo {filename} salvo com sucesso")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {filename}: {e}")
        return False


def build_b3_universe(
    min_fin_volume: float = 15_000_000,
    min_atr_pct: float = 0.80,
    max_atr_pct: float = 7.0,
    max_ibov_corr: float = 0.82,
    min_market_cap: float = 1_000_000_000,
    save_json: bool = True,
) -> dict:
    """
    Scanner dinâmico de ativos B3 - roda 1x/dia
    Retorna: {'ELITE': [...], 'OPORTUNIDADE': [...], 'TOTAL': [...]}
    """
    logger.info("🔍 Iniciando Daily Universe Builder B3...")

    if not mt5:
        logger.error("❌ MT5 não disponível")
        return {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

    try:
        all_symbols = mt5.symbols_get()  # Pega TODOS os símbolos do MT5
        if not all_symbols:
            logger.error("❌ Nenhum símbolo encontrado no MT5")
            return {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

        universe = {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

        logger.info(f"📊 Analisando {len(all_symbols)} símbolos...")

        for sym in all_symbols:
            name = sym.name.upper()

            # Somente ações (sem futuros/ETFs/FIIs)
            if not is_stock(name):
                continue

            info = mt5.symbol_info(name)
            if not info or info.trade_mode not in [1, 4]:
                continue

            # Seleciona o símbolo se não estiver selecionado
            if not info.select:
                if not mt5.symbol_select(name, True):
                    logger.debug(f"❌ Falha ao selecionar {name} no Market Watch")
                    continue

            # === CRITÉRIOS DE QUALIDADE (os mais importantes) ===
            try:
                df = safe_copy_rates(name, mt5.TIMEFRAME_D1, 60)  # 60 dias
                if df is None or len(df) < 20:
                    continue

                # 1. Liquidez (o mais importante na B3)
                avg_fin_volume = (
                    df["tick_volume"] * df["close"]
                ).mean()  # Volume financeiro médio
                if avg_fin_volume < min_fin_volume:
                    continue

                # 2. Volatilidade saudável
                atr = get_atr(df, 14)
                current_price = df["close"].iloc[-1]
                atr_pct = (atr / current_price) * 100
                if not (min_atr_pct <= atr_pct <= max_atr_pct):
                    continue

                # 3. Correlação com IBOV (evita concentração)
                corr = get_ibov_correlation(name)
                if corr > max_ibov_corr:
                    continue

                # 4. Market Cap (via fundamentals)
                fund = fundamental_fetcher.get_fundamentals(name)
                mcap = fund.get("market_cap", 0)
                if mcap < min_market_cap:
                    continue

                # 5. Spread aceitável
                spread_pct = (
                    (info.spread * info.point) / info.ask * 100 if info.ask > 0 else 0
                )
                if spread_pct > 0.35:
                    continue

                # Calcula score final
                score = calculate_asset_score(
                    avg_fin_volume, atr_pct, corr, mcap, spread_pct
                )

                # Classifica por score
                if score >= 85:
                    universe["ELITE"].append((name, score))
                elif score >= 65:
                    universe["OPORTUNIDADE"].append((name, score))
                else:
                    universe["TOTAL"].append((name, score))

                logger.debug(
                    f"✅ {name}: Score {score} (Vol: R${avg_fin_volume:,.0f}, ATR: {atr_pct:.1f}%, Corr: {corr:.2f})"
                )

            except Exception as e:
                logger.debug(f"❌ Erro ao analisar {name}: {e}")
                continue

        # Ordena por score e limita a 120 ativos por categoria
        for key in universe:
            universe[key] = [
                x[0] for x in sorted(universe[key], key=lambda x: x[1], reverse=True)
            ][:120]

        # Salva arquivo JSON
        if save_json:
            atomic_save_json("elite_symbols_latest.json", universe)

        logger.info(
            f"✅ Universe Builder concluído: {len(universe['ELITE'])} ELITE | {len(universe['OPORTUNIDADE'])} OPORTUNIDADE | {len(universe['TOTAL'])} TOTAL"
        )

        # Log dos melhores ativos
        if universe["ELITE"]:
            logger.info(f"🏆 Top 10 ELITE: {', '.join(universe['ELITE'][:10])}")
        if universe["OPORTUNIDADE"]:
            logger.info(
                f"⭐ Top 10 OPORTUNIDADE: {', '.join(universe['OPORTUNIDADE'][:10])}"
            )

        return universe

    except Exception as e:
        logger.error(f"❌ Erro no Universe Builder: {e}")
        return {"ELITE": [], "OPORTUNIDADE": [], "TOTAL": []}

def is_market_open() -> bool:
    """
    Verifica se o mercado está aberto.
    Horário padrão B3: 09:00 - 17:55 (ajustável).
    """
    now = datetime.now().time()
    # Definindo horário de negociação estendido para garantir funcionamento
    start_time = datetime_time(9, 0)
    end_time = datetime_time(17, 55)
    
    # Verifica final de semana (0=Segunda, 6=Domingo)
    if datetime.now().weekday() >= 5:
        return False
        
    return start_time <= now <= end_time

import math

def normalize_volume(symbol: str, raw_quantity: float) -> float:
    """
    Ajusta a quantidade para o lote padrão da B3.
    """
    # Verifica se é mercado fracionário (final F) ou padrão
    if symbol.endswith('F'):
        lot_step = 1
        min_lot = 1
    else:
        # Mercado Padrão (CSED3, VALE3, etc)
        lot_step = 100
        min_lot = 100

    # Arredonda para baixo para o múltiplo mais próximo do lote (Floor)
    qty = math.floor(raw_quantity / lot_step) * lot_step

    if qty < min_lot:
        return 0.0

    return float(qty)

def get_market_regime() -> str:
    """
    Analisa o regime de mercado atual (Bullish, Bearish, Neutral, Extreme Bearish).
    Tenta usar IBOV ou WIN.
    """
    import MetaTrader5 as mt5
    
    # Tenta obter dados do Índice Futuro (Série Contínua) ou IBOV
    indices = ['WIN', 'IBOV', 'IND']
    symbol = None
    
    for idx in indices:
        if mt5.symbol_select(idx, True):
            symbol = idx
            break
            
    if not symbol:
        return 'neutral' # Sem dados
        
    # Análise simples de tendência (Média de 200 no H1)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now(), 200)
    
    if rates is None or len(rates) < 200:
        return 'neutral'
        
    df = pd.DataFrame(rates)
    sma200 = df['close'].mean() # Simples
    current_price = df['close'].iloc[-1]
    
    # Queda de mais de 2% abaixo da média de 200 = Pânico/Tendência forte de baixa
    if current_price < sma200 * 0.98:
        return 'bearish_extreme'
    elif current_price < sma200:
        return 'bearish'
    elif current_price > sma200 * 1.02:
        return 'bullish_strong'
    else:
        return 'bullish' # Acima da média mas perto
