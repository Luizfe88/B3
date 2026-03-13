import MetaTrader5 as mt5
import logging
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("ExecutionEngine")

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderParams:
    symbol: str
    side: OrderSide
    volume: float
    price: float
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 123456
    comment: str = "XP3 Agent"
    type_filling: int = mt5.ORDER_FILLING_IOC

class ExecutionEngine:
    """
    Camada de execução responsável pela comunicação direta com o MT5.
    Gerencia conexão, envio de ordens e verificação de execução.
    Thread-safe.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._connected = False

    def connect(self) -> bool:
        with self._lock:
            # Se já estiver conectado e o terminal responder, não reinicia
            if self._connected:
                info = mt5.terminal_info()
                if info and info.connected:
                    return True

            import utils
            if not utils.safe_mt5_initialize():
                self._connected = False
                return False
            
            # Pequeno delay para estabilização da conexão com o servidor
            time.sleep(1.0)

            # Verifica conexão efetiva com o servidor de trade
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("⚠️ Falha ao obter informações do terminal.")
                self._connected = False
                return False
            elif not terminal_info.connected:
                 logger.warning("⚠️ Terminal MT5 inicializado, mas desconectado do servidor (Verifique login/proxy).")
                 self._connected = False
                 return False
            
            self._connected = True
            return True

    def is_connected(self) -> bool:
        """Verifica se o terminal está efetivamente conectado ao servidor."""
        with self._lock:
            if not self._connected:
                return False
            info = mt5.terminal_info()
            return info is not None and info.connected

    def safe_symbol_select(self, symbol: str, select: bool = True, max_retries: int = 3) -> bool:
        """
        Encapsula mt5.symbol_select com mecanismo de skip rápido e retry.
        """
        # --- FAST SKIP: Se o símbolo nem existe no servidor, pula imediatamente ---
        with self._lock:
            sym_info = mt5.symbol_info(symbol)
            if sym_info is None:
                logger.warning(f"❌ Símbolo {symbol} NÃO encontrado no servidor. Pulando este ciclo.")
                return False

        if not self.is_connected():
            if not self.connect():
                return False

        for attempt in range(max_retries):
            with self._lock:
                if mt5.symbol_select(symbol, select):
                    return True
                
                err = mt5.last_error()
                # Code -1: Terminal: Call failed (Geralmente congestionamento de IPC)
                if err[0] == -1:
                    logger.warning(f"⚠️ [Attempt {attempt+1}/{max_retries}] {symbol} select failed (IPC load). Waiting...")
                    time.sleep(1.0 * (attempt + 1))
                    
                    if attempt == max_retries - 1:
                        if not self.is_connected():
                            self.connect()
                else:
                    # Outro erro (Ex: limite de ativos no MarketWatch atingido)
                    logger.error(f"❌ Falha ao selecionar {symbol}: {err}")
                    return False
        
        return False

    def shutdown(self):
        with self._lock:
            mt5.shutdown()
            self._connected = False
            logger.info("🛑 MT5 desconectado")

    def send_order(self, order: OrderParams) -> Dict[str, Any]:
        """
        Envia uma ordem para o MT5.
        Retorna dicionário com resultado ou erro.
        """
        if not self.is_connected():
            if not self.connect():
                return {"status": "error", "message": "MT5 not connected"}

        action = mt5.TRADE_ACTION_DEAL
        type_order = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL

        # ─── Pré-validação: stops_level mínimo do broker (evita Code 10016) ──────
        tick = mt5.symbol_info_tick(order.symbol)
        ref_price = tick.ask if order.side == OrderSide.BUY else tick.bid if tick else 0.0
        if ref_price > 0:
            try:
                import utils
                order.sl, order.tp = utils.validate_stops_level(order.symbol, order.side.value, ref_price, order.sl, order.tp)
            except Exception as e:
                logger.error(f"Erro ao validar stops level para {order.symbol}: {e}")
        # ─────────────────────────────────────────────────────────────────────────

        request = {
            "action": action,
            "symbol": order.symbol,
            "volume": float(order.volume),
            "type": type_order,
            "price": float(order.price),
            "sl": float(order.sl),
            "tp": float(order.tp),
            "deviation": 20,
            "magic": order.magic,
            "comment": order.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": order.type_filling,
        }

        with self._lock:
            # Check price freshness logic could go here
            result = mt5.order_send(request)
            
        if result is None:
            err = mt5.last_error()
            logger.error(f"❌ Erro crítico no envio de ordem para {order.symbol}: {err}")
            return {"status": "error", "message": f"MT5 internal error: {err}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"❌ Ordem rejeitada ({order.symbol}): {result.comment} (Code: {result.retcode})")
            return {
                "status": "error", 
                "message": f"MT5 Error: {result.comment}",
                "retcode": result.retcode
            }

        logger.info(f"✅ Ordem executada: {order.symbol} {order.side.value} {order.volume} @ {result.price} (Ticket: {result.order})")
        
        # LOGAR TRADE NO TXT (INTEGRAÇÃO)
        try:
             import utils
             if result.retcode == mt5.TRADE_RETCODE_DONE:
                 trade_data = {
                     "ticket": result.order, # ID da ordem gerada
                     "time": time.strftime("%H:%M:%S", time.localtime()),
                     "symbol": order.symbol,
                     "type": order.side.value,
                     "volume": order.volume,
                     "price": result.price if hasattr(result, 'price') else order.price,
                     "profit": 0.0, # Entrada não tem lucro ainda
                     "comment": order.comment
                 }
                 utils.log_trade_to_txt(trade_data)
        except Exception as e:
             logger.error(f"Erro ao logar trade TXT: {e}")

        return {
            "status": "success",
            "message": "Order executed successfully",
            "order_id": result.order,
            "price": result.price if hasattr(result, 'price') else order.price,
            "volume": result.volume if hasattr(result, 'volume') else order.volume
        }

    def close_position(self, ticket: int, symbol: str, volume: Optional[float] = None) -> bool:
        """
        Fecha uma posição específica pelo ticket.
        """
        with self._lock:
            # Obter detalhes da posição para saber o lado oposto
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.warning(f"⚠️ Posição {ticket} não encontrada para fechamento")
                return False
            
            pos = positions[0]
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
            
            vol = volume if volume else pos.volume
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(vol),
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "Agent Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Falha ao fechar {ticket}: {result.comment}")
                return False
                
            logger.info(f"✅ Posição {ticket} fechada com sucesso ({request.get('comment', 'Agent Close')})")
            return True

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """
        Modifica o Stop Loss de uma posição aberta.
        """
        with self._lock:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            pos = positions[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": pos.symbol,
                "sl": float(new_sl),
                "tp": pos.tp,
                "magic": pos.magic,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"❌ Falha ao modificar SL do ticket {ticket}: {result.comment}")
                return False
            return True

    def get_positions(self, symbol: Optional[str] = None) -> list:
        if not self.is_connected():
            self.connect()
        with self._lock:
            if symbol:
                return list(mt5.positions_get(symbol=symbol) or [])
            return list(mt5.positions_get() or [])

    def get_symbol_info(self, symbol: str):
        with self._lock:
            return mt5.symbol_info(symbol)

    # ========================
    # 🌐 OPTIMAL EXECUTION (Pilar 3)
    # ========================
    
    def get_avg_volume_profile(self, symbol: str, lookback_days: int = 5) -> pd.Series:
        """Calcula o perfil de volume intra-day (M5) dos últimos dias."""
        try:
            import utils
            rates = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 12 * 9 * lookback_days)
            if rates is None or rates.empty:
                return pd.Series()
            
            rates['time_only'] = rates['time'].dt.time
            profile = rates.groupby('time_only')['tick_volume'].mean()
            return profile
        except Exception as e:
            logger.error(f"Erro ao calcular perfil de volume para {symbol}: {e}")
            return pd.Series()

    def almgren_chriss_slicing(self, total_volume: float, duration_slices: int, risk_aversion: float = 0.1) -> np.ndarray:
        """Implementação Almgren-Chriss simplificada."""
        volatility = 0.02 # Proxy
        liquidity_coeff = 0.01 # Proxy
        kappa = np.sqrt(risk_aversion * volatility**2 / liquidity_coeff)
        t = np.arange(duration_slices + 1)
        N, T = total_volume, duration_slices
        n_t = N * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        slices = -np.diff(n_t)
        return slices * (N / np.sum(slices)) if np.sum(slices) != 0 else np.array([total_volume])

    def execute_advanced(self, symbol: str, side: OrderSide, total_volume: float, duration_min: int, comment: str):
        """Executa ordem fatiada (VWAP/TWAP) via thread separada para não bloquear o bot."""
        def _task():
            try:
                logger.info(f"🚀 [ASYNC SLICE] Iniciando execução {symbol} | Total: {total_volume} | Duração: {duration_min}m")
                slices_count = max(1, duration_min)
                interval = 60 # 1 minuto entre fatias
                slice_vol = total_volume / slices_count
                
                for i in range(slices_count):
                    # No primeiro lote envia na hora, nos demais aguarda o intervalo
                    if i > 0: 
                        time.sleep(interval)
                    
                    import utils
                    norm_vol = utils.normalize_volume(symbol, slice_vol)
                    if norm_vol <= 0: continue
                    
                    # Refresh de cotação para garantir preço justo em cada fatia
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        logger.warning(f"⚠️ [SLICE] Falha ao obter tick para {symbol} na fatia {i+1}")
                        continue
                        
                    price = tick.ask if side == OrderSide.BUY else tick.bid
                    order = OrderParams(
                        symbol=symbol, 
                        side=side, 
                        volume=norm_vol, 
                        price=price, 
                        comment=f"{comment}_S{i+1}"
                    )
                    
                    logger.info(f"📤 [SLICE] Enviando fatia {i+1}/{slices_count} de {symbol} Vol: {norm_vol}")
                    self.send_order(order)
                
                logger.info(f"✅ [ASYNC SLICE] Execução avançada de {symbol} finalizada.")
            except Exception as e:
                logger.error(f"❌ [SLICE ERROR] Erro na thread de execução para {symbol}: {e}")
        
        # Daemon=True garante que se o bot morrer, a thread morre junto
        threading.Thread(target=_task, daemon=True, name=f"Slicing_{symbol}").start()

    def execute_smart_order(self, symbol: str, side: OrderSide, volume: float, price: float, sl: float, tp: float, comment: str):
        """Decide entre execução direta ou fatiada baseado na agressividade (eta)."""
        try:
            from calibration_manager import calibration_manager
            calib = calibration_manager.get_calibrated_params(symbol)
            eta = calib.get("eta_aggression", 5.0) # Default 5.0 (agressivo)
            
            import utils
            rates = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 12) # Última hora
            avg_vol_5m = rates['tick_volume'].mean() if rates is not None and not rates.empty else 100
            
            # Se o volume da ordem for maior que eta * volume médio de 5min, fatiamos
            safe_volume_limit = avg_vol_5m * eta
            
            # --- SLIPPAGE GUARD: Aborta se o volume for extremo (10x o volume seguro) ---
            extreme_liquidity_threshold = safe_volume_limit * 10
            if volume > extreme_liquidity_threshold:
                logger.error(f"🚨 [LIQUIDITY CRASH] {symbol}: Volume solicitado ({volume}) é extremo vs liquidez segura ({extreme_liquidity_threshold:.1f}). ABORTANDO EXECUÇÃO.")
                return False

            if volume > safe_volume_limit:
                logger.info(f"⚖️ [SMART] Volume {volume} > {avg_vol_5m} * {eta:.1f}. Fatiando execução.")
                self.execute_advanced(symbol, side, volume, 10, comment)
                return True
            else:
                order = OrderParams(symbol=symbol, side=side, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
                res = self.send_order(order)
                return res.get("status") == "success"
        except Exception as e:
            logger.error(f"Erro no smart order ({symbol}): {e}")
            order = OrderParams(symbol=symbol, side=side, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
            res = self.send_order(order)
            return res.get("status") == "success"
