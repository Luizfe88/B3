
import MetaTrader5 as mt5
import logging
import time
import threading
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

            import config
            mt5_acc = getattr(config, "MT5_ACCOUNT", 0)
            init_params = {
                "path": getattr(config, "MT5_TERMINAL_PATH", None),
                "login": int(mt5_acc) if mt5_acc is not None else 0,
                "password": str(getattr(config, "MT5_PASSWORD", "") or ""),
                "server": str(getattr(config, "MT5_SERVER", "") or ""),
                "timeout": 10000
            }
            
            # Filtra None (MT5_TERMINAL_PATH pode ser None)
            init_params = {k: v for k, v in init_params.items() if v is not None}

            if not mt5.initialize(**init_params):
                logger.error(f"❌ Falha ao inicializar MT5: {mt5.last_error()}")
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
            logger.info(f"✅ Conectado ao MetaTrader 5 (Conta: {init_params.get('login')})")
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
                logger.warning(f"❌ Símbolo {symbol} NÃO encontrado no servidor. Pulando permanentemente.")
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
                
            logger.info(f"✅ Posição {ticket} fechada com sucesso")
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
