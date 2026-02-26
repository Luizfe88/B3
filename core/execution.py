
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
            if not mt5.initialize():
                logger.error(f"❌ Falha ao inicializar MT5: {mt5.last_error()}")
                self._connected = False
                return False
            
            # Check terminal connection status
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("⚠️ Falha ao obter informações do terminal.")
            elif not terminal_info.connected:
                 logger.warning("⚠️ Terminal MT5 inicializado, mas desconectado do servidor.")
            
            self._connected = True
            logger.info("✅ Conectado ao MetaTrader 5")
            return True

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
        if not self._connected:
            if not self.connect():
                return {"status": "error", "message": "MT5 not connected"}

        action = mt5.TRADE_ACTION_DEAL
        type_order = mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
        
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

    def get_positions(self, symbol: Optional[str] = None) -> list:
        with self._lock:
            if symbol:
                return list(mt5.positions_get(symbol=symbol) or [])
            return list(mt5.positions_get() or [])

    def get_symbol_info(self, symbol: str):
        with self._lock:
            return mt5.symbol_info(symbol)
