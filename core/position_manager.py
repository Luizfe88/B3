
import logging
from typing import Dict, Any, List
from .execution import ExecutionEngine, OrderParams, OrderSide

logger = logging.getLogger("PositionManager")

class PositionManager:
    """
    Gerencia o portfólio de posições.
    Responsável por calcular risco, stops e atualizações.
    """
    def __init__(self, execution_engine: ExecutionEngine):
        self.execution = execution_engine
        self.active_positions = {}
        
    def get_open_positions(self) -> List[Dict[str, Any]]:
        raw_positions = self.execution.get_positions()
        # Converte para formato interno
        return [self._convert_position(p) for p in raw_positions]

    def _convert_position(self, pos) -> Dict[str, Any]:
        return {
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "entry_price": pos.price_open,
            "current_price": pos.price_current,
            "sl": pos.sl,
            "tp": pos.tp,
            "profit": pos.profit,
            "type": "BUY" if pos.type == 0 else "SELL"
        }

    def close_all(self, reason: str = "Emergency Close"):
        """
        Fecha todas as posições abertas.
        """
        positions = self.get_open_positions()
        if not positions:
            logger.info("✅ Nenhuma posição para fechar.")
            return

        logger.warning(f"🚨 Fechando {len(positions)} posições por motivo: {reason}")
        
        for p in positions:
            # Check for futures if needed (user requirement)
            if self._is_future(p['symbol']):
                continue
                
            self.execution.close_position(p['ticket'], p['symbol'])
            
    def _is_future(self, symbol: str) -> bool:
        # Lógica simplificada de futuros (pode ser melhorada com regex)
        prefixes = ["WIN", "WDO", "IND", "DOL"]
        return any(symbol.startswith(pre) for pre in prefixes)

    def update_stops(self):
        """
        Atualiza SL/TP dinamicamente (Trailing Stop).
        """
        positions = self.get_open_positions()
        for p in positions:
            # Implementar lógica de trailing stop aqui
            pass
            
    def check_risk_limits(self) -> bool:
        """
        Verifica se limites globais de risco foram atingidos.
        """
        # Implementar verificação de perda diária máxima
        return True
