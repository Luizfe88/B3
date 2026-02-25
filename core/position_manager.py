
import logging
from typing import Dict, Any, List
from .execution import ExecutionEngine, OrderParams, OrderSide

logger = logging.getLogger("PositionManager")

class PositionManager:
    """
    Gerencia o portfólio de posições.
    Responsável por calcular risco, stops e atualizações.
    """
    def __init__(self, execution_engine: ExecutionEngine, magic_number: int = 123456):
        self.execution = execution_engine
        self.magic_number = magic_number
        self.active_positions = {}
        
    def get_open_positions(self, filter_magic: bool = True) -> List[Dict[str, Any]]:
        raw_positions = self.execution.get_positions()
        
        filtered = []
        for p in raw_positions:
            # Se filter_magic=True, só retorna posições do nosso robô
            if filter_magic and p.magic != self.magic_number:
                continue
            filtered.append(self._convert_position(p))
            
        return filtered

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
            "magic": pos.magic,
            "type": "BUY" if pos.type == 0 else "SELL"
        }

    def close_all(self, reason: str = "Emergency Close"):
        """
        Fecha todas as posições abertas GERENCIADAS PELO BOT.
        Ignora posições manuais ou de outros robôs (magic number diferente).
        """
        positions = self.get_open_positions(filter_magic=True) # Só pega as nossas
        
        if not positions:
            logger.info("✅ Nenhuma posição gerenciada pelo bot para fechar.")
            return

        logger.warning(f"🚨 Fechando {len(positions)} posições por motivo: {reason}")
        
        for p in positions:
            # Check for futures if needed (user requirement)
            if self._is_future(p['symbol']):
                logger.info(f"➡️ Pulando futuro: {p['symbol']}")
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
