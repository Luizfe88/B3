
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .execution import ExecutionEngine, OrderParams, OrderSide
import utils
import config

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
            "type": "BUY" if pos.type == 0 else "SELL",
            "time": pos.time # Importante para throttle
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
            # Chama a função de trailing do utils (que já tem a lógica de ATR)
            utils.manage_dynamic_trailing(p['symbol'], p['ticket'])
            
    def check_risk_limits(self) -> bool:
        """
        Verifica se limites globais de risco foram atingidos.
        """
        # Implementar verificação de perda diária máxima
        return True

    def get_total_exposure(self) -> float:
        """
        Calcula a exposição financeira total (soma de todas as posições abertas).
        """
        positions = self.get_open_positions(filter_magic=True)
        total = 0.0
        for p in positions:
            # Exposição = Volume * Preço Atual
            # (Para futuros, precisaria multiplicar pelo contrato, mas simplificado aqui)
            total += p['volume'] * p['current_price']
            
        # Loga as posições que estão somando
        if total > 0:
            logger.info(f"📊 Exposição atual: R$ {total:.2f} (em {len(positions)} posições)")
            
        return total

    def count_recent_entries(self, minutes: int = 60) -> int:
        """
        Conta quantas posições foram abertas nos últimos X minutos.
        Nota: mt5.positions_get retorna 'time' como timestamp de abertura.
        """
        raw_positions = self.execution.get_positions()
        count = 0
        limit_time = datetime.now().timestamp() - (minutes * 60)
        
        for p in raw_positions:
            if p.magic == self.magic_number:
                if p.time >= limit_time:
                    count += 1
        return count
