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

    ⚠️ FIX RACE CONDITION: Rastreia ordens recentemente enviadas que ainda não
       aparecem no MT5 para evitar duplicação de exposição.
    """

    def __init__(self, execution_engine: ExecutionEngine, magic_number: int = 123456):
        self.execution = execution_engine
        self.magic_number = magic_number
        self.active_positions = {}
        # Rastreamento de ordens pendentes (últimos 3 segundos)
        self.pending_orders = []  # Lista de (timestamp, symbol, volume, price)

    def get_open_positions(self, filter_magic: bool = True) -> List[Dict[str, Any]]:
        raw_positions = self.execution.get_positions()

        filtered = []
        for p in raw_positions:
            # Se filter_magic=True, só retorna posições do nosso robô (Magic Number 2026)
            # O bot usa magic=2026 fixo no utils.py, mas aqui estava magic_number=123456
            # Vamos alinhar para aceitar ambos ou apenas o correto.

            # Ajuste crítico: Se o utils.py usa 2026, devemos filtrar por ele também.
            if filter_magic:
                if p.magic not in [self.magic_number, 2026]:
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
            "time": pos.time,  # Importante para throttle
        }

    def close_all(self, reason: str = "Emergency Close"):
        """
        Fecha todas as posições abertas GERENCIADAS PELO BOT.
        Ignora posições manuais ou de outros robôs (magic number diferente).
        """
        positions = self.get_open_positions(filter_magic=True)  # Só pega as nossas

        if not positions:
            logger.info("✅ Nenhuma posição gerenciada pelo bot para fechar.")
            return

        logger.warning(f"🚨 Fechando {len(positions)} posições por motivo: {reason}")

        for p in positions:
            # Check for futures if needed (user requirement)
            if self._is_future(p["symbol"]):
                logger.info(f"➡️ Pulando futuro: {p['symbol']}")
                continue

            self.execution.close_position(p["ticket"], p["symbol"])

    def _is_future(self, symbol: str) -> bool:
        # Lógica simplificada de futuros (pode ser melhorada com regex)
        prefixes = ["WIN", "WDO", "IND", "DOL"]
        return any(symbol.startswith(pre) for pre in prefixes)

    def update_stops(self):
        """
        Atualiza SL/TP dinamicamente utilizando a engine centralizada TrailingStopConfig (5 Níveis).
        Inclui travamento inteligente e Circuit Breaker para não devolver lucro.
        """
        from trailing_stop import calculate_dynamic_stop, TrailingStopConfig
        import MetaTrader5 as mt5  # Garantir acesso ao timeframe e symbols
        
        positions = self.get_open_positions()
        if not positions:
            return
            
        try:
            ts_config = TrailingStopConfig()
            for p in positions:
                try:
                    p_sym = p.get("symbol", "")
                    p_type = p.get("type", "BUY")
                    entry_price = p.get("entry_price", 0.0) or p.get("price_open", 0.0)
                    current_sl  = p.get("sl", 0.0)
                    cur_price   = p.get("current_price", 0.0) or p.get("price_current", 0.0)
                    ticket      = p.get("ticket", None)

                    if not p_sym or not ticket or entry_price <= 0:
                        continue

                    # ATR via candles do símbolo (M15)
                    ts_candles = utils.safe_copy_rates(p_sym, utils.TIMEFRAME_BASE, 30)
                    if ts_candles is None or ts_candles.empty:
                        continue
                    
                    ind_ts = utils.quick_indicators_custom(p_sym, utils.TIMEFRAME_BASE, df=ts_candles)
                    atr = ind_ts.get("atr", 0.0) if ind_ts else 0.0
                    adx = ind_ts.get("adx", 0.0) if ind_ts else 0.0

                    if atr <= 0:
                        continue

                    # max_price: extensão máxima em favor da posição
                    if p_type == "BUY":
                        max_price = p.get("price_max", cur_price) or cur_price
                        position_side = 1
                    else:  # SELL
                        max_price = p.get("price_min", cur_price) or cur_price
                        position_side = -1

                    last_candle = ts_candles.iloc[-1]

                    new_sl, ts_reason = calculate_dynamic_stop(
                        current_price=cur_price,
                        entry_price=entry_price,
                        current_stop_price=current_sl,
                        max_price_reached=max_price,
                        atr=atr,
                        position_side=position_side,
                        config=ts_config,
                        candle_low=float(last_candle["low"]),
                        candle_high=float(last_candle["high"]),
                        adx=adx,
                    )

                    # Só move SL se melhorou (protege lucro — nunca piora)
                    sl_improved = (
                        (position_side == 1  and new_sl > current_sl) or
                        (position_side == -1 and new_sl < current_sl and new_sl > 0)
                    )
                    if sl_improved and ts_reason != "HOLD" and ts_reason != "ATR_ZERO":
                        
                        # Fix Stop Levels Rejection Rule here before submitting
                        import utils
                        new_sl, _ = utils.validate_stops_level(p_sym, p_type, cur_price, new_sl, p.get("tp", 0.0))
                        
                        logger.info(
                            f"🔒 Trailing Stop {p_sym} [{ts_reason}]: SL {current_sl:.4f} → {new_sl:.4f}"
                        )
                        # Assumindo que execution possua modify_sl, ou precisamos criar se não houver (mas parece haver no código original)
                        if hasattr(self.execution, "modify_sl"):
                            self.execution.modify_sl(ticket, new_sl)
                        else:
                            # Se não existir modify_sl explícito, enviamos um TRADE_ACTION_SLTP
                            import MetaTrader5 as mt5_api
                            req = {
                                "action": mt5_api.TRADE_ACTION_SLTP,
                                "position": ticket,
                                "symbol": p_sym,
                                "sl": new_sl,
                                "tp": p.get("tp", 0.0),
                            }
                            mt5_api.order_send(req)

                except Exception as ts_err:
                    logger.debug(f"⚠️ Trailing stop erro em {p.get('symbol','?')}: {ts_err}")

        except Exception as ts_outer_err:
            logger.error(f"❌ Erro no loop de trailing stop: {ts_outer_err}")

    def check_risk_limits(self) -> bool:
        """
        Verifica se limites globais de risco foram atingidos.
        """
        # Implementar verificação de perda diária máxima
        return True

    def get_total_exposure(self) -> float:
        """
        Calcula a exposição financeira total (soma de todas as posições abertas + PENDENTES).
        ⚠️ ATUALIZADO: Inclui ordens recentemente enviadas que ainda não aparecem no MT5.

        Isto resolve o problema de race condition onde múltiplas ordens são enviadas
        antes do MT5 registrar a primeira posição.
        """
        # Posições confirmadas
        positions = self.get_open_positions(filter_magic=True)
        confirmed_exposure = sum(p["volume"] * p["current_price"] for p in positions)

        # Posições pendentes (últimos 3 segundos)
        pending_exp_dict = self.get_pending_exposure()
        pending_exposure = sum(pending_exp_dict.values())

        total = confirmed_exposure + pending_exposure

        # Log detalhado para debug
        if total > 0:
            logger.info(
                f"📊 Exposição Total: R${total:.2f} "
                f"(Confirmada: R${confirmed_exposure:.2f} + Pendente: R${pending_exposure:.2f})"
            )

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

    def register_pending_order(self, symbol: str, volume: float, price: float):
        """
        Registra uma ordem que foi enviada mas pode não estar refletida no MT5 ainda.
        Importante para evitar race condition de múltiplas ordens simultâneas.
        """
        now = datetime.now()
        self.pending_orders.append(
            {"timestamp": now, "symbol": symbol, "volume": volume, "price": price}
        )
        logger.info(f"📤 Ordem pendente registrada: {symbol} x{volume} @ R${price:.2f}")

    def clean_pending_orders(self):
        """
        Remove ordens pendentes que já têm mais de 3 segundos.
        Assume que o MT5 já atualizou sua posição até então.
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=3)

        before_count = len(self.pending_orders)
        self.pending_orders = [
            order for order in self.pending_orders if order["timestamp"] > cutoff
        ]

        removed = before_count - len(self.pending_orders)
        if removed > 0:
            logger.debug(f"🧹 Limpas {removed} ordens pendentes (> 3 segundos)")

    def get_pending_exposure(self) -> Dict[str, float]:
        """
        Calcula exposição de ordens pendentes por símbolo.
        Retorna dict: {symbol: exposure_in_reais}
        """
        self.clean_pending_orders()

        pending_exp = {}
        for order in self.pending_orders:
            symbol = order["symbol"]
            exposure = order["volume"] * order["price"]
            pending_exp[symbol] = pending_exp.get(symbol, 0.0) + exposure

        if pending_exp:
            logger.debug(f"⏳ Exposição pendente: {pending_exp}")

        return pending_exp
