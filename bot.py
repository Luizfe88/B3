
import os
import sys
import logging
import time
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from logging.handlers import TimedRotatingFileHandler

# Configuração de logs com rotação a cada 3 horas
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = "logs/trading_agents.log"

# Garante diretório de logs
os.makedirs("logs", exist_ok=True)

# Handler de Arquivo com Rotação (3 horas)
# O padrão TimedRotatingFileHandler adiciona .YYYY-MM-DD_HH-MM ao final
# Para ter trading_agents_2023-10-27_12-00.log precisamos customizar ou aceitar o padrão .log.data
# Vamos usar o padrão, mas configurando o sufixo para ser amigável.
file_handler = TimedRotatingFileHandler(log_file, when="H", interval=3, backupCount=10, encoding='utf-8')
file_handler.suffix = "%Y-%m-%d_%H-%M.log" # Define o formato do sufixo (ex: 2023-10-27_12-00.log)
# O TimedRotatingFileHandler por padrão anexa o sufixo DEPOIS da extensão original (ex: file.log.2023...)
# Para fazer exatamante file_DATE.log é mais complexo, mas ajustando o sufixo já ajuda.
# Vamos forçar uma nomenclatura mais limpa sobrescrevendo o namer.

def custom_namer(name):
    # name vem como "logs/trading_agents.log.2023-10-27_12-00.log"
    # Queremos "logs/trading_agents_2023-10-27_12-00.log"
    base, ext, date_part = name.rsplit(".", 2) 
    return f"{base}_{date_part}.log"

# file_handler.namer = custom_namer # (Opcional, pode ser complexo de manter cross-platform)
file_handler.setFormatter(log_formatter)

# Handler de Console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Configura logger raiz
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Forçar encoding UTF-8 no stdout para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger("MainBot")

# Imports do sistema
from core.execution import ExecutionEngine, OrderParams, OrderSide
from core.position_manager import PositionManager
from agents.fund_manager import FundManager
import config
import utils
import MetaTrader5 as mt5

def main():
    logger.info("🚀 Iniciando TradingAgents-B3 Framework...")
    
    # 1. Inicialização de Infraestrutura
    execution = ExecutionEngine()
    if not execution.connect():
        logger.critical("❌ Falha crítica: Não foi possível conectar ao MT5. Abortando.")
        return

    position_manager = PositionManager(execution)
    fund_manager = FundManager()
    
    # 2. Loop Principal
    logger.info("✅ Sistema online. Preparando Market Watch...")
    
    # Inicializa Market Watch para garantir visibilidade
    # Itera sobre todos os símbolos monitorados e força a seleção
    # Isso evita erros de "Símbolo não selecionado" durante o loop
    all_symbols = config.MONITORED_SYMBOLS
    valid_symbols = []
    
    for sym in all_symbols:
        if mt5.symbol_select(sym, True):
            valid_symbols.append(sym)
        else:
            logger.warning(f"⚠️ Falha inicial ao selecionar {sym} - Removido da lista de execução.")
            
    logger.info(f"📋 Market Watch inicializado: {len(valid_symbols)}/{len(all_symbols)} ativos válidos e prontos.")

    logger.info("🚀 Iniciando loop de mercado.")
    
    try:
        while True:
            # Verifica se deve fechar posições no final do dia
            now = datetime.now()
            current_time = now.time()
            
            # Define horário de fechamento (Sexta vs Outros dias)
            if now.weekday() == 4: # 4 = Sexta-feira
                close_time_str = config.FRIDAY_CLOSE_ALL_BY # "17:15"
            else:
                close_time_str = config.CLOSE_ALL_BY # "17:45"
                
            close_time = datetime.strptime(close_time_str, "%H:%M").time()
            
            if current_time >= close_time:
                 # Se ainda tiver posições abertas, fecha tudo
                 open_pos = position_manager.get_open_positions()
                 if open_pos:
                     logger.info("⏰ Horário de fechamento diário atingido. Zerando carteira...")
                     position_manager.close_all(reason="End of Day")
                 else:
                     logger.info("💤 Mercado fechado ou horário limite atingido. Aguardando...")
                 
                 time.sleep(60)
                 continue

            # Verifica horário de mercado
            if not utils.is_market_open():
                logger.info("💤 Mercado fechado. Aguardando...")
                time.sleep(60)
                continue
                
            # Verifica conexões
            if not execution.connect():
                logger.warning("⚠️ MT5 desconectado. Tentando reconectar...")
                time.sleep(5)
                continue

            # Obtém lista de ativos (Apenas os válidos)
            symbols = valid_symbols
            
            # Check de quantidade de posições antes do loop
            open_positions_list = position_manager.get_open_positions()
            open_count = len(open_positions_list)
            
            # Log de auditoria de posições
            if open_count > 0:
                 logger.debug(f"🔍 Posições Abertas ({open_count}): {[f'{p['symbol']} ({p['magic']})' for p in open_positions_list]}")

            if open_count >= config.MAX_CONCURRENT_POSITIONS:
                logger.info(f"🛑 Limite de posições atingido ({open_count}/{config.MAX_CONCURRENT_POSITIONS}).")
                
                # Se exceder (ex: por posições manuais ou erro anterior), tenta reduzir?
                # Por enquanto, apenas atualiza stops das existentes e aguarda
                position_manager.update_stops()
                
                logger.info("💤 Aguardando liberação de slots...")
                time.sleep(60)
                continue
            
            for symbol in symbols:
                try:
                    # 1. Coleta dados de mercado (Market Data)
                    if not mt5.symbol_select(symbol, True):
                        # Tenta novamente após um curto delay (pode ser problema de rede momentâneo)
                        time.sleep(0.1)
                        if not mt5.symbol_select(symbol, True):
                            # Tenta reconectar se falhar duas vezes
                            logger.warning(f"⚠️ Falha ao selecionar {symbol}. Tentando reconexão com MT5...")
                            if execution.connect():
                                if mt5.symbol_select(symbol, True):
                                    logger.info(f"✅ {symbol} selecionado após reconexão.")
                                else:
                                    logger.warning(f"⚠️ Não foi possível selecionar {symbol} no MT5 mesmo após reconexão. Erro: {mt5.last_error()}. Pulando.")
                                    continue
                            else:
                                logger.error("❌ Falha na reconexão com MT5 durante o loop.")
                                continue
                    
                    # Candles (últimos 100 M15)
                    candles = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
                    if candles is None or candles.empty:
                        logger.warning(f"⚠️ Dados insuficientes (candles) para {symbol}. Pulando.")
                        continue
                        
                    # Preço atual e Ticks
                    tick_info = mt5.symbol_info_tick(symbol)
                    current_price = tick_info.last if tick_info else candles['close'].iloc[-1]

                    # Ticks (últimos 1000 ticks) - Usando horário do servidor para evitar problemas de fuso
                    try:
                        if tick_info:
                            server_time = datetime.fromtimestamp(tick_info.time)
                            # Pega ticks da última hora baseada no servidor
                            ticks = mt5.copy_ticks_range(symbol, server_time - timedelta(hours=1), server_time, mt5.COPY_TICKS_ALL)
                        else:
                            # Fallback se não tiver tick info (mercado fechado/sem dados recentes)
                            # Tenta pegar ultimos 1000 a partir de agora (menos confiável se fuso errado)
                            ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(hours=1), 1000, mt5.COPY_TICKS_ALL)
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao obter ticks para {symbol}: {e}")
                        ticks = []
                    
                    # Dados Globais de Risco
                    account_info = mt5.account_info()
                    equity = account_info.equity if account_info else 1000.0
                    
                    # Calcula exposições por setor
                    sector_exposure = {}
                    open_positions = position_manager.get_open_positions()
                    for p in open_positions:
                        p_sym = p['symbol']
                        p_sector = config.SECTOR_MAP.get(p_sym, "OUTROS")
                        p_exp = p['volume'] * p['current_price']
                        sector_exposure[f'sector_exposure_{p_sector}'] = sector_exposure.get(f'sector_exposure_{p_sector}', 0.0) + p_exp

                    market_data = {
                        "price": current_price,
                        "ticks": ticks if ticks is not None else [],
                        "candles": candles,
                        "equity": equity,
                        "total_exposure": position_manager.get_total_exposure(),
                        "recent_entries_count": position_manager.count_recent_entries(minutes=60),
                        "ibov_trend": utils.get_market_regime(),
                        **sector_exposure # Adiciona exposições setoriais ao contexto
                    }
                    
                    # 2. Decisão do Fund Manager (Agentes)
                    decision = fund_manager.decide(symbol, market_data)
                    
                    # 3. Execução
                    # Verifica account info para gestão de risco e lote
                    account_info = mt5.account_info()
                    if account_info:
                        equity = account_info.equity
                    else:
                        equity = 1000.0 # Fallback
                    
                    if decision["action"] == "BUY":
                        # Valida se já tem posição
                        open_positions = position_manager.get_open_positions()
                        for p in open_positions:
                            if p['symbol'] == symbol:
                                if p['type'] == 'SELL':
                                    logger.info(f"🔄 Invertendo mão em {symbol} (SELL -> BUY)")
                                    execution.close_position(p['ticket'], symbol)
                                else:
                                    logger.info(f"⏭️ Posição de COMPRA já existente em {symbol}. Mantendo.")
                                    continue # Já comprado, não faz nada (poderia aumentar posição)

                        # Cálculo de Lote:
                        # 1. Base: Config do capital (configurável)
                        # 2. Ajuste: size_multiplier do agente
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        size_multiplier = decision.get("size", 0.0)
                        
                        target_exposure = equity * base_allocation_pct * size_multiplier
                        
                        # Correção: Garante lote mínimo de 100 se exposição > 0
                        # Se o preço for muito alto e equity baixo, pode dar 0. 
                        # Vamos forçar o cálculo correto de lotes.
                        if current_price > 0:
                            raw_qty = target_exposure / current_price
                        else:
                            raw_qty = 0

                        final_volume = utils.normalize_volume(symbol, raw_qty)
                        
                        # Se volume ficou 0 mas a decisão é forte e tem capital, tenta lote mínimo
                        if final_volume == 0 and size_multiplier > 0 and equity > 1000:
                            min_lot = 100
                            cost = min_lot * current_price
                            if cost <= equity * 0.95: # Margem de segurança
                                final_volume = float(min_lot)
                                logger.info(f"⚠️ Volume ajustado para lote mínimo ({final_volume}) em {symbol}")
                        
                        if final_volume <= 0:
                             logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                             continue
                        
                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(symbol, mt5.TIMEFRAME_M15, df=candles)
                        sl, tp = utils.calculate_dynamic_sl_tp(symbol, "BUY", current_price, ind)
                        
                        if not sl or sl <= 0:
                             logger.warning(f"⚠️ SL inválido para {symbol}. Bloqueando ordem.")
                             continue

                        # Double Check de Posições ANTES de enviar ordem
                        current_open = len(position_manager.get_open_positions())
                        if current_open >= config.MAX_CONCURRENT_POSITIONS:
                             logger.warning(f"🛑 [FAILSAFE] Limite de posições atingido ({current_open}/{config.MAX_CONCURRENT_POSITIONS}) antes de BUY em {symbol}. Abortando.")
                             continue

                        # Cria ordem
                        order = OrderParams(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            volume=final_volume,
                            price=0.0, # Market order
                            sl=sl, # SL calculado
                            tp=tp  # TP calculado
                        )
                        execution.send_order(order)
                        
                    elif decision["action"] == "SELL":
                        # Valida se já tem posição
                        open_positions = position_manager.get_open_positions()
                        for p in open_positions:
                            if p['symbol'] == symbol:
                                if p['type'] == 'BUY':
                                    logger.info(f"🔄 Invertendo mão em {symbol} (BUY -> SELL)")
                                    execution.close_position(p['ticket'], symbol)
                                else:
                                    logger.info(f"⏭️ Posição de VENDA já existente em {symbol}. Mantendo.")
                                    continue

                        # Cálculo de Lote (Mesma lógica)
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        size_multiplier = decision.get("size", 0.0)
                        
                        target_exposure = equity * base_allocation_pct * size_multiplier
                        
                        if current_price > 0:
                            raw_qty = target_exposure / current_price
                        else:
                            raw_qty = 0
                            
                        final_volume = utils.normalize_volume(symbol, raw_qty)
                        
                        # Se volume ficou 0 mas a decisão é forte e tem capital, tenta lote mínimo
                        if final_volume == 0 and size_multiplier > 0 and equity > 1000:
                            min_lot = 100
                            cost = min_lot * current_price
                            if cost <= equity * 0.95:
                                final_volume = float(min_lot)
                                logger.info(f"⚠️ Volume VENDA ajustado para lote mínimo ({final_volume}) em {symbol}")
                        
                        if final_volume <= 0:
                             logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                             continue
                        
                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(symbol, mt5.TIMEFRAME_M15, df=candles)
                        sl, tp = utils.calculate_dynamic_sl_tp(symbol, "SELL", current_price, ind)
                        
                        if not sl or sl <= 0:
                             logger.warning(f"⚠️ SL inválido para {symbol}. Bloqueando ordem.")
                             continue
                            
                        # Double Check de Posições ANTES de enviar ordem
                        current_open = len(position_manager.get_open_positions())
                        if current_open >= config.MAX_CONCURRENT_POSITIONS:
                             logger.warning(f"🛑 [FAILSAFE] Limite de posições atingido ({current_open}/{config.MAX_CONCURRENT_POSITIONS}) antes de SELL em {symbol}. Abortando.")
                             continue

                        # Cria ordem
                        order = OrderParams(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            volume=final_volume,
                            price=0.0, # Market order
                            sl=sl, # SL calculado
                            tp=tp  # TP calculado
                        )
                        execution.send_order(order)
                    
                    elif decision["action"] == "HOLD":
                        logger.info(f"⏸️ {symbol}: HOLD - Motivo: {decision.get('reason', 'N/A')}")
                        
                except Exception as e:
                    logger.error(f"❌ Erro no loop para {symbol}: {e}")
            
            # Gerenciamento de posições abertas
            position_manager.update_stops()
            
            # Sleep para evitar sobrecarga (Timeframe M15/H1 sugerido)
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("🛑 Parada manual solicitada.")
    finally:
        execution.shutdown()
        logger.info("👋 Bot finalizado.")

if __name__ == "__main__":
    main()
