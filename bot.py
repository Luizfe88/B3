import os
import sys
import logging
import time
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from logging.handlers import TimedRotatingFileHandler
from trailing_stop import calculate_dynamic_stop, TrailingStopConfig
from opportunity_ranker import opportunity_ranker

# Configuração de logs com rotação a cada 3 horas
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log_file = "logs/trading_agents.log"

# Garante diretório de logs
os.makedirs("logs", exist_ok=True)

# Handler de Arquivo com Rotação (3 horas)
# O padrão TimedRotatingFileHandler adiciona .YYYY-MM-DD_HH-MM ao final
# Para ter trading_agents_2023-10-27_12-00.log precisamos customizar ou aceitar o padrão .log.data
# Vamos usar o padrão, mas configurando o sufixo para ser amigável.
file_handler = TimedRotatingFileHandler(
    log_file, when="H", interval=3, backupCount=10, encoding="utf-8"
)
file_handler.suffix = (
    "%Y-%m-%d_%H-%M.log"  # Define o formato do sufixo (ex: 2023-10-27_12-00.log)
)
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
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("MainBot")

# Imports do sistema
from core.execution import ExecutionEngine, OrderParams, OrderSide
from core.position_manager import PositionManager
from agents.fund_manager import FundManager
import config
import utils
import MetaTrader5 as mt5
from calibration_manager import calibration_manager


def main():
    logger.info("🚀 Iniciando TradingAgents-B3 Framework...")

    # 1. Inicialização de Infraestrutura
    # Inicia a thread do Telegram para ouvir comandos em background
    from telegram_handler import start_telegram_polling
    telegram_thread = threading.Thread(target=start_telegram_polling, daemon=True, name="TelegramPollingThread")
    telegram_thread.start()
    
    # Inicia o monitoramento da inteligência adaptativa
    from adaptive_intelligence import adaptive_intelligence
    if config.get_config().get("adaptive_intelligence", {}).get("enabled", False):
        adaptive_intelligence.start_monitoring()

    execution = ExecutionEngine()
    if not execution.connect():
        logger.critical(
            "❌ Falha crítica: Não foi possível conectar ao MT5. Abortando."
        )
        return

    position_manager = PositionManager(execution)
    fund_manager = FundManager()

    # Exibe resumo da calibração no console
    print(calibration_manager.get_summary(), flush=True)

    # 2. Loop Principal
    logger.info("✅ Sistema online. Preparando Market Watch...")

    # Inicializa Market Watch para garantir visibilidade
    # Itera sobre todos os símbolos monitorados e força a seleção
    # Isso evita erros de "Símbolo não selecionado" durante o loop
    all_symbols = config.MONITORED_SYMBOLS
    valid_symbols = []

    for sym in all_symbols:
        if not utils.is_valid_b3_ticker(sym):
            logger.warning(f"🚨 [{sym}] Ticker inválido para o mercado à vista. Removido da lista de scans.")
            continue
            
        if execution.safe_symbol_select(sym, True):
            valid_symbols.append(sym)
        else:
            logger.warning(
                f"⚠️ Falha inicial ao selecionar {sym} - Removido da lista de execução."
            )

    logger.info(
        f"📋 Market Watch inicializado: {len(valid_symbols)}/{len(all_symbols)} ativos válidos e prontos."
    )

    logger.info("🚀 Iniciando loop de mercado.")

    try:
        last_report_date = None
        while True:
            # Verifica se deve fechar posições no final do dia
            now = datetime.now()
            current_time = now.time()

            # Define horário de fechamento (Sexta vs Outros dias)
            if now.weekday() == 4:  # 4 = Sexta-feira
                close_time_str = config.FRIDAY_CLOSE_ALL_BY  # "17:15"
            else:
                close_time_str = config.CLOSE_ALL_BY  # "17:45"

            close_time = datetime.strptime(close_time_str, "%H:%M").time()

            if current_time >= close_time:
                # Se ainda tiver posições abertas, fecha tudo
                open_pos = position_manager.get_open_positions()
                if open_pos:
                    logger.info(
                        "⏰ Horário de fechamento diário atingido. Zerando carteira..."
                    )
                    position_manager.close_all(reason="End of Day")
                else:
                    logger.info(
                        "💤 Mercado fechado ou horário limite atingido. Aguardando..."
                    )
                
                # Envia o relatório de aprendizado apenas uma vez no dia
                if last_report_date != now.date():
                    try:
                        from telegram_handler import send_daily_learning_report
                        send_daily_learning_report()
                        last_report_date = now.date()
                    except Exception as e:
                        logger.error(f"Erro ao enviar relatório diário: {e}")

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
                logger.debug(
                    f"🔍 Posições Abertas ({open_count}): {[f'{p['symbol']} ({p['magic']})' for p in open_positions_list]}"
                )

            if open_count >= config.MAX_CONCURRENT_POSITIONS:
                logger.info(
                    f"🛑 Limite de posições atingido ({open_count}/{config.MAX_CONCURRENT_POSITIONS})."
                )

                # Se exceder (ex: por posições manuais ou erro anterior), tenta reduzir?
                # Por enquanto, apenas atualiza stops das existentes e aguarda
                position_manager.update_stops()

                logger.info("💤 Aguardando liberação de slots...")
                time.sleep(60)
                continue

            # 1. Atualiza trailing stops das posições existentes ANTES de abrir novas (Crucial para stop móvel)
            position_manager.update_stops()

            # 2. Contexto Global de Mercado (Fora do loop para evitar redundância)
            ibov_regime = utils.get_market_regime()
            logger.info(f"🌍 Contexto de Mercado: IBOV {ibov_regime.upper()}")
            
            opportunities = [] # Array para acumular scans
            total_symbols = len(symbols)

            for i, symbol in enumerate(valid_symbols):
                # ✅ IGNORE: Pula índices futuros e mini-contratos conforme configuração 'Ações Apenas'
                if utils.is_future(symbol):
                    logger.warning(f"⚠️ {symbol} identificado como futuro. Ignorando conforme configuração 'Ações Apenas'.")
                    continue

                # Log de progresso (a cada 25 ativos ou no início/fim)
                if (i + 1) % 25 == 0 or i == 0 or (i + 1) == total_symbols:
                    logger.info(f"🔍 Scan em progresso: {i+1}/{total_symbols} ativos processados...")

                try:
                    # 1. Coleta dados de mercado (Market Data)
                    if not execution.safe_symbol_select(symbol, True):
                        logger.warning(f"⚠️ {symbol} indisponível. Pulando este ciclo.")
                        continue

                    # 1.1 Candles (utiliza timeframe calibrado ou M15) - Cache do terminal geralmente rápido
                    params = calibration_manager.get_calibrated_params(symbol)
                    sym_tf_str = params.get("timeframe", "M15")
                    sym_tf = utils.str_to_tf(sym_tf_str)
                    
                    candles = utils.safe_copy_rates(symbol, sym_tf, 100)
                    if candles is None or candles.empty:
                        logger.warning(
                            f"⚠️ Dados insuficientes (candles {sym_tf_str}) para {symbol}. Pulando."
                        )
                        continue

                    # 1.2 Preço atual (Fast call)
                    tick_info = mt5.symbol_info_tick(symbol)
                    current_price = (
                        tick_info.last if tick_info and tick_info.last > 0 else (tick_info.bid if tick_info else candles["close"].iloc[-1])
                    )

                    # 1.3 Ticks (Últimos 800 negócios - Janela Fechada com Time do Servidor)
                    last_tick = mt5.symbol_info_tick(symbol)
                    if last_tick is not None:
                        # 1. Ancoramos o tempo atual no relógio do servidor da corretora
                        server_time = datetime.fromtimestamp(last_tick.time)
                        
                        # 2. Criamos uma janela abrangente (últimas 2 horas reais da B3)
                        start_time = server_time - timedelta(hours=2)
                        end_time = server_time + timedelta(minutes=1)
                        
                        try:
                            # 3. Puxamos negócios agressivos da janela fechada
                            ticks_data = mt5.copy_ticks_range(symbol, start_time, end_time, mt5.COPY_TICKS_TRADE)
                            
                            # 4. Fatiamos apenas os últimos 800 (Foco no fluxo recente)
                            if ticks_data is not None and len(ticks_data) > 0:
                                ticks = ticks_data[-800:]
                            else:
                                ticks = []
                                
                        except Exception as e:
                            logger.error(f"⚠️ Erro ao buscar ticks_range para {symbol}: {e}")
                            ticks = []
                    else:
                        ticks = []

                    # Dados Globais de Risco
                    account_info = mt5.account_info()
                    equity = account_info.equity if account_info else 1000.0

                    # Calcula exposições por setor
                    sector_exposure = {}
                    open_positions = position_manager.get_open_positions()
                    for p in open_positions:
                        p_sym = p["symbol"]
                        p_sector = config.SECTOR_MAP.get(p_sym, "OUTROS")
                        p_exp = p["volume"] * p["current_price"]
                        sector_exposure[f"sector_exposure_{p_sector}"] = (
                            sector_exposure.get(f"sector_exposure_{p_sector}", 0.0)
                            + p_exp
                        )

                    market_data = {
                        "price": current_price,
                        "ticks": ticks if ticks is not None else [],
                        "candles": candles,
                        "equity": equity,
                        "total_exposure": position_manager.get_total_exposure(),
                        "recent_entries_count": position_manager.count_recent_entries(
                            minutes=60
                        ),
                        "ibov_trend": ibov_regime,
                        **sector_exposure,  # Adiciona exposições setoriais ao contexto
                    }

                    # 2. Decisão do Fund Manager (Agentes)
                    decision = fund_manager.decide(symbol, market_data)

                    # Se aprovado ou apenas bloqueado pelo almoço (para ranking), armazena
                    if decision["action"] in ["BUY", "SELL"] or decision.get("lunch_filter"):
                        opportunities.append((symbol, decision))
                        
                        if decision.get("lunch_filter"):
                            logger.info(f"🥪 [SCAN] {symbol} Bloqueado pelo almoço, mas incluído no Ranking Parcial.")
                        else:
                            logger.info(f"🔎 [SCAN] {symbol} marcou possível {decision['action']}. Aguardando Ranking...")
                
                except Exception as e:
                    logger.error(f"❌ Erro ao analisar {symbol}: {e}")
                    import traceback
                    traceback.print_exc()


            # =======================================================
            # 💡 FASE 2: RANK E EXECUÇÃO (Global Context)
            # =======================================================
            if opportunities:
                logger.info(f"🏆 Avaliando {len(opportunities)} oportunidades detectadas neste ciclo.")
                
                # Rankeia baseado no Conviction Score
                ranked_opportunities = opportunity_ranker.rank_opportunities(opportunities)
                
                # Executa apenas as Top N melhores, baseado no limite dinâmico de conexões/novas trades
                open_positions_post_scan = position_manager.get_open_positions()
                slots_livres = config.MAX_CONCURRENT_POSITIONS - len(open_positions_post_scan)
                max_new = getattr(config, "MAX_NEW_POSITIONS_PER_HOUR", 4)
                vagas_reais = min(slots_livres, max_new)

                logger.info(f"📈 Limite de ação neste ciclo: top {vagas_reais} operações.")

                # Dicionário para contar posições por setor no ciclo atual
                # para evitar estourar o limite de 2 durante a execução sequencial
                current_sector_counts = {}
                for p in open_positions_post_scan:
                    sec = config.SECTOR_MAP.get(p["symbol"], "Outros")
                    current_sector_counts[sec] = current_sector_counts.get(sec, 0) + 1

                for rank_idx, (symbol, decision) in enumerate(ranked_opportunities):
                    if rank_idx >= vagas_reais:
                        logger.info(f"✂️ {symbol} ignorado (Ranking #{rank_idx+1} fora do limite top {vagas_reais}).")
                        continue

                    # --- VALIDAÇÃO DE SETOR (Limite de 2) ---
                    symbol_sector = config.SECTOR_MAP.get(symbol, "Outros")
                    current_sec_count = current_sector_counts.get(symbol_sector, 0)
                    
                    if current_sec_count >= config.MAX_PER_SECTOR:
                        logger.warning(
                            f"⚠️ {symbol} ignorado (Limite de {config.MAX_PER_SECTOR} posições por setor atingido para {symbol_sector})."
                        )
                        continue

                    logger.info(f"🚀 Iniciando execução rankeada #{rank_idx+1}: {symbol} -> {decision['action']}")
                    
                    # 3. Execução Final
                    if decision["action"] == "BUY":
                        # ── Busca dados frescos do símbolo (evita usar price/candles do último scan) ──
                        _tick = mt5.symbol_info_tick(symbol)
                        if _tick is None:
                            logger.warning(f"⚠️ Sem tick para {symbol} na execução. Pulando.")
                            continue
                        current_price = _tick.ask  # Para BUY, usamos o ask
                        candles = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
                        if candles is None or candles.empty:
                            logger.warning(f"⚠️ Sem candles para {symbol} na execução. Pulando.")
                            continue
                        # ────────────────────────────────────────────────────────────────────────────

                        # --- SMART SIZING & PYRAMIDING (Centralizado no PositionManager) ---
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        target_exposure = equity * base_allocation_pct * decision.get("size", 1.0)
                        
                        raw_qty = target_exposure / current_price if current_price > 0 else 0
                        qty_normalized = utils.normalize_volume(symbol, raw_qty)

                        final_volume, can_execute, msg = position_manager.validate_and_size_order(
                            symbol, "BUY", qty_normalized, decision.get("conviction", 0.0)
                        )

                        if not can_execute:
                            if "Abortado" in msg:
                                logger.warning(msg)
                            else:
                                logger.info(f"✂️ {symbol} ignorado: {msg}")
                            continue

                        # Se volume ficou 0 mas a decisão permite, tenta lote mínimo (Failsafe)
                        if final_volume == 0 and equity > 1000:
                            min_lot = 100
                            cost = min_lot * current_price
                            if cost <= equity * 0.95:
                                final_volume = float(min_lot)
                                logger.info(f"⚠️ Volume BUY ajustado para lote mínimo ({final_volume}) em {symbol}")

                        if final_volume <= 0:
                            logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                            continue

                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(
                            symbol, mt5.TIMEFRAME_M15, df=candles
                        )
                        
                        if ind.get("error"):
                            logger.warning(f"⚠️ {symbol} abortado por erro nos indicadores: {ind.get('error')}")
                            continue

                        sl, tp = utils.calculate_dynamic_sl_tp(
                            symbol, "BUY", current_price, ind
                        )

                        if not sl or sl <= 0:
                            logger.warning(
                                f"⚠️ SL inválido para {symbol}. Bloqueando ordem."
                            )
                            continue

                        # Double Check de Posições ANTES de enviar ordem
                        current_open = len(position_manager.get_open_positions())
                        if current_open >= config.MAX_CONCURRENT_POSITIONS:
                            logger.warning(
                                f"🛑 [FAILSAFE] Limite de posições atingido ({current_open}/{config.MAX_CONCURRENT_POSITIONS}) antes de BUY em {symbol}. Abortando."
                            )
                            continue

                        # Execução Inteligente (Pillar 3: Optimal Execution)
                        success = execution.execute_smart_order(
                            symbol,
                            OrderSide.BUY,
                            final_volume,
                            current_price,
                            sl,
                            tp,
                            comment=f"XP3_QUANT_{size_multiplier:.1f}",
                        )

                        # ⏱️ FIX RACE CONDITION: Registra ordem pendente e aguarda confirmação no MT5
                        position_manager.register_pending_order(
                            symbol, final_volume, current_price
                        )
                        logger.info(
                            f"⏳ Aguardando confirmação de {symbol} no MT5 (1.5s)..."
                        )
                        time.sleep(1.5)  # Dá tempo para o MT5 registrar a posição
                        
                        # Incrementa contador do setor após execução bem-sucedida
                        current_sector_counts[symbol_sector] = current_sector_counts.get(symbol_sector, 0) + 1

                    elif decision["action"] == "SELL":
                        # ── Busca dados frescos do símbolo (evita usar price/candles do último scan) ──
                        _tick = mt5.symbol_info_tick(symbol)
                        if _tick is None:
                            logger.warning(f"⚠️ Sem tick para {symbol} na execução. Pulando.")
                            continue
                        current_price = _tick.bid  # Para SELL, usamos o bid
                        candles = utils.safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 100)
                        if candles is None or candles.empty:
                            logger.warning(f"⚠️ Sem candles para {symbol} na execução. Pulando.")
                            continue
                        # ────────────────────────────────────────────────────────────────────────────

                        # --- SMART SIZING & PYRAMIDING (Centralizado no PositionManager) ---
                        base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT
                        target_exposure = equity * base_allocation_pct * decision.get("size", 1.0)
                        
                        raw_qty = target_exposure / current_price if current_price > 0 else 0
                        qty_normalized = utils.normalize_volume(symbol, raw_qty)

                        final_volume, can_execute, msg = position_manager.validate_and_size_order(
                            symbol, "SELL", qty_normalized, decision.get("conviction", 0.0)
                        )

                        if not can_execute:
                            if "Abortado" in msg:
                                logger.warning(msg)
                            else:
                                logger.info(f"✂️ {symbol} ignorado: {msg}")
                            continue

                        # Se volume ficou 0 mas a decisão permite, tenta lote mínimo (Failsafe)
                        if final_volume == 0 and equity > 1000:
                            min_lot = 100
                            cost = min_lot * current_price
                            if cost <= equity * 0.95:
                                final_volume = float(min_lot)
                                logger.info(f"⚠️ Volume SELL ajustado para lote mínimo ({final_volume}) em {symbol}")

                        if final_volume <= 0:
                            logger.warning(f"⚠️ Volume calculado para {symbol} inválido ({final_volume}). Ignorando.")
                            continue

                        # Cálculo de SL/TP Dinâmico
                        ind = utils.quick_indicators_custom(
                            symbol, mt5.TIMEFRAME_M15, df=candles
                        )

                        if ind.get("error"):
                            logger.warning(f"⚠️ {symbol} abortado por erro nos indicadores: {ind.get('error')}")
                            continue

                        sl, tp = utils.calculate_dynamic_sl_tp(
                            symbol, "SELL", current_price, ind
                        )

                        if not sl or sl <= 0:
                            logger.warning(
                                f"⚠️ SL inválido para {symbol}. Bloqueando ordem."
                            )
                            continue

                        # Double Check de Posições ANTES de enviar ordem
                        current_open = len(position_manager.get_open_positions())
                        if current_open >= config.MAX_CONCURRENT_POSITIONS:
                            logger.warning(
                                f"🛑 [FAILSAFE] Limite de posições atingido ({current_open}/{config.MAX_CONCURRENT_POSITIONS}) antes de SELL em {symbol}. Abortando."
                            )
                            continue

                        # Execução Inteligente (Pillar 3: Optimal Execution)
                        success = execution.execute_smart_order(
                            symbol,
                            OrderSide.SELL,
                            final_volume,
                            current_price,
                            sl,
                            tp,
                            comment=f"XP3_QUANT_{size_multiplier:.1f}",
                        )

                        # ⏱️ FIX RACE CONDITION: Registra ordem pendente e aguarda confirmação no MT5
                        position_manager.register_pending_order(
                            symbol, final_volume, current_price
                        )
                        logger.info(
                            f"⏳ Aguardando confirmação de {symbol} no MT5 (1.5s)..."
                        )
                        time.sleep(1.5)  # Dá tempo para o MT5 registrar a posição

                        # Incrementa contador do setor após execução bem-sucedida
                        current_sector_counts[symbol_sector] = current_sector_counts.get(symbol_sector, 0) + 1

            else:
                logger.info("📭 Nenhum sinal relevante detectado neste ciclo de scan.")

            # Gerenciamento de posições abertas
            position_manager.update_stops()

            # Verifica e loga trades fechados (histórico recente)
            utils.check_and_log_closed_trades()

            # Sleep para evitar sobrecarga (Timeframe M15/H1 sugerido)
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("🛑 Parada manual solicitada.")
    finally:
        execution.shutdown()
        logger.info("👋 Bot finalizado.")


if __name__ == "__main__":
    main()
