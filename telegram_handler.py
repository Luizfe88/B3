# telegram_handler.py

import telebot
import logging
from datetime import datetime
import MetaTrader5 as mt5
import config
from utils import send_telegram_message  # opcional, se quiser usar sua função
from news_filter import (
    get_next_high_impact_event,
    check_news_blackout,
    get_upcoming_events,
)

logger = logging.getLogger("telegram")

# Só cria o bot se Telegram estiver habilitado
if getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    CHAT_ID = config.TELEGRAM_CHAT_ID  # Para envios automáticos
else:
    bot = None
    CHAT_ID = None

# ==================== HANDLERS ====================

if bot:  # Só registra handlers se o bot foi criado

    @bot.message_handler(commands=["start", "help"])
    def handle_help(message):
        help_text = """
🤖 <b>XP3 PRO - Comandos Disponíveis</b>

📊 <b>Informações</b>
/status         → Status do bot e conexão
/lucro          → Lucro do dia e posições
/health         → Latência, memória e status do sistema
/proximoevento  → Próximo evento econômico importante
/blackout ou /news → Status de blackout por notícia
/aprendizado    → Relatório Diário de Aprendizado do Bot

ℹ️ Bot opera automaticamente na B3.
        """
        bot.reply_to(message, help_text, parse_mode="HTML")

    @bot.message_handler(commands=["status"])
    def handle_status(message):
        if not mt5.terminal_info() or not mt5.terminal_info().connected:
            status = "🔴 <b>MT5 DESCONECTADO</b>"
        else:
            acc = mt5.account_info()
            balance = acc.balance if acc else 0
            equity = acc.equity if acc else 0
            positions_count = len(mt5.positions_get() or [])

            status = (
                f"🤖 <b>XP3 PRO - STATUS</b>\n\n"
                f"✅ <b>Conectado ao MT5</b>\n"
                f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
                f"💰 Balance: R$ {balance:,.2f}\n"
                f"📈 Equity:  R$ {equity:,.2f}\n"
                f"📊 Posições abertas: {positions_count}"
            )

        bot.reply_to(message, status, parse_mode="HTML")

    @bot.message_handler(commands=["lucro"])
    def handle_lucro(message):
        acc = mt5.account_info()
        if not acc:
            bot.reply_to(message, "❌ Não conectado ao MT5")
            return

        profit_today = acc.profit
        positions = mt5.positions_get() or []

        msg = (
            f"📊 <b>RESUMO DO DIA</b>\n\n"
            f"💰 Lucro realizado + flutuante: <b>{profit_today:+.2f} R$</b>\n"
            f"📈 Posições abertas: {len(positions)}\n"
        )

        if positions:
            msg += "\n<b>Posições atuais:</b>\n"
            for p in positions[:8]:
                emoji = "🟢" if p.profit >= 0 else "🔴"
                msg += (
                    f"{emoji} {p.symbol} | Vol: {p.volume} | P&L: {p.profit:+.2f} R$\n"
                )

        bot.reply_to(message, msg, parse_mode="HTML")

    @bot.message_handler(commands=["proximoevento"])
    def handle_proximoevento(message):
        event_msg = get_next_high_impact_event()
        emoji = (
            "🔴" if "em" in event_msg.lower() and "min" in event_msg.lower() else "🟢"
        )
        full_msg = f"{emoji} <b>PRÓXIMO EVENTO</b>\n\n{event_msg}"
        bot.reply_to(message, full_msg, parse_mode="HTML")

    @bot.message_handler(commands=["blackout", "news"])
    def handle_blackout(message):
        blocked, reason = check_news_blackout()
        upcoming = get_upcoming_events(hours_ahead=8)

        if blocked:
            status = f"🚫 <b>BOT EM BLACKOUT</b>\n\n{reason}\n\nEntradas bloqueadas até passar o evento."
        else:
            if upcoming:
                ev = upcoming[0]
                mins = int((ev["time"] - datetime.now()).total_seconds() / 60)
                emoji = "🔴" if ev["impact"] == "High" else "🟡"
                status = (
                    f"✅ <b>TRADING LIBERADO</b>\n\n"
                    f"{emoji} Próximo: <b>{ev['title']}</b>\n"
                    f"⏰ Em {mins} minutos ({ev['impact']} impacto)"
                )
            else:
                status = (
                    "✅ <b>TRADING LIBERADO</b>\n\nSem eventos nas próximas 8 horas."
                )

        bot.reply_to(message, status, parse_mode="HTML")

    @bot.message_handler(commands=["aprendizado"])
    def handle_aprendizado(message):
        msg = build_learning_report()
        bot.reply_to(message, msg, parse_mode="HTML")
        # Também gera o .txt sob demanda
        save_file_path = save_learning_report_to_file(msg)
        if save_file_path:
            # Envia a mensagem com o path do log
            bot.reply_to(message, f"📁 Relatório detalhado também foi salvo em: `{save_file_path}`", parse_mode="Markdown")

    @bot.message_handler(commands=["health"])
    def handle_health(message):
        """
        Retorna status de saúde do sistema:
        - Latência com a corretora
        - Status da conexão MT5
        """
        try:
            start = datetime.now()
            terminal_info = mt5.terminal_info()
            latency_ms = (datetime.now() - start).total_seconds() * 1000

            if not terminal_info or not terminal_info.connected:
                health_msg = "🔴 <b>HEALTH CHECK - CRÍTICO</b>\n\n❌ MT5 DESCONECTADO"
            else:
                acc = mt5.account_info()
                if not acc:
                    health_msg = (
                        "🔴 <b>HEALTH CHECK - CRÍTICO</b>\n\n❌ Conta não encontrada"
                    )
                else:
                    # Simula uso de memória (substitua por psutil se quiser real)
                    import psutil

                    mem = psutil.virtual_memory()
                    cpu = psutil.cpu_percent(interval=1)

                    health_msg = (
                        f"🤖 <b>XP3 PRO - HEALTH CHECK</b>\n\n"
                        f"✅ <b>MT5 Conectado</b>\n"
                        f"⏱️ Latência: {latency_ms:.1f} ms\n"
                        f"💾 Memória: {mem.percent:.1f}% usada\n"
                        f"⚡ CPU: {cpu:.1f}%\n"
                        f"📊 Conexão: {'Online' if terminal_info.connected else 'Offline'}"
                    )

            bot.reply_to(message, health_msg, parse_mode="HTML")

        except ImportError:
            # Fallback sem psutil
            health_msg = (
                f"🤖 <b>XP3 PRO - HEALTH CHECK</b>\n\n"
                f"✅ <b>MT5 Conectado</b>\n"
                f"⏱️ Latência: {latency_ms:.1f} ms\n"
                f"📊 Conexão: {'Online' if terminal_info.connected else 'Offline'}"
            )
            bot.reply_to(message, health_msg, parse_mode="HTML")

        except Exception as e:
            bot.reply_to(message, f"❌ Erro no health check: {str(e)}")


def send_telegram_alert(message_text: str, parse_mode="HTML"):
    """
    Função auxiliar para enviar alertas automáticos (ex: entradas, saídas, erros críticos).
    """
    if not bot or not CHAT_ID:
        return False
    try:
        bot.send_message(CHAT_ID, message_text, parse_mode=parse_mode)
        return True
    except Exception as e:
        logger.error(f"Erro ao enviar mensagem Telegram: {e}")
        return False


def build_learning_report() -> str:
    """Constrói o texto do relatório diário do Adaptive Intelligence"""
    try:
        from adaptive_intelligence import adaptive_intelligence
        report = adaptive_intelligence.get_performance_report()

        if report.get("status") != "Ativo":
            return f"❌ <b>Relatório de Aprendizado</b>\n\nStatus: {report.get('status', 'Inativo')}"

        params = report.get("current_parameters", {})
        metrics = report.get("performance_metrics", {})
        state = report.get("market_state", {})

        last_adj_str = report.get("last_adjustment", "")
        last_adj_time = last_adj_str.split("T")[1][:8] if "T" in last_adj_str else last_adj_str

        msg = (
            f"🧠 <b>XP3 PRO - APRENDIZADO DIÁRIO</b>\n\n"
            f"🔧 Ajustes realizados hoje: <b>{report.get('total_adjustments', 0)}</b>\n"
            f"⏰ Último ajuste: {last_adj_time}\n\n"
        )

        # Adiciona detalhes da última mudança
        adj_details = report.get("last_adjustment_details")
        if adj_details:
            old = adj_details.get("old_params", {})
            new = adj_details.get("new_params", {})
            recs = adj_details.get("recommendations", {})

            msg += "🔄 <b>Última Mudança Dinâmica</b>\n"
            
            changes = []
            if old.get("ml_confidence_threshold") != new.get("ml_confidence_threshold"):
                changes.append(f"• ML Conf: {old.get('ml_confidence_threshold', 0):.2f} ➔ <b>{new.get('ml_confidence_threshold', 0):.2f}</b>")
            if old.get("kelly_fraction_multiplier") != new.get("kelly_fraction_multiplier"):
                changes.append(f"• Kelly Mult: {old.get('kelly_fraction_multiplier', 0):.2f}x ➔ <b>{new.get('kelly_fraction_multiplier', 0):.2f}x</b>")
            if old.get("spread_filter_multiplier") != new.get("spread_filter_multiplier"):
                changes.append(f"• Spread Lim: {old.get('spread_filter_multiplier', 0):.2f}x ➔ <b>{new.get('spread_filter_multiplier', 0):.2f}x</b>")
            
            if changes:
                msg += "\n".join(changes) + "\n"

            # Tradução amigável dos motivos
            motivos_map = {
                "increase_confidence_threshold": "Queda no Winrate",
                "decrease_confidence_threshold": "Alta no Winrate",
                "reduce_kelly_multiplier": "Alta Volatilidade/Risco",
                "increase_kelly_multiplier": "Mercado Favorável",
                "increase_sl_distance": "Ruído de Volatilidade",
                "reduce_position_sizes": "Sharpe Ratio Baixo",
                "increase_spread_filter": "Spread Elevado",
                "clamp_sl_max_percent": "Anomalia de Volume"
            }
            
            motivos = [motivos_map.get(k, k) for k in recs.keys()]
            if motivos:
                msg += f"💡 <b>Motivo:</b> {', '.join(motivos)}\n"
            msg += "\n"

        msg += (
            f"📊 <b>Métricas de Desempenho (Adaptive)</b>\n"
            f"• Winrate 24h: {metrics.get('avg_winrate_24h', 0):.1%}\n"
            f"• Tendência WR: {metrics.get('winrate_trend', '')}\n"
            f"• Sharpe 4h: {metrics.get('avg_sharpe_4h', 0):.2f}\n\n"
            f"⚙️ <b>Parâmetros em Vigor</b>\n"
            f"• Confiança ML: {params.get('ml_confidence_threshold', 0):.2f}\n"
            f"• Kelly Mult: {params.get('kelly_fraction_multiplier', 0):.2f}x\n"
            f"• Spread Limite: {params.get('spread_filter_multiplier', 0):.2f}x\n\n"
            f"🌍 <b>Leitura de Mercado</b>\n"
            f"• Volatilidade: {state.get('volatility_level', '')}\n"
            f"• Correlação: {state.get('correlation_level', '')}\n\n"
            f"🔮 <b>Lições para o Próximo Pregão</b>\n"
        )
        
        projections = report.get("tomorrow_projections", [])
        if projections:
            for p in projections:
                msg += f"• {p}\n"
        else:
            msg += "• Aguardando mais dados para projeções consistentes.\n"
            
        return msg
    except Exception as e:
        logger.error(f"Erro ao construir relatório de aprendizado: {e}")
        return f"❌ Erro ao gerar relatório: {str(e)}"

def save_learning_report_to_file(msg: str):
    import os
    from datetime import datetime
    try:
        os.makedirs("relatorios", exist_ok=True)
        filename = f"relatorios/aprendizado_{datetime.now().strftime('%Y-%m-%d')}.txt"
        
        # Remove tags HTML para salvar no txt
        import re
        clean_msg = re.sub(r'<[^>]+>', '', msg)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(clean_msg)
        logger.info(f"Relatório de aprendizado salvo em {filename}")
        return filename
    except Exception as e:
        logger.error(f"Erro ao salvar relatório de aprendizado no disco: {e}")
        return None

def send_daily_learning_report():
    """
    Chamado no final do dia para enviar o relatório automático.
    """
    logger.info("Enviando relatório diário de aprendizado para o Telegram...")
    msg = build_learning_report()
    save_learning_report_to_file(msg)
    send_telegram_alert(msg)


def start_telegram_polling():
    """
    Inicia o polling do bot Telegram (se habilitado).
    Chame isso em uma thread separada ou no início do main().
    """
    if not bot:
        logger.info("Telegram desabilitado. Polling não iniciado.")
        return

    logger.info("Iniciando polling Telegram...")
    try:
        bot.polling(none_stop=True, timeout=60)
    except Exception as e:
        logger.error(f"Erro no polling Telegram: {e}")


if __name__ == "__main__":
    start_telegram_polling()
