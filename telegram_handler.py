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
