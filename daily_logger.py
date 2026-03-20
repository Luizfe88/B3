import logging
from datetime import datetime, timedelta
from threading import Timer
from pathlib import Path
import MetaTrader5 as mt5
import config
import utils
from database import get_trades_by_date

logger = logging.getLogger("daily_logger")


def _format_currency(v: float) -> str:
    return f"R$ {v:,.2f}"


def generate_cvm_report() -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    trades = get_trades_by_date(today_str)
    acc = mt5.account_info()
    equity = float(getattr(acc, "equity", 0.0) or 0.0)
    exposure = float(utils.calculate_total_exposure())
    total_pnl = float(trades["pnl_money"].sum()) if hasattr(trades, "sum") else 0.0
    wins = (
        int(len(trades[trades["pnl_money"] > 0])) if hasattr(trades, "__len__") else 0
    )
    losses = (
        int(len(trades[trades["pnl_money"] < 0])) if hasattr(trades, "__len__") else 0
    )
    total_trades = int(len(trades)) if hasattr(trades, "__len__") else 0
    lines = []
    lines.append(f"RELATÓRIO CVM — {datetime.now().strftime('%d/%m/%Y')}")
    lines.append(f"Equity: {_format_currency(equity)}")
    lines.append(f"Exposição: {_format_currency(exposure)}")
    lines.append(f"PnL: {_format_currency(total_pnl)}")
    lines.append(f"Trades: {total_trades} (✅{wins} ❌{losses})")
    if total_trades > 0:
        lines.append("Detalhes:")
        for _, row in trades.iterrows():
            ts = str(row.get("timestamp", ""))[:19]
            sym = str(row.get("symbol", ""))
            side = str(row.get("side", ""))
            vol = float(row.get("volume", 0.0) or 0.0)
            ep = float(row.get("entry_price", 0.0) or 0.0)
            xp = float(row.get("exit_price", 0.0) or 0.0)
            pnl = float(row.get("pnl_money", 0.0) or 0.0)
            reason = str(row.get("reason", "") or "")
            lines.append(
                f"{ts} | {sym} | {side} | Vol: {vol:.0f} | Entrada: {ep:.2f} | Saída: {xp:.2f} | PnL: {_format_currency(pnl)} | {reason}"
            )
    return "\n".join(lines)


def save_cvm_report_txt(text: str) -> Path:
    out_dir = Path("logs/cvm")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"cvm_report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return filename


def send_cvm_report_telegram(text: str):
    if getattr(config, "ENABLE_TELEGRAM_NOTIF", False):
        try:
            utils.send_telegram_message(f"📑 <b>RELATÓRIO CVM DIÁRIO</b>\n\n{text}")
            logger.info("Relatório CVM enviado via Telegram")
        except Exception as e:
            logger.warning(f"Erro ao enviar relatório CVM: {e}")


def _run_once():
    try:
        report = generate_cvm_report()
        save_cvm_report_txt(report)
        send_cvm_report_telegram(report)
        logger.info("Relatório CVM diário gerado")
    except Exception as e:
        logger.error(f"Erro no relatório CVM: {e}", exc_info=True)

def _run_daily_telegram_bundle():
    """Executa o bundle de relatórios do Telegram às 17:00"""
    try:
        from telegram_handler import send_daily_reports_bundle
        send_daily_reports_bundle()
        logger.info("Relatórios diários (17:00) enviados via Telegram")
    except Exception as e:
        logger.error(f"Erro ao enviar relatórios diários das 17:00: {e}", exc_info=True)


def _seconds_until(hour: int, minute: int = 0) -> float:
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def start_cvm_daily_scheduler():
    """Inicia os agendamentos diários (17:00 e 18:00)"""
    
    # 1. Relatório CVM (Já existente às 18:00)
    delay_cvm = _seconds_until(18, 0)
    def _schedule_cvm_next():
        _run_once()
        start_cvm_daily_scheduler()
    
    t_cvm = Timer(delay_cvm, _schedule_cvm_next)
    t_cvm.daemon = True
    t_cvm.start()

    # 2. Relatórios XP3 (Aprendizado + Didático às 17:00)
    delay_xp3 = _seconds_until(17, 0)
    def _schedule_xp3_next():
        _run_daily_telegram_bundle()
        # Note: O agendamento recursivo é feito pelo start_cvm_daily_scheduler() que relança ambos
        pass
    
    t_xp3 = Timer(delay_xp3, _schedule_xp3_next)
    t_xp3.daemon = True
    t_xp3.start()
    
    logger.info("Agendadores diários iniciados: 17:00 (Relatórios XP3) e 18:00 (CVM)")
