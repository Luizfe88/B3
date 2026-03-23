import psutil
import subprocess
import time
import os
import datetime
import platform
import MetaTrader5 as mt5

# ==================== CONFIGURAÇÕES ====================
BOT_SCRIPT = "bot.py"  # Nome do seu script principal
SUPERVISOR_SCRIPT = "run_supervisor.py"  # Agendador de IA
LOG_FILE = "xp3_bot.log"  # Arquivo de log do bot
MAX_INACTIVITY_SECONDS = 180  # 3 minutos sem log = suspeito
CHECK_INTERVAL = 30  # Verifica a cada 30s
MT5_TIMEOUT_SECONDS = 10  # Timeout para testar MT5
# ======================================================


def is_script_running(script_name):
    """Verifica se um script python específico está ativo"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["cmdline"]:
                cmd = " ".join(proc.info["cmdline"])
                if (
                    "python" in proc.info["name"].lower()
                    and script_name in cmd
                    and "botwatchdog.py" not in cmd
                ):
                    return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def is_bot_running():
    return is_script_running(BOT_SCRIPT)


def is_supervisor_running():
    return is_script_running(SUPERVISOR_SCRIPT)


def get_log_last_modified():
    if not os.path.exists(LOG_FILE):
        return None
    return os.path.getmtime(LOG_FILE)


def is_mt5_connected():
    try:
        if mt5.initialize(timeout=MT5_TIMEOUT_SECONDS * 1000):
            connected = (
                mt5.terminal_info() is not None and mt5.account_info() is not None
            )
            mt5.shutdown()
            return connected
    except Exception:
        return False
    return False


def kill_bot(pid):
    try:
        proc = psutil.Process(pid)
        proc.kill()
        print(f"[{datetime.datetime.now()}] ⚡ Processo morto (PID {pid})")
    except Exception:
        pass


def start_script_in_new_window(script_name):
    """Inicia um script em uma NOVA JANELA DE TERMINAL visível"""
    system = platform.system()
    print(f"[{datetime.datetime.now()}] 🚀 Iniciando {script_name} em NOVA JANELA ({system})...")

    if system == "Windows":
        subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", "python", script_name])
    elif system == "Linux":
        terminals = [
            ["gnome-terminal", "--", "python3", script_name],
            ["konsole", "-e", "python3", script_name],
            ["xfce4-terminal", "-e", f"python3 {script_name}"],
            ["xterm", "-e", f"python3 {script_name}"],
        ]
        for term_cmd in terminals:
            try:
                subprocess.Popen(term_cmd)
                return
            except FileNotFoundError:
                continue
    elif system == "Darwin":  # macOS
        subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "python3 {script_name}"'])
    else:
        print(f"⚠️ Sistema operacional {system} não suportado para nova janela.")


def main():
    print(f"[{datetime.datetime.now()}] 🐶 Watchdog iniciado")
    print(f"   → Monitorando Bot: {BOT_SCRIPT}")
    print(f"   → Monitorando Supervisor: {SUPERVISOR_SCRIPT}")

    last_log_time = get_log_last_modified()

    while True:
        current_time = datetime.datetime.now()
        
        # 1. VERIFICA O BOT PRINCIPAL
        bot_pid = is_bot_running()
        if bot_pid is None:
            print(f"[{current_time}] ❌ Bot parado → Abrindo...")
            start_script_in_new_window(BOT_SCRIPT)
            time.sleep(5)
        
        # 2. VERIFICA O SUPERVISOR (NOVIDADE)
        sup_pid = is_supervisor_running()
        if sup_pid is None:
            print(f"[{current_time}] ❌ Supervisor parado → Abrindo...")
            start_script_in_new_window(SUPERVISOR_SCRIPT)
            time.sleep(5)

        # 3. VERIFICA SAÚDE DO BOT (FREEZE E CONEXÃO)
        bot_pid = is_bot_running() # Re-checa o PID após possível reinício
        if bot_pid:
            # Check log freeze
            current_log_time = get_log_last_modified()
            if (current_log_time and last_log_time and 
                (current_log_time - last_log_time) > MAX_INACTIVITY_SECONDS):
                print(f"[{current_time}] ⏰ Bot Freeze detectado → Reiniciando")
                kill_bot(bot_pid)
                start_script_in_new_window(BOT_SCRIPT)
                time.sleep(10)
                current_log_time = get_log_last_modified()
            
            # Check MT5 connection
            if not is_mt5_connected():
                print(f"[{current_time}] 📡 MT5 desconectado → Reiniciando Bot")
                kill_bot(bot_pid)
                start_script_in_new_window(BOT_SCRIPT)
                time.sleep(10)
                current_log_time = get_log_last_modified()

            last_log_time = current_log_time

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
