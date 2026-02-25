import MetaTrader5 as mt5
import time
import datetime

def debug_account():
    print("=== Diagnóstico de Conta MT5 ===")
    
    # Tenta inicializar
    if not mt5.initialize():
        print(f"❌ Falha no mt5.initialize(): {mt5.last_error()}")
        return

    # Info do Terminal
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"Terminal Conectado: {terminal_info.connected}")
        print(f"Caminho: {terminal_info.path}")
        print(f"Trade Allowed: {terminal_info.trade_allowed}")
    else:
        print("❌ Falha ao obter terminal_info")

    # Info da Conta
    account_info = mt5.account_info()
    if account_info:
        print("\n=== Informações da Conta ===")
        print(f"Login: {account_info.login}")
        print(f"Servidor: {account_info.server}")
        print(f"Moeda: {account_info.currency}")
        print(f"Alavancagem: {account_info.leverage}")
        print(f"Tipo: {'Real' if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL else 'Demo'}")
        print("-" * 30)
        print(f"Saldo (Balance): {account_info.balance}")
        print(f"Equity (Patrimônio): {account_info.equity}")
        print(f"Margem Livre: {account_info.margin_free}")
        print("-" * 30)
    else:
        print("\n❌ mt5.account_info() retornou None!")
        print(f"Erro: {mt5.last_error()}")

    mt5.shutdown()

if __name__ == "__main__":
    debug_account()
