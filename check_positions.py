import os
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd

load_dotenv()
login = os.getenv('MT5_LOGIN')
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

if not login or not password or not server:
    print("Erro: Credenciais MT5 não encontradas no .env")
    exit(1)

if not mt5.initialize(login=int(login), password=password, server=server):
    print(f"Erro ao inicializar MT5: {mt5.last_error()}")
    exit(1)

positions = mt5.positions_get()
if not positions:
    print("Nenhuma posição aberta no momento.")
else:
    print(f"Total de {len(positions)} posições abertas:")
    for p in positions:
        print(f"Ticket: {p.ticket}, Símbolo: {p.symbol}, Volume: {p.volume}, Preço Aberto: {p.price_open}, SL: {p.sl}, TP: {p.tp}, Preço Atual: {p.price_current}, Lucro: {p.profit}")

mt5.shutdown()
