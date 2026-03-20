import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def check():
    if not mt5.initialize():
        print("Falha ao inicializar MT5")
        return

    indices = ['WIN', 'IBOV', 'IND']
    for symbol in indices:
        if mt5.symbol_select(symbol, True):
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, datetime.now(), 200)
            if rates is not None and len(rates) >= 200:
                df = pd.DataFrame(rates)
                sma200 = df['close'].mean()
                current_price = df['close'].iloc[-1]
                ratio = current_price / sma200
                diff_pct = (ratio - 1) * 100
                
                status = "NORMAL"
                if current_price < sma200 * 0.98:
                    status = "BEARISH_EXTREME (PÂNICO)"
                elif current_price < sma200:
                    status = "BEARISH"
                
                print(f"Símbolo: {symbol}")
                print(f"  Preço Atual: {current_price}")
                print(f"  SMA200 (H1): {sma200:.2f}")
                print(f"  Diferença: {diff_pct:.2f}%")
                print(f"  Status: {status}")
                print("-" * 20)
            else:
                print(f"Símbolo {symbol} sem dados suficientes.")
        else:
            print(f"Símbolo {symbol} não encontrado.")
    
    mt5.shutdown()

if __name__ == "__main__":
    check()
