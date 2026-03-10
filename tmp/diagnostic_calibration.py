import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

# Adiciona o diretório atual ao path
sys.path.append(r"c:\Users\luizf\Documents\xp3v5")

from otimizador_semanal import (
    load_all_symbols, 
    safe_mt5_initialize, 
    check_liquidity_dynamic,
    get_ibov_data
)
import config

def diagnostic():
    print("--- Diagnóstico de Liquidez e Dados ---")
    
    # Simula o ambiente da rotina semanal
    os.environ["XP3_LOAD_ALL_MT5"] = "1"
    os.environ["XP3_SANDBOX"] = "0"  # Força modo produção para ver filtros reais
    
    if not safe_mt5_initialize():
        print("Erro: Não foi possível conectar ao MT5.")
        return

    try:
        symbols = load_all_symbols()
        print(f"Total de símbolos no Market Watch: {len(symbols)}")
        
        # Pega IBOV para o check de liquidez
        ibov_df = get_ibov_data()
        
        test_syms = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "AZUL4", "MGLU3"]
        # Filtra apenas os que existem no Market Watch para o teste
        test_syms = [s for s in test_syms if s in symbols]
        
        if not test_syms:
            test_syms = symbols[:10]
            
        print(f"\nTestando liquidez para: {test_syms}")
        
        results = []
        for sym in test_syms:
            is_ok, reason, metrics = check_liquidity_dynamic(sym, ibov_df)
            results.append({
                "Symbol": sym,
                "Is_Ok": is_ok,
                "Reason": reason,
                "Avg_Fin": metrics.get("avg_fin", 0),
                "ADX_Thr": metrics.get("adx_threshold", 0)
            })
            
        df_res = pd.DataFrame(results)
        print("\nResultado do Check de Liquidez:")
        print(df_res.to_string())
        
        # Verifica se há algum filtro de data que pode estar limpando results
        print("\nVerificando se há dados para os primeiros 5 símbolos...")
        for sym in test_syms[:5]:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 100)
            if rates is not None and len(rates) > 0:
                print(f"✅ {sym}: {len(rates)} barras encontradas.")
            else:
                print(f"❌ {sym}: Nenhuma barra encontrada.")
                
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    diagnostic()
