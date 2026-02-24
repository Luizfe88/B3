#!/usr/bin/env python3
"""
Limpar cache Redis e testar novamente
"""

import sys

sys.path.append(".")

from utils import ConcurrentMarketScanner
import redis

# Limpar cache Redis
try:
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushdb()
    print("✅ Cache Redis limpo")
except Exception as e:
    print(f"❌ Erro ao limpar cache: {e}")

# Testar novamente
print("\n=== Teste após limpar cache ===")
try:
    scanner = ConcurrentMarketScanner()
    resultados = scanner._scan_futures_fast(["WINQ26"])

    if resultados and "WINQ26" in resultados:
        ind = resultados["WINQ26"]
        print(f"Indicadores WINQ26:")
        for key, valor in ind.items():
            print(f"  {key}: {valor}")

        # Verificar se ema_diff está presente
        if "ema_diff" in ind:
            print(f"✅ ema_diff calculado: {ind['ema_diff']:.4f}")

            # Testar interpretação da tendência
            ema_diff = ind["ema_diff"]
            if ema_diff > 0.02:
                print(f"📈 Tendência: Alta forte ({ema_diff*100:.1f}%)")
            elif ema_diff > 0.01:
                print(f"📊 Tendência: Alta moderada ({ema_diff*100:.1f}%)")
            elif ema_diff > -0.01:
                print(f"📉 Tendência: Lateral ({ema_diff*100:.1f}%)")
            elif ema_diff > -0.02:
                print(f"📉 Tendência: Baixa moderada ({ema_diff*100:.1f}%)")
            else:
                print(f"📉 Tendência: Baixa forte ({ema_diff*100:.1f}%)")
        else:
            print("❌ ema_diff não encontrado!")
    else:
        print("❌ Nenhum resultado do scanner")

except Exception as e:
    print(f"❌ Erro scanner: {e}")

print("\n🎯 Teste concluído!")
