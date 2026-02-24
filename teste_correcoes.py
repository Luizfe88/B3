#!/usr/bin/env python3
"""
Teste das correções de ema_diff e cotações
"""

import sys

sys.path.append(".")

from utils import ConcurrentMarketScanner, FilterChain
import MetaTrader5 as mt5

# Testar MT5
print("=== Teste MT5 ===")
try:
    mt5.initialize()
    print(f"✅ MT5 Conectado: {mt5.terminal_info().connected}")
    print(f"Servidor: {mt5.terminal_info().name}")

    # Testar alguns ativos
    ativos_teste = ["PETR4", "VALE3", "ITUB4"]
    for ativo in ativos_teste:
        tick = mt5.symbol_info_tick(ativo)
        if tick:
            print(f"✅ {ativo}: Ask={tick.ask}, Bid={tick.bid}")
        else:
            print(f"❌ {ativo}: Sem cotação")

    mt5.shutdown()
except Exception as e:
    print(f"❌ Erro MT5: {e}")

print("\n=== Teste Market Scanner ===")
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
        else:
            print("❌ ema_diff não encontrado!")
    else:
        print("❌ Nenhum resultado do scanner")

except Exception as e:
    print(f"❌ Erro scanner: {e}")

print("\n=== Teste FilterChain ===")
try:
    # Criar indicadores de teste
    indicadores_teste = {
        "rsi": 55.0,
        "adx": 25.0,
        "volume_ratio": 1.2,
        "ema_diff": 0.015,  # 1.5% de diferença
        "atr": 0.5,
    }

    fc = FilterChain()
    aprovado, motivo = fc.validate("TESTE3", "BUY", indicadores_teste)

    print(f"Resultado: {'✅ Aprovado' if aprovado else '❌ Reprovado'}")
    if not aprovado:
        print(f"Motivo: {motivo}")

except Exception as e:
    print(f"❌ Erro FilterChain: {e}")

print("\n=== Teste Tendência ===")
# Testar diferentes valores de ema_diff
testes_tendencia = [
    (0.025, "Alta forte"),
    (0.015, "Alta moderada"),
    (0.005, "Lateral alta"),
    (0.0, "Lateral"),
    (-0.005, "Lateral baixa"),
    (-0.015, "Baixa moderada"),
    (-0.025, "Baixa forte"),
]

for ema_diff, descricao in testes_tendencia:
    indicadores = {
        "rsi": 55.0,
        "adx": 25.0,
        "volume_ratio": 1.2,
        "ema_diff": ema_diff,
        "atr": 0.5,
    }

    fc = FilterChain()
    aprovado, motivo = fc.validate("TESTE3", "BUY", indicadores)

    print(f"ema_diff {ema_diff:+.3f} ({descricao}): {'✅' if aprovado else '❌'}")
    if not aprovado and "Tendência lateral" in motivo:
        print(f"  → Detectado: {motivo}")

print("\n🎯 Teste concluído!")
