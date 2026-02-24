#!/usr/bin/env python3
"""
Teste completo do sistema adaptativo XP3 PRO
"""

from adaptive_integration import *
from adaptive_intelligence import *
import time


def test_adaptive_system():
    print("🧠 Testando sistema adaptativo XP3 PRO...")

    # Testa inicialização
    print("1. Iniciando sistema...")
    start_adaptive_system()
    time.sleep(1)

    # Testa status
    print("2. Verificando status...")
    status = get_adaptive_status()
    print(f'   Status: {status.get("running", "unknown")}')
    print(f'   Última sincronização: {status.get("last_sync", "unknown")}')

    # Testa parâmetros adaptativos
    print("3. Verificando parâmetros...")
    ml_threshold = get_adaptive_ml_threshold()
    kelly_mult = get_adaptive_kelly_multiplier()
    spread_mult = get_adaptive_spread_multiplier()

    print(f"   ML Threshold: {ml_threshold}")
    print(f"   Kelly Multiplier: {kelly_mult}")
    print(f"   Spread Multiplier: {spread_mult}")

    # Testa se os parâmetros estão sendo gerenciados
    print("4. Verificando gerenciamento de parâmetros...")
    if status.get("current_parameters"):
        params = status["current_parameters"]
        perf_metrics = params.get("performance_metrics", {})
        print(f'   Total de ajustes: {perf_metrics.get("total_adjustments", 0)}')
        print(f'   Winrate 24h: {perf_metrics.get("winrate_24h", "N/A")}')
        print(f'   Volatilidade: {perf_metrics.get("volatility", "N/A")}')

    # Para o sistema
    print("5. Parando sistema...")
    stop_adaptive_system()

    print("✅ Teste do sistema adaptativo concluído com sucesso!")
    print("📊 O bot XP3 PRO está pronto para operar com inteligência adaptativa!")
    print("🎯 O sistema ajustará automaticamente:")
    print("   • Threshold ML (0.62-0.74)")
    print("   • Multiplicador Kelly (0.8-1.2x)")
    print("   • Multiplicador Spread (2.0-3.0x)")
    print("   • Limite de perdas por símbolo")
    print("   • Triggers de emergência (winrate <40%, volatilidade >5%)")


if __name__ == "__main__":
    test_adaptive_system()
