#!/usr/bin/env python3
"""
Test script para o B3 Daily Universe Builder
"""
import sys
import os
import logging
from datetime import datetime

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_universe_builder():
    """Testa o Universe Builder de forma isolada"""
    try:
        # Importa os módulos necessários
        print("🧪 Iniciando teste do Universe Builder...")

        # Tenta importar MT5 e utils
        try:
            import MetaTrader5 as mt5
            import utils
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            print("⚠️  Certifique-se de que MT5 está instalado e disponível")
            return False

        # Testa conexão MT5
        print("🔌 Testando conexão MT5...")
        if not mt5.initialize():
            print(f"❌ Falha ao conectar MT5: {mt5.last_error()}")
            return False

        print("✅ MT5 conectado com sucesso!")

        # Testa o Universe Builder com parâmetros mais relaxados para teste
        print("🎯 Executando Universe Builder (modo teste)...")
        universe = utils.build_b3_universe(
            min_fin_volume=5_000_000,  # Menor volume para teste
            min_atr_pct=0.5,  # ATR mínimo menor
            max_atr_pct=10.0,  # ATR máximo maior
            max_ibov_corr=0.9,  # Correlação máxima maior
            min_market_cap=500_000_000,  # Market cap mínimo menor
            save_json=True,
        )

        # Verifica resultados
        if universe and universe.get("ELITE"):
            print(
                f"\n🎉 SUCESSO! Universe Builder encontrou {len(universe['ELITE'])} ativos ELITE"
            )
            print(
                f"⭐ Também encontrou {len(universe.get('OPORTUNIDADE', []))} oportunidades"
            )
            print(f"📊 Total de ativos analisados: {len(universe.get('TOTAL', []))}")

            print(f"\n🏆 Top 10 Ativos ELITE:")
            for i, symbol in enumerate(universe["ELITE"][:10], 1):
                print(f"  {i:2d}. {symbol}")

            print(f"\n📈 Top 10 Oportunidades:")
            for i, symbol in enumerate(universe.get("OPORTUNIDADE", [])[:10], 1):
                print(f"  {i:2d}. {symbol}")

            # Verifica se arquivo foi salvo
            if os.path.exists("elite_symbols_latest.json"):
                print(f"\n💾 Arquivo elite_symbols_latest.json salvo com sucesso!")

                # Testa carregamento do arquivo
                loaded_universe = utils.load_elite_symbols_from_json()
                if loaded_universe:
                    print(
                        f"✅ Arquivo carregado com sucesso: {len(loaded_universe.get('ELITE', []))} ativos ELITE"
                    )
                else:
                    print("❌ Erro ao carregar arquivo salvo")
            else:
                print("❌ Arquivo não foi salvo")

            return True

        else:
            print("❌ Universe Builder não encontrou nenhum ativo ELITE")
            print("📋 Verificando todos os ativos encontrados...")

            if universe:
                print(f"TOTAL: {len(universe.get('TOTAL', []))}")
                print(f"OPORTUNIDADE: {len(universe.get('OPORTUNIDADE', []))}")
                print(f"ELITE: {len(universe.get('ELITE', []))}")

                if universe.get("TOTAL"):
                    print(f"Top 5 TOTAL: {universe['TOTAL'][:5]}")

            return False

    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Desconecta MT5
        try:
            import MetaTrader5 as mt5

            mt5.shutdown()
            print("\n🔌 MT5 desconectado")
        except:
            pass


def test_individual_functions():
    """Testa funções individuais do Universe Builder"""
    print("\n🧪 Testando funções individuais...")

    try:
        import utils
        import pandas as pd
        import numpy as np

        # Testa calculate_asset_score
        print("📊 Testando calculate_asset_score...")
        score = utils.calculate_asset_score(
            volume=20_000_000,  # R$ 20M de volume
            atr_pct=3.0,  # 3% ATR
            corr=0.5,  # 50% correlação
            mcap=50_000_000_000,  # R$ 50B market cap
            spread_pct=0.15,  # 0.15% spread
        )
        print(f"✅ Score calculado: {score}/100")

        # Testa is_future
        print("\n🔮 Testando is_future...")
        test_symbols = ["PETR4", "VALE3", "WINZ26", "WDOZ26", "IBOV"]
        for symbol in test_symbols:
            is_fut = utils.is_future(symbol)
            print(f"  {symbol}: {'✅ Futuro' if is_fut else '❌ Ação'}")

        # Testa fundamental_fetcher
        print("\n💰 Testando fundamental_fetcher...")
        test_stocks = ["PETR4", "VALE3", "ITUB4", "XYZ3"]
        for stock in test_stocks:
            fund = utils.fundamental_fetcher.get_fundamentals(stock)
            print(
                f"  {stock}: Market Cap R$ {fund.get('market_cap', 0):,.0f} ({fund.get('sector', 'Unknown')})"
            )

        print("\n✅ Testes individuais concluídos!")
        return True

    except Exception as e:
        print(f"❌ Erro nos testes individuais: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 B3 Daily Universe Builder - Test Suite")
    print("=" * 50)
    print(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Testa funções individuais primeiro
    individual_ok = test_individual_functions()

    # Pergunta se quer testar com MT5 (requer conexão)
    print("\n" + "=" * 50)
    response = input("Deseja testar com conexão MT5? (s/n): ").strip().lower()

    if response == "s":
        mt5_ok = test_universe_builder()

        if individual_ok and mt5_ok:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            sys.exit(0)
        else:
            print("\n❌ ALGUNS TESTES FALHARAM")
            sys.exit(1)
    else:
        if individual_ok:
            print("\n🎉 TESTES INDIVIDUAIS PASSARAM!")
            sys.exit(0)
        else:
            print("\n❌ TESTES INDIVIDUAIS FALHARAM")
            sys.exit(1)
