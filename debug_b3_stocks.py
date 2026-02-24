#!/usr/bin/env python3
"""
Debug script para verificar símbolos B3 reais (não BDRs)
"""
import MetaTrader5 as mt5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_real_b3_stocks():
    """Procura por ações B3 reais (não BDRs)"""
    try:
        if not mt5.initialize():
            logger.error(f"Falha ao conectar MT5: {mt5.last_error()}")
            return

        logger.info("✅ MT5 conectado")

        # Pega todos os símbolos
        all_symbols = mt5.symbols_get()
        logger.info(f"📊 Total de símbolos: {len(all_symbols)}")

        # Procura por padrões diferentes
        b3_patterns = []

        for sym in all_symbols:
            name = sym.name.upper()

            # Verifica diferentes padrões B3
            patterns = {
                "termina_com_numero": name[-1].isdigit() and len(name) <= 6,
                "petrobras": "PETR" in name,
                "vale": "VALE" in name,
                "itau": "ITUB" in name or "ITAU" in name,
                "bb": "BBDC" in name or "BBAS" in name,
                "abev": "ABEV" in name,
                "wege": "WEGE" in name,
                "magalu": "MGLU" in name,
                "via": "VIIA" in name or "VVAR" in name,
                "americanas": "AMER" in name,
                "gol": "GOLL" in name,
                "azul": "AZUL" in name,
                "cvc": "CVCB" in name,
                "renner": "LREN" in name,
                "hering": "HGTX" in name,
                "suzano": "SUZB" in name,
                "fibria": "FIBR" in name,
                "kroton": "KROT" in name,
                "cogna": "COGN" in name,
                "positivo": "POSI" in name,
            }

            if any(patterns.values()):
                try:
                    info = mt5.symbol_info(name)
                    if info:
                        b3_patterns.append(
                            {
                                "symbol": name,
                                "select": info.select,
                                "trade_mode": info.trade_mode,
                                "visible": info.visible,
                                "description": info.description,
                                "patterns": {k: v for k, v in patterns.items() if v},
                            }
                        )

                        if len(b3_patterns) <= 20:  # Primeiros 20 para debug
                            logger.info(
                                f"{name:8} | Select: {info.select:5} | Mode: {info.trade_mode} | Desc: {info.description[:30]}"
                            )

                except Exception as e:
                    logger.debug(f"Erro ao verificar {name}: {e}")

        logger.info(f"\n🏆 Total de possíveis B3: {len(b3_patterns)}")

        # Filtra apenas os selecionáveis
        selectable_b3 = [
            s for s in b3_patterns if s["select"] and s["trade_mode"] in [1, 4]
        ]
        logger.info(f"📈 B3 selecionáveis: {len(selectable_b3)}")

        # Mostra exemplos
        if selectable_b3:
            logger.info(f"\n✅ Exemplos de ações B3 encontradas:")
            for stock in selectable_b3[:10]:
                logger.info(f"  {stock['symbol']:8} | {stock['description'][:40]}")

        # Testa símbolos específicos
        test_symbols = [
            "PETR4",
            "VALE3",
            "ITUB4",
            "BBDC4",
            "ABEV3",
            "WEGE3",
            "MGLU3",
            "VIIA3",
            "AMER3",
            "GOLL4",
        ]
        logger.info(f"\n🔍 Verificando símbolos específicos:")
        for symbol in test_symbols:
            info = mt5.symbol_info(symbol)
            if info:
                logger.info(
                    f"{symbol}: ✅ Select={info.select}, Mode={info.trade_mode}, Desc='{info.description}'"
                )
            else:
                logger.info(f"{symbol}: ❌ Não encontrado")

    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback

        traceback.print_exc()

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    find_real_b3_stocks()
