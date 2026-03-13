import re
import sys
import os

# Simula a função que implementamos no utils.py
def is_valid_b3_ticker(symbol: str) -> bool:
    if not symbol:
        return False
    pattern = r"^[A-Z]{4}(3|4|5|6|11)$"
    return bool(re.match(pattern, symbol.upper().strip()))

def test_ticker_filter():
    test_cases = {
        "PETR4": True,    # PN
        "VALE3": True,    # ON
        "SANB11": True,   # Unit
        "CPLE6": True,    # PNB
        "BRSR5": True,    # PNA
        "GOLL54": False,  # Opção (Ticker de derivativo detectado no log)
        "PETRH26": False, # Opção de compra
        "WDOJ25": False,  # Futuro
        "GOLD11": True,   # ETF (segue padrão Unit)
        "BOVA11": True,   # ETF
        "AAPL34": False,  # BDR (terminado em 34 não incluído na regra 3,4,5,6,11)
        "ABCD12": False,  # Inválido
    }
    
    passed = 0
    failed = 0
    
    print("[TEST] Iniciando Testes de Filtro de Tickers B3...")
    for sym, expected in test_cases.items():
        result = is_valid_b3_ticker(sym)
        status = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"   [{status}] {sym}: expected {expected}, got {result}")
    
    print(f"\n📊 Resultado: {passed} Passou, {failed} Falhou")
    return failed == 0

if __name__ == "__main__":
    if test_ticker_filter():
        sys.exit(0)
    else:
        sys.exit(1)
