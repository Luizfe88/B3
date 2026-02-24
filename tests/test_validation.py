import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Adiciona o diretório raiz ao path para permitir importações dos módulos do bot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Agora importa os módulos necessários
try:
    import utils
    from validation import calculate_kelly_position_size
    import config
except ImportError as e:
    print(f"Erro de importação: {e}")
    # Se os imports falharem, é provável que o path esteja incorreto.
    # Adicione um print para depuração.
    print("sys.path:", sys.path)
    pytest.skip(
        "Não foi possível importar os módulos do bot, pulando testes.",
        allow_module_level=True,
    )


# Mock para simular a resposta do mt5.account_info()
@pytest.fixture
def mock_mt5_account():
    mock_account = MagicMock()
    mock_account.balance = 100000.0
    mock_account.equity = 100000.0
    return mock_account


# Mock para simular o a resposta do mt5.symbol_info()
@pytest.fixture
def mock_mt5_symbol_info():
    mock_info = MagicMock()
    mock_info.volume_min = 100.0
    mock_info.volume_max = 10000.0
    return mock_info


@patch("utils.get_symbol_performance")
@patch("utils.detect_market_regime")
@patch("bot.daily_max_equity", 100000.0)  # Mocking global variable from bot
@patch("validation.mt5.account_info")
@patch("validation.mt5.symbol_info")
def test_calculate_kelly_position_size_basic_case(
    mock_symbol_info, mock_account_info, mock_regime, mock_perf
):
    """
    Testa o cálculo de tamanho de posição com o Critério de Kelly
    para um caso de uso padrão e previsível.
    """
    # Configuração do Mock
    mock_account_info.return_value = MagicMock(balance=100000, equity=100000)
    mock_symbol_info.return_value = MagicMock(volume_min=100.0, volume_max=50000.0)
    mock_regime.return_value = "RISK_ON"
    mock_perf.return_value = {"win_rate": 0.60, "avg_rr": 2.0, "total_trades": 50}

    # Parâmetros para o teste
    symbol = "PETR4"
    entry_price = 30.0
    sl = 29.1  # Risco de 0.90 por ação (3%)
    tp = 32.7  # Recompensa de 2.7 por ação (R:R de 3.0)

    # Execução
    volume = calculate_kelly_position_size(symbol, entry_price, sl, tp, "BUY")

    # Verificação
    # Com os mocks, o cálculo deve ser determinístico.
    # A expectativa é que o volume seja um valor > 0 e múltiplo de 100.
    assert volume > 0
    assert volume % 100 == 0

    # Um cálculo aproximado para validação:
    # WR=0.6, RR=3 -> p=0.6, q=0.4, b=3 -> Kelly = (0.6 * 3 - 0.4) / 3 = 1.4 / 3 = 0.46
    # Kelly Ajustado (vol_adjust=1, streak=1, dd=1) -> 0.46
    # Final Kelly (x0.15) -> 0.069
    # Posição = 100000 * 0.069 = 6900
    # Volume = 6900 / 30.0 = 230 -> arredondado para 200
    assert volume == 200.0


@patch("utils.get_symbol_performance")
@patch("utils.detect_market_regime")
@patch("bot.daily_max_equity", 95000.0)
@patch("validation.mt5.account_info")
@patch("validation.mt5.symbol_info")
def test_kelly_reduces_volume_on_drawdown(
    mock_symbol_info, mock_account_info, mock_regime, mock_perf
):
    """
    Testa se o Critério de Kelly reduz o tamanho da posição quando a conta
    está em um drawdown significativo.
    """
    # Configuração do Mock
    # Simulando um DD > 3% (100k -> 95k)
    mock_account_info.return_value = MagicMock(balance=100000, equity=95000)
    mock_symbol_info.return_value = MagicMock(volume_min=100.0, volume_max=50000.0)
    mock_regime.return_value = "RISK_ON"
    mock_perf.return_value = {"win_rate": 0.60, "avg_rr": 2.0, "total_trades": 50}

    # Parâmetros
    symbol = "VALE3"
    entry_price = 60.0
    sl = 58.0
    tp = 66.0

    # Execução
    volume = calculate_kelly_position_size(symbol, entry_price, sl, tp, "BUY")

    # Verificação
    # O DD > 3% deve ativar o dd_adjustment = 0.5, reduzindo o volume final pela metade.
    # O volume aqui deve ser menor que o do teste anterior, mesmo com parâmetros similares.
    # Kelly Frac = 0.35
    # Adj. Kelly = 0.35 * 1.0 (vol) * 1.0 (streak) * 0.5 (dd) = 0.175
    # Final Kelly (x0.15) = 0.02625
    # Posição = 95000 * 0.02625 = 2493.75
    # Volume = 2493.75 / 60 = 41.56 -> arredondado para 0, pois é < 100.
    assert volume == 0
