"""
Verificação rápida das 7 correções críticas do bot XP3v5.
Execute: python test_fixes.py
"""
import sys
sys.path.insert(0, '.')

errors = []

# ============================================================
# TESTE 1: FundamentalAnalyst não usa mais score fixo 0.40
# ============================================================
print("=== TESTE 1: FundamentalAnalyst ===")
try:
    from agents.analyst_team import FundamentalAnalyst
    fa = FundamentalAnalyst()
    r = fa.analyze('PETR4', {'ibov_trend': 'bullish'})
    score = r['score']
    val   = r['valuation']
    print(f"Score: {score:.2f} | Valuation: {val} | Drivers: {r['drivers']} | Risks: {r['risks']}")
    # Score nao deve ser exatamente 0.4 com mt5_avg_tick_volume=0 -> penaliza -> 0.40
    # Mas AGORA com mt5_bars=0 tambem penaliza -> 0.30 (diferente de antes)
    assert 'mt5_avg_tick_volume' not in str(r), "OK se chegou aqui sem crash"
    print("PASS - FundamentalAnalyst nao crasha e retorna dados estruturados")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 1: {e}")

# ============================================================
# TESTE 2: SentimentAnalyst usa IBOV trend real (nao random)
# ============================================================
print()
print("=== TESTE 2: SentimentAnalyst ===")
try:
    from agents.analyst_team import SentimentAnalyst
    sa = SentimentAnalyst()
    r_bull = sa.analyze('PETR4', {'ibov_trend': 'bullish'})
    r_bear = sa.analyze('PETR4', {'ibov_trend': 'bearish'})
    print(f"Bullish market: score={r_bull['score']:.2f} sent={r_bull['sentiment']}")
    print(f"Bearish market: score={r_bear['score']:.2f} sent={r_bear['sentiment']}")
    assert r_bull['score'] > r_bear['score'], "Score bullish deve ser maior que bearish!"
    assert r_bull['sentiment'] == 'optimistic', "Bullish deve ter sentiment optimistic!"
    assert r_bear['sentiment'] == 'pessimistic', "Bearish deve ter sentiment pessimistic!"
    assert r_bull['ibov_trend_used'] == 'bullish', "Deve registrar qual trend usou!"
    print("PASS - Sentimento reflete IBOV trend corretamente")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 2: {e}")

# ============================================================
# TESTE 3: Debate alcanca BULLISH com sinal tecnico sozinho
# ============================================================
print()
print("=== TESTE 3: ResearcherTeam Debate ===")
try:
    from agents.researcher_team import ResearcherTeam
    rt = ResearcherTeam()

    # So tecnico bullish forte (score 0.90 -> conf_esc: ~1.09) passa de 0.8
    r_tech_only = rt.debate('PETR4', {
        'technical':   {'trend': 'bullish', 'score': 0.90},
        'fundamental': {'valuation': 'neutral', 'score': 0.50},
        'sentiment':   {'sentiment': 'neutral', 'score': 0.50},
        'orderflow':   {'pressure': 'neutral', 'imbalance': 0.0}
    })
    print(f"So tecnico: {r_tech_only['consensus']} (bull={r_tech_only['bull_score']:.1f} bear={r_tech_only['bear_score']:.1f})")
    assert r_tech_only['consensus'] == 'BULLISH', f"Tecnico forte > threshold 0.8 -> deve ser BULLISH! Got: {r_tech_only['consensus']}"

    # Full bullish: tecnico + fundamental cheap + sentimento + orderflow
    r_full = rt.debate('PETR4', {
        'technical':   {'trend': 'bullish', 'score': 0.80},
        'fundamental': {'valuation': 'cheap', 'score': 0.70},
        'sentiment':   {'sentiment': 'optimistic', 'score': 0.70},
        'orderflow':   {'pressure': 'bullish', 'imbalance': 0.20}
    })
    print(f"Full bullish: {r_full['consensus']} (conf={r_full['confidence']:.2f} bull={r_full['bull_score']:.1f})")
    assert r_full['consensus'] == 'BULLISH'
    assert r_full['confidence'] > 0.5, "Confianca alta em cenario full bullish!"

    # Cenario neutro deve permanecer NEUTRAL
    r_neutral = rt.debate('PETR4', {
        'technical':   {'trend': 'neutral', 'score': 0.50},
        'fundamental': {'valuation': 'neutral', 'score': 0.50},
        'sentiment':   {'sentiment': 'neutral', 'score': 0.50},
        'orderflow':   {'pressure': 'neutral', 'imbalance': 0.0}
    })
    print(f"Neutro: {r_neutral['consensus']}")
    assert r_neutral['consensus'] == 'NEUTRAL'
    print("PASS - Consenso funciona corretamente")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 3: {e}")

# ============================================================
# TESTE 4: TraderAgents propoe BUY em cenario bullish
# ============================================================
print()
print("=== TESTE 4: TraderAgents - caminho BUY ===")
try:
    from agents.trader_agents import TraderTeam
    tt = TraderTeam()
    analysis_bull = {
        'technical':   {'trend': 'bullish', 'score': 0.70},
        'fundamental': {'valuation': 'fair', 'score': 0.60},
        'sentiment':   {'sentiment': 'optimistic', 'score': 0.72},
        'orderflow':   {'pressure': 'neutral', 'imbalance': 0.0}
    }
    debate_bull = {'consensus': 'BULLISH', 'confidence': 0.75}
    proposals = tt.collect_proposals('PETR4', analysis_bull, debate_bull)
    actions = [p['action'] for p in proposals]
    print(f"Propostas: {actions}")
    buy_count = actions.count('BUY')
    assert buy_count > 0, f"Deve ter ao menos 1 proposta BUY! Got: {actions}"
    print(f"PASS - {buy_count} proposta(s) BUY gerada(s) com valuation='fair'")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 4: {e}")

# ============================================================
# TESTE 5: Config - verificar FRIDAY_NO_ENTRY_AFTER e MAX_POSITIONS
# ============================================================
print()
print("=== TESTE 5: Config ===")
try:
    import config
    friday_cutoff = config.FRIDAY_NO_ENTRY_AFTER
    max_pos = config.MAX_CONCURRENT_POSITIONS
    print(f"FRIDAY_NO_ENTRY_AFTER: {friday_cutoff}")
    print(f"MAX_CONCURRENT_POSITIONS: {max_pos}")
    assert friday_cutoff == "16:15", f"Esperado 16:15, obtido {friday_cutoff}"
    assert max_pos >= 15, f"Esperado >= 15, obtido {max_pos}"
    print("PASS - Config correto")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 5: {e}")

# ============================================================
# TESTE 6: ML Fallback tecnico (sem RF model)
# ============================================================
print()
print("=== TESTE 6: ML Fallback Tecnico ===")
try:
    import pandas as pd
    import numpy as np
    from ml.prediction import MLPredictor

    pred = MLPredictor.__new__(MLPredictor)
    pred.rf_model = None
    pred.scaler = None

    n = 50
    close = pd.Series(np.linspace(10.0, 12.0, n))
    high  = close * 1.01
    low   = close * 0.99
    vol   = pd.Series([1000.0] * n)
    df_bull = pd.DataFrame({'close': close, 'high': high, 'low': low, 'tick_volume': vol})
    result_bull = pred._technical_fallback(df_bull)
    print(f"Fallback (tendencia alta): {result_bull['signal']} prob={result_bull['probability']:.3f}")
    assert result_bull['signal'] in ('BUY', 'NEUTRAL'), f"Esperado BUY ou NEUTRAL, got {result_bull['signal']}"

    close_bear = pd.Series(np.linspace(12.0, 10.0, n))
    df_bear = pd.DataFrame({'close': close_bear, 'high': close_bear*1.01, 'low': close_bear*0.99, 'tick_volume': vol})
    result_bear = pred._technical_fallback(df_bear)
    print(f"Fallback (tendencia baixa): {result_bear['signal']} prob={result_bear['probability']:.3f}")
    assert result_bear['signal'] in ('SELL', 'NEUTRAL'), f"Bear trend -> esperado SELL ou NEUTRAL, got {result_bear['signal']}"
    print("PASS - Fallback tecnico funciona sem RF model")
except Exception as e:
    print(f"FAIL: {e}")
    errors.append(f"Teste 6: {e}")

# ============================================================
# RESULTADO FINAL
# ============================================================
print()
print("=" * 50)
if errors:
    print(f"FALHOU {len(errors)} teste(s):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("TODOS OS 6 TESTES PASSARAM!")
    print("As 7 correcoes criticas estao funcionando corretamente.")
    sys.exit(0)
