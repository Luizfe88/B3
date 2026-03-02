import sys
sys.path.insert(0, '.')
import logging

logging.basicConfig(level=logging.INFO)

from opportunity_ranker import opportunity_ranker

# Simulação de oportunidades do bot (decision format)
opps = [
    # 1. Ativo razoável, mas Debate não unânime (diff 1.0)
    ("BBAS3", {
        "action": "BUY",
        "debate_info": {"diff": 1.0},
        "analysis": {
            "technical": {"score": 0.65},
            "orderflow": {"imbalance": 0.1},
            "sentiment": {"score": 0.55}
        }
    }),
    # 2. Excelente ML e Debate forte (diff 2.0)
    ("PETR4", {
        "action": "BUY",
        "debate_info": {"diff": 2.0},
        "analysis": {
            "technical": {"score": 0.85},
            "orderflow": {"imbalance": 0.3},
            "sentiment": {"score": 0.80}
        }
    }),
    # 3. Super Order Flow, ML neutro
    ("VALE3", {
        "action": "SELL",
        "debate_info": {"diff": 1.2},
        "analysis": {
            "technical": {"score": 0.40},
            "orderflow": {"imbalance": 0.8},
            "sentiment": {"score": 0.50}
        }
    })
]

print("=== DEPOIS DO RANKING ===")
ranked = opportunity_ranker.rank_opportunities(opps)

for idx, (sym, dec) in enumerate(ranked):
    print(f"Top {idx+1}: {sym} - Action: {dec['action']}")

# O esperado é PETR4 como Top 1 (Maior debate e ML forte)
print("\nVerificando se PETR4 ficou em 1º...")
assert ranked[0][0] == "PETR4", "Erro: PETR4 devia ser o 1º pela pontuação"
print("SUCCESS - Ranker Global Operacional")
