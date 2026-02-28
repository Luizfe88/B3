# 🔧 TROUBLESHOOTING & CHECKLIST DE IMPLEMENTAÇÃO

---

## ✅ PRÉ-DEPLOYMENT CHECKLIST

- [ ] Backup de todos os arquivos críticos feito
- [ ] Código-fonte em versionamento (git commit)
- [ ] Testes locais executados com sucesso: `python test_security_fixes.py`
- [ ] Bot parado (não está trading ao fazer alterações)
- [ ] Arquivos de configuração (config.yaml, config.py) não mudaram
- [ ] Imports adicionados se necessário (verifique datetime, timedelta)

---

## 📋 PROCEDIMENTO DE IMPLEMENTAÇÃO PASSO-A-PASSO

### Passo 1: Backup
```bash
# Windows
copy core\position_manager.py core\position_manager.py.backup
copy agents\risk_team.py agents\risk_team.py.backup
copy bot.py bot.py.backup
```

### Passo 2: Aplicar Mudanças

Usar o arquivo `QUICK_REFERENCE_DIFFS.md` para copiar/colar as mudanças nos 3 arquivos:

1. **`core/position_manager.py`**: Adicionar método + atualizar `get_total_exposure()`
2. **`agents/risk_team.py`**: Reescrever método `validate_trade()`
3. **`bot.py`**: Adicionar `sleep(1.5)` em 2 lugares

### Passo 3: Validar Sintaxe
```bash
# Windows PowerShell
python -m py_compile core\position_manager.py
python -m py_compile agents\risk_team.py
python -m py_compile bot.py

# Deve retornar sem erros
```

### Passo 4: Executar Testes
```bash
python test_security_fixes.py
# Deve retornar "✅ PRONTO PARA PRODUÇÃO"
```

### Passo 5: Iniciar em Paper/Demo (24-48 horas)
```bash
# Inicie o bot em conta de demonstração/paper primeiro
python bot.py --mode=paper  # (se suportado)
```

### Passo 6: Monitorar Logs
```bash
# Procurar por estes padrões nos logs:
# ✅ "📤 Ordem pendente registrada"
# ✅ "⏳ Aguardando confirmação"
# ✅ "📊 Exposição Total: R$ XXX (Confirmada + Pendente)"
# ❌ Se não ver → verificar implementação
```

---

## 🚨 PROBLEMAS COMUNS E SOLUÇÕES

### Problema 1: AttributeError: 'PositionManager' has no attribute 'pending_orders'

**Causa:** `__init__` não foi atualizado

**Solução:**
```python
# Verificar que __init__ tem:
self.pending_orders = []
```

---

### Problema 2: NameError: name 'register_pending_order' is not defined

**Causa:** Método não foi adicionado à classe

**Solução:**
```python
# Verificar que PositionManager tem estes métodos:
def register_pending_order(self, symbol: str, volume: float, price: float):
def clean_pending_orders(self):
def get_pending_exposure(self) -> Dict[str, float]:
```

---

### Problema 3: Bot fica travado por 1.5s a cada ordem

**Causa:** Esperado! É intencional (race condition fix)

**Solução:**  
Se não quiser esperar tanto:
- Reduzir para `time.sleep(1.0)` (menos conservador)
- Reduzir para `time.sleep(0.8)` (teste com sua latência e MT5)
- Não remover! Sem isso volta o bug.

---

### Problema 4: Exposição pendente não limpa após 3s

**Causa:** `clean_pending_orders()` não está sendo chamado

**Solução:**
```python
# Verificar que get_pending_exposure() chama:
def get_pending_exposure(self):
    self.clean_pending_orders()  # ← Isto deve estar aqui!
    # ... resto ...
```

---

### Problema 5: RiskTeam rejeita mais ordens que esperado

**Causa:** `effective_risk_pct` está maior que antes

**Razão:** Agora é calculado corretamente! Antes era bugado.

**Esperado:**
- Antes: 50 ordens de 2% cada = 100% aprovadas (mas realmente 120%+!)
- Depois: ~62 ordens de 2.4% cada = 150% total (correto)

**Solução:** É comportamento esperado. Testar em paper.

---

### Problema 6: Vejo logs de "Limpas X ordens (>3 segundos)"

**Causa:** Esperado!

**O que significa:**
- Ordem foi registrada como pendente
- MT5 confirmou dentro de 3 segundos
- Sistema limpou automaticamente

**Ação:** Nenhuma, é normal.

---

### Problema 7: Não vejo "📤 Ordem pendente registrada" nos logs

**Causa:** `register_pending_order()` não está sendo chamado

**Solução:**
```python
# bot.py após send_order() deve ter:
position_manager.register_pending_order(symbol, final_volume, current_price)
# Se não tiver → adicionar
```

---

### Problema 8: Testes falham com "AssertionError"

**Causa:** Código não foi aplicado corretamente

**Solução:**
```bash
# Executar teste verboso para ver onde falha
python -m pytest test_security_fixes.py -v

# Comparar código com QUICK_REFERENCE_DIFFS.md
# Procurar diferenças na indentação ou sintaxe
```

---

### Problema 9: ImportError: cannot import datetime

**Causa:** Falta import em position_manager.py

**Solução:**
```python
# Verificar imports no topo do arquivo:
from datetime import datetime, timedelta

# Se faltar → adicionar
```

---

## 📊 MONITORAMENTO EM PRODUÇÃO

### Métricas para Acompanhar

1. **Taxa de Aprovação de Ordens**
   - Esperado: 62-65 ordens de 100+ (150% limite / 2.4% por ordem)
   - Se > 70: algo errado
   - Se < 50: verificar rejeições

2. **Exposição Total**
   - Máximo: 150% da conta
   - Mínimo observado: 0% (quando sem posições)
   - Deve ser consistente quando comparado com MT5

3. **Latência de Ordens**
   - Sem delay: ~5ms por ordem
   - Com delay: ~1500ms por ordem
   - Esperado no novo sistema

4. **Race Condition Events**
   - Deve ser 0 (ou seja, nunca ver "120% short")
   - Se > 0: alert imediato

---

### Logs para Monitorar Continuamente

```bash
# Procurar por estes padrões diariamente:

# ✅ BONS SINAIS:
grep "Exposição Total:" logs/trading_agents.log
# Deve mostrar números crescentes conforme novas ordens

grep "Ordem pendente registrada" logs/trading_agents.log  
# Deve mostrar cada ordem enviada

grep "Limpas.*ordens" logs/trading_agents.log
# Deve haver algumas limpezas (ordens > 3s)

# ❌ SINAIS DE ALERTA:
grep "Exposição limite atingido" logs/trading_agents.log
# Se muitas → diminuir size_multiplier ou aumentar limite

grep "Risco efetivo excessivo" logs/trading_agents.log
# Se muitas → verificar se RiskyTrader está agressivo demais
```

---

## 🧪 TESTES ADICIONAIS RECOMENDADOS

### Teste Manual 1: Verificar Rastreamento
```python
# No terminal Python:
from core.position_manager import PositionManager
from core.execution import ExecutionEngine

exec = ExecutionEngine()
pm = PositionManager(exec)

# Simular 3 ordens
pm.register_pending_order("BBAS3", 1000, 30.0)
print(pm.get_pending_exposure())  # {'BBAS3': 30000}

pm.register_pending_order("BRML3", 500, 25.0)
print(pm.get_pending_exposure())  # {'BBAS3': 30000, 'BRML3': 12500}

# Exposição total deve incluir pending
total = pm.get_total_exposure()  # Confirmada + Pending
```

### Teste Manual 2: Verificar Size Multiplier
```python
# No terminal Python:
from agents.risk_team import RiskGuardian

rg = RiskGuardian("TestGuard", tolerance=0.5)

proposal = {
    "action": "BUY",
    "size_multiplier": 1.2,  # RiskyTrader
}

market_context = {
    "equity": 100000,
    "total_exposure": 0,
    "price": 30.0,
    "recent_entries_count": 0,
    "ibov_trend": "neutral"
}

result = rg.validate_trade("BBAS3", proposal, market_context)
print(result)  # {'approved': True, 'adjusted_proposal': {...}}

# Verificar que exposição calculada é 2.4%, não 24%
```

---

## 🎯 METAS DE SUCESSO

**Métricas de Sucesso (após 48h em papel/demo):**

| Métrica | Target | Critério |
|---------|--------|----------|
| Crashes do bot | 0 | Nenhuma falha |
| Ordens rejeitadas > limite | 0 | Proteção funcionando |
| Exposição máxima observada | 130-150% | Dentro do esperado |
| Race condition events | 0 | Nunca ocorrer |
| Latência média | ~1500ms | Com delay |
| Taxa de aprovação | 60-65% | Do total de símbolos |

---

## 📞 SUPORTE DE DEBUG

Se algo der errado:

1. **Verifique o checklist acima**
2. **Execute `test_security_fixes.py`** (testes unitários)
3. **Compare código com `QUICK_REFERENCE_DIFFS.md`** (linha por linha)
4. **Ative debug logging:**
   ```python
   # Em bot.py ou config:
   logging.basicConfig(level=logging.DEBUG)  # Mais verbose
   ```
5. **Procure por padrões de erro nos logs:**
   - `❌ ` = erro
   - `⚠️ ` = aviso
   - `📤` = ordem pendente
   - `🧹` = limpeza

---

## 🔄 ROLLBACK (se necessário)

Se tudo der errado, restaurar é simples:

```bash
# Windows
copy core\position_manager.py.backup core\position_manager.py
copy agents\risk_team.py.backup agents\risk_team.py
copy bot.py.backup bot.py

# Reiniciar bot
python bot.py
```

---

**Última Atualização:** 28/02/2026  
**Status:** ✅ Completo e Validado
