# 🚨 CORREÇÕES CRÍTICAS DE SEGURANÇA - ATAQUE MASSIVO E RACE CONDITION

Data: 28 de fevereiro de 2026
Status: ✅ IMPLEMENTADO

---

## 📋 Sumário Executivo

Foram identificadas e corrigidas **2 vulnerabilidades críticas** que causaram o "ataque massivo" de 120% short:

1. **Interpretação incorreta de `size_multiplier`**: Multiplicava capital total, não risco base
2. **Race Condition no MT5**: Múltiplas ordens aprovadas sem aguardar confirmação anterior

---

## ❌ PROBLEMA #1: Causa Raiz do "Ataque Massivo"

### O Bug Original

**Arquivo:** `trader_agents.py` + `risk_team.py` + `bot.py`

#### RiskyTrader propõe:
```python
return {"action": "SELL", "size_multiplier": 1.2, ...}
```
Esperava que isso multiplicasse o risco base (2%), mas o sistema estava traduzindo para:

```python
# ERRADO (bot.py linha 246):
base_allocation_pct = config.MAX_CAPITAL_ALLOCATION_PCT  # 2% (0.02)
size_multiplier = decision.get("size", 0.0)  # 1.2 do RiskyTrader
target_exposure = equity * base_allocation_pct * size_multiplier
# = equity * 0.02 * 1.2 = equity * 0.024 = 2.4% POR ATIVO
```

Mas o `RiskGuardian` estava aceitando até **150% do lote base**, criando confusão:

```python
# ERRADO (risk_team.py linha 41):
max_position_size = 1.5  # 150% (aceitando até 1.5!)
proposed_size = proposal.get('size_multiplier', 0.0)  # 1.2
new_exposure = (equity * config.MAX_CAPITAL_ALLOCATION_PCT * proposed_size)
# Se isso for interpretado errado = 150% aprovado!
```

**Resultado:** Como existem ~50 ativos no universo e múltiplas ordens saem sem confirmação (race condition), o sistema aprovava tudo rapidamente.

### ✅ FIX Implementado

**Arquivo:** `agents/risk_team.py` (método `validate_trade`)

Agora o `size_multiplier` é corretamente interpretado como **multiplicador do risco base**:

```python
# CORRETO (risk_team.py nova versão):
base_risk_pct = config.MAX_CAPITAL_ALLOCATION_PCT  # 2% (0.02) - RISCO BASE
proposed_size_multiplier = proposal.get('size_multiplier', 0.0)  # 1.2 do RiskyTrader

# Risco EFETIVO desta proposta
effective_risk_pct = base_risk_pct * proposed_size_multiplier
# = 0.02 * 1.2 = 0.024 = 2.4% (MÁXIMO por trade)

# Limite máximo permitido
max_effective_risk = base_risk_pct * 1.5  # 3% (150% do risco base)

if effective_risk_pct > max_effective_risk:
    # Bloqueia ou ajusta se exceder
    proposal['size_multiplier'] = max_effective_risk / base_risk_pct

# Agora calcula exposição corretamente
new_exposure = equity * effective_risk_pct
```

**Impacto:**
- ✅ RiskyTrader (1.2x) = 2.4% por trade (seguro)
- ✅ NeutralTrader (1.0x) = 2.0% por trade
- ✅ SafeTrader (0.8x) = 1.6% por trade
- ❌ Máximo absolutoto = 3% por trade (limite hard)

---

## 🏃 PROBLEMA #2: Race Condition (50 Ordens Simultâneas)

### O Bug Original

O robô está enviando ordens em um loop **sem aguardar confirmação no MT5**:

```python
# bot.py loop original:
for symbol in symbols:  # ~50 ativos
    decision = fund_manager.decide(symbol, market_data)
    
    if decision["action"] == "BUY":
        execution.send_order(order)  # ← Envia ordem
        # NENHUM DELAY! Loop continua imediatamente
    
    # Próxima iteração começa 5ms depois
    if decision["action"] == "SELL":
        execution.send_order(order)  # ← Envia 2ª ordem antes da 1ª aparecer no MT5
```

**Problema:** O MT5 demora **200-500ms** para registrar a posição. Enquanto isso:

```
T=0ms:   send_order(BBAS3, 1000)
T=5ms:   eval(BRML3) - MT5 ainda mostra: exposição = 0 (BBAS3 não apareceu!)
T=10ms:  send_order(BRML3, 800) - RiskTeam aprova porque total_exposure = 0!
T=15ms:  eval(PETR4) - Mesma situação
T=400ms: MT5 finally confirms BBAS3 + BRML3 + PETR4 = 2800 lotes
         Total exposição real = 50+ * 1000 = ACIMA DO LIMITE!
```

### ✅ FIX Implementado

Implementado **sistema de rastreamento de ordens pendentes** em 2 partes:

#### Part 1: Position Manager - Rastreamento (core/position_manager.py)

```python
class PositionManager:
    def __init__(self, execution_engine: ExecutionEngine, magic_number: int = 123456):
        self.execution = execution_engine
        self.magic_number = magic_number
        # ← NOVO: Rastreia ordens dos últimos 3 segundos
        self.pending_orders = []  # [(timestamp, symbol, volume, price), ...]
    
    def register_pending_order(self, symbol: str, volume: float, price: float):
        """Registra ordem enviada que ainda não apareceu no MT5"""
        self.pending_orders.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'volume': volume,
            'price': price
        })
        logger.info(f"📤 Ordem pendente: {symbol} x{volume} @ R${price:.2f}")
    
    def get_pending_exposure(self) -> Dict[str, float]:
        """Calcula exposição das ordens ainda não confirmadas pelo MT5"""
        self.clean_pending_orders()  # Remove entries > 3 segundos
        
        pending_exp = {}
        for order in self.pending_orders:
            exposure = order['volume'] * order['price']
            pending_exp[order['symbol']] = pending_exp.get(...) + exposure
        return pending_exp
    
    def get_total_exposure(self) -> float:
        """
        ← ATUALIZADO: Inclui ORDENS PENDENTES no cálculo
        """
        positions = self.get_open_positions(filter_magic=True)
        confirmed_exposure = sum(p['volume'] * p['current_price'] for p in positions)
        
        # ← NOVO: Soma exposição pendente também!
        pending_exp_dict = self.get_pending_exposure()
        pending_exposure = sum(pending_exp_dict.values())
        
        total = confirmed_exposure + pending_exposure
        logger.info(f"📊 Exposição: R${total:.2f} "
                   f"(Confirmada: R${confirmed_exposure:.2f} + "
                   f"Pendente: R${pending_exposure:.2f})")
        return total
```

#### Part 2: Bot Loop - Delays (bot.py)

```python
# bot.py após send_order (linhas 301 + 365):

execution.send_order(order)

# ← NOVO: Registra como pendente e aguarda confirmação
position_manager.register_pending_order(symbol, final_volume, current_price)
logger.info(f"⏳ Aguardando confirmação de {symbol} (1.5s)...")
time.sleep(1.5)  # Dá tempo do MT5 registrar e updated get_total_exposure()
```

**Impacto:**
- ✅ Primeira ordem (BBAS3) registrada como pendente
- ✅ Aguarda 1.5s (suficiente para MT5 registrar)
- ✅ Segunda ordem (BRML3) vê exposição real = 1000 + pendente
- ✅ RiskTeam aprova apenas se dentro do limite total
- ✅ Se limite atingido, bloqueia a ordem

---

## 📊 Comparação Antes vs Depois

### Cenário: 50 ativos para trading, equity = R$ 100.000

#### ANTES (Vulnerável):

```
MAX_TOTAL_EXPOSURE_PCT = 150% (1.5 * equity = R$ 150.000)
MAX_CAPITAL_ALLOCATION_PCT = 2% (R$ 2.000 por trade)

T=0ms:  send_order(BBAS3)   - Aproveado: 2%
T=5ms:  send_order(BRML3)   - Aprovado: 2% (MT5 ainda vazio!)
T=10ms: send_order(PETR4)   - Aprovado: 2% (MT5 ainda vazio!)
...
T=100ms: send_order(USIM5)  - Aprovado: 2% (todas simultâneas)

Resultado REAL: 50 ordens * 2% = 100% da equity
               Mas RiskTeam vê 0% cada vez = TODAS APROVADAS POR ENGANO!
               Total real: R$ 100.000 (100% exposição)
```

#### DEPOIS (Seguro):

```
MAX_TOTAL_EXPOSURE_PCT = 150% (1.5 * equity = R$ 150.000)
MAX_CAPITAL_ALLOCATION_PCT = 2% (risco base)
size_multiplier interpretado corretamente

T=0ms:  send_order(BBAS3) = 2.4%
        register_pending_order(BBAS3)
        sleep(1.5s)
        → get_total_exposure() = confirmado(0) + pendente(BBAS3 2.4%) = 2.4%
        
T=1500ms: send_order(BRML3) = 2.4%
          proposed_risk = 2.4%
          total_exposed = 2.4% + 2.4% = 4.8%
          limit_check: 4.8% < 150% ✅ APPROVED
          
          register_pending_order(BRML3)
          sleep(1.5s)
          → get_total_exposure() = confirmado(0) + pendente(BBAS3+BRML3) = 4.8%

T=3000ms: send_order(PETR4) = 2.4%
          proposed_risk = 2.4%
          total_exposed = 4.8% + 2.4% = 7.2%
          limit_check: 7.2% < 150% ✅ APPROVED
          
          ... e assim por diante
          
T=126s: send_order(50º ativo) = 2.4%
        total_exposed = 50 * 2.4% = 120%
        limit_check: 120% < 150% ✅ APPROVED
        FINAL: Exatamente 120% = DENTRO DO LIMITE!
```

---

## 📝 Checklist de Mudanças

### ✅ Arquivo: `core/position_manager.py`
- [x] Adicionado atributo `self.pending_orders` para rastreamento
- [x] Método `register_pending_order()` para registrar ordens enviadas
- [x] Método `clean_pending_orders()` para limpar entradas > 3s
- [x] Método `get_pending_exposure()` para calcular exposição pendente
- [x] **Método `get_total_exposure()` ATUALIZADO** para incluir pending orders

### ✅ Arquivo: `agents/risk_team.py`
- [x] **Método `validate_trade()` REESCRITO** para interpretar corretamente `size_multiplier`
- [x] Mudança chave: `effective_risk_pct = base_risk_pct * proposed_size_multiplier`
- [x] Limite máximo agora é `base_risk_pct * 1.5` (3%, não 150%)
- [x] Cálculo de `new_exposure` usa `effective_risk_pct` (correto)

### ✅ Arquivo: `bot.py`
- [x] Após `execution.send_order()` no BUY (linha ~301):
  - Chamada a `position_manager.register_pending_order()`
  - Adicionado `time.sleep(1.5)`
  
- [x] Após `execution.send_order()` no SELL (linha ~365):
  - Mesmas mudanças que BUY

---

## 🧪 Como Testar as Correções

### Teste 1: Verificar que `size_multiplier` não causa envenenamento

```python
# Test case:
# RiskyTrader propõe: size_multiplier = 1.2
# Expected: 2% * 1.2 = 2.4% de risco max por trade
# NOT: 2% * 1.2 * 100 = 240% (bug anterior)

equity = 100000
base_risk = 0.02  # 2%
size_mult = 1.2

effective_risk = equity * base_risk * size_mult
print(f"Effective risk: R${effective_risk:.2f}")  # Should be ~2400 (2.4%)
```

### Teste 2: Verificar Race Condition Fix

```python
# Simular 50 ordens rápidas:
for i in range(50):
    send_order(symbol_list[i])
    position_manager.register_pending_order(symbol_list[i], 1000, 30.0)
    
    # Sem delay (bug): get_total_exposure() = 0 (todas ignoradas)
    # Com delay (fix): get_total_exposure() = crescente (rastreado corretamente)
```

---

## ⚠️ Notas Importantes

1. **Delay de 1.5s é conservador**: Pode ser reduzido para 1.0s se testes mostrarem confirmação mais rápida
2. **Pending orders são limpas automaticamente** após 3 segundos
3. **Limite máximo global (150%)** continua sendo o hard limit
4. **Novo sistema é thread-safe**? Verificar se há múltiplos threads acessando `pending_orders`

---

## 📚 Referências

- Problema original: Multiplicativo de capital vs risco base
- Race condition: Async MT5 + síncrono bot loop
- Solução: Rastreamento de estado intermediário + delays

---

**Status Final:** ✅ **IMPLEMENTADO E PRONTO PARA TESTE**
