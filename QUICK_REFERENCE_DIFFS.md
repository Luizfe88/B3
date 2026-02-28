# 📝 QUICK REFERENCE: Diffs Exatos das Mudanças

Copie e cole para referência rápida do que foi mudado.

---

## 1️⃣ `core/position_manager.py` - Rastreamento de Ordens Pendentes

### MUDANÇA 1A: Adicionar atributo no `__init__`
```python
# ADICIONADO em __init__:
self.pending_orders = []  # Lista de (timestamp, symbol, volume, price)
```

### MUDANÇA 1B: Novos Métodos (adicionar classe)
```python
def register_pending_order(self, symbol: str, volume: float, price: float):
    """
    Registra uma ordem que foi enviada mas pode não estar refletida no MT5 ainda.
    Importante para evitar race condition de múltiplas ordens simultâneas.
    """
    now = datetime.now()
    self.pending_orders.append({
        'timestamp': now,
        'symbol': symbol,
        'volume': volume,
        'price': price
    })
    logger.info(f"📤 Ordem pendente registrada: {symbol} x{volume} @ R${price:.2f}")

def clean_pending_orders(self):
    """
    Remove ordens pendentes que já têm mais de 3 segundos.
    Assume que o MT5 já atualizou sua posição até então.
    """
    now = datetime.now()
    cutoff = now - timedelta(seconds=3)
    
    before_count = len(self.pending_orders)
    self.pending_orders = [
        order for order in self.pending_orders 
        if order['timestamp'] > cutoff
    ]
    
    removed = before_count - len(self.pending_orders)
    if removed > 0:
        logger.debug(f"🧹 Limpas {removed} ordens pendentes (> 3 segundos)")

def get_pending_exposure(self) -> Dict[str, float]:
    """
    Calcula exposição de ordens pendentes por símbolo.
    Retorna dict: {symbol: exposure_in_reais}
    """
    self.clean_pending_orders()
    
    pending_exp = {}
    for order in self.pending_orders:
        symbol = order['symbol']
        exposure = order['volume'] * order['price']
        pending_exp[symbol] = pending_exp.get(symbol, 0.0) + exposure
    
    if pending_exp:
        logger.debug(f"⏳ Exposição pendente: {pending_exp}")
    
    return pending_exp
```

### MUDANÇA 1C: Atualizar `get_total_exposure()`
```python
# ANTES:
def get_total_exposure(self) -> float:
    """
    Calcula a exposição financeira total (soma de todas as posições abertas).
    """
    positions = self.get_open_positions(filter_magic=True)
    total = 0.0
    for p in positions:
        total += p['volume'] * p['current_price']
    if total > 0:
        logger.info(f"📊 Exposição atual: R$ {total:.2f} (em {len(positions)} posições)")
    return total

# DEPOIS:
def get_total_exposure(self) -> float:
    """
    Calcula a exposição financeira total (soma de todas as posições abertas + PENDENTES).
    ⚠️ ATUALIZADO: Inclui ordens recentemente enviadas que ainda não aparecem no MT5.
    
    Isto resolve o problema de race condition onde múltiplas ordens são enviadas
    antes do MT5 registrar a primeira posição.
    """
    # Posições confirmadas
    positions = self.get_open_positions(filter_magic=True)
    confirmed_exposure = sum(p['volume'] * p['current_price'] for p in positions)
    
    # Posições pendentes (últimos 3 segundos)
    pending_exp_dict = self.get_pending_exposure()
    pending_exposure = sum(pending_exp_dict.values())
    
    total = confirmed_exposure + pending_exposure
    
    # Log detalhado para debug
    if total > 0:
        logger.info(f"📊 Exposição Total: R${total:.2f} "
                   f"(Confirmada: R${confirmed_exposure:.2f} + Pendente: R${pending_exposure:.2f})")
        
    return total
```

---

## 2️⃣ `agents/risk_team.py` - Interpretação Correta de Size Multiplier

### MUDANÇA 2: Reescrever método `validate_trade()`

```python
# ANTES:
def validate_trade(self, symbol: str, proposal: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida se o trade proposto respeita os limites de risco.
    Considera:
    - Drawdown máximo
    - Correlação
    - Exposição setorial
    - Tamanho da posição
    """
    logger.info(f"👮 [{self.name}] Validando risco para {symbol}...")
    
    # Simulação de verificação
    max_position_size = 1.5 # 150% do lote base (flexibilidade)
    proposed_size = proposal.get('size_multiplier', 0.0)
    
    if proposed_size > max_position_size:
        logger.warning(f"❌ [{self.name}] Tamanho excessivo ({proposed_size:.2%}). Ajustando.")
        proposal['size_multiplier'] = max_position_size
        proposal['adjusted'] = True
        
    # 1. Limite Global de Exposição Financeira
    total_exposure = market_context.get('total_exposure', 0.0)
    equity = market_context.get('equity', 1000.0)
    max_exposure = equity * config.MAX_TOTAL_EXPOSURE_PCT
    
    # Estima exposição da nova ordem
    current_price = market_context.get('price', 0.0)
    new_exposure = (equity * config.MAX_CAPITAL_ALLOCATION_PCT * proposed_size)
    
    # ... resto do código ...

# DEPOIS:
def validate_trade(self, symbol: str, proposal: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida se o trade proposto respeita os limites de risco.
    Considera:
    - Drawdown máximo
    - Correlação
    - Exposição setorial
    - Tamanho da posição
    
    ⚠️ CORRIGIDO: size_multiplier agora multiplica o RISCO BASE (2%), não o capital total
    Ex: RiskyTrader propõe 1.2 -> 2% * 1.2 = 2.4% da conta.
    """
    logger.info(f"👮 [{self.name}] Validando risco para {symbol}...")
    
    # CORREÇÃO CRÍTICA: size_multiplier é multiplicador do risco base, não do capital
    # Risco base é definido em config.MAX_CAPITAL_ALLOCATION_PCT (padrão: 2%)
    base_risk_pct = config.MAX_CAPITAL_ALLOCATION_PCT  # Ex: 0.02 (2%)
    proposed_size_multiplier = proposal.get('size_multiplier', 0.0)
    
    # Calcula o risco efetivo desta proposta
    effective_risk_pct = base_risk_pct * proposed_size_multiplier
    
    # Limita o risco máximo que um trader individual pode tomar
    # SafeTrader: 0.8 * 2% = 1.6%, RiskyTrader: 1.2 * 2% = 2.4%, etc
    max_effective_risk = base_risk_pct * 1.5  # Máximo 150% do risco base = 3% por trade
    
    if effective_risk_pct > max_effective_risk:
        logger.warning(f"❌ [{self.name}] Risco efetivo excessivo ({effective_risk_pct:.2%} > {max_effective_risk:.2%}). Ajustando.")
        # Ajusta o multiplicador de tamanho para atingir o máximo permitido
        proposal['size_multiplier'] = max_effective_risk / base_risk_pct
        proposal['adjusted'] = True
        effective_risk_pct = max_effective_risk
        
    # Agora verifica exposição FINANCEIRA com base no risco ajustado
    equity = market_context.get('equity', 1000.0)
    total_exposure = market_context.get('total_exposure', 0.0)
    max_exposure = equity * config.MAX_TOTAL_EXPOSURE_PCT
    
    # Estima exposição da nova ordem baseada no risco efetivo
    new_exposure = equity * effective_risk_pct
    
    if (total_exposure + new_exposure) > max_exposure:
         logger.warning(f"❌ [{self.name}] Limite Global de Exposição atingido! "
                        f"(Atual: R${total_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_exposure:.2f} | Equity: R${equity:.2f})")
         return {"approved": False, "reason": f"Exposure Limit (Eq: {equity:.0f})"}

    # 2. Throttle (Limite de novas posições por hora)
    recent_entries = market_context.get('recent_entries_count', 0)
    if recent_entries >= config.MAX_NEW_POSITIONS_PER_HOUR:
         logger.warning(f"❌ [{self.name}] Throttle ativado! ({recent_entries} novas posições na última hora)")
         return {"approved": False, "reason": "Entry Throttle Active"}

    # 3. Limite de Exposição Setorial (25% do Capital)
    sector = config.SECTOR_MAP.get(symbol, "OUTROS")
    current_sector_exposure = market_context.get(f'sector_exposure_{sector}', 0.0)
    max_sector_exposure = equity * config.MAX_SECTOR_ALLOCATION_PCT
    
    if (current_sector_exposure + new_exposure) > max_sector_exposure:
         logger.warning(f"❌ [{self.name}] Limite de Setor ({sector}) atingido! "
                        f"(Atual: R${current_sector_exposure:.2f} + Novo: R${new_exposure:.2f} > Limite: R${max_sector_exposure:.2f})")
         return {"approved": False, "reason": f"Sector Limit ({sector})"}

    # 4. Market Regime Guard (Filtro de Pânico)
    if config.MARKET_REGIME_FILTER:
        ibov_trend = market_context.get('ibov_trend', 'neutral')
        if ibov_trend == 'bearish_extreme' and proposal.get('action') == 'BUY':
             logger.warning(f"⚠️ [{self.name}] Market Regime Guard: Bloqueando COMPRA em pânico.")
             return {"approved": False, "reason": "Market Panic Mode"}

    # 4. Verificação de correlação com IBOV
    corr = market_context.get('ibov_correlation', 0.5)
    if corr > 0.8 and self.tolerance < 0.5:
        logger.warning(f"⚠️ [{self.name}] Alta correlação com mercado em queda. Bloqueando.")
        return {"approved": False, "reason": "High correlation risk"}
        
    return {"approved": True, "adjusted_proposal": proposal}
```

---

## 3️⃣ `bot.py` - Adicionar Delays e Rastreamento

### MUDANÇA 3A: Após `execution.send_order()` no BUY (~linha 296)

```python
# ANTES:
execution.send_order(order)

# DEPOIS:
execution.send_order(order)

# ⏱️ FIX RACE CONDITION: Registra ordem pendente e aguarda confirmação no MT5
position_manager.register_pending_order(symbol, final_volume, current_price)
logger.info(f"⏳ Aguardando confirmação de {symbol} no MT5 (1.5s)...")
time.sleep(1.5)  # Dá tempo para o MT5 registrar a posição
```

### MUDANÇA 3B: Após `execution.send_order()` no SELL (~linha 358)

```python
# ANTES:
execution.send_order(order)

# DEPOIS:
execution.send_order(order)

# ⏱️ FIX RACE CONDITION: Registra ordem pendente e aguarda confirmação no MT5
position_manager.register_pending_order(symbol, final_volume, current_price)
logger.info(f"⏳ Aguardando confirmação de {symbol} no MT5 (1.5s)...")
time.sleep(1.5)  # Dá tempo para o MT5 registrar a posição
```

---

## ✅ Verificação de Mudanças

Para validar que tudo está correto:

```bash
# Verificar tamanho dos arquivos (deve ter aumentado um pouco)
ls -lh core/position_manager.py agents/risk_team.py bot.py

# Executar testes de validação
python test_security_fixes.py

# Procurar por "pending_orders" para confirmar implementação
grep -n "pending_orders" core/position_manager.py
grep -n "register_pending_order" bot.py
grep -n "effective_risk_pct" agents/risk_team.py
```

---

## 🔍 Como Verificar se Está Funcionando (Logs)

Após iniciar o bot, você deve ver nos logs:

```
INFO - 📤 Ordem pendente registrada: BBAS3 x1000 @ R$30.50
INFO - ⏳ Aguardando confirmação de BBAS3 no MT5 (1.5s)...
INFO - 📊 Exposição Total: R$30500.00 (Confirmada: R$0.00 + Pendente: R$30500.00)
INFO - ✅ [FundManager] Decisão Final para BRML3: BUY (Size: 2.40%)
```

---

**Última Atualização:** 28/02/2026
