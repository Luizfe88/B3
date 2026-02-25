# XP3v5 - Framework de Trading Multi-Agente para B3

## 📊 Visão Geral

XP3v5 é um sistema de trading quantitativo de última geração projetado para a B3 (Bolsa de Valores Brasileira). Ele implementa a arquitetura **TradingAgents-B3**, simulando uma mesa de trading profissional com agentes autônomos especializados trabalhando em conjunto.

## 🧠 Arquitetura Multi-Agente (TradingAgents-B3)

O sistema opera através de um pipeline de decisão hierárquico e colaborativo:

### 1. Analyst Team (Equipe de Análise)
4 agentes especialistas operam em paralelo:
- **Fundamental Analyst**: Analisa balanços, múltiplos e dados macroeconômicos.
- **Sentiment Analyst**: Monitora notícias e redes sociais brasileiras.
- **Technical Analyst**: Utiliza modelos de Machine Learning (Random Forest) treinados em dados reais.
- **OrderFlow Analyst**: Analisa fluxo de ordens (Tape Reading) e agressão de mercado.

### 2. Researcher Team (Debate)
- Realiza um debate "Bull vs Bear" obrigatório antes de qualquer decisão.
- Gera um consenso baseado em evidências conflitantes.

### 3. Trader Agents (Propostas)
3 perfis de traders propõem ações baseadas no consenso:
- **Risky Trader**: Busca oportunidades de alto retorno/risco.
- **Neutral Trader**: Equilibra risco e retorno.
- **Safe Trader**: Prioriza proteção de capital.

### 4. Risk Management Team (Guardiões)
- Valida todas as propostas contra limites rígidos de risco.
- Controla drawdown, exposição setorial e correlação com IBOV.

### 5. Fund Manager (Decisor Final)
- Orquestra todo o fluxo e toma a decisão final de execução.
- Executa ordens através da camada de infraestrutura robusta.

## 🚀 Principais Funcionalidades

### Inteligência Artificial Real
- **Feature Store**: Engenharia de 42 features técnicas baseadas em dados reais do MT5.
- **ML Training**: Pipeline de treinamento (RandomForest/XGBoost) com validação temporal (Walk-Forward).
- **Order Flow**: Análise de pressão de compra/venda em tempo real.

### Gestão de Risco Profissional
- **Kelly Criterion**: Dimensionamento de posição dinâmico.
- **Circuit Breakers**: Pausa automática em alta volatilidade.
- **Setorização**: Limite de exposição por setor da economia.

## 📁 Estrutura do Projeto

```
xp3v5/
├── agents/                      # Equipes de Agentes
│   ├── analyst_team.py          # Analistas (Fund, Sent, Tech, OrderFlow)
│   ├── researcher_team.py       # Debate Bull vs Bear
│   ├── trader_agents.py         # Traders (Risky, Neutral, Safe)
│   ├── risk_team.py             # Risk Guardians
│   └── fund_manager.py          # Decisor Final
├── core/                        # Infraestrutura
│   ├── execution.py             # Camada de Execução MT5
│   └── position_manager.py      # Gestão de Portfólio
├── ml/                          # Machine Learning
│   ├── feature_store.py         # Engenharia de Features
│   ├── training.py              # Treinamento de Modelos
│   └── prediction.py            # Inferência Online
├── bot.py                       # Entry Point (Orquestrador)
└── config.py                    # Configurações do sistema
```

## 🔧 Instalação & Configuração

### Pré-requisitos
- Python 3.10+
- MetaTrader 5 (Terminal instalado e logado)
- Conta B3 (Demo ou Real)

### Instalação
```bash
# Clone o repositório
git clone https://github.com/Luizfe88/B3.git
cd B3

# Instale as dependências
pip install -r requirements.txt

# Configure o ambiente
cp config.example.py config.py
```

### Executando o Bot
```bash
python bot.py
```

## 📈 Machine Learning Workflow

1. **Coleta de Dados**: O `FeatureStore` extrai dados históricos do MT5.
2. **Treinamento**: Execute `python ml/training.py` para treinar modelos para seus ativos.
3. **Inferência**: O `TechnicalAnalyst` carrega os modelos automaticamente durante a operação.

## 🛡️ Aviso de Risco

Este software é uma ferramenta de pesquisa e automação. Trading envolve risco significativo de perda financeira.
- Teste exaustivamente em conta DEMO.
- Nunca opere dinheiro que você não pode perder.
- O autor não se responsabiliza por perdas financeiras.

## 📄 Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes.
