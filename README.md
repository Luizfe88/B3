# XP3v5 - Bot de Trading Quantitativo para B3

## 📊 Visão Geral

XP3v5 é um bot de trading quantitativo avançado desenvolvido para operar na B3 (Bolsa de Valores Brasileira). O sistema utiliza machine learning, análise técnica e gestão de risco adaptativa para tomar decisões de trading automatizadas.

## 🚀 Principais Funcionalidades

### Inteligência Artificial & Machine Learning
- **Modelo de Predição**: Random Forest com análise de 42 features técnicas
- **Score de Confiança**: Probabilidade de 0-1 para direção de movimento
- **Otimização Adaptativa**: Parâmetros ajustados dinamicamente baseado em performance
- **Universe Builder**: Seleção automática dos melhores ativos para operar

### Gestão de Risco Avançada
- **Kelly Criterion**: Cálculo dinâmico de tamanho de posição
- **Drawdown Control**: Limite máximo de 3% ao dia
- **Setorização**: Limite de exposição por setor (máx. 3 ativos)
- **Circuit Breaker**: Pausa automática em condições adversas
- **Anti-Chop**: Cooldown após perdas consecutivas

### Filtros de Mercado
- **IBOV Trend Analysis**: Adaptação estratégica baseada na tendência do Ibovespa
- **ADX Filter**: Confirmação de força de tendência
- **Volume Analysis**: Análise de volume relativo e absoluto
- **Spread Control**: Filtragem de ativos com spread elevado
- **Market Hours**: Operação apenas em horário definido (10:20 - 16:40)

### Sistema de Logs & Monitoramento
- **Rejection Logger**: Registro detalhado de sinais rejeitados
- **Daily Analysis**: Análise diária de performance
- **Telegram Integration**: Notificações em tempo real
- **Dashboard Web**: Interface de monitoramento em Streamlit

## 📁 Estrutura do Projeto

```
xp3v5/
├── bot.py                    # Core do bot de trading
├── config.py                 # Configurações do sistema
├── database.py              # Gerenciamento de dados SQLite
├── utils.py                 # Funções utilitárias
├── risk_manager.py          # Gestão de risco e Kelly Criterion
├── ml_signals.py            # Geração de sinais ML
├── ml_optimizer.py          # Otimização de parâmetros ML
├── universe_builder.py      # Construção do universo de ativos
├── rejection_logger.py      # Registro de sinais rejeitados
├── daily_analysis_logger.py # Análise diária de performance
├── telegram_handler.py      # Integração com Telegram
├── dashboard.py             # Dashboard web em Streamlit
├── backtest.py              # Sistema de backtesting
├── optimizer.py             # Otimização de estratégias
├── tests/                   # Testes unitários
├── logs/                    # Arquivos de log
├── data/                    # Dados históricos
└── optimizer_output/        # Resultados de otimização
```

## 🔧 Instalação & Configuração

### Pré-requisitos
- Python 3.8+
- MetaTrader 5 (para dados em tempo real)
- Conta na B3 com API de corretora compatível

### Instalação
```bash
# Clone o repositório
git clone https://github.com/Luizfe88/B3.git
cd B3

# Instale as dependências
pip install -r requirements.txt

# Configure o ambiente
cp config.example.py config.py
# Edite config.py com suas credenciais e preferências
```

### Configuração Inicial
1. **MetaTrader 5**: Instale e configure MT5 com sua corretora
2. **API Credentials**: Configure credenciais da corretora em `config.py`
3. **Telegram Bot**: Crie um bot no Telegram para notificações
4. **Parâmetros**: Ajuste os parâmetros iniciais no arquivo de configuração

## 📈 Como Funciona

### 1. Análise de Mercado
- Coleta dados de 60 ativos selecionados
- Calcula 42 indicadores técnicos
- Gera scores de confiança via Random Forest
- Filtra ativos baseado em critérios de qualidade

### 2. Tomada de Decisão
- **Entrada Long**: Score ML > 0.60, ADX > 15, sinal técnico confirmado
- **Entrada Short**: Score ML > 0.60, ADX > 15, sinal técnico confirmado
- **Exit**: TP/SL dinâmico ou sinal contrário
- **Position Size**: Kelly Criterion com limite máximo de 20%

### 3. Gestão de Risco
- **Por Trade**: Máximo 2% do capital por operação
- **Por Dia**: Máximo 3% de drawdown diário
- **Por Setor**: Máximo 3 ativos por setor
- **Total**: Máximo 15 posições simultâneas

### 4. Monitoramento
- Logs detalhados de todas as operações
- Dashboard web em tempo real
- Notificações Telegram para eventos importantes
- Análise diária de performance

## 🎯 Performance & Resultados

### Métricas Chave
- **Sharpe Ratio**: Target > 1.5
- **Maximum Drawdown**: < 15%
- **Win Rate**: Target 55-65%
- **Profit Factor**: Target > 1.3
- **Kelly Efficiency**: Otimização contínua

### Otimização
- Otimização diária de parâmetros
- Walk-forward analysis
- Monte Carlo simulation
- Stress testing em diferentes cenários

## 🛡️ Segurança & Compliance

### Controles de Segurança
- **Circuit Breaker**: Pausa em perdas consecutivas
- **Market Hours**: Operação apenas em horário permitido
- **Blacklist**: Ativos proibidos automaticamente
- **Position Limits**: Limites rígidos por posição e total

### Auditoria & Compliance
- Logs completos de todas as operações
- Rastreabilidade total das decisões
- Conformidade com regulamentações da B3
- Relatórios automáticos de performance

## 🚀 Executando o Bot

### Modo Produção
```bash
python bot.py
```

### Modo Teste (Paper Trading)
```bash
python bot.py --paper
```

### Backtesting
```bash
python backtest.py --start-date 2024-01-01 --end-date 2024-12-31
```

### Otimização
```bash
python optimizer.py --symbols WIN* --days 60
```

## 📊 Dashboard & Monitoramento

### Dashboard Web
Acesse o dashboard em: `http://localhost:8501`

### Métricas Disponíveis
- Performance em tempo real
- Posições abertas
- Histórico de trades
- Análise de setores
- Estatísticas de risco
- Logs de rejeições

## 🧪 Testes & Validação

### Testes Unitários
```bash
pytest tests/
```

### Validação de Sinais
```bash
python test_validation.py
```

### Teste de Universo
```bash
python test_universe_builder.py
```

## 📋 Requisitos do Sistema

### Hardware Mínimo
- CPU: 4 cores
- RAM: 8GB
- Disco: 50GB livres
- Internet: Conexão estável

### Software
- Python 3.8+
- MetaTrader 5
- SQLite 3.x
- Streamlit (para dashboard)

## 🔗 Integrações

### MetaTrader 5
- Dados em tempo real
- Execução de ordens
- Gestão de posições

### Telegram
- Notificações instantâneas
- Comandos remotos
- Status do sistema

### APIs de Dados
- Yahoo Finance (dados históricos)
- Alpha Vantage (fundamentalista)
- B3 (dados de mercado)

## 📚 Documentação Adicional

### Guias Detalhados
- [Configuração Inicial](docs/setup.md)
- [Estratégias de Trading](docs/strategies.md)
- [Gestão de Risco](docs/risk-management.md)
- [API Reference](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)

### Vídeos & Tutoriais
- Configuração passo a passo
- Estratégias de otimização
- Interpretação de métricas
- Gestão de risco avançada

## 🤝 Contribuindo

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

### Guidelines
- Siga o padrão de código existente
- Adicione testes para novas funcionalidades
- Documente mudanças significativas
- Respeite as regras de gestão de risco

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ⚠️ Disclaimer

**Aviso Importante**: Este é um sistema de trading automatizado que envolve risco significativo de perda. Nenhuma garantia é oferecida quanto à performance futura. Use por sua conta e risco.

- **Risco de Perda**: Você pode perder parte ou todo o seu capital
- **Teste Antes**: Sempre teste em modo paper trading antes de usar capital real
- **Gestão de Risco**: Nunca opere com mais do que pode perder
- **Acompanhamento**: Monitore o sistema constantemente
- **Regulamentação**: Certifique-se de estar em conformidade com as regulamentações locais

## 📞 Suporte & Contato

### Issues & Bugs
Reporte bugs e problemas em: [Issues](https://github.com/Luizfe88/B3/issues)

### Features & Sugestões
Sugira melhorias em: [Discussions](https://github.com/Luizfe88/B3/discussions)

### Comunidade
Participe da comunidade em: [Discussions](https://github.com/Luizfe88/B3/discussions)

---

**Desenvolvido com ❤️ para a comunidade de trading brasileira**