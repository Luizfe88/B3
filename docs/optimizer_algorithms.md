# Suite de Otimização: Arquitetura e Algoritmos

## Visão Geral
- Componentes principais: optimizer.py, optimizer_optuna.py, ml_optimizer.py.
- Objetivo: otimizar parâmetros de estratégia e treinar modelos ML com validação cruzada e seleção de hiperparâmetros.
- Reprodutibilidade: seeds fixas (42) em todos os métodos estocásticos.

## Pipelines e Validação
- Treinamento ML usa Pipeline(StandardScaler, Estimador) com StratifiedKFold(5) e Grid/RandomizedSearchCV.
- Métricas: accuracy e log loss; feature importance salva quando disponível.

## Algoritmos de Otimização
- Gradient Descent (GD): gradiente numérico com diferenças finitas; passos adaptativos; restrições e arredondamento para domínios inteiros.
- Genetic Algorithms (GA): população inicial aleatória; seleção por score; crossover uniforme; mutação probabilística; limites e coerência aplicados.
- Simulated Annealing (SA): vizinhança com perturbações discretas/contínuas; aceitação por critério de Boltzmann; resfriamento linear.

## Espaço de Parâmetros
- ema_short: [5, 20], ema_long: [24, 144]
- rsi_period: [7, 21], adx_period: [10, 20]
- adx_threshold: [5.0, 30.0], rsi_low: [20.0, 45.0], rsi_high: [55.0, 80.0]
- mom_min: [0.0, 0.005], use_rsi/use_adx: {0,1}, exit_max_bars: [0, 200]
- Invariantes: ema_short < ema_long; rsi_low < rsi_high.

## Função Objetivo
- Avaliada via evaluate_params_wfo com janelas IS/OOS e métricas híbridas: score = mean_oos − 0.6·max_dd − 0.2·std_oos.
- Filtros: MIN_TRADES_OOS e MAX_DD_OOS.

## Visualização
- compare_optimizers gera comparação de score e tempo; salva em optimizer_output/comparison_<SYMBOL>.png se matplotlib disponível.

## Testes
- tests/test_optimizers.py valida GD, GA, SA e pipeline ML.
- Dataset sintético com OHLC e volume; checagens de finitude e execução.

## Integração Optuna
- optimizer_optuna.py utiliza TPE com seed=42 e validação temporal (TimeSeriesSplit).
- Hiperparâmetros e probabilidade ML opcionais; retorno com best_params, best_score e status.

## Boas Práticas
- Sem comentários no código de produção; documentação separada.
- Logging informativo e defensivo; salvamento de artefatos e parâmetros.
- Falhas de dependências tratadas com fallbacks.
