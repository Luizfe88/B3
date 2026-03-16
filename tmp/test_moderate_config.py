import sys
import yaml

print("--- VALIDANDO ARQUIVOS APÓS REFATORACAO MODERADA ---")

try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    print(f"Kelly Fraction Base: {config_data['kelly_fraction']['base']} (Expected: 0.50)")
    print(f"Max Ruin Probability: {config_data['kelly_fraction']['max_ruin_probability']} (Expected: 0.01)")
    print(f"Max Losses Default: {config_data['risk_limits']['max_losses_per_symbol_default']} (Expected: 3)")
    print(f"Break Even Trigger: {config_data['risk_limits']['break_even_trigger_brl']} (Expected: 50.0)")
    print(f"ML Confidence Base: {config_data['ml_model']['confidence_base']} (Expected: 0.60)")
    
    import config_risk
    print(f"\nMax Consecutive Losses: {config_risk.CIRCUIT_BREAKERS['max_consecutive_losses']} (Expected: 4)")
    print(f"Intraday DD Limit: {config_risk.CIRCUIT_BREAKERS['intraday_dd_limit']} (Expected: 0.15)")
    print(f"Prefilter DD Max: {config_risk.MARKOWITZ_RULES['prefilter_dd_max']} (Expected: 0.70)")
    
    import ml_signals
    pred = ml_signals.MLSignalPredictor()
    print(f"\nMLSignalPredictor Base Threshold: {pred.base_threshold:.2f} (Expected: 0.60)")
    print(f"MLSignalPredictor Dynamic Threshold: {pred.get_dynamic_threshold('TEST3'):.2f} (Expected: ~0.60)")

    print("\n✅ TODAS AS VARIÁVEIS CARREGADAS COM SUCESSO.")
    
except Exception as e:
    print(f"❌ ERRO DURANTE VALIDAÇÃO: {e}")
    sys.exit(1)
