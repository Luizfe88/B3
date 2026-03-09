
import database
from adaptive_intelligence import adaptive_intelligence
import telegram_handler
import logging

# Configura log para ver o que acontece
logging.basicConfig(level=logging.INFO)

def generate_report():
    print("--- Syncing and Generating B3 Report ---")
    
    # 1. Garante que os dados estão sincronizados e limpos (B3 only)
    database.sync_trades_from_mt5()
    database.cleanup_invalid_symbols()
    
    # 2. Força a coleta de métricas no Adaptive Intelligence
    # Como o bot real coleta a cada 15 min, aqui forçamos uma coleta manual
    adaptive_intelligence.running = True # Força para o report sair completo
    metrics = adaptive_intelligence._collect_current_metrics()
    if metrics:
        adaptive_intelligence.metrics_history.append(metrics)
        print(f"Metrics collected. Winrate 24h: {metrics.winrate_24h:.1%}")
    else:
        print("Failed to collect metrics.")
        return

    # 3. Constrói o relatório formatado
    report_msg = telegram_handler.build_learning_report()
    
    # Salva no arquivo para garantir integridade
    save_path = telegram_handler.save_learning_report_to_file(report_msg)
    
    print("\n--- LEARNING REPORT UPDATED (B3 Only) ---")
    # Tenta imprimir limpando o que o console não aguenta
    import re
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', report_msg)
    clean_text = clean_text.replace("<b>", "").replace("</b>", "")
    print(clean_text)
    print("--------------------------------------------------")
    print(f"Report also saved to: {save_path}")

if __name__ == "__main__":
    generate_report()
