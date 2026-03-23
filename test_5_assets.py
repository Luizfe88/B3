import os
import sys

# 1. Defina os ativos que deseja testar (separados por vírgula, sem espaços)
# Exemplo com 5 ativos:
os.environ["XP3_TARGET_SYMBOLS"] = "PETR4,VALE3,ITUB4,BBDC4,EQTL3"

# 2. Configure outras variáveis de ambiente se necessário
os.environ["XP3_SANDBOX"] = "1"        # 1 para teste rápido, 0 para produção completa
os.environ["XP3_LOAD_ALL_MT5"] = "0"   # Usa apenas os ativos acima

# 3. Importa e executa o otimizador
try:
    from otimizador_semanal import run_optimizer
    
    if __name__ == "__main__":
        print(f"[START] Iniciando teste de calibração para: {os.environ['XP3_TARGET_SYMBOLS']}")
        run_optimizer()
        print("✅ Teste concluído!")
except ImportError:
    print("❌ Erro: Certifique-se de que o script está na pasta raiz do projeto.")
except Exception as e:
    print(f"[ERROR] Ocorreu um erro: {e}")
