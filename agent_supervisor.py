import os
import time
import logging
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Carrega variaveis do .env (incluindo a GEMINI_API_KEY)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent_supervisor")

# Bibliotecas do projeto
import database
from telegram_handler import send_telegram_alert

# Configuracoes do Gemini (Flash Latest para maior compatibilidade e estabilidade)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-flash-latest"  # Estavel e rapido no Free Tier
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

LOG_FILE = "xp3_bot.log"

def fetch_daily_data():
    today_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Coletando trades para a data: {today_str}")

    try:
        trades_df = database.get_trades_by_date(today_str)
        if trades_df.empty:
            return "Nenhum trade realizado hoje."

        total_trades = len(trades_df)
        trades_df['pnl_money'] = trades_df['pnl_money'].fillna(0.0)
        
        wins = len(trades_df[trades_df['pnl_money'] > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = trades_df['pnl_money'].sum()
        
        resumo_ativo = trades_df.groupby('symbol')['pnl_money'].sum().sort_values()
        pior_ativo = resumo_ativo.index[0] if len(resumo_ativo) > 0 else "N/A"
        melhor_ativo = resumo_ativo.index[-1] if len(resumo_ativo) > 0 else "N/A"

        return f"""
[DADOS OPERACIONAIS - {today_str}]
- Total de Trades: {total_trades}
- Vitorias: {wins} | Derrotas: {losses}
- Winrate: {win_rate:.1f}%
- Resultado Financeiro Total: R$ {total_pnl:.2f}
- Pior Ativo: {pior_ativo} (R$ {resumo_ativo.get(pior_ativo, 0):.2f})
- Melhor Ativo: {melhor_ativo} (R$ {resumo_ativo.get(melhor_ativo, 0):.2f})
    """
    except Exception as e:
        logger.error(f"Erro ao buscar dados do banco: {e}")
        return "Erro ao acessar base de dados de trades."

def fetch_daily_errors():
    today_str = datetime.now().strftime("%Y-%m-%d")
    errors = []
    try:
        if not os.path.exists(LOG_FILE):
            return "Arquivo xp3_bot.log nao encontrado."
            
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-2000:]
            for line in lines:
                if today_str in line and ("ERROR" in line or "Exception" in line):
                    errors.append(line.strip())
                    
        return "[LOGS DE ERRO DO DIA]\n" + ("\n".join(errors[-10:]) if errors else "Nenhum erro grave registrado hoje.")

    except Exception as e:
        logger.error(f"Erro ao ler log: {e}")
        return f"Erro ao tentar ler logs: {e}"

def call_gemini_council(dados, erros, retries=3):
    """Uma unica chamada consolidada para o Conselho de Agentes (evita 429 no Free Tier)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "SUA_CHAVE_AQUI":
        return "Erro: API Key do Gemini nao configurada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    
    prompt = f"""
Voce e o Conselho Consultivo de IA do bot de trading XP3.
Analise os dados abaixo sob tres perspectivas (Quant, Infra e Risco) e gere um relatorio final unificado.

DADOS DO DIA:
{dados}

LOGS DE ERRO:
{erros}

SUAS PERSONAS:
1. Analista Quantitativo: Foca em lucro, fator de lucro e winrate.
2. Engenheiro de Infra: Foca em erros de conexao, estabilidade e timeouts.
3. CRO (Chief Risk Officer): Cético, foca em perdas e recomendacao de parada.

OBJETIVO:
Crie uma MENSAGEM FINAL PARA O TELEGRAM do CEO (em Portugues Brasileiro).
Seja executivo, direto e use emojis. Separe em:
📊 Visao Quant:
🛠️ Infraestrutura:
🛡️ Gestao de Risco:
🎯 Veredito Final (Acao imediata):
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "topP": 0.8, "topK": 40}
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 429:
                wait_time = (attempt + 1) * 30
                logger.warning(f"Limite 429 atingido. Esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return "Erro: Resposta vazia do Gemini."
        except Exception as e:
            if attempt == retries - 1:
                return f"Falha na IA: {e}"
            time.sleep(5)
    return "Erro nas tentativas de conexao."

def main():
    logger.info("🤖 Iniciando CONSELHO CONSOLIDADO XP3 (GEMINI API)...")
    
    dados = fetch_daily_data()
    erros = fetch_daily_errors()
    
    # Chama o conselho unificado em uma unica requisicao (Muito melhor para Free Tier)
    cumulative_pnl = database.get_cumulative_pnl(since_date="2026-03-20")
    
    # Busca o saldo inicial do config_risk se possível, senão 2000
    import config_risk
    initial_v_equity = getattr(config_risk, "HOTFIX_SMALL_ACCOUNT", {}).get("force_virtual_equity", 2000.0)
    total_wallet = initial_v_equity + cumulative_pnl

    # Adiciona contexto de acumulado aos dados passados para a IA
    dados_com_contexto = dados + f"\n\n[RESUMO ACUMULADO]\n- PnL Total (desde o início): R$ {cumulative_pnl:.2f}\n- Saldo Total da Carteira: R$ {total_wallet:.2f}"

    final_telegram_msg = call_gemini_council(dados_com_contexto, erros)
    
    header = "🔮 <b>CONSELHO DE AGENTES IA (CONSULTA UNICA)</b>\n\n"
    telegram_message = header + final_telegram_msg

    logger.info("Enviando veredito final do Conselho para o Telegram...")
    success = send_telegram_alert(telegram_message)
    
    if success:
        logger.info("✅ Conselho concluido com sucesso!")
    else:
        logger.error("❌ Falha ao enviar para o Telegram.")

if __name__ == "__main__":
    main()
