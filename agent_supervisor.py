import os
import time
import logging
import requests
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Carrega variaveis do .env (incluindo a GEMINI_API_KEY)
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent_supervisor")

# Bibliotecas do projeto
import database
import utils
import config
import MetaTrader5 as mt5
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

        # Separa trades Abertas de Fechadas
        open_trades_df = trades_df[trades_df['exit_price'].isna()].copy()
        closed_trades_df = trades_df[trades_df['exit_price'].notna()].copy()

        num_open = len(open_trades_df)
        num_closed = len(closed_trades_df)
        
        # Cálculos para trades fechadas
        wins = len(closed_trades_df[closed_trades_df['pnl_money'] > 0])
        losses = num_closed - wins
        win_rate = (wins / num_closed) * 100 if num_closed > 0 else 0
        total_pnl_closed = closed_trades_df['pnl_money'].sum() if num_closed > 0 else 0.0
        
        resumo_ativo = closed_trades_df.groupby('symbol')['pnl_money'].sum().sort_values() if num_closed > 0 else pd.Series()
        pior_ativo = resumo_ativo.index[0] if not resumo_ativo.empty else "N/A"
        melhor_ativo = resumo_ativo.index[-1] if not resumo_ativo.empty else "N/A"

        # Formata lista de ativos abertos
        ativos_abertos = ", ".join(open_trades_df['symbol'].unique()) if num_open > 0 else "Nenhum"

        return f"""
[DADOS OPERACIONAIS - {today_str}]
- Posições Abertas (Em andamento): {num_open} ({ativos_abertos})
- Trades Finalizados hoje: {num_closed}
- Vitórias: {wins} | Derrotas: {losses}
- Winrate (finalizados): {win_rate:.1f}%
- Resultado Financeiro (fechados): R$ {total_pnl_closed:.2f}
- Pior Ativo (fechado): {pior_ativo}
- Melhor Ativo (fechado): {melhor_ativo}
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

def fetch_mt5_data():
    """Coleta dados em tempo real diretamente do MetaTrader 5"""
    logger.info("Conectando ao MT5 para coletar dados em tempo real...")
    
    if not utils.ensure_mt5_connected():
        return "[MT5 OFFLINE] Não foi possível conectar ao terminal para ler posições abertas."

    try:
        acc = mt5.account_info()
        positions = mt5.positions_get()
        
        if not acc:
            return "[MT5 ERRO] Conectado, mas falha ao ler account_info."

        total_profit = acc.profit
        balance = acc.balance
        equity = acc.equity
        
        resumo = f"""
[DADOS EM TEMPO REAL (MT5)]
- Saldo (Balance): R$ {balance:.2f}
- Patrimônio (Equity): R$ {equity:.2f}
- Lucro Flutuante Total: R$ {total_profit:.2f}
- Posições Abertas no Terminal: {len(positions) if positions else 0}
"""
        if positions:
            resumo += "\nDETALHE DAS POSIÇÕES:\n"
            for p in positions:
                resumo += f"- {p.symbol}: {p.volume} lotes | Lucro: R$ {p.profit:.2f} | Preço Entrada: {p.price_open} | Atual: {p.price_current}\n"
        
        return resumo
    except Exception as e:
        logger.error(f"Erro ao coletar dados do MT5: {e}")
        return f"[MT5 ERRO] Falha técnica na coleta: {e}"

def call_gemini_council(dados_db, dados_mt5, erros, params_atuais, retries=3):
    """Uma unica chamada consolidada para o Conselho de Agentes (evita 429 no Free Tier)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "SUA_CHAVE_AQUI":
        return "Erro: API Key do Gemini nao configurada."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    
    prompt = f"""
Voce e o Conselho Consultivo de IA do bot de trading XP3.
Analise os dados abaixo sob quatro perspectivas (Quant, Infra, Risco, e Mercado) e gere um relatorio final unificado.

DADOS HISTORICOS DO DIA (BANCO DE DADOS):
{dados_db}

DADOS EM TEMPO REAL (TERMINAL MT5):
{dados_mt5}

LOGS DE ERRO:
{erros}

PARAMETROS ATUAIS DO SISTEMA:
{params_atuais}

SUAS PERSONAS:
1. Analista Quantitativo: Foca em lucro realizado vs flutuante, fator de lucro e winrate.
2. Engenheiro de Infra: Foca em erros de conexao, estabilidade do terminal e latencia.
3. CRO (Chief Risk Officer): Foca na exposicao atual (Equity vs Balance), perdas flutuantes e se deve manter ou fechar as operacoes.
4. Analista de Mercado (Market Analyst): Analisa as condicoes de mercado inferidas e sugere ajustes (manter, testar, aumentar, diminuir) nos parametros do bot (ADX, EMA, RSI, ML_Threshold, Stop Loss - SL, Take Profit - TP) para otimizar desempenho.

OBJETIVO:
Crie uma MENSAGEM FINAL PARA O TELEGRAM do CEO (em Portugues Brasileiro).
Seja executivo, direto e use emojis. Separe em:
📊 Visao Quant:
🛠️ Infraestrutura:
🛡️ Gestao de Risco: (Analise se as trades abertas estao seguras)
⚙️ Ajustes Sugeridos (Market Analyst): (Avalie os parametros atuais de ADX, EMA, ML, e discuta mudancas baseadas no mercado/winrate atual)
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
    
    dados_db = fetch_daily_data()
    dados_mt5 = fetch_mt5_data()
    erros = fetch_daily_errors()
    
    # Chama o conselho unificado em uma unica requisicao (Muito melhor para Free Tier)
    cumulative_pnl = database.get_cumulative_pnl(since_date="2026-03-20")
    
    # Busca o saldo inicial do config_risk se possível, senão 2000
    import config_risk
    initial_v_equity = getattr(config_risk, "HOTFIX_SMALL_ACCOUNT", {}).get("force_virtual_equity", 2000.0)
    total_wallet = initial_v_equity + cumulative_pnl

    # Adiciona contexto de acumulado aos dados passados para a IA
    dados_db_completo = dados_db + f"\n\n[RESUMO ACUMULADO]\n- PnL Total (desde o início): R$ {cumulative_pnl:.2f}\n- Saldo Total da Carteira Virtual: R$ {total_wallet:.2f}"

    # Levanta os parametros vigentes do config.py
    params_atuais = f"""
- ADX Atual Base: {config.ADAPTIVE_ADX_MIN}
- EMA Longa (Tendencia): {config.MACRO_EMA_LONG}
- ML Threshold Confianca: {config.ADAPTIVE_ML_THRESHOLD}
- RSI limits: Padroes ML (Variaveis dinâmicas nos scripts de sinais)
- Stop Loss Multiplier (Risco): {config.ADAPTIVE_SL_MULTIPLIER}
- Take Profit Multiplier (Risco): {config.ADAPTIVE_TP_MULTIPLIER}
- Kelly Fraction (Multiplicador): {config.ADAPTIVE_KELLY_MULTIPLIER}
"""

    final_telegram_msg = call_gemini_council(dados_db_completo, dados_mt5, erros, params_atuais)
    
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
