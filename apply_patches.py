
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
 
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
 
def tag_ok(msg):    print(f"  [OK]  {msg}")
def tag_skip(msg):  print(f"  [SKIP] {msg}")
def tag_warn(msg):  print(f"  [WARN] {msg}")
def tag_err(msg):   print(f"  [ERR]  {msg}")
def tag_info(msg):  print(f"  [INFO] {msg}")
 
PATCHES = [
    {
        "id": "P1",
        "description": "insufficient_setups: return -score_final -> return score_final",
        "old": (
            '            if metrics.get("insufficient_setups"):\n'
            '                penalty = 10.0\n'
            '                bonus = 0.0\n'
            '                score_final = bonus - penalty\n'
            '                log_rejection(symbol, trial.number, "INSUFICIENT_SETUPS", f"Setups={metrics.get(\'setups_identified\')} | Penalty applied.")\n'
            '                return -score_final'
        ),
        "new": (
            '            if metrics.get("insufficient_setups"):\n'
            '                penalty = 10.0\n'
            '                bonus = 0.0\n'
            '                score_final = bonus - penalty          # = -10.0  (trial ruim)\n'
            '                log_rejection(symbol, trial.number, "INSUFICIENT_SETUPS", f"Setups={metrics.get(\'setups_identified\')} | Penalty applied.")\n'
            '                return score_final                     # FIX P1: era -score_final (+10); maximize escolhia o pior'
        ),
        "new_marker": "FIX P1",
    },
    {
        "id": "P2",
        "description": "Não zerar profit_factor/win_rate quando DD >= 95%",
        "old": (
            '    # Comentário: E.2 Pós-backtest checks\n'
            '    if metrics["total_trades"] == 0 or metrics["max_drawdown"] >= 0.95:\n'
            '        metrics["calmar"] = 0.0\n'
            '        metrics["profit_factor"] = 0.0\n'
            '        metrics["win_rate"] = 0.0'
        ),
        "new": (
            '    # FIX P2: E.2 Pós-backtest checks\n'
            '    # Zero absoluto só quando não houve nenhum trade.\n'
            '    # Com DD >= 95% mantemos profit_factor e win_rate reais para diagnóstico;\n'
            '    # apenas o Calmar (que divide por DD) é zerado para evitar divisão ruidosa.\n'
            '    if metrics["total_trades"] == 0:\n'
            '        metrics["calmar"] = 0.0\n'
            '        metrics["profit_factor"] = 0.0\n'
            '        metrics["win_rate"] = 0.0\n'
            '    elif metrics["max_drawdown"] >= 0.95:\n'
            '        metrics["calmar"] = 0.0          # Calmar zerado: DD extremo distorce a fórmula\n'
            '        # profit_factor e win_rate mantidos: diagnóstico precisa dos valores reais'
        ),
        "new_marker": "FIX P2",
    },
    {
        "id": "P3a",
        "description": "Declarar _warned_symbols no topo do modulo",
        "old": "import xgboost as xgb\n\nlogger = logging.getLogger(__name__)",
        "new": (
            "import xgboost as xgb\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "# FIX P3: controle de warnings unicos por ativo por sessao\n"
            "_warned_symbols: set = set()"
        ),
        "new_marker": "_warned_symbols: set = set()",
    },
    {
        "id": "P3b",
        "description": "Warning de tendencia: flood 200+ linhas -> 1 por ativo por sessão",
        "old": (
            '    # Check de Tendência: ema_s > ema_l em pelo menos 30% das barras\n'
            '    trend_freq = np.sum(ema_s > ema_l) / len(close)\n'
            '    if trend_freq < 0.30:\n'
            '        logger.warning(\n'
            '            f"[WARN] {symbol}: Mercado sem tendência clara (Alta em apenas {trend_freq:.1%})"\n'
            '        )'
        ),
        "new": (
            '    # FIX P3b: Check de Tendência — warning único por ativo por sessão\n'
            '    trend_freq = np.sum(ema_s > ema_l) / len(close)\n'
            '    if trend_freq < 0.30 and symbol not in _warned_symbols:\n'
            '        _warned_symbols.add(symbol)\n'
            '        logger.warning(\n'
            '            f"[WARN] {symbol}: Mercado sem tendência clara "\n'
            '            f"(Alta em apenas {trend_freq:.1%}). Este aviso não se repetirá para este ativo nesta sessão."\n'
            '        )'
        ),
        "new_marker": "FIX P3b",
    },
    {
        "id": "P4",
        "description": "Threshold de setups minimos: 5 -> 3 (VALE3, ITUB4, mid-caps)",
        "old": (
            '    if total_setups_estimados < 5:\n'
            '        return {\n'
            '            "total_trades": 0,\n'
            '            "setups_identified": int(total_setups_estimados),\n'
            '            "total_return": 0.0,\n'
            '            "max_drawdown": 0.0,\n'
            '            "sharpe": 0.0,\n'
            '            "profit_factor": 0.0,\n'
            '            "insufficient_setups": True\n'
            '        }'
        ),
        "new": (
            '    # FIX P4: reduzido de 5 para 3 — ativos de baixa frequência (VALE3, mid-caps)\n'
            '    # têm naturalmente menos setups por janela e não devem ser descartados cedo.\n'
            '    if total_setups_estimados < 3:\n'
            '        return {\n'
            '            "total_trades": 0,\n'
            '            "setups_identified": int(total_setups_estimados),\n'
            '            "total_return": 0.0,\n'
            '            "max_drawdown": 0.0,\n'
            '            "sharpe": 0.0,\n'
            '            "profit_factor": 0.0,\n'
            '            "insufficient_setups": True\n'
            '        }'
        ),
        "new_marker": "FIX P4",
    },
    {
        "id": "P5",
        "description": "Guard pf >= 0.80 no SNIPER EM OBSERVAÇÃO (evita aprovação com PF destrutivo)",
        "old": (
            '        elif rr_estimado >= 1.4 and trades >= 5:\n'
            '            msg = [\n'
            '                "VEREDITO: SNIPER EM OBSERVAÇÃO (R/R Alto)",\n'
            '                f"   Motivo: R/R de {rr_estimado:.2f} é excelente, apesar das métricas borderline.",\n'
            '                "   Ação: Monitorar comportamento em tempo real.",\n'
            '            ]\n'
            '            for m in msg:\n'
            '                print(m)\n'
            '                lines.append(m)\n'
            '        else:\n'
            '            msg = [\n'
            '                "VEREDITO: REPROVADO",\n'
            '                "   Motivo: Poucos trades e lucro irrelevante. Não vale o risco.",\n'
            '            ]'
        ),
        "new": (
            '        elif rr_estimado >= 1.4 and trades >= 5 and pf >= 0.80:\n'
            '            # FIX P5: exige PF >= 0.80 — R/R alto com PF < 0.80 destrói capital\n'
            '            msg = [\n'
                '                "VEREDITO: SNIPER EM OBSERVAÇÃO (R/R Alto)",\n'
                '                f"   Motivo: R/R de {rr_estimado:.2f} é excelente, apesar das métricas borderline.",\n'
                '                "   Ação: Monitorar comportamento em tempo real.",\n'
                '            ]\n'
                '            for m in msg:\n'
                '                print(m)\n'
                '                lines.append(m)\n'
                '        else:\n'
                '            pf_msg = f" | PF={pf:.2f} < 0.80 (destruindo capital)" if pf < 0.80 else ""\n'
                '            msg = [\n'
                '                "VEREDITO: REPROVADO",\n'
                '                f"   Motivo: Sem edge real.{pf_msg} Trades={trades} | R/R={rr_estimado:.2f}.",\n'
                '            ]'
        ),
        "new_marker": "FIX P5",
    },
    {
        "id": "P6",
        "description": "Teto de penalidade (4.0) + rebalanceamento de pesos TREND/SIDEWAYS",
        "old": (
            '            bonus = 0.0\n'
            '            if wr > 0.55 and pf > 1.5:\n'
            '                bonus += 0.5\n'
            '                reason.append("BONUS_QUALITY")\n'
            '            if trades > 12 and wr > 0.50:\n'
            '                bonus += 0.3\n'
            '                reason.append("BONUS_FREQUENCY")\n'
            '            if dd < 0.30:\n'
            '                bonus += 0.2\n'
            '                reason.append("BONUS_LOW_DD")\n'
            '\n'
            '            # ✅ SCORE POR REGIME\n'
            '            if regime == "TREND":\n'
                '                # Foco: Profit Factor e Retorno (Surfar a onda)\n'
                '                score_final = (pf * 1.5) + (metrics.get("total_return", 0.0) * 2.0) + bonus - penalty\n'
                '            elif regime == "SIDEWAYS":\n'
                '                # Foco: Win Rate e Sortino (Consistência em baixa volatilidade)\n'
                '                score_final = (wr * 3.0) + (metrics.get("sortino", 0.0) * 1.2) + bonus - penalty\n'
                '                # Penaliza ADX alto se estiver buscando perfil lateral\n'
                '                if params.get("adx_threshold", 0) > 20: \n'
                '                    score_final -= 1.0\n'
                '            elif regime == "PROTECTION":\n'
                '                # Foco: Capital Preservation (Mínimo DD)\n'
                '                score_final = ((1.0 - dd) * 4.0) + (pf * 1.0) + bonus - penalty\n'
                '                # Penaliza trades excessivos se o objetivo é proteção\n'
                '                if trades > 30: score_final -= 0.5\n'
                '            else:\n'
                '                # Fallback Sharpista\n'
                '                score_final = (metrics.get("sharpe", 0.0) * 2.0) + bonus - penalty'
        ),
        "new": (
            '            bonus = 0.0\n'
            '            if wr > 0.55 and pf > 1.5:\n'
            '                bonus += 0.5\n'
            '                reason.append("BONUS_QUALITY")\n'
            '            if trades > 12 and wr > 0.50:\n'
            '                bonus += 0.3\n'
            '                reason.append("BONUS_FREQUENCY")\n'
            '            if dd < 0.30:\n'
                '                bonus += 0.2\n'
                '                reason.append("BONUS_LOW_DD")\n'
                '\n'
                '            # FIX P6a: teto de penalidade - impede que ativos estruturalmente\n'
                '            # dificeis (PETR4, mid-caps) nunca encontrem territorio positivo.\n'
                '            # Sem teto: penalty > 7 com reward maximo ~2 -> espaco sempre negativo.\n'
                '            # Com teto 4.0: um trial com PF=1.0 e 8 trades limpos sempre pontua > 0.\n'
                '            penalty = min(penalty, 4.0)\n'
                '            reason.append(f"PenaltyCapped={penalty:.2f}")\n'
                '\n'
                '            # FIX P6b: SCORE POR REGIME - caps em todas as metricas para evitar explosao\n'
                '            # pf > 10 acontece com 2-3 trades e perda minima -> cap em 5.0 (estrategia excelente = 5)\n'
                '            # sortino > 20 acontece com down_std ≈ 0 -> cap em 10.0 (sortino excepcional = 10)\n'
                '            # total_return > 2.0 = position sizing overflow -> cap em 2.0 (200% na janela)\n'
                '            pf_capped           = min(pf, 5.0)\n'
                '            sortino_capped      = min(metrics.get("sortino", 0.0), 10.0)\n'
                '            total_return_capped = min(metrics.get("total_return", 0.0), 2.0)\n'
                '\n'
                '            if regime == "TREND":\n'
                '                score_final = (pf_capped * 2.0) + (total_return_capped * 1.2) + bonus - penalty\n'
                '            elif regime == "SIDEWAYS":\n'
                '                score_final = (wr * 3.0) + (sortino_capped * 2.0) + bonus - penalty\n'
                '                if params.get("adx_threshold", 0) > 20:\n'
                '                    score_final -= 1.0\n'
                '            elif regime == "PROTECTION":\n'
                '                score_final = ((1.0 - dd) * 4.0) + (pf_capped * 1.0) + bonus - penalty\n'
                '                if trades > 30: score_final -= 0.5\n'
                '            else:\n'
                '                score_final = (metrics.get("sharpe", 0.0) * 2.0) + bonus - penalty'
        ),
        "new_marker": "FIX P6",
    },
]
 
PENDING  = "PENDING"
APPLIED  = "APPLIED"
CONFLICT = "CONFLICT"
 
def classify(content, patch):
    marker_found = patch["new_marker"] in content
    if marker_found: return APPLIED
    old_found = patch["old"] in content
    if old_found:    return PENDING
    return CONFLICT
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    default="optimizer_optuna.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status",  action="store_true")
    args = parser.parse_args()
 
    target = Path(args.file)
    if not target.exists():
        tag_err(f"Arquivo não encontrado: {target}")
        sys.exit(1)
 
    print(f"\n{BOLD}{'-'*64}{RESET}")
    print(f"{BOLD}  XP3 Pro - Patch Engine  |  {target.name}{RESET}")
    print(f"{BOLD}{'-'*64}{RESET}\n")
 
    content = target.read_text(encoding="utf-8")
    states  = {p["id"]: classify(content, p) for p in PATCHES}
 
    print(f"{CYAN}{BOLD}[ Status dos Patches ]{RESET}")
    for pid, state in states.items():
        p = next(x for x in PATCHES if x["id"] == pid)
        desc = p["description"][:68]
        if state == APPLIED:   tag_skip(f"[{pid}] ja aplicado   - {desc}")
        elif state == PENDING: tag_warn(f"[{pid}] PENDENTE      - {desc}")
        else:                  tag_err( f"[{pid}] CONFLITO      - {desc}")
 
    n_p = sum(1 for s in states.values() if s == PENDING)
    n_a = sum(1 for s in states.values() if s == APPLIED)
    n_c = sum(1 for s in states.values() if s == CONFLICT)
    print(f"\\n  {GREEN}{n_a} ja aplicado(s){RESET}  {YELLOW}{n_p} pendente(s){RESET}  {RED}{n_c} conflito(s){RESET}\\n")
 
    if args.status:
        sys.exit(0)
 
    if n_c > 0:
        tag_err(f"{n_c} conflito(s) detectado(s) - revise manualmente antes de continuar.")
        sys.exit(1)
 
    if n_p == 0:
        print(f"{GREEN}Todos os patches já estão aplicados. Nada a fazer.{RESET}\n")
        sys.exit(0)
 
    print(f"{CYAN}{BOLD}[ Aplicando {n_p} patch(es) pendente(s) ]{RESET}")
    applied_ids, skipped_ids = [], []
    for p in PATCHES:
        if states[p["id"]] == PENDING:
            content = content.replace(p["old"], p["new"], 1)
            applied_ids.append(p["id"])
            tag_ok(f"[{p['id']}] Aplicado: {p['description'][:68]}")
        else:
            skipped_ids.append(p["id"])
            tag_skip(f"[{p['id']}] Ignorado (já aplicado)")
 
    if args.dry_run:
        print(f"\n{YELLOW}Modo dry-run: nenhuma alteração gravada.{RESET}\n")
        sys.exit(0)
 
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.stem}.BACKUP_{ts}{target.suffix}")
    shutil.copy2(target, backup)
    tag_info(f"Backup criado: {backup.name}")
    target.write_text(content, encoding="utf-8")
 
    print(f"\n{BOLD}{'-'*64}{RESET}")
    print(f"{GREEN}{BOLD}  [OK]  Concluido!{RESET}")
    print(f"{'-'*64}")
    print(f"  Arquivo  : {target}")
    print(f"  Backup   : {backup.name}")
    print(f"  Aplicados: {' '.join(applied_ids)}")
    print(f"  Ignorados: {' '.join(skipped_ids) or '—'}")
    print(f"{'-'*64}")
 
if __name__ == "__main__":
    main()
