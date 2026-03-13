"""
Calibration Manager v1.0 - XP3 PRO QUANT-REFORM
Gerencia a persistência de parâmetros calibrados e ajustes de Kelly.
"""

import json
import os
import logging
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import database
import config

logger = logging.getLogger(__name__)

CALIBRATIONS_FILE = "calibrations.json"

class CalibrationManager:
    """Gerenciador de calibrações e Kelly Dinâmico"""
    
    def __init__(self, file_path=CALIBRATIONS_FILE):
        self.file_path = file_path
        self.calibrations = self.load()

    def load(self) -> Dict[str, Any]:
        """Carrega calibrações do disco"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar calibrações: {e}")
        return {"symbols": {}, "clusters": {}, "global": {}}

    def load_clusters_map(self) -> Dict[str, int]:
        """Carrega mapeamento de clusters (símbolo -> ID)"""
        if os.path.exists("clusters.json"):
            try:
                with open("clusters.json", "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar clusters.json: {e}")
        return {}

    def save(self):
        """Salva calibrações no disco"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.calibrations, f, indent=4)
            logger.info(f"✅ Calibrações salvas em {self.file_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar calibrações: {e}")

    def update_symbol_kelly(self, symbol: str, lookback_days: int = 30):
        """
        Calcula o Kelly dinâmico baseado no histórico real do banco de dados.
        """
        try:
            conn = sqlite3.connect(database.DB_PATH)
            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Busca todos os pnl_pct do símbolo
            query = f"""
                SELECT pnl_pct 
                FROM trades 
                WHERE symbol = ? AND date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty or len(df) < 5:
                logger.debug(f"Histórico insuficiente para Kelly em {symbol}")
                return

            pnls = df['pnl_pct'].values
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            
            win_rate = len(wins) / len(pnls)
            avg_win = np.mean(wins) if len(wins) > 0 else 0.015
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.012
            
            # Kelly clássico
            if avg_loss > 0:
                k_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
            else:
                k_fraction = 0.0
                
            # Limita a 1.0 (não operamos mais que 100% de capital por ativo via Kelly)
            # Mas o RiskManager filtrará pelo Hard Cap de 10%
            k_fraction = max(0.0, min(1.0, k_fraction))
            
            if symbol not in self.calibrations["symbols"]:
                self.calibrations["symbols"][symbol] = {}
                
            self.calibrations["symbols"][symbol]["kelly"] = {
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "fraction": float(k_fraction),
                "updated_at": datetime.now().isoformat()
            }
            logger.info(f"📊 Kelly atualizado para {symbol}: WR={win_rate:.1%}, Fraction={k_fraction:.2f}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar Kelly para {symbol}: {e}")

    def get_calibrated_params(self, symbol: str, basket_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Retorna os parâmetros mesclando:
        1. Multi-Regime do arquivo individual (Se existir)
        2. Parâmetros específicos do símbolo (calibrations.json)
        3. Parâmetros da cesta (cluster)
        4. Parâmetros globais do config
        """
        params = {}
        
        # Tenta descobrir basket_id se não fornecido
        if basket_id is None:
            cluster_map = self.load_clusters_map()
            basket_id = cluster_map.get(symbol, 1) # Fallback para Mid-Cap
            
        # 1. Base da Cesta (ID: 0=A, 1=B, 2=C)
        basket_key = f"basket_{basket_id}"
        basket_params = self.calibrations.get("clusters", {}).get(basket_key, {})
        params.update(basket_params)
        
        # 2. Calibração Individual (Preferência para o NOVO Sistema Multi-Regime)
        ind_dir = "calibrations_individual"
        ind_file = os.path.join(ind_dir, f"{symbol}.json")
        
        if os.path.exists(ind_file):
            try:
                with open(ind_file, 'r', encoding='utf-8') as f:
                    ind_data = json.load(f)
                
                # Se for o novo formato com regimes
                if "regimes" in ind_data and "current_active" in ind_data:
                    regime = ind_data["current_active"]
                    regime_params = ind_data["regimes"].get(regime, {}).get("best_params", {})
                    if regime_params:
                        params.update(regime_params)
                        # Preserva metadados
                        params["active_regime"] = regime
                        params["timeframe"] = ind_data.get("timeframe", params.get("timeframe", "M15"))
                        params["verdict"] = ind_data.get("verdict", "OPTIMIZED")
                        logger.debug(f"🎯 {symbol}: Usando perfil {regime} do arquivo individual.")
                else:
                    # Formato antigo ou default
                    params.update({k: v for k, v in ind_data.items() if k != "kelly"})
            except Exception as e:
                logger.error(f"Erro ao ler calibração individual de {symbol}: {e}")

        # 3. Específicos do Símbolo (calibrations.json - Legado/Fallback)
        symbol_params = self.calibrations.get("symbols", {}).get(symbol, {})
        # Apenas atualiza se não foi sobrescrevido pelo arquivo individual
        for k, v in symbol_params.items():
            if k != "kelly" and k not in params:
                params[k] = v
        
        # 4. Kelly (se disponível)
        if "kelly" in symbol_params:
            params["kelly"] = symbol_params["kelly"]
        elif os.path.exists(ind_file):
            # Tenta buscar Kelly do individual se não tiver no global
             try:
                with open(ind_file, 'r', encoding='utf-8') as f:
                    ind_data = json.load(f)
                    if "kelly" in ind_data:
                        params["kelly"] = ind_data["kelly"]
             except: pass
            
        return params

    def get_summary(self) -> str:
        """Retorna uma string formatada com o resumo das calibrações carregadas."""
        sym_count = len(self.calibrations.get("symbols", {}))
        cluster_count = len(self.calibrations.get("clusters", {}))
        
        status = "✅ ATIVA" if sym_count > 0 or cluster_count > 0 else "⚠️ PADRÃO (Sem calibração)"
        
        summary = [
            "\n" + "="*50,
            "📊 SISTEMA DE CALIBRAÇÃO DINÂMICA (WFA)",
            f"Status: {status}",
            f"Símbolos Calibrados: {sym_count}",
            f"Cestas (Clusters): {cluster_count}",
            "="*50 + "\n"
        ]
        return "\n".join(summary)

    def set_basket_params(self, basket_id: int, params: Dict[str, Any]):
        """Define parâmetros para uma cesta inteira"""
        basket_key = f"basket_{basket_id}"
        if "clusters" not in self.calibrations:
            self.calibrations["clusters"] = {}
        self.calibrations["clusters"][basket_key] = params

    def update_from_results(self, all_results: Dict[str, Any]):
        """
        Consolida resultados da otimização no arquivo de calibrações.
        """
        for symbol, result in all_results.items():
            params = result.get("selected_params") or result.get("best_params")
            if not params:
                continue
                
            if symbol not in self.calibrations["symbols"]:
                self.calibrations["symbols"][symbol] = {}
            
            # Atualiza parâmetros (exceto Kelly que é gerido internamente)
            for k, v in params.items():
                if k != "kelly":
                    self.calibrations["symbols"][symbol][k] = v
            
            # ✅ NOVO: Salva o timeframe selecionado e veredito
            if "timeframe" in result:
                self.calibrations["symbols"][symbol]["timeframe"] = result["timeframe"]
            
            if "verdict" in result:
                self.calibrations["symbols"][symbol]["verdict"] = result["verdict"]
                self.calibrations["symbols"][symbol]["last_calib_date"] = datetime.now().strftime("%Y-%m-%d")
            
            # Agenda atualização do Kelly para este símbolo
            self.update_symbol_kelly(symbol)
            
        self.save()
        logger.info(f"✅ Calibrations processadas para {len(all_results)} ativos.")

# Instância global
calibration_manager = CalibrationManager()
