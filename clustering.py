"""
Clustering Module v1.0 - XP3 PRO QUANT-REFORM
Agrupa ativos por liquidez (ADV) e volatilidade para calibração dinâmica.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class TickerClusterer:
    """Classificador de ativos baseado em liquidez e volatilidade"""
    
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_map = {} # symbol -> basket_id (0, 1, 2)
        self.basket_labels = {
            0: "Cesta B (Mid-Caps)",
            1: "Cesta A (Alta Liquidez)",
            2: "Cesta C (Baixa Liquidez)"
        }

    def train(self, data: pd.DataFrame):
        """
        Treina o clusterizador usando as colunas 'adv' (Average Daily Volume) 
        e 'volatility' (Std Dev de retornos).
        """
        if data.empty or len(data) < self.n_clusters:
            logger.warning("Dados insuficientes para clusterização.")
            return

        # Normalização logarítmica para ADV (pode variar ordens de magnitude)
        features = data[['adv', 'volatility']].copy()
        features['adv_log'] = np.log1p(features['adv'])
        
        # Fit KMeans
        X = features[['adv_log', 'volatility']].values
        self.model.fit(X)
        
        # Atribui nomes às cestas baseadas no ADV médio de cada cluster
        centers = self.model.cluster_centers_
        adv_centers = centers[:, 0] # Index 0 é adv_log
        
        # Ordena clusters por ADV (A=Maior, B=Médio, C=Menor)
        sorted_indices = np.argsort(adv_centers)[::-1] # Decrescente
        
        # Mapeamento do KMeans label para nossa Cesta A, B, C
        label_to_basket = {
            sorted_indices[0]: 0, # Alta (A) -> No código usaremos 0, 1, 2 internamente
            sorted_indices[1]: 1, # Média (B)
            sorted_indices[2]: 2  # Baixa (C)
        }
        
        self.basket_id_map = label_to_basket
        self.basket_labels = {
            0: "Cesta A (Alta Liquidez)",
            1: "Cesta B (Mid-Caps)",
            2: "Cesta C (Baixa Liquidez)"
        }
        
        # Mapeia cada símbolo
        data['cluster'] = self.model.labels_
        data['basket_id'] = data['cluster'].map(label_to_basket)
        
        self.cluster_map = data['basket_id'].to_dict()
        logger.info(f"✅ Clusterização concluída para {len(data)} ativos.")
        for bid, label in self.basket_labels.items():
            count = sum(1 for v in self.cluster_map.values() if v == bid)
            logger.info(f" - {label}: {count} ativos")

    def get_basket(self, symbol: str) -> int:
        """Retorna o ID da cesta (0=A, 1=B, 2=C). Fallback para 1 (B)"""
        return self.cluster_map.get(symbol, 1)

    def get_label(self, symbol: str) -> str:
        """Retorna o nome da cesta"""
        bid = self.get_basket(symbol)
        return self.basket_labels.get(bid, "Cesta B (Mid-Caps)")

    def save(self, path="clusters.json"):
        import json
        with open(path, 'w') as f:
            json.dump({
                "clusters": self.cluster_map,
                "labels": self.basket_labels
            }, f, indent=4)

    @classmethod
    def load(cls, path="clusters.json"):
        import json
        if not os.path.exists(path):
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
            obj = cls()
            obj.cluster_map = data.get("clusters", {})
            obj.basket_labels = {int(k): v for k, v in data.get("labels", {}).items()}
            return obj
