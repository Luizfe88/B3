"""
Script de retreinamento do RandomForest (ml/rf_signal.pkl) com dados balanceados.

Problema: o modelo atual foi treinado predominantemente em mercado bearish (2022-2023)
e produz consistentemente 61-62% de probabilidade SELL mesmo em bull market.

Solução: treinar com janela de dados recentes (últimos 6 meses de bull market B3)
usando dados sintéticos balanceados BUY/SELL/HOLD + dados históricos reais do MT5.

Execute: python retrain_rf_model.py
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RF_Retrain")

# Adiciona raiz do projeto no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_balanced_synthetic_data(n_samples: int = 3000) -> tuple:
    """
    Gera dados sintéticos balanceados representando 3 regimes de mercado:
    - BUY regime:  EMA positiva, RSI baixo-médio, ADX alto, volume subindo
    - SELL regime: EMA negativa, RSI alto, ADX alto, volume subindo
    - HOLD regime: ADX baixo, EMA neutra, volume normal

    Classes: 0=BUY, 1=SELL, 2=HOLD
    Features (16): rsi, adx, atr_pct, volume_ratio, momentum, ema_diff,
                   macd, price_vs_vwap, pe_ratio, roe, market_cap,
                   sentiment, imbalance, cvd, vix_br, book_imbalance
    """
    np.random.seed(42)
    n_each = n_samples // 3
    X_list, y_list = [], []

    # ── BUY regime (label=0) ──────────────────────────────────────────────
    n_buy = n_each + (n_samples - 3 * n_each)  # arredondamento
    rsi_buy        = np.random.normal(42, 8, n_buy).clip(20, 65)
    adx_buy        = np.random.normal(30, 8, n_buy).clip(20, 55)
    atr_pct_buy    = np.random.normal(0.018, 0.006, n_buy).clip(0.005, 0.04)
    vol_ratio_buy  = np.random.normal(1.4, 0.4, n_buy).clip(0.5, 3.0)
    momentum_buy   = np.random.normal(0.012, 0.005, n_buy)
    ema_diff_buy   = np.random.normal(0.004, 0.002, n_buy)     # EMA9 > EMA21
    macd_buy       = np.random.normal(0.30, 0.15, n_buy)
    pv_vwap_buy    = np.random.normal(0.005, 0.003, n_buy)
    # Features externas (zeros pois não disponíveis no MT5)
    ext_buy = np.zeros((n_buy, 8))
    X_buy = np.column_stack([rsi_buy, adx_buy, atr_pct_buy, vol_ratio_buy,
                              momentum_buy, ema_diff_buy, macd_buy, pv_vwap_buy])
    X_buy = np.hstack([X_buy, ext_buy])
    X_list.append(X_buy)
    y_list.extend([0] * n_buy)

    # ── SELL regime (label=1) ────────────────────────────────────────────
    rsi_sell       = np.random.normal(62, 8, n_each).clip(50, 85)
    adx_sell       = np.random.normal(28, 8, n_each).clip(18, 55)
    atr_pct_sell   = np.random.normal(0.022, 0.007, n_each).clip(0.005, 0.05)
    vol_ratio_sell = np.random.normal(1.3, 0.4, n_each).clip(0.5, 3.0)
    momentum_sell  = np.random.normal(-0.010, 0.005, n_each)
    ema_diff_sell  = np.random.normal(-0.004, 0.002, n_each)   # EMA9 < EMA21
    macd_sell      = np.random.normal(-0.25, 0.15, n_each)
    pv_vwap_sell   = np.random.normal(-0.004, 0.003, n_each)
    ext_sell = np.zeros((n_each, 8))
    X_sell = np.column_stack([rsi_sell, adx_sell, atr_pct_sell, vol_ratio_sell,
                               momentum_sell, ema_diff_sell, macd_sell, pv_vwap_sell])
    X_sell = np.hstack([X_sell, ext_sell])
    X_list.append(X_sell)
    y_list.extend([1] * n_each)

    # ── HOLD/NEUTRAL regime (label=2) ────────────────────────────────────
    rsi_hold      = np.random.normal(50, 10, n_each).clip(30, 70)
    adx_hold      = np.random.normal(18, 5, n_each).clip(5, 30)   # ADX baixo = sem tendência
    atr_pct_hold  = np.random.normal(0.012, 0.004, n_each).clip(0.003, 0.025)
    vol_ratio_h   = np.random.normal(0.9, 0.3, n_each).clip(0.3, 1.8)
    momentum_h    = np.random.normal(0.001, 0.003, n_each)
    ema_diff_h    = np.random.normal(0.000, 0.001, n_each)
    macd_h        = np.random.normal(0.00, 0.10, n_each)
    pv_vwap_h     = np.random.normal(0.000, 0.002, n_each)
    ext_hold = np.zeros((n_each, 8))
    X_hold = np.column_stack([rsi_hold, adx_hold, atr_pct_hold, vol_ratio_h,
                               momentum_h, ema_diff_h, macd_h, pv_vwap_h])
    X_hold = np.hstack([X_hold, ext_hold])
    X_list.append(X_hold)
    y_list.extend([2] * n_each)

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Adiciona ruído realista
    X += np.random.normal(0, 0.002, X.shape)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def load_historical_mt5_data() -> tuple:
    """
    Tenta carregar trades históricos reais do banco para adicionar ao treinamento.
    Retorna (X, y) ou (None, None) se não disponível.
    """
    try:
        import database
        from ml.prediction import MLPredictor
        import utils
        import MetaTrader5 as mt5

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # últimos 6 meses

        trades = database.get_all_trades(start_date, end_date)
        if not trades or len(trades) < 30:
            logger.warning(f"Poucos trades reais ({len(trades) if trades else 0}). Usando apenas sintético.")
            return None, None

        logger.info(f"Carregando {len(trades)} trades históricos...")

        pred = MLPredictor()
        X_real, y_real = [], []

        for trade in trades:
            try:
                # Busca candles do período de entrada
                candles = utils.safe_copy_rates(trade.symbol, mt5.TIMEFRAME_M15, 100)
                if candles is None or candles.empty:
                    continue

                indicators = pred.compute_indicators(candles)
                if not indicators:
                    continue

                features = pred.extract_features(trade.symbol, indicators)

                # Label: 0=BUY sucesso, 1=SELL sucesso, 2=perdeu dinheiro
                if trade.pnl > 0:
                    label = 0 if trade.side == 'BUY' else 1
                else:
                    label = 2  # trade perdedor → HOLD era melhor

                X_real.append(features)
                y_real.append(label)
            except Exception:
                continue

        if len(X_real) < 20:
            return None, None

        X_real = np.array(X_real, dtype=np.float32)
        y_real = np.array(y_real)
        logger.info(f"Dados reais carregados: {len(X_real)} amostras | classes: {np.bincount(y_real)}")
        return X_real, y_real

    except Exception as e:
        logger.warning(f"Não foi possível carregar dados reais: {e}")
        return None, None


def train_and_save():
    logger.info("=" * 60)
    logger.info("RETREINAMENTO RF — Modelo Balanceado BUY/SELL/HOLD")
    logger.info("=" * 60)

    # 1. Dados sintéticos balanceados
    logger.info("Gerando dados sintéticos balanceados (1000 amostras por classe)...")
    X_synth, y_synth = generate_balanced_synthetic_data(n_samples=3000)

    # 2. Tenta enriquecer com dados reais
    X_real, y_real = load_historical_mt5_data()
    if X_real is not None:
        X = np.vstack([X_synth, X_real])
        y = np.concatenate([y_synth, y_real])
        logger.info(f"Dataset final: {len(X)} amostras (sintético + real)")
    else:
        X, y = X_synth, y_synth
        logger.info(f"Dataset final: {len(X)} amostras (apenas sintético)")

    # 3. Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Pesos de classe para balanceamento extra
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weight_dict}")

    # 5. Cross-validation temporal
    tscv = TimeSeriesSplit(n_splits=5)

    # 6. Modelos candidatos
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )

    # 7. Avalia e escolhe o melhor
    rf_scores = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='f1_macro')
    gb_scores = cross_val_score(gb, X_scaled, y, cv=tscv, scoring='f1_macro')
    logger.info(f"RF  — F1-macro CV: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    logger.info(f"GB  — F1-macro CV: {gb_scores.mean():.3f} ± {gb_scores.std():.3f}")

    # Treina o melhor no conjunto completo
    if rf_scores.mean() >= gb_scores.mean():
        best = rf
        model_name = "RandomForest"
    else:
        best = gb
        model_name = "GradientBoosting"

    best.fit(X_scaled, y)
    logger.info(f"Modelo escolhido: {model_name}")

    # 8. Report final
    y_pred = best.predict(X_scaled)
    print("\n" + classification_report(y, y_pred, target_names=["BUY", "SELL", "HOLD"]))
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=["BUY", "SELL", "HOLD"], columns=["BUY", "SELL", "HOLD"]))

    # 9. Verifica distribuição de predições (não deve ser 100% SELL!)
    pred_counts = np.bincount(y_pred, minlength=3)
    pred_pcts   = pred_counts / len(y_pred) * 100
    logger.info(f"Dist. predições — BUY: {pred_pcts[0]:.1f}% | SELL: {pred_pcts[1]:.1f}% | HOLD: {pred_pcts[2]:.1f}%")
    if pred_pcts[1] > 70:
        logger.warning("⚠️ Modelo ainda muito bearish! Revise os dados de treinamento.")
    else:
        logger.info("✅ Distribuição de predições balanceada.")

    # 10. Salva
    rf_path     = os.path.join(MODELS_DIR, "rf_signal.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(best, rf_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Modelo salvo em {rf_path}")
    logger.info(f"✅ Scaler salvo em {scaler_path}")

    return best, scaler


if __name__ == "__main__":
    train_and_save()
    logger.info("\n🎉 Retreinamento concluído! Reinicie o bot para usar o novo modelo.")
