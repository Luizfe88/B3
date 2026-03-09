import sqlite3
import os
from datetime import datetime, timedelta
import pandas as pd
import logging
import threading

logger = logging.getLogger("database")

DB_PATH = "xp3_trades.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket INTEGER UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            volume REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            sl REAL,
            tp REAL,
            pnl_money REAL,
            pnl_pct REAL,
            reason TEXT,
            ml_reward REAL,
            strategy TEXT DEFAULT 'ELITE',
            ml_confidence REAL DEFAULT 0.0,
            ml_prediction TEXT DEFAULT '',
            atr_pct REAL DEFAULT 0.0,
            vix_level REAL DEFAULT 0.0,
            order_flow_delta REAL DEFAULT 0.0,
            duration_minutes INTEGER DEFAULT 0,
            exit_time DATETIME,
            ab_group TEXT DEFAULT 'A'
        )
    """
    )
    conn.commit()
    conn.close()


class StateManager:
    def __init__(self, db_path: str = "trading_state.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = (
            os.name and threading.RLock()
            if hasattr(__import__("threading"), "RLock")
            else None
        )
        self._init_schema()

    def _init_schema(self):
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_state (
                trading_date DATE PRIMARY KEY,
                equity_start REAL NOT NULL,
                equity_max REAL NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins_count INTEGER DEFAULT 0,
                loss_streak INTEGER DEFAULT 0,
                circuit_breaker_active BOOLEAN DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (equity_start > 0),
                CHECK (equity_max >= equity_start),
                CHECK (trades_count >= 0),
                CHECK (wins_count >= 0),
                CHECK (loss_streak >= 0)
            )
        """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS symbol_limits (
                symbol TEXT NOT NULL,
                trading_date DATE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                losses_count INTEGER DEFAULT 0,
                last_sl_time TIMESTAMP,
                cooldown_until TIMESTAMP,
                PRIMARY KEY (symbol, trading_date),
                CHECK (trades_count >= losses_count)
            )
        """
        )
        self.db.commit()

    def save_state_atomic(self, state: dict):
        cur = self.db.cursor()
        try:
            self.db.execute("BEGIN TRANSACTION")
            cur.execute(
                """
                INSERT OR REPLACE INTO daily_state 
                (trading_date, equity_start, equity_max, trades_count, wins_count, loss_streak, circuit_breaker_active, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    state.get("trading_date"),
                    float(state.get("equity_start", 0) or 0),
                    float(state.get("equity_max", 0) or 0),
                    int(state.get("trades_count", 0) or 0),
                    int(state.get("wins_count", 0) or 0),
                    int(state.get("loss_streak", 0) or 0),
                    int(bool(state.get("circuit_breaker_active", False))),
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def get_today_state(self) -> dict:
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM daily_state WHERE trading_date = date(?)",
            self.db,
            params=(today,),
        )
        if len(df) == 0:
            return {}
        row = df.iloc[0].to_dict()
        return row

    def reset_daily_if_needed(self):
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM daily_state WHERE trading_date = date(?)",
            self.db,
            params=(today,),
        )
        if len(df) == 0:
            self.save_state_atomic(
                {
                    "trading_date": today,
                    "equity_start": 1.0,
                    "equity_max": 1.0,
                    "trades_count": 0,
                    "wins_count": 0,
                    "loss_streak": 0,
                    "circuit_breaker_active": False,
                }
            )

    def update_symbol_limits(
        self, symbol: str, trades_delta: int = 0, losses_delta: int = 0
    ):
        today = datetime.now().date().isoformat()
        cur = self.db.cursor()
        cur.execute(
            """
            INSERT INTO symbol_limits (symbol, trading_date, trades_count, losses_count)
            VALUES (?, date(?), 0, 0)
            ON CONFLICT(symbol, trading_date) DO NOTHING
        """,
            (symbol, today),
        )
        cur.execute(
            """
            UPDATE symbol_limits
            SET trades_count = trades_count + ?, losses_count = losses_count + ?, last_sl_time = CASE WHEN ? > 0 THEN CURRENT_TIMESTAMP ELSE last_sl_time END
            WHERE symbol = ? AND trading_date = date(?)
        """,
            (int(trades_delta), int(losses_delta), int(losses_delta), symbol, today),
        )
        self.db.commit()

    def get_symbol_limits(self, symbol: str) -> dict:
        today = datetime.now().date().isoformat()
        df = pd.read_sql_query(
            "SELECT * FROM symbol_limits WHERE symbol = ? AND trading_date = date(?)",
            self.db,
            params=(symbol, today),
        )
        if len(df) == 0:
            return {"trades_count": 0, "losses_count": 0}
        return df.iloc[0].to_dict()


def migrate_db():
    """Adiciona novas colunas se não existirem (migrate schema)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    new_columns = [
        ("ticket", "INTEGER UNIQUE"),
        ("strategy", "TEXT DEFAULT 'ELITE'"),
        ("ml_confidence", "REAL DEFAULT 0.0"),
        ("ml_prediction", "TEXT DEFAULT ''"),
        ("atr_pct", "REAL DEFAULT 0.0"),
        ("vix_level", "REAL DEFAULT 0.0"),
        ("order_flow_delta", "REAL DEFAULT 0.0"),
        ("duration_minutes", "INTEGER DEFAULT 0"),
        ("exit_time", "DATETIME"),
        ("ab_group", "TEXT DEFAULT 'A'"),
    ]

    for col_name, col_def in new_columns:
        try:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
            logger.info(f"✅ Coluna {col_name} adicionada")
        except sqlite3.OperationalError:
            pass  # Coluna já existe

    conn.commit()
    conn.close()


def save_trade(
    symbol,
    side,
    volume,
    entry_price,
    exit_price=None,
    sl=None,
    tp=None,
    pnl_money=0.0,
    pnl_pct=0.0,
    reason="",
    ml_reward=0.0,
    ticket=None,
    strategy="ELITE",
    ml_confidence=0.0,
    ml_prediction="",
    atr_pct=0.0,
    vix_level=0.0,
    order_flow_delta=0.0,
    duration_minutes=0,
    ab_group="A",
    exit_time=None,
    timestamp=None,
):
    init_db()
    migrate_db()

    conn = sqlite3.connect(DB_PATH)
    try:
        # Tenta atualizar se o ticket existir, senão insere
        if ticket:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM trades WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            if row:
                # Update existente
                conn.execute(
                    """
                    UPDATE trades 
                    SET exit_price = ?, sl = ?, tp = ?, pnl_money = ?, pnl_pct = ?, 
                        exit_time = ?, duration_minutes = ?
                    WHERE ticket = ?
                """,
                    (exit_price, sl, tp, pnl_money, pnl_pct, 
                     exit_time if exit_time else (datetime.now() if exit_price else None), duration_minutes, ticket),
                )
                conn.commit()
                return

        # Insert novo
        conn.execute(
            """
            INSERT INTO trades 
            (symbol, side, volume, entry_price, exit_price, sl, tp, pnl_money, pnl_pct, 
             reason, ml_reward, strategy, ml_confidence, ml_prediction, atr_pct,
             vix_level, order_flow_delta, duration_minutes, exit_time, ab_group, ticket, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                side,
                volume,
                entry_price,
                exit_price,
                sl,
                tp,
                pnl_money,
                pnl_pct,
                reason,
                ml_reward,
                strategy,
                ml_confidence,
                ml_prediction,
                atr_pct,
                vix_level,
                order_flow_delta,
                duration_minutes,
                exit_time if exit_time else (datetime.now() if exit_price else None),
                ab_group,
                ticket,
                timestamp if timestamp else datetime.now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_trades_by_date(target_date_str: str):
    """Busca todos os trades de uma data específica (formato 'YYYY-MM-DD')."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM trades WHERE date(timestamp) = date(?)"
    df = pd.read_sql_query(query, conn, params=(target_date_str,))
    conn.close()
    return df


def get_win_rate_report(lookback_days: int = 30) -> dict:
    """
    Gera relatório de win rate geral e por estratégia.
    """
    init_db()
    migrate_db()
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Win rate geral
    df = pd.read_sql_query(
        f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl_money) as total_pnl,
            AVG(pnl_pct) as avg_pnl_pct
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
    """,
        conn,
    )

    # Win rate por estratégia
    df_strategy = pd.read_sql_query(
        f"""
        SELECT 
            strategy,
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
            AVG(ml_confidence) as avg_ml_conf
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        GROUP BY strategy
    """,
        conn,
    )

    # Win rate por grupo AB
    df_ab = pd.read_sql_query(
        f"""
        SELECT 
            ab_group,
            COUNT(*) as total,
            SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins
        FROM trades
        WHERE date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        GROUP BY ab_group
    """,
        conn,
    )

    conn.close()

    total = df["total"].iloc[0] if len(df) > 0 else 0
    wins = df["wins"].iloc[0] if len(df) > 0 else 0
    total = int(total or 0)
    wins = int(wins or 0)

    result = {
        "period_days": lookback_days,
        "total_trades": total,
        "wins": wins,
        "losses": int(total - wins),
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "total_pnl": float(df["total_pnl"].iloc[0] or 0),
        "avg_pnl_pct": float(df["avg_pnl_pct"].iloc[0] or 0),
        "by_strategy": {},
        "by_ab_group": {},
    }

    for _, row in df_strategy.iterrows():
        strat = row["strategy"] or "UNKNOWN"
        wr = (row["wins"] / row["total"] * 100) if row["total"] > 0 else 0
        result["by_strategy"][strat] = {
            "total": int(row["total"]),
            "wins": int(row["wins"]),
            "win_rate": wr,
            "avg_ml_conf": float(row["avg_ml_conf"] or 0),
        }

    for _, row in df_ab.iterrows():
        grp = row["ab_group"] or "A"
        wr = (row["wins"] / row["total"] * 100) if row["total"] > 0 else 0
        result["by_ab_group"][grp] = {
            "total": int(row["total"]),
            "wins": int(row["wins"]),
            "win_rate": wr,
        }

    return result


def get_symbol_statistics(symbol: str, lookback_days: int = 30) -> dict:
    """📊 Estatísticas reais do símbolo."""
    try:
        init_db()
        migrate_db()
        conn = sqlite3.connect(DB_PATH)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        df = pd.read_sql_query(
            f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN pnl_money > 0 THEN 1 ELSE 0 END) as wins,
                AVG(ABS(tp - entry_price) / NULLIF(ABS(sl - entry_price), 0)) as avg_rr
            FROM trades
            WHERE symbol = ? AND date(timestamp) >= date('{cutoff}') AND exit_price IS NOT NULL
        """,
            conn,
            params=(symbol,),
        )
        conn.close()

        if len(df) > 0 and df["total"].iloc[0] >= 10:
            total = df["total"].iloc[0]
            wins = df["wins"].iloc[0]
            return {
                "win_rate": wins / total if total > 0 else 0.55,
                "avg_rr": float(df["avg_rr"].iloc[0] or 2.0),
                "total_trades": int(total),
                "last_updated": datetime.now(),
            }

        return {"win_rate": 0.55, "avg_rr": 2.0, "total_trades": 0}

    except Exception as e:
        logger.error(f"Erro stats {symbol}: {e}")
        return {"win_rate": 0.55, "avg_rr": 2.0, "total_trades": 0}


def _ensure_adaptive_config_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            ml_threshold REAL,
            kelly_multiplier REAL,
            spread_multiplier REAL,
            max_losses_per_symbol INTEGER
        )
    """
    )
    conn.commit()


def save_adaptive_config(config_data: dict) -> None:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_adaptive_config_table(conn)
        ts = config_data.get("timestamp") or datetime.now()
        if isinstance(ts, datetime):
            ts_value = ts.isoformat(sep=" ", timespec="seconds")
        else:
            ts_value = str(ts)

        conn.execute(
            """
            INSERT INTO adaptive_config
            (timestamp, ml_threshold, kelly_multiplier, spread_multiplier, max_losses_per_symbol)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ts_value,
                float(config_data.get("ml_threshold", 0.0) or 0.0),
                float(config_data.get("kelly_multiplier", 0.0) or 0.0),
                float(config_data.get("spread_multiplier", 0.0) or 0.0),
                int(config_data.get("max_losses_per_symbol", 0) or 0),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_adaptive_config() -> dict:
    """Retorna os últimos parâmetros salvos do Adaptive Intelligence, ou None se vazio."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_adaptive_config_table(conn)
        df = pd.read_sql_query(
            "SELECT * FROM adaptive_config ORDER BY timestamp DESC LIMIT 1",
            conn
        )
        if len(df) == 0:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Erro ao buscar o último adaptive_config: {e}")
        return None
    finally:
        conn.close()


def _ensure_adaptive_adjustments_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_adjustments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            old_parameters TEXT,
            new_parameters TEXT,
            recommendations TEXT,
            market_conditions TEXT
        )
        """
    )
    conn.commit()


def save_adaptive_adjustment(adjustment_data: dict) -> None:
    """Salva um histórico completo de ajuste de parâmetros."""
    import json
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_adaptive_adjustments_table(conn)
        ts = adjustment_data.get("timestamp") or datetime.now()
        if isinstance(ts, datetime):
            ts_value = ts.isoformat(sep=" ", timespec="seconds")
        else:
            ts_value = str(ts)

        old_params = json.dumps(adjustment_data.get("old_parameters", {}))
        new_params = json.dumps(adjustment_data.get("new_parameters", {}))
        recommendations = json.dumps(adjustment_data.get("recommendations", {}))
        market_conditions = json.dumps(adjustment_data.get("market_conditions", {}))

        conn.execute(
            """
            INSERT INTO adaptive_adjustments
            (timestamp, old_parameters, new_parameters, recommendations, market_conditions)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts_value, old_params, new_params, recommendations, market_conditions),
        )
        conn.commit()

        # Também salva no adaptive_config para que os valores padrão sejam atualizados
        new_p = adjustment_data.get("new_parameters", {})
        save_adaptive_config({
            "timestamp": ts_value,
            "ml_threshold": new_p.get("ml_confidence_threshold", 0.0),
            "kelly_multiplier": new_p.get("kelly_fraction_multiplier", 0.0),
            "spread_multiplier": new_p.get("spread_filter_multiplier", 0.0),
            "max_losses_per_symbol": new_p.get("max_losses_per_symbol", 0)
        })

    except Exception as e:
        logger.error(f"Erro ao salvar adaptive_adjustment: {e}")
    finally:
        conn.close()


def get_recent_adaptive_adjustments(limit: int = 100) -> list:
    """Busca o histórico mais recente de ajustes do Adaptive Intelligence."""
    import json
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_adaptive_adjustments_table(conn)
        df = pd.read_sql_query(
            f"SELECT * FROM adaptive_adjustments ORDER BY timestamp ASC LIMIT {limit}",
            conn
        )
        
        if len(df) == 0:
            return []
            
        adjustments = []
        for _, row in df.iterrows():
            try:
                # Convert string format isoformat to datetime
                ts_value = row["timestamp"]
                # Replace ' ' with 'T' before python 3.11 if it comes that way, 
                # but fromisoformat handles space separated in 3.11+
                try:
                    ts = datetime.fromisoformat(ts_value)
                except ValueError:
                    # quick fix
                    ts = datetime.fromisoformat(ts_value.replace(" ", "T"))

                adj = {
                    "timestamp": ts,
                    "old_parameters": json.loads(row["old_parameters"] or "{}"),
                    "new_parameters": json.loads(row["new_parameters"] or "{}"),
                    "recommendations": json.loads(row["recommendations"] or "{}"),
                    "market_conditions": json.loads(row["market_conditions"] or "{}")
                }
                
                # Adapta ao formato que parameter_history usa
                adj_format = {
                    "timestamp": adj["timestamp"],
                    "old_params": adj["old_parameters"],
                    "new_params": adj["new_parameters"],
                    "recommendations": adj["recommendations"]
                }
                
                adjustments.append(adj_format)
            except Exception as loop_e:
                logger.error(f"Erro ao parsear row de adjustment: {loop_e}")
                
        return adjustments
    except Exception as e:
        logger.error(f"Erro ao buscar recentes adaptive_adjustments: {e}")
        return []
    finally:
        conn.close()



def cleanup_invalid_symbols():
    """Remove trades that are not B3 stocks according to config.MONITORED_SYMBOLS."""
    import config
    allowed = config.MONITORED_SYMBOLS
    if not allowed:
        return
        
    conn = sqlite3.connect(DB_PATH)
    try:
        # Pega todos os símbolos no banco
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM trades", conn)
        invalid = [s for s in df['symbol'].tolist() if s not in allowed]
        
        if invalid:
            query = "DELETE FROM trades WHERE symbol IN ({})".format(','.join(['?']*len(invalid)))
            conn.execute(query, tuple(invalid))
            conn.commit()
            logger.info(f"🧹 Database cleanup: Removed {len(invalid)} non-B3 assets: {invalid}")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
    finally:
        conn.close()


def sync_trades_from_mt5():
    """
    Sincroniza o banco de dados SQLite com o histórico do MT5.
    Garante que trades fechados manualmente ou por falhas de log sejam capturados.
    """
    import MetaTrader5 as mt5
    import config
    
    logger.info("🔄 Iniciando sincronização MT5 -> Database...")
    
    # Inicializa MT5 se necessário
    if not mt5.initialize():
        init_params = {
            "path": getattr(config, "MT5_TERMINAL_PATH", None),
            "login": int(config.MT5_ACCOUNT) if config.MT5_ACCOUNT else 0,
            "password": str(config.MT5_PASSWORD or ""),
            "server": str(config.MT5_SERVER or ""),
            "timeout": 10000
        }
        init_params = {k: v for k, v in init_params.items() if v is not None}
        if not mt5.initialize(**init_params):
            logger.error(f"❌ Falha ao sincronizar: MT5 não inicializado {mt5.last_error()}")
            return

    try:
        # Busca trades desde o início do dia
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        to_date = datetime.now() + timedelta(minutes=1)
        
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            logger.info("✅ Nenhum deal encontrado para sincronizar hoje.")
            return

        for deal in deals:
            # FILTRO CRÍTICO: Somente Ações B3 (conforme MONITORED_SYMBOLS)
            if deal.symbol not in config.MONITORED_SYMBOLS:
                continue

            # Entry 0 = IN (Abertura), 1 = OUT (Fechamento)
            if deal.entry == 0: # Entrada
                save_trade(
                    symbol=deal.symbol,
                    side="BUY" if deal.type == mt5.ORDER_TYPE_BUY else "SELL",
                    volume=deal.volume,
                    entry_price=deal.price,
                    ticket=deal.position_id, # Usamos position_id para agrupar entry/exit
                    timestamp=datetime.fromtimestamp(deal.time)
                )
            elif deal.entry == 1 or deal.entry == 2: # Saída ou In/Out
                # Para saída, atualizamos o registro existente
                save_trade(
                    symbol=deal.symbol,
                    side="BUY" if deal.type == mt5.ORDER_TYPE_BUY else "SELL",
                    volume=deal.volume,
                    entry_price=0, # Não usado no update
                    exit_price=deal.price,
                    pnl_money=deal.profit + deal.commission + deal.swap,
                    ticket=deal.position_id,
                    exit_time=datetime.fromtimestamp(deal.time)
                )

        logger.info(f"✅ Sincronização concluída: {len(deals)} deals processados.")
    except Exception as e:
        logger.error(f"❌ Erro na sincronização: {e}")


def get_trades_since(start_time: datetime):
    init_db()
    migrate_db()
    
    # Sincroniza antes de buscar para garantir winrate real no relatório
    try:
        sync_trades_from_mt5()
        cleanup_invalid_symbols() # Garante que nada estranho ficou no banco
    except Exception as e:
        logger.warning(f"⚠️ Sincronização falhou antes de get_trades_since: {e}")

    if not isinstance(start_time, datetime):
        raise TypeError("start_time deve ser datetime")

    start_value = start_time.isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM trades
            WHERE exit_time IS NOT NULL
              AND datetime(exit_time) >= datetime(?)
            ORDER BY datetime(exit_time) DESC
            """,
            conn,
            params=(start_value,),
        )
    finally:
        conn.close()

    if len(df) == 0:
        return []

    trades = []
    for rec in df.to_dict(orient="records"):
        exit_time_value = rec.get("exit_time")
        if isinstance(exit_time_value, str) and exit_time_value:
            try:
                rec["exit_time"] = datetime.fromisoformat(exit_time_value)
            except Exception:
                pass

        timestamp_value = rec.get("timestamp")
        if isinstance(timestamp_value, str) and timestamp_value:
            try:
                rec["timestamp"] = datetime.fromisoformat(timestamp_value)
            except Exception:
                pass

        if "net_pnl" not in rec:
            rec["net_pnl"] = rec.get("pnl_money", 0.0)

        trades.append(rec)

    return trades
