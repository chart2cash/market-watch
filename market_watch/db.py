from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool


SQLITE_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS watchlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watchlist_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        watchlist_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        target_buy REAL,
        alert_below REAL,
        alert_above REAL,
        thesis TEXT,
        conviction INTEGER NOT NULL DEFAULT 3,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(watchlist_id, symbol),
        FOREIGN KEY(watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
        quantity REAL NOT NULL CHECK(quantity > 0),
        price REAL NOT NULL CHECK(price >= 0),
        fees REAL NOT NULL DEFAULT 0,
        trade_date TEXT NOT NULL,
        account TEXT,
        strategy TEXT,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        note_type TEXT NOT NULL DEFAULT 'General',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        alert_type TEXT NOT NULL CHECK(alert_type IN ('BELOW','ABOVE')),
        threshold REAL NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        notes TEXT,
        last_triggered_at TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

POSTGRES_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS watchlists (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watchlist_items (
        id BIGSERIAL PRIMARY KEY,
        watchlist_id BIGINT NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
        symbol TEXT NOT NULL,
        target_buy DOUBLE PRECISION,
        alert_below DOUBLE PRECISION,
        alert_above DOUBLE PRECISION,
        thesis TEXT,
        conviction INTEGER NOT NULL DEFAULT 3 CHECK(conviction BETWEEN 1 AND 5),
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(watchlist_id, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
        quantity DOUBLE PRECISION NOT NULL CHECK(quantity > 0),
        price DOUBLE PRECISION NOT NULL CHECK(price >= 0),
        fees DOUBLE PRECISION NOT NULL DEFAULT 0,
        trade_date TEXT NOT NULL,
        account TEXT,
        strategy TEXT,
        notes TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_notes (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        note_type TEXT NOT NULL DEFAULT 'General',
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        alert_type TEXT NOT NULL CHECK(alert_type IN ('BELOW','ABOVE')),
        threshold DOUBLE PRECISION NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        notes TEXT,
        last_triggered_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


class Database:
    """A small persistence layer that works with SQLite or hosted PostgreSQL."""

    def __init__(self, path: Path | None = None, database_url: str = ""):
        self.path = Path(path) if path is not None else None
        self.database_url = self._normalize_url(database_url.strip()) if database_url else ""
        self.is_postgres = bool(self.database_url)
        self.engine = self._build_engine()
        self.initialize()

    @staticmethod
    def _normalize_url(url: str) -> str:
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if ("supabase.co" in url or "pooler.supabase.com" in url) and "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        return url

    def _build_engine(self) -> Engine:
        if self.is_postgres:
            return create_engine(
                self.database_url,
                future=True,
                pool_pre_ping=True,
                poolclass=NullPool,
            )

        if self.path is None:
            raise ValueError("A SQLite path is required when DATABASE_URL is not configured.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{self.path}", future=True)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    @property
    def storage_label(self) -> str:
        return "Hosted PostgreSQL" if self.is_postgres else "Local SQLite"

    def initialize(self) -> None:
        schema = POSTGRES_SCHEMA if self.is_postgres else SQLITE_SCHEMA
        with self.engine.begin() as conn:
            for statement in schema:
                conn.execute(text(statement))
            conn.execute(text("INSERT INTO watchlists(name) VALUES (:name) ON CONFLICT(name) DO NOTHING"), {"name": "Core Watchlist"})
            conn.execute(text("INSERT INTO watchlists(name) VALUES (:name) ON CONFLICT(name) DO NOTHING"), {"name": "Speculative Growth"})
            core_id = conn.execute(text("SELECT id FROM watchlists WHERE name=:name"), {"name": "Core Watchlist"}).scalar_one()
            speculative_id = conn.execute(text("SELECT id FROM watchlists WHERE name=:name"), {"name": "Speculative Growth"}).scalar_one()
            seed_items = [
                (core_id, "SPY", None, None, None, "Broad-market trend reference", 5),
                (core_id, "QQQ", None, None, None, "Growth and technology trend reference", 5),
                (speculative_id, "IONQ", 36.0, 36.0, 55.0, "Long-term quantum thesis; demand-zone re-entry focus", 5),
                (speculative_id, "RKLB", None, None, None, "Space infrastructure watch", 3),
                (speculative_id, "PLTR", None, None, None, "High-growth software watch", 4),
            ]
            insert = text(
                """
                INSERT INTO watchlist_items
                (watchlist_id,symbol,target_buy,alert_below,alert_above,thesis,conviction)
                VALUES (:watchlist_id,:symbol,:target_buy,:alert_below,:alert_above,:thesis,:conviction)
                ON CONFLICT(watchlist_id,symbol) DO NOTHING
                """
            )
            conn.execute(
                insert,
                [
                    {
                        "watchlist_id": row[0],
                        "symbol": row[1],
                        "target_buy": row[2],
                        "alert_below": row[3],
                        "alert_above": row[4],
                        "thesis": row[5],
                        "conviction": row[6],
                    }
                    for row in seed_items
                ],
            )

    @staticmethod
    def _named_sql(sql: str, params: tuple = ()) -> tuple[str, dict]:
        parts = sql.split("?")
        expected = len(parts) - 1
        if expected != len(params):
            if expected == 0 and not params:
                return sql, {}
            raise ValueError(f"SQL expected {expected} parameters but received {len(params)}.")
        if expected == 0:
            return sql, {}
        rebuilt = parts[0]
        values: dict[str, object] = {}
        for index, value in enumerate(params):
            key = f"p{index}"
            rebuilt += f":{key}" + parts[index + 1]
            values[key] = value
        return rebuilt, values

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        named_sql, values = self._named_sql(sql, params)
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(named_sql), conn, params=values)

    def execute(self, sql: str, params: tuple = ()) -> None:
        named_sql, values = self._named_sql(sql, params)
        with self.engine.begin() as conn:
            conn.execute(text(named_sql), values)

    def executemany(self, sql: str, params: Iterable[tuple]) -> None:
        rows = list(params)
        if not rows:
            return
        named_sql, _ = self._named_sql(sql, rows[0])
        values = [self._named_sql(sql, row)[1] for row in rows]
        with self.engine.begin() as conn:
            conn.execute(text(named_sql), values)

    def watchlists(self) -> pd.DataFrame:
        return self.query("SELECT * FROM watchlists ORDER BY name")

    def watchlist_items(self, watchlist_id: int | None = None) -> pd.DataFrame:
        sql = """
        SELECT wi.*, w.name AS watchlist_name
        FROM watchlist_items wi JOIN watchlists w ON w.id=wi.watchlist_id
        """
        params: tuple = ()
        if watchlist_id is not None:
            sql += " WHERE wi.watchlist_id=?"
            params = (watchlist_id,)
        sql += " ORDER BY w.name, wi.symbol"
        return self.query(sql, params)

    def trades(self) -> pd.DataFrame:
        return self.query("SELECT * FROM trades ORDER BY trade_date DESC, id DESC")

    def notes(self, symbol: str | None = None) -> pd.DataFrame:
        if symbol:
            return self.query(
                "SELECT * FROM research_notes WHERE symbol=? ORDER BY created_at DESC, id DESC",
                (symbol.upper(),),
            )
        return self.query("SELECT * FROM research_notes ORDER BY created_at DESC, id DESC")

    def alerts(self) -> pd.DataFrame:
        return self.query("SELECT * FROM alerts ORDER BY active DESC, symbol, threshold")
