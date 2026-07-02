import os
from pathlib import Path

import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/tech_news",
)

_connection_pool = None


def get_pool():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(1, 20, DATABASE_URL)
    return _connection_pool


def get_conn():
    return get_pool().getconn()


def return_conn(conn):
    get_pool().putconn(conn)


def init_db():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        conn.commit()
        print("  [DB] Tables initialised")
    except psycopg2.Error as e:
        print(f"  [DB ERROR] Failed to initialise tables: {e}")
        raise
    finally:
        return_conn(conn)


def ensure_source(name: str, source_type: str, category: str | None = None) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sources WHERE name = %s", (name,))
            row = cur.fetchone()
            if row:
                return row[0]

            cur.execute(
                "INSERT INTO sources (name, source_type, category) VALUES (%s, %s, %s) RETURNING id",
                (name, source_type, category),
            )
            conn.commit()
            sid = cur.fetchone()[0]
            print(f"  [DB] Created source '{name}' (id={sid}, type={source_type})")
            return sid
    finally:
        return_conn(conn)
