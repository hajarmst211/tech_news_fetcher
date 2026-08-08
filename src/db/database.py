import os
from collections import defaultdict
from pathlib import Path

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from dotenv import load_dotenv 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

load_dotenv(PROJECT_ROOT.parent / ".env") 
DATABASE_URL = os.getenv("DATABASE_URL")

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

            migrations = [
                "ALTER TABLE sources ADD COLUMN IF NOT EXISTS last_status_code INTEGER",
                "ALTER TABLE items ADD COLUMN IF NOT EXISTS topics TEXT[]",
                "ALTER TABLE items ADD COLUMN IF NOT EXISTS sentiment JSONB",
                "CREATE INDEX IF NOT EXISTS idx_items_topics ON items USING GIN (topics)",
            ]
            for migration in migrations:
                cur.execute(migration)
        conn.commit()
        print("  [DB] Tables initialised")
    except psycopg2.Error as e:
        print(f"  [DB ERROR] Failed to initialise tables: {e}")
        raise
    finally:
        return_conn(conn)


def update_source_status(name: str, status: str, status_code: int | None = None) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources
                SET last_fetch_status = %s,
                    last_status_code = %s,
                    last_fetched_at = now()
                WHERE name = %s
                """,
                (status, status_code, name),
            )
            if cur.rowcount == 0:
                print(f"  [WARN] update_source_status: no source row found for name={name!r}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  [ERROR] update_source_status failed for {name!r}: {e}")
        raise
    finally:
            get_pool().putconn(conn)
    try:
            get_pool().putconn(conn)
    except Exception as e:
            print(f"  [ERROR] Failed to return connection to pool for {name!r}: {e}")


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


def search_today_items(query: str) -> list[dict]:
    sql = """
        SELECT
            i.id,
            i.title,
            i.summary,
            i.topics,
            i.url,
            i.published_at,
            s.name AS source_name,
            (
                SELECT json_object_agg(sentiment_label, cnt)
                FROM (
                    SELECT sentiment_label, count(*) AS cnt
                    FROM comments c
                    WHERE c.item_id = i.id AND c.sentiment_label IS NOT NULL
                    GROUP BY sentiment_label
                ) sub
            ) AS comment_sentiment,
            (SELECT count(*) FROM comments c WHERE c.item_id = i.id) AS comment_count
        FROM items i
        JOIN sources s ON s.id = i.source_id
        WHERE i.fetched_at::date = CURRENT_DATE
          AND (
              %(q)s = ''
              OR i.title ILIKE %(qp)s
              OR i.summary ILIKE %(qp)s
              OR i.content ILIKE %(qp)s
          )
        ORDER BY i.published_at DESC NULLS LAST
        LIMIT 50
    """
    params = {"q": query, "qp": f"%{query}%"}
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        return_conn(conn)

    results = []
    for row in rows:
        topics = row["topics"] or []
        results.append({
            "id": row["id"],
            "title": row["title"],
            "summary": row["summary"],
            "topics": topics[:3],
            "url": row["url"],
            "published_at": row["published_at"].isoformat() if row["published_at"] else None,
            "source_name": row["source_name"],
            "comment_count": row["comment_count"],
            "comment_sentiment": row["comment_sentiment"] or {},
        })
    return results


def get_item_texts(days: int) -> list[str]:
    sql = """
        SELECT title, coalesce(summary, '') AS summary, coalesce(content, '') AS content
        FROM items
        WHERE fetched_at >= now() - (%(days)s || ' days')::interval
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"days": days})
            rows = cur.fetchall()
    finally:
        return_conn(conn)
    return [f"{r['title']} {r['summary']} {r['content']}" for r in rows]


def top_topics(days: int, limit: int) -> list[dict]:
    sql = """
        SELECT topic, count(*) AS cnt
        FROM (
            SELECT unnest(topics) AS topic
            FROM items
            WHERE fetched_at >= now() - (%(days)s || ' days')::interval
        ) t
        GROUP BY topic
        ORDER BY cnt DESC
        LIMIT %(limit)s
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"days": days, "limit": limit})
            rows = cur.fetchall()
    finally:
        return_conn(conn)
    return [{"topic": r["topic"], "count": r["cnt"]} for r in rows]


def distinct_topics() -> list[str]:
    sql = """
        SELECT DISTINCT unnest(topics) AS topic
        FROM items
        WHERE topics IS NOT NULL
        ORDER BY topic
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        return_conn(conn)
    return [r[0] for r in rows]


def sentiment_trend(topic: str, days: int) -> dict:
    sql = """
        SELECT
            date_trunc('day', c.published_at)::date AS day,
            c.sentiment_label,
            count(*) AS cnt
        FROM comments c
        JOIN items i ON i.id = c.item_id
        WHERE c.sentiment_label IS NOT NULL
          AND c.published_at IS NOT NULL
          AND c.published_at >= now() - (%(days)s || ' days')::interval
          AND (%(topic)s = '' OR %(topic)s = ANY(i.topics))
        GROUP BY day, c.sentiment_label
        ORDER BY day
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"days": days, "topic": topic})
            rows = cur.fetchall()
    finally:
        return_conn(conn)

    by_day = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for r in rows:
        by_day[r["day"].isoformat()][r["sentiment_label"]] = r["cnt"]

    days_sorted = sorted(by_day.keys())
    return {
        "days": days_sorted,
        "positive": [by_day[d]["positive"] for d in days_sorted],
        "neutral": [by_day[d]["neutral"] for d in days_sorted],
        "negative": [by_day[d]["negative"] for d in days_sorted],
    }
