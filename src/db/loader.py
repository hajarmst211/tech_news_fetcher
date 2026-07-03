import hashlib
from datetime import datetime

import psycopg2
from psycopg2.extras import Json

from .database import get_conn, return_conn


def _extract_metrics(record: dict) -> dict:
    metric_keys = {
        "comments_count", "score", "stars", "reactions",
        "points", "descendants", "positive_reactions_count",
        "public_reactions_count",
    }
    metrics = {}
    for k in list(record.keys()):
        if k in metric_keys and isinstance(record[k], (int, float)):
            metrics[k] = record.pop(k)
    return metrics


def _extract_tags(record: dict) -> list[str] | None:
    for key in ("tags", "tag_list", "categories"):
        val = record.pop(key, None)
        if val:
            if isinstance(val, str):
                return [t.strip() for t in val.split(",") if t.strip()]
            if isinstance(val, list):
                return [str(t) for t in val if t]
    return None


def _compute_content_hash(content: str | None) -> str | None:
    if not content:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_item(record: dict, source_type: str) -> dict:
    raw = dict(record)

    external_id = str(raw.pop("id", raw.pop("external_id", raw.pop("tag_name", ""))))
    title = raw.pop("title", raw.pop("name", ""))
    summary = raw.pop("summary", raw.pop("description", ""))
    url = raw.pop("url", raw.pop("link", raw.pop("html_url", "")))
    author = _extract_author(raw)
    published_at = _parse_timestamp(raw.pop("published_at", raw.pop("published", raw.pop("created_at", raw.pop("date", None)))))
    updated_at = _parse_timestamp(raw.pop("updated_at", raw.pop("updated", raw.pop("last_modified", None))))
    content = raw.pop("content", raw.pop("body_markdown", raw.pop("body", raw.pop("text", raw.pop("article_content", None)))))
    content_hash = _compute_content_hash(content)

    tags = _extract_tags(raw)
    metrics = _extract_metrics(raw)
    extra = {k: v for k, v in raw.items() if v is not None and k != "author"}

    return {
        "external_id": external_id,
        "title": title,
        "summary": summary,
        "url": url,
        "author": author,
        "published_at": published_at,
        "updated_at": updated_at,
        "content": content,
        "content_hash": content_hash,
        "tags": tags,
        "metrics": metrics,
        "extra": extra,
    }


def _extract_author(raw: dict) -> str | None:
    author = raw.pop("author", None)
    if author and isinstance(author, str):
        return author
    authors = raw.pop("authors", None)
    if authors and isinstance(authors, list):
        return ", ".join(str(a) for a in authors if a)
    by = raw.pop("by", None)
    if by and isinstance(by, str):
        return by
    creator = raw.pop("creator", None)
    if creator and isinstance(creator, str):
        return creator
    return None


def _parse_timestamp(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return datetime.utcfromtimestamp(val).isoformat()
        except (OSError, ValueError):
            return None
    if isinstance(val, str):
        return val
    return None


def insert_items(source_id: int, records: list[dict], source_type: str) -> int:
    if not records:
        return 0

    conn = get_conn()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for rec in records:
                norm = _normalize_item(rec, source_type)
                try:
                    cur.execute(
                        """
                        INSERT INTO items
                            (source_id, external_id, title, summary, url, author,
                             published_at, updated_at, content, content_hash,
                             tags, metrics, extra)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (source_id, external_id) DO NOTHING
                        """,
                        (
                            source_id,
                            norm["external_id"],
                            norm["title"],
                            norm["summary"],
                            norm["url"],
                            norm["author"],
                            norm["published_at"],
                            norm["updated_at"],
                            norm["content"],
                            norm["content_hash"],
                            norm["tags"],
                            Json(norm["metrics"]) if norm["metrics"] else None,
                            Json(norm["extra"]) if norm["extra"] else None,
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                except psycopg2.Error as e:
                    print(f"  [DB ERROR] Skipping item {norm.get('external_id')}: {e}")
        conn.commit()
    finally:
        return_conn(conn)

    if inserted:
        print(f"  [DB] Inserted {inserted}/{len(records)} items")
    return inserted


def insert_vulnerabilities(source_id: int, records: list[dict]) -> int:
    if not records:
        return 0

    conn = get_conn()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for rec in records:
                try:
                    cur.execute(
                        """
                        INSERT INTO vulnerabilities
                            (cve_id, source_id, description, published_at,
                             last_modified, cvss_score, cvss_severity,
                             cvss_vector, raw)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (cve_id) DO NOTHING
                        """,
                        (
                            rec["cve_id"],
                            source_id,
                            rec.get("description"),
                            rec.get("published_at"),
                            rec.get("last_modified"),
                            rec.get("cvss_score"),
                            rec.get("cvss_severity"),
                            rec.get("cvss_vector"),
                            Json(rec.get("raw")),
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                except psycopg2.Error as e:
                    print(f"  [DB ERROR] Skipping vulnerability {rec.get('cve_id')}: {e}")
        conn.commit()
    finally:
        return_conn(conn)

    if inserted:
        print(f"  [DB] Inserted {inserted}/{len(records)} vulnerabilities")
    return inserted


def insert_hn_seen_ids(hn_ids: list[int]) -> int:
    if not hn_ids:
        return 0

    conn = get_conn()
    inserted = 0
    try:
        with conn.cursor() as cur:
            for hid in hn_ids:
                try:
                    cur.execute(
                        "INSERT INTO hn_seen_ids (hn_id) VALUES (%s) ON CONFLICT DO NOTHING",
                        (hid,),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                except psycopg2.Error as e:
                    print(f"  [DB ERROR] Skipping hn_id {hid}: {e}")
        conn.commit()
    finally:
        return_conn(conn)

    if inserted:
        print(f"  [DB] Inserted {inserted} new HN IDs")
    return inserted
