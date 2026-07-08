#!/usr/bin/env python3
"""Generate comprehensive stats.json from the database for the dashboard."""

import json
import os
from collections import defaultdict
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

load_dotenv()

OUTPUT = "stats.json"

def main():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    stats = {}

    # ── Sources ──
    cur.execute("""
        SELECT s.name, s.source_type, s.category,
               COUNT(i.id) AS item_count
        FROM sources s
        LEFT JOIN items i ON i.source_id = s.id
        GROUP BY s.id, s.name, s.source_type, s.category
        ORDER BY s.name
    """)
    sources = []
    total_sources_with_items = 0
    source_type_counts = defaultdict(int)
    for name, stype, cat, item_count in cur.fetchall():
        sources.append({
            "name": name,
            "type": stype,
            "category": cat,
            "items": item_count,
        })
        source_type_counts[stype] += 1
        if item_count > 0:
            total_sources_with_items += 1

    stats["sources"] = {
        "total": len(sources),
        "with_items": total_sources_with_items,
        "by_type": dict(source_type_counts),
        "list": sources,
    }

    # ── Item counts ──
    cur.execute("SELECT COUNT(*) FROM items")
    total_items = cur.fetchone()[0]

    def safe_count(table):
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]
        except Exception:
            conn.rollback()
            return 0

    total_comments = safe_count("comments")
    total_vulns = safe_count("vulnerabilities")
    total_hn_seen = safe_count("hn_seen_ids")

    stats["db_counts"] = {
        "items": total_items,
        "comments": total_comments,
        "vulnerabilities": total_vulns,
        "hn_seen_ids": total_hn_seen,
    }

    # ── Summary / Content breakdown ──
    cur.execute("""
        SELECT
            CASE
                WHEN summary IS NOT NULL AND TRIM(summary) != ''
                     AND content IS NOT NULL AND TRIM(content) != ''
                    THEN 'summary AND content'
                WHEN summary IS NOT NULL AND TRIM(summary) != ''
                     AND (content IS NULL OR TRIM(content) = '')
                    THEN 'summary, no content'
                WHEN (summary IS NULL OR TRIM(summary) = '')
                     AND content IS NOT NULL AND TRIM(content) != ''
                    THEN 'no summary, content'
                ELSE 'no summary, no content'
            END AS bucket,
            COUNT(*) AS cnt
        FROM items
        GROUP BY bucket
        ORDER BY bucket
    """)
    breakdown = {}
    for bucket, cnt in cur.fetchall():
        breakdown[bucket] = cnt
    for b in ("summary AND content", "summary, no content", "no summary, content", "no summary, no content"):
        breakdown.setdefault(b, 0)

    stats["summary_content_breakdown"] = breakdown

    # ── Summarizer candidates (items the summarizer could process) ──
    cur.execute("""
        SELECT COUNT(*) FROM items
        WHERE (summary IS NULL OR TRIM(summary) = '')
          AND content IS NOT NULL AND TRIM(content) != ''
    """)
    stats["summarizer_candidates"] = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM items
        WHERE summary IS NOT NULL AND TRIM(summary) != ''
    """)
    stats["items_with_summary"] = cur.fetchone()[0]

    # ── Summarizer before/after impact ──
    # "Before" state estimates what the DB looked like before the summarizer ran:
    #   - items now with summary AND content had content but no summary
    #   - items with summary, no content never had content (RSS descriptions etc.)
    #   - items with neither are unchanged
    before_no_summary_content = breakdown.get("no summary, content", 0) + breakdown.get("summary AND content", 0)
    before_summary_no_content = breakdown.get("summary, no content", 0)
    before_neither = breakdown.get("no summary, no content", 0)

    stats["summarizer_impact"] = {
        "before": {
            "content_no_summary": before_no_summary_content,
            "summary_no_content": before_summary_no_content,
            "neither": before_neither,
        },
        "after": {
            "summary_and_content": breakdown.get("summary AND content", 0),
            "summary_no_content": breakdown.get("summary, no content", 0),
            "content_no_summary": breakdown.get("no summary, content", 0),
            "neither": breakdown.get("no summary, no content", 0),
        },
    }

    # ── Items with comments in DB ──
    stats["items_with_comments"] = safe_count("comments")
    try:
        cur.execute("""
            SELECT COUNT(DISTINCT item_id) FROM comments
        """)
        stats["items_with_comments"] = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        stats["items_with_comments"] = 0

    try:
        cur.execute("""
            SELECT s.name, COUNT(c.id) AS comment_count
            FROM comments c
            JOIN items i ON i.id = c.item_id
            JOIN sources s ON s.id = i.source_id
            GROUP BY s.name
            ORDER BY comment_count DESC
        """)
        stats["comments_by_source"] = [{"source": row[0], "count": row[1]} for row in cur.fetchall()]
    except Exception:
        conn.rollback()
        stats["comments_by_source"] = []

    # ── Themes ──
    cur.execute("""
        SELECT theme, COUNT(*) AS cnt
        FROM items
        WHERE theme IS NOT NULL AND TRIM(theme) != ''
        GROUP BY theme
        ORDER BY cnt DESC, theme
    """)
    themes = [{"theme": row[0], "count": row[1]} for row in cur.fetchall()]
    stats["themes"] = {
        "total_distinct": len(themes),
        "list": themes,
    }

    # ── Items per source type ──
    cur.execute("""
        SELECT s.source_type, COUNT(i.id) AS cnt
        FROM sources s
        LEFT JOIN items i ON i.source_id = s.id
        GROUP BY s.source_type
        ORDER BY cnt DESC
    """)
    stats["items_by_source_type"] = dict(cur.fetchall())

    # ── Items per source (top N) ──
    cur.execute("""
        SELECT s.name, COUNT(i.id) AS cnt
        FROM sources s
        JOIN items i ON i.source_id = s.id
        GROUP BY s.name
        ORDER BY cnt DESC
        LIMIT 20
    """)
    stats["top_sources_by_items"] = [{"name": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.close()
    conn.close()

    with open(OUTPUT, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Stats written to {OUTPUT}")
    print(f"  Sources: {stats['sources']['total']} total, {stats['sources']['with_items']} with items")
    print(f"  Items: {total_items}, Comments: {total_comments}, Vulns: {total_vulns}, HN seen: {total_hn_seen}")
    print(f"  Summary/content breakdown: {breakdown}")
    print(f"  Distinct themes: {stats['themes']['total_distinct']}")

if __name__ == "__main__":
    main()
