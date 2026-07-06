import json
from pathlib import Path
from dotenv import load_dotenv
import os
import psycopg2
from collections import OrderedDict

load_dotenv()
DATA_DIR = Path("data/2026-07-03")
DB_URL = os.environ["DATABASE_URL"]

SOURCE_GROUPING = {
    "devto": "Dev.to",
    "arxiv": "ArXiv",
    "github": "GitHub",
    "hacker_news": "Hacker News",
    "reddit": "Reddit",
    "infoq": "InfoQ",
    "dzone": "DZone",
    "microsoft_net_blog": "Microsoft .NET Blog",
    "venturebeat": "VentureBeat",
    "light_reading": "Light Reading",
    "the_hacker_news": "The Hacker News",
    "dark_reading": "Dark Reading",
    "schneier_on_security": "Schneier on Security",
    "national_vulnerability_database": "NVD",
}

def group_source(filename: str) -> tuple[str, str]:
    stem = filename.replace(".json", "")
    is_comments = "_comments" in stem
    base = stem.replace("_comments", "")
    for prefix, group_name in SOURCE_GROUPING.items():
        if base.startswith(prefix) or base == prefix:
            return group_name, stem
    return stem.split("_")[0].title(), stem

files_data: dict[str, dict] = OrderedDict()
comments_sources: set = set()

for f in sorted(DATA_DIR.glob("*.json")):
    group, stem = group_source(f.name)
    is_comments = "_comments" in stem
    items = json.loads(f.read_text())
    if not isinstance(items, list):
        items = [items]
    count = sum(1 for i in items if isinstance(i, dict))

    if group not in files_data:
        files_data[group] = {"files": [], "has_comments": False}
    files_data[group]["files"].append({"file": f.name, "items": count, "is_comments": is_comments})
    if is_comments:
        files_data[group]["has_comments"] = True

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

# Items per table
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
table_counts = {}
for t in tables:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        table_counts[t] = cur.fetchone()[0]
    except Exception:
        table_counts[t] = 0

# Summary/content categories (excluding devto comments)
cur.execute("""
    SELECT category, COUNT(*) AS count FROM (
        SELECT
            CASE
                WHEN i.summary IS NULL AND i.content IS NULL THEN 'no summary, no content'
                WHEN i.summary IS NULL AND i.content IS NOT NULL THEN 'no summary, content'
                WHEN i.summary IS NOT NULL AND i.content IS NULL THEN 'summary, no content'
                WHEN i.summary IS NOT NULL AND i.content IS NOT NULL THEN 'summary, content'
            END AS category
        FROM items i
        JOIN sources s ON i.source_id = s.id
        WHERE s.name NOT LIKE '%%_comments'
    ) sub
    GROUP BY category
    ORDER BY category
""")
db_categories = {row[0]: row[1] for row in cur.fetchall()}

# Total items in DB (excluding comments sources)
cur.execute("""
    SELECT COUNT(*) FROM items i
    JOIN sources s ON i.source_id = s.id
    WHERE s.name NOT LIKE '%%_comments'
""")
db_total = cur.fetchone()[0]

cur.close()
conn.close()

output = {
    "files": files_data,
    "db": {
        "table_counts": table_counts,
        "categories": db_categories,
        "total_items_non_comments": db_total,
    }
}

Path("stats.json").write_text(json.dumps(output, indent=2))
print("stats.json generated")
