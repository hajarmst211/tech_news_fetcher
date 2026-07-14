import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db.database import init_db, ensure_source, get_conn, return_conn
from db.loader import (
    insert_items, 
    insert_vulnerabilities, 
    insert_hn_seen_ids, 
    insert_comments, 
    update_source_metadata
)


def normalize_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Returns only the 'YYYY-MM-DD' portion
        return dt.date().isoformat()
    except (ValueError, TypeError):
        if isinstance(date_str, str) and len(date_str) >= 10:
            if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                return date_str[:10]
        return date_str


EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed / Supplement
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000200D"             # Zero-width joiner
    "]+", flags=re.UNICODE
)


def clean_text(text: str, remove_emojis: bool = False) -> str:
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^(#{1,6})\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    if remove_emojis:
        text = EMOJI_PATTERN.sub('', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_value(value):
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, str):
        return clean_text(value, remove_emojis=True)
    return value


def dedup_records(records: list[dict], key: str = "id") -> list[dict]:
    seen = set()
    result = []
    for rec in records:
        k = rec.get(key)
        if k is not None and k not in seen:
            seen.add(k)
            result.append(rec)
    return result


def extract_nvd_fields(cve_item: dict) -> dict:
    cve = cve_item.get("cve", {})

    en_desc = ""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            en_desc = d.get("value", "")
            break

    cvss = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key)
        if entries:
            cvss = entries[0].get("cvssData")
            break

    return {
        "cve_id": cve.get("id", ""),
        "description": en_desc,
        "published_at": normalize_date(cve.get("published")),
        "last_modified": normalize_date(cve.get("lastModified")),
        "cvss_score": cvss.get("baseScore") if cvss else None,
        "cvss_severity": cvss.get("baseSeverity") if cvss else None,
        "cvss_vector": cvss.get("vectorString") if cvss else None,
        "raw": cve_item,
    }


SOURCE_HANDLERS = {
    "nvd": extract_nvd_fields,
}


def detect_source_type(filename_stem: str) -> str:
    stem = filename_stem.lower()
    if "national_vulnerability_database" in stem or stem == "nvd":
        return "nvd"
    if stem.startswith("devto"):
        return "devto"
    if stem.startswith("arxiv"):
        return "arxiv"
    if "github" in stem:
        return "github_release"
    if "hacker_news" in stem or stem.startswith("hn_"):
        return "hn"
    return "rss"


def extract_category(filename_stem: str) -> str | None:
    stem = filename_stem.lower()
    for prefix in ("devto_", "arxiv_", "reddit_"):
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            return rest if rest else None
    return None


def clean_record(record: dict, source_type: str | None = None) -> dict:
    if source_type and source_type in SOURCE_HANDLERS:
        return SOURCE_HANDLERS[source_type](record)
    return clean_value(record)


# ── File processing ──────────────────────────────────────────────

def _process_comments_file(stem: str, data: dict, source_type: str) -> None:
    parent_stem = stem.removesuffix("_comments")
    if parent_stem == stem:
        print(f"  [WARN] Cannot determine parent source for '{stem}'")
        return

    parent_category = extract_category(parent_stem)
    parent_source_id = ensure_source(parent_stem, source_type, parent_category)
    ensure_source(stem, source_type, extract_category(stem))

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            article_ids = list(data.keys())
            cur.execute(
                f"SELECT id, external_id FROM items WHERE source_id = %s AND external_id = ANY (%s)",
                (parent_source_id, list(article_ids)),
            )
            item_map = {row[1]: row[0] for row in cur.fetchall()}
    finally:
        return_conn(conn)

    if not item_map:
        print(f"  [WARN] No parent items found for '{stem}' — skipping")
        return

    records = []
    for article_eid, comments in data.items():
        item_id = item_map.get(str(article_eid))
        if item_id is None:
            continue

        if isinstance(comments, dict):
            comments = comments.values()
        for comment in comments:
            if not isinstance(comment, dict):
                continue

            external_id = comment.get("id_code", "")
            body_text = clean_text(comment.get("body_html", ""))
            published_at = normalize_date(comment.get("created_at"))

            extra = {}
            for k, v in comment.items():
                if k not in ("id_code", "created_at", "body_html", "user", "author"):
                    extra[k] = v

            records.append({
                "item_id": item_id,
                "external_id": external_id,
                "author": comment.get("author"),
                "body_text": body_text,
                "published_at": published_at,
                "extra": extra or None,
            })

    if records:
        insert_comments(records)
        print(f"  [OK]   Inserted {len(records)} comments for '{stem}'")


def clean_file(filepath: Path) -> None:
    print(f"\n  Processing: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    stem = filepath.stem
    source_type = detect_source_type(stem)
    source_name = stem
    category = extract_category(stem)

    fetched_at = None
    status = None
    status_code = None
    records = []

    # Unwrap structured metadata wrappers
    if isinstance(raw_data, dict):
        fetched_at = normalize_date(raw_data.get("fetched_at"))
        status = raw_data.get("status") or raw_data.get("fetch_status")
        status_code = raw_data.get("status_code") or raw_data.get("last_status_code")

        payload = None
        for key in ("data", "results", "items", "comments", "articles"):
            if key in raw_data and isinstance(raw_data[key], (list, dict)):
                payload = raw_data[key]
                break

        if payload is not None:
            records_data = payload
        else:
            # If there's no metadata key signature, consider the dictionary itself a single item
            if not any(k in raw_data for k in ("fetched_at", "status", "status_code")):
                records_data = raw_data
            else:
                records_data = []
    else:
        records_data = raw_data

    # Comment sub-processing
    if stem.endswith("_comments"):
        if isinstance(records_data, dict):
            _process_comments_file(stem, records_data, source_type)
            if fetched_at or status or status_code:
                comment_source_id = ensure_source(stem, source_type, category)
                update_source_metadata(comment_source_id, fetched_at, status, status_code)
        else:
            print(f"  [WARN] Expected dict format for comments in '{stem}', got {type(records_data).__name__}")
        return

    # Base item sources processing
    source_id = ensure_source(source_name, source_type, category)

    if fetched_at or status or status_code:
        update_source_metadata(source_id, fetched_at, status, status_code)

    if not records_data:
        print(f"  [INFO] No records found inside '{stem}' payload")
        return

    if source_type == "nvd" and isinstance(records_data, list):
        records = [clean_record(rec, "nvd") for rec in records_data]
        records = dedup_records(records, key="cve_id")
        insert_vulnerabilities(source_id, records)

    elif source_type == "hn" and "new_stories" in stem.lower() and isinstance(records_data, list):
        hn_ids = [int(x) for x in records_data if isinstance(x, (int, float, str)) and str(x).isdigit()]
        if hn_ids:
            insert_hn_seen_ids(hn_ids)

    elif isinstance(records_data, list):
        records = [clean_record(rec) for rec in records_data]
        records = dedup_records(records, key="id")
        insert_items(source_id, records, source_type)

    elif isinstance(records_data, dict):
        record = clean_record(records_data, source_type)
        if source_type == "nvd":
            insert_vulnerabilities(source_id, [record])
        else:
            insert_items(source_id, [record], source_type)


def clean_all(date_str: str | None = None) -> None:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    raw_dir = PROJECT_ROOT / "data" / date_str

    if not raw_dir.exists():
        print(f"No data directory found: {raw_dir}")
        return

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {raw_dir}")
        return

    init_db()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(clean_file, f): f for f in files}
        for future in as_completed(futures):
            f = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [ERROR] Failed to process {f.name}: {e}")


if __name__ == "__main__":
    clean_all()