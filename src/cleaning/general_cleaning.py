import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import init_db, ensure_source
from db.loader import insert_items, insert_vulnerabilities, insert_hn_seen_ids


# ── Helpers ──────────────────────────────────────────────────────

def normalize_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except (ValueError, TypeError):
        return date_str


# ── 1. Text cleaning (HTML + Markdown) ──────────────────────────

def clean_text(text: str) -> str:
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
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_value(value):
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value


# ── 2. Batch dedup (before DB insert) ───────────────────────────

def dedup_records(records: list[dict], key: str = "id") -> list[dict]:
    seen = set()
    result = []
    for rec in records:
        k = rec.get(key)
        if k is not None and k not in seen:
            seen.add(k)
            result.append(rec)
    return result


# ── 3. Source-specific: NVD ─────────────────────────────────────

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


# ── 4. Orchestration — one entry point per record ───────────────

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

def clean_file(filepath: Path) -> None:
    print(f"\n  Processing: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    stem = filepath.stem
    source_type = detect_source_type(stem)
    source_name = stem
    category = extract_category(stem)

    source_id = ensure_source(source_name, source_type, category)

    if source_type == "nvd" and isinstance(data, list):
        records = [clean_record(rec, "nvd") for rec in data]
        records = dedup_records(records, key="cve_id")
        insert_vulnerabilities(source_id, records)

    elif source_type == "hn" and "new_stories" in stem.lower() and isinstance(data, list):
        hn_ids = [int(x) for x in data if isinstance(x, (int, float, str)) and str(x).isdigit()]
        if hn_ids:
            insert_hn_seen_ids(hn_ids)

    elif isinstance(data, list):
        records = [clean_record(rec) for rec in data]
        records = dedup_records(records, key="id")
        insert_items(source_id, records, source_type)

    elif isinstance(data, dict):
        record = clean_record(data, source_type)
        if source_type == "nvd":
            insert_vulnerabilities(source_id, [record])
        else:
            insert_items(source_id, [record], source_type)


def clean_all(date_str: str | None = None) -> None:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    raw_dir = PROJECT_ROOT / "data" / "raw" / date_str

    if not raw_dir.exists():
        print(f"No raw data directory found: {raw_dir}")
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
