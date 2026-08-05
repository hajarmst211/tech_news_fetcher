#!/usr/bin/env python3
"""Unified article processing stage of the pipeline.

For every newly fetched article the following steps run, in order:

1. Deduplication   — query the database for existing article URLs and skip any
                     article whose URL is already stored.
2. Cleaning        — pass the raw text of new articles through the cleaning
                     function (HTML/Markdown stripped, emojis removed, dates
                     normalized).
3. Summarization   — reuse the summary provided by the source API/feed if one
                     exists; otherwise generate one with the SGR summarizer.
4. Topic extraction — extract key topics (keywords/phrases) from the cleaned
                     text with TextRank.
5. Sentiment       — if the article has comments, classify each comment with
                     the SVM model and compute the aggregated sentiment
                     percentages for the article.
6. Database insert — save the finalized article (metadata, cleaned content,
                     summary, topics, aggregate sentiment), its individual
                     comments with their sentiment labels, and any NVD/HN
                     records.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"

for _path in (
    SRC,
    SRC / "cleaning",
    SRC / "functionalities" / "summary" / "SGR",
    SRC / "functionalities" / "theme_reconginition" / "text_rank",
    SRC / "functionalities" / "sentiment_analysis",
):
    sys.path.insert(0, str(_path))

from general_cleaning import (
    clean_record,
    clean_text,
    dedup_records,
    detect_source_type,
    extract_category,
    normalize_date,
)
from db.database import ensure_source, get_conn, init_db, return_conn
from db.loader import (
    insert_comments,
    insert_hn_seen_ids,
    insert_processed_items,
    insert_vulnerabilities,
    update_source_metadata,
)

from semantic_graph_reduction import SemanticGraphReducer
from script import extract_top_topics
from svm import get_classifier

SVM_CACHE_PATH = SRC / "functionalities" / "sentiment_analysis" / "data" / "svm_model.joblib"

_sgr = None
_comment_classifier = None


def _get_sgr() -> SemanticGraphReducer:
    global _sgr
    if _sgr is None:
        _sgr = SemanticGraphReducer()
    return _sgr


def _get_svm_classifier():
    global _comment_classifier
    if _comment_classifier is None:
        _comment_classifier = get_classifier(str(SVM_CACHE_PATH))
    return _comment_classifier


def _extract_url(record: dict) -> str | None:
    for key in ("url", "link", "html_url", "repo_url"):
        val = record.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    return None


def _record_key(record: dict) -> str | None:
    for key in ("id", "external_id", "tag_name", "repo_url", "url", "link", "html_url"):
        val = record.get(key)
        if val is not None and str(val):
            return str(val)
    return None


def _dedup_records(records: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for rec in records:
        key = _record_key(rec)
        if key is not None and key not in seen:
            seen.add(key)
            result.append(rec)
    return result


def _extract_summary(record: dict) -> str | None:
    for key in ("summary", "description"):
        val = record.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_content(record: dict) -> str | None:
    for key in ("content", "body_markdown", "body", "text", "article_content", "readme"):
        val = record.get(key)
        if isinstance(val, str):
            if val.strip():
                return val
        elif isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("value", "")))
            joined = "\n".join(p for p in parts if p)
            if joined.strip():
                return joined
    return None


def _existing_urls() -> set[str]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM items WHERE url IS NOT NULL AND url != ''")
            return {row[0] for row in cur.fetchall()}
    finally:
        return_conn(conn)


def _load_json_files(date_str: str) -> dict[str, Path]:
    raw_dir = PROJECT_ROOT / "data" / date_str
    if not raw_dir.exists():
        print(f"No data directory found: {raw_dir}")
        return {}
    files = {f.stem: f for f in sorted(raw_dir.glob("*.json"))}
    if not files:
        print(f"No JSON files found in {raw_dir}")
        return {}
    return files


def _load_comments(files: dict[str, Path]) -> dict[str, dict]:
    comments_by_parent = {}
    for stem, filepath in files.items():
        if not stem.endswith("_comments"):
            continue
        parent_stem = stem[: -len("_comments")]
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                comments_by_parent[parent_stem] = data
        except Exception as e:
            print(f"  [ERROR] Failed to read comments file {filepath.name}: {e}")
    return comments_by_parent


def _unwrap(raw):
    fetched_at = None
    status = None
    status_code = None
    records_data = None

    if isinstance(raw, dict):
        fetched_at = normalize_date(raw.get("fetched_at"))
        status = raw.get("status") or raw.get("fetch_status")
        status_code = raw.get("status_code") or raw.get("last_status_code")

        for key in ("data", "results", "items", "comments", "articles", "vulnerabilities"):
            if key in raw and isinstance(raw[key], (list, dict)):
                records_data = raw[key]
                break

        if records_data is None:
            if not any(k in raw for k in ("fetched_at", "status", "status_code")):
                records_data = raw
            else:
                records_data = []
    else:
        records_data = raw

    return records_data, fetched_at, status, status_code


def _build_enriched(record: dict, comments: list[dict] | None) -> tuple[dict, list[dict]]:
    """Clean + summarize + extract topics + classify comments for one article."""
    content = _extract_content(record)
    summary = _extract_summary(record)

    if summary:
        print(f"    [SUMMARY] using source-provided summary ({len(summary)} chars)")
    else:
        text_for_summary = content or record.get("title", "")
        if text_for_summary:
            try:
                summary = _get_sgr().summarize(text_for_summary)
                print(f"    [SUMMARY] SGR generated summary ({len(summary or '')} chars)")
            except Exception as e:
                print(f"    [WARN] SGR summarization failed: {e}")

    enriched = dict(record)
    enriched["summary"] = summary or None

    if not enriched.get("id") and not enriched.get("external_id"):
        enriched["external_id"] = (
            enriched.get("repo_url") or enriched.get("url")
            or enriched.get("link") or enriched.get("html_url") or ""
        )

    topic_text = content or summary or record.get("title", "")
    try:
        topics = extract_top_topics(topic_text)
    except Exception as e:
        print(f"    [WARN] TextRank topic extraction failed: {e}")
        topics = []
    enriched["topics"] = topics
    if topics:
        print(f"    [TOPICS] {', '.join(topics)}")

    comment_records, aggregate = _classify_comments(comments)
    if aggregate:
        enriched["sentiment"] = aggregate

    return enriched, comment_records


def _classify_comments(comments: list[dict] | None) -> tuple[list[dict], dict | None]:
    if not comments:
        return [], None

    cleaned = []
    texts = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        external_id = str(comment.get("id_code", ""))
        body_text = clean_text(comment.get("body_html", ""))
        published_at = normalize_date(comment.get("created_at"))

        extra = {}
        for k, v in comment.items():
            if k not in ("id_code", "created_at", "body_html", "user", "author"):
                extra[k] = v

        cleaned.append({
            "external_id": external_id,
            "author": comment.get("author"),
            "body_text": body_text,
            "published_at": published_at,
            "extra": extra or None,
        })
        texts.append(body_text)

    if not texts:
        return [], None

    classifier = _get_svm_classifier()
    predictions = classifier.predict_with_confidence(texts)

    label_counts = {"positive": 0, "neutral": 0, "negative": 0}
    comment_records = []
    for rec, (label, confidence) in zip(cleaned, predictions):
        norm_label = str(label).strip().lower()
        if norm_label not in label_counts:
            norm_label = "neutral"
        label_counts[norm_label] += 1
        rec["sentiment_label"] = norm_label
        rec["sentiment_score"] = float(confidence)
        comment_records.append(rec)

    total = len(comment_records)
    aggregate = {
        "total": total,
        "positive": round(label_counts["positive"] / total * 100, 1),
        "neutral": round(label_counts["neutral"] / total * 100, 1),
        "negative": round(label_counts["negative"] / total * 100, 1),
        "overall": max(label_counts, key=label_counts.get),
    }
    print(
        f"    [SENTIMENT] {total} comments — "
        f"{aggregate['positive']:.0f}% positive, {aggregate['neutral']:.0f}% neutral, "
        f"{aggregate['negative']:.0f}% negative"
    )
    return comment_records, aggregate


def _process_file(stem: str, filepath: Path, comments_map: dict, seen_urls: set[str]) -> None:
    print(f"\n  Processing: {filepath.name}")

    if stem.endswith("_comments"):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    source_type = detect_source_type(stem)
    source_name = stem
    category = extract_category(stem)

    records_data, fetched_at, status, status_code = _unwrap(raw)

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
        return

    if source_type == "hn" and "new_stories" in stem.lower() and isinstance(records_data, list):
        hn_ids = [int(x) for x in records_data if isinstance(x, (int, float, str)) and str(x).isdigit()]
        if hn_ids:
            insert_hn_seen_ids(hn_ids)
        return

    if not isinstance(records_data, list):
        records_data = [records_data]

    records = [clean_record(rec) for rec in records_data if isinstance(rec, dict)]
    records = _dedup_records(records)

    if not records:
        print(f"  [INFO] No article records in '{stem}'")
        return

    comments_for_articles = comments_map.get(stem, {})
    enriched_records = []
    pending_comments = []

    for rec in records:
        external_id = str(rec.get("id", rec.get("external_id", "")))
        url = _extract_url(rec)

        if url and url in seen_urls:
            print(f"    [SKIP] Duplicate URL (already in DB): {url}")
            continue

        article_comments = comments_for_articles.get(external_id)
        enriched, comment_records = _build_enriched(rec, article_comments)
        enriched_records.append(enriched)
        pending_comments.append((external_id, comment_records))

        if url:
            seen_urls.add(url)

    if not enriched_records:
        print(f"  [INFO] All {len(records)} articles in '{stem}' already exist — nothing new")
        return

    item_ids = insert_processed_items(source_id, enriched_records)

    comment_inserts = []
    for external_id, comment_records in pending_comments:
        item_id = item_ids.get(external_id)
        if item_id is None:
            continue
        for comment in comment_records:
            comment["item_id"] = item_id
            comment_inserts.append(comment)

    if comment_inserts:
        insert_comments(comment_inserts)


def process(date_str: str | None = None) -> None:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    files = _load_json_files(date_str)
    if not files:
        return

    print("Initializing database...")
    init_db()

    print("Querying existing article URLs for deduplication...")
    seen_urls = _existing_urls()
    print(f"  {len(seen_urls)} URLs already in database")

    comments_map = _load_comments(files)

    print("\n=== Processing articles (dedup → clean → summarize → topics → sentiment → insert) ===")
    for stem in sorted(files):
        if stem.endswith("_comments"):
            continue
        try:
            _process_file(stem, files[stem], comments_map, seen_urls)
        except Exception as e:
            print(f"  [ERROR] Failed to process {files[stem].name}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    process()
