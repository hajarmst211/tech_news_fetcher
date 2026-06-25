#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

from lxml.html import document_fromstring

PROJECT_ROOT = Path(__file__).resolve().parent
DATE_STR = datetime.now().strftime("%Y-%m-%d")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / DATE_STR
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned" / DATE_STR
OUTPUT_FILE = PROJECT_ROOT / "cleaning_stats_output.md"


def count_items(data) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return 1
    return 0


def _has_html_tags(text: str) -> bool:
    try:
        doc = document_fromstring(text)
        return len(doc.body) > 0
    except Exception:
        return False


def _count_tags_in_text(text: str) -> int:
    try:
        doc = document_fromstring(text)
        return len(list(doc.iter())) - 2
    except Exception:
        return 0


def count_html_fields(data) -> int:
    count = 0
    if isinstance(data, dict):
        for v in data.values():
            count += count_html_fields(v)
    elif isinstance(data, list):
        for item in data:
            count += count_html_fields(item)
    elif isinstance(data, str) and _has_html_tags(data):
        count += 1
    return count


def count_html_tags(data) -> int:
    count = 0
    if isinstance(data, dict):
        for v in data.values():
            count += count_html_tags(v)
    elif isinstance(data, list):
        for item in data:
            count += count_html_tags(item)
    elif isinstance(data, str):
        count += _count_tags_in_text(data)
    return count


def analyze_file(filename: str) -> dict:
    raw_path = RAW_DIR / filename
    cleaned_path = CLEANED_DIR / filename

    with open(raw_path) as f:
        raw_data = json.load(f)
    with open(cleaned_path) as f:
        cleaned_data = json.load(f)

    raw_tags = count_html_tags(raw_data)
    cleaned_tags = count_html_tags(cleaned_data)

    return {
        "filename": filename,
        "raw_items": count_items(raw_data),
        "cleaned_items": count_items(cleaned_data),
        "html_fields_raw": count_html_fields(raw_data),
        "html_fields_cleaned": count_html_fields(cleaned_data),
        "html_tags_raw": raw_tags,
        "html_tags_removed": raw_tags - cleaned_tags,
    }


def generate_report(stats_list: list[dict]) -> str:
    lines = [
        "# Cleaning Statistics Report",
        f"**Date:** {DATE_STR}",
        "",
        "## Per-File Statistics",
        "",
        "| File | Raw Items | Cleaned Items | HTML Fields (raw) | HTML Fields (cleaned) | Fields Cleaned | HTML Tags (raw) | HTML Tags Removed |",
        "|------|-----------|---------------|-------------------|----------------------|----------------|-----------------|-------------------|",
    ]

    totals = {
        "raw_items": 0, "cleaned_items": 0, "html_fields_raw": 0,
        "html_fields_cleaned": 0, "html_tags_raw": 0, "html_tags_removed": 0,
    }

    for s in stats_list:
        for k in totals:
            totals[k] += s[k]
        lines.append(
            f"| {s['filename']} | {s['raw_items']} | {s['cleaned_items']} "
            f"| {s['html_fields_raw']} | {s['html_fields_cleaned']} "
            f"| {s['html_fields_raw'] - s['html_fields_cleaned']} "
            f"| {s['html_tags_raw']} | {s['html_tags_removed']} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Date:** {DATE_STR}",
        f"- **Files analyzed:** {len(stats_list)}",
        f"- **Total raw items:** {totals['raw_items']}",
        f"- **Total cleaned items:** {totals['cleaned_items']}",
        f"- **Items removed:** {totals['raw_items'] - totals['cleaned_items']}",
        f"- **Total HTML fields in raw data:** {totals['html_fields_raw']}",
        f"- **Total HTML fields after cleaning:** {totals['html_fields_cleaned']}",
        f"- **Total HTML fields cleaned:** {totals['html_fields_raw'] - totals['html_fields_cleaned']}",
        f"- **Total HTML tags in raw data:** {totals['html_tags_raw']}",
        f"- **Total HTML tags removed:** {totals['html_tags_removed']}",
        "",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    if not RAW_DIR.exists():
        print(f"No raw data directory found at {RAW_DIR}")
        return

    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {RAW_DIR}")
        return

    stats_list = []
    for f in json_files:
        cleaned = CLEANED_DIR / f.name
        if cleaned.exists():
            stats_list.append(analyze_file(f.name))
        else:
            print(f"  [WARN] No cleaned version for {f.name}")

    if not stats_list:
        print("No files to analyze")
        return

    report = generate_report(stats_list)

    OUTPUT_FILE.write_text(report)
    print(report)
    print(f"Report written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
