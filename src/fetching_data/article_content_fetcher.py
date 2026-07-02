#!/usr/bin/env python3
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import trafilatura
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_article_url(entry: dict) -> str | None:
    for key in ("url", "link", "html_url"):
        val = entry.get(key)
        if val and isinstance(val, str) and val.startswith("http"):
            return val
    return None


def _fetch_article_text(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(downloaded, include_links=False, include_images=False, include_tables=False)
            if text and len(text.strip()) > 100 and "enable JavaScript" not in text[:200].lower():
                return text.strip()
    except Exception:
        pass

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            text = page.inner_text("body")
            browser.close()
            if text:
                return text.strip()
    except Exception:
        pass

    return None


def _enrich_entry(entry: dict) -> None:
    existing = entry.get("content")
    if existing and isinstance(existing, str) and len(existing) > 100:
        return
    url = _get_article_url(entry)
    if not url:
        return
    text = _fetch_article_text(url)
    if text:
        entry["content"] = text


def _process_file(filepath: Path) -> None:
    print(f"\n  Processing: {filepath.name}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = [(i, e) for i, e in enumerate(data) if isinstance(e, dict) and _get_article_url(e)]
    elif isinstance(data, dict):
        url = _get_article_url(data)
        if url:
            entries = [(0, data)]
        else:
            print(f"  [SKIP] No article URL found in dict")
            return
    else:
        print(f"  [SKIP] Unexpected type: {type(data).__name__}")
        return

    if not entries:
        print(f"  [SKIP] No entries with article URLs")
        return

    print(f"  Found {len(entries)} items with URLs")

    def _fetch_and_attach(idx: int, entry: dict) -> int:
        _enrich_entry(entry)
        time.sleep(1)
        return idx

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_and_attach, i, e): i for i, e in entries}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                print(f"    [{idx + 1}/{len(entries)}] Done")
            except Exception as e:
                print(f"    [{idx + 1}/{len(entries)}] Error: {e}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Count successes
    successes = sum(1 for _, e in entries if isinstance(e.get("content"), str) and len(e["content"]) > 100)
    print(f"  [DONE] {successes}/{len(entries)} articles fetched")
    print(f"  [SAVED] {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    date_str = args.date
    raw_dir = PROJECT_ROOT / "data" / "raw" / date_str

    if not raw_dir.exists():
        print(f"No raw data directory found: {raw_dir}")
        return

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {raw_dir}")
        return

    print(f"Processing {len(files)} files from {raw_dir}")
    for f in files:
        _process_file(f)


if __name__ == "__main__":
    main()
