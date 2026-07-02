import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import feedparser
import yaml
from general_api_fetcher import GeneralApiFetcher

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_sources() -> list[dict]:
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def _sanitize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name.strip('_')


def _save_json(data, source_name: str) -> Path | None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    raw_dir = PROJECT_ROOT / "data" / "raw" / date_str
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_name(source_name) + ".json"
    filepath = raw_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {filepath}")
    return filepath


def _entry_to_dict(entry) -> dict:
    item = {
        "title": entry.get("title"),
        "link": entry.get("link"),
        "published": entry.get("published"),
        "updated": entry.get("updated"),
        "summary": entry.get("summary"),
        "id": entry.get("id"),
    }
    if "content" in entry:
        item["content"] = [
            {"type": c.get("type"), "value": c.get("value")}
            for c in entry.content
        ]
    if "tags" in entry:
        item["tags"] = [t.get("term") for t in entry.tags]
    return item


def fetch_rss_feed(source: dict) -> None:
    name = source["name"]
    base_url = source["base_url"]
    endpoint = source["endpoint"]
    headers = source.get("headers", {})

    ssl_verify = source.get("ssl_verify", True)
    fetcher = GeneralApiFetcher(base_url=base_url, headers=headers, timeout=15, ssl_verify=ssl_verify)
    raw_xml = fetcher.request_raw(endpoint)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if raw_xml is None:
        print(f"  [FAIL] {name} — request returned no data")
        return

    feed = feedparser.parse(raw_xml)

    if feed.bozo and not feed.entries:
        print(f"  [FAIL] {name} — feed parse error: {feed.bozo_exception}")
        return

    print(f"  Found {len(feed.entries)} entries")
    for i, entry in enumerate(feed.entries[:10]):
        title = entry.get("title", "N/A")
        link = entry.get("link", "N/A")
        published = entry.get("published", entry.get("updated", "N/A"))
        print(f"  [{i+1}] {title}")
        print(f"       {link}")
        if published:
            print(f"       Published: {published}")

    if not feed.entries:
        print(f"  [WARN] {name} — 0 entries (feed may be empty)")
        return

    entries_data = [_entry_to_dict(e) for e in feed.entries]
    _save_json(entries_data, name)
    print(f"  [OK]   {name} — {len(feed.entries)} entries saved")


def main() -> None:
    sources = load_sources()
    rss_sources = [s for s in sources if s.get("type") == "rss"]

    if not rss_sources:
        print("No RSS sources found in config.")
        return

    reddit_sources = [s for s in rss_sources if "reddit.com" in s.get("base_url", "")]
    other_sources = [s for s in rss_sources if "reddit.com" not in s.get("base_url", "")]

    if other_sources:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_rss_feed, s) for s in other_sources]
            for future in as_completed(futures):
                future.result()

    for i, s in enumerate(reddit_sources):
        if i > 0:
            time.sleep(3)
        fetch_rss_feed(s)


if __name__ == "__main__":
    main()
