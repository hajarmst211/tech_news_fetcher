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
    raw_dir = PROJECT_ROOT / "data" / date_str
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_name(source_name) + ".json"
    filepath = raw_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {filepath}")
    return filepath


def _entry_to_dict(entry) -> dict:
    now = datetime.now().isoformat()
    updated = entry.get("updated")
    item = {
        "title": entry.get("title"),
        "link": entry.get("link"),
        "published": entry.get("published"),
        "updated": updated,
        "updated_at": updated,
        "fetched_at": now,
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


def _flatten_reddit_comments(children: list) -> list[dict]:
    result = []
    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        created = data.get("created_utc")
        comment = {
            "id_code": data.get("id"),
            "author": data.get("author"),
            "body_html": data.get("body"),
            "created_at": datetime.utcfromtimestamp(created).isoformat() if isinstance(created, (int, float)) else None,
            "parent_id": data.get("parent_id"),
        }
        result.append(comment)
        replies = data.get("replies")
        if isinstance(replies, dict) and replies.get("kind") == "Listing":
            more_children = replies.get("data", {}).get("children", [])
            if more_children:
                result.extend(_flatten_reddit_comments(more_children))
    return result


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

    if "infoq" in name.lower():
        for entry in entries_data:
            entry.pop("id", None)

    if "light reading" in name.lower():
        for entry in entries_data:
            entry.pop("published", None)
            entry.pop("updated", None)
            entry.pop("id", None)

    if "microsoft .net" in name.lower():
        for entry in entries_data:
            entry.pop("published", None)
            entry.pop("updated", None)
            entry.pop("id", None)

    if "schneier" in name.lower():
        for entry in entries_data:
            entry.pop("id", None)
            entry.pop("content", None)
            summary = entry.pop("summary", None)
            if summary:
                entry["content"] = summary

    if "reddit" in name.lower():
        for entry in entries_data:
            entry.pop("content", None)
            summary = entry.pop("summary", None)
            if summary:
                entry["content"] = summary
            post_id_match = re.search(r'/comments/([^/]+)/', entry.get("link", ""))
            if post_id_match:
                entry["id"] = post_id_match.group(1)
            else:
                entry.pop("id", None)

        all_comments = {}
        for entry in entries_data:
            link = entry.get("link", "")
            post_id = entry.get("id", "")
            if not link or not post_id:
                continue
            time.sleep(0.5)
            comments_url = link.rstrip("/") + ".json"
            comments_data = fetcher.request(comments_url)
            if comments_data and isinstance(comments_data, list) and len(comments_data) == 2:
                children = comments_data[1].get("data", {}).get("children", [])
                flat = _flatten_reddit_comments(children)
                if flat:
                    all_comments[post_id] = flat
                    print(f"    Fetched {len(flat)} comments for post {post_id}")

        if all_comments:
            _save_json(all_comments, f"{name} (Comments)")
            print(f"  [OK]   {name} — comments saved ({len(all_comments)} posts with comments)")

    if "venturebeat" in name.lower():
        for entry in entries_data:
            entry.pop("published", None)
            entry.pop("updated", None)
            summary = entry.pop("summary", None)
            if summary:
                entry["content"] = summary

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
