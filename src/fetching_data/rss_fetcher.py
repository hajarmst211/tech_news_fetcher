from pathlib import Path

import feedparser
import yaml
from general_api_fetcher import GeneralApiFetcher

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"


def load_sources() -> list[dict]:
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


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
        return

    feed = feedparser.parse(raw_xml)

    if feed.bozo and not feed.entries:
        print(f"  [ERROR] Feed parse error: {feed.bozo_exception}")
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


def main() -> None:
    sources = load_sources()
    rss_sources = [s for s in sources if s.get("type") == "rss"]

    if not rss_sources:
        print("No RSS sources found in config.")
        return

    for source in rss_sources:
        fetch_rss_feed(source)


if __name__ == "__main__":
    main()
