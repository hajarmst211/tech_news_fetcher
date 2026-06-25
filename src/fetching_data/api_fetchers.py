import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from dotenv import load_dotenv

from general_api_fetcher import GeneralApiFetcher

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"

HN_TOP_STORIES_TO_FETCH = 10


def load_sources() -> list[dict]:
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("sources", [])


def fetch_api_source(source: dict) -> None:
    name = source["name"]
    base_url = source["base_url"]
    endpoint = source["endpoint"]
    params = source.get("params", {})
    headers = source.get("headers", {}).copy()
    env_api_key = source.get("env_api_key")

    if env_api_key:
        key_value = os.getenv(env_api_key)
        if key_value:
            headers["apiKey"] = key_value
        else:
            print(f"  [WARN] Env var '{env_api_key}' not set for '{name}'")

    ssl_verify = source.get("ssl_verify", True)
    fetcher = GeneralApiFetcher(base_url=base_url, headers=headers, timeout=15, ssl_verify=ssl_verify)
    response_type = source.get("response_type", "json")

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if response_type == "xml":
        raw = fetcher.request_raw(endpoint, params=params)
        if raw is None:
            return
        _print_xml_source(raw)
    else:
        data = fetcher.request(endpoint, params=params)
        if data is None:
            return
        _print_json_source(data)


def _print_xml_source(raw: str) -> None:
    try:
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//atom:entry", ns)
        if not entries:
            entries = root.findall(".//entry")
        print(f"  Found {len(entries)} entries")
        for i, entry in enumerate(entries[:10]):
            title_el = entry.find("atom:title", ns)
            if title_el is None:
                title_el = entry.find("title")
            id_el = entry.find("atom:id", ns)
            if id_el is None:
                id_el = entry.find("id")
            title = title_el.text.strip() if title_el is not None and title_el.text else "N/A"
            link = id_el.text.strip() if id_el is not None and id_el.text else "N/A"
            print(f"  [{i+1}] {title}")
            print(f"       {link}")
    except ET.ParseError as e:
        print(f"  [ERROR] XML parse failed: {e}")


def _print_json_source(data) -> None:
    if isinstance(data, list):
        valid = [d for d in data if isinstance(d, dict)]
        print(f"  Found {len(valid)} items")
        for i, item in enumerate(valid[:10]):
            title = item.get("title") or "N/A"
            url = item.get("url") or item.get("html_url") or item.get("link") or "N/A"
            print(f"  [{i+1}] {title}")
            print(f"       {url}")
    elif isinstance(data, dict):
        if "vulnerabilities" in data and isinstance(data["vulnerabilities"], list):
            vulns = data["vulnerabilities"]
            print(f"  Found {len(vulns)} items")
            for i, vuln in enumerate(vulns[:10]):
                if "cve" in vuln:
                    cve = vuln["cve"]
                    cve_id = cve.get("id", "N/A")
                    descs = cve.get("descriptions", [])
                    desc = descs[0].get("value", "") if descs else ""
                    refs = cve.get("references", [])
                    ref_url = refs[0].get("url", "") if refs else ""
                    print(f"  [{i+1}] {cve_id}: {desc[:120]}")
                    print(f"       {ref_url}")
                else:
                    cve_id = vuln.get("cveID", "N/A")
                    vname = vuln.get("vulnerabilityName", "N/A")
                    sdesc = vuln.get("shortDescription", "")
                    notes = vuln.get("notes", "")
                    print(f"  [{i+1}] {cve_id}: {vname} — {sdesc[:100]}")
                    print(f"       {notes}")
        else:
            children = (
                data.get("data", {}).get("children", [])
                if "data" in data and isinstance(data["data"], dict)
                else []
            )
            if children:
                print(f"  Found {len(children)} items")
                for i, child in enumerate(children[:10]):
                    kid = child.get("data", {}) if isinstance(child, dict) else {}
                    title = kid.get("title", "N/A")
                    url = kid.get("url", "N/A")
                    print(f"  [{i+1}] {title}")
                    print(f"       {url}")
            else:
                title = data.get("title", data.get("name", "N/A"))
                url = data.get("html_url", data.get("url", "N/A"))
                print(f"  Title: {title}")
                print(f"  URL:   {url}")
    else:
        print(f"  {data}")


def fetch_hacker_news(fetcher: GeneralApiFetcher) -> None:
    print(f"\n{'='*60}")
    print("  Hacker News — Top Stories Details")
    print(f"{'='*60}")

    ids_data = fetcher.request("/v0/newstories.json", params={"print": "pretty"})
    if not ids_data or not isinstance(ids_data, list):
        print("  [ERROR] Could not fetch HN story IDs")
        return

    print(f"  Found {len(ids_data)} story IDs, fetching top {HN_TOP_STORIES_TO_FETCH}")
    for story_id in ids_data[:HN_TOP_STORIES_TO_FETCH]:
        item = fetcher.request(f"/v0/item/{story_id}.json")
        if item and isinstance(item, dict):
            title = item.get("title", "N/A")
            url = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
            print(f"  [{story_id}] {title}")
            print(f"         {url}")


def main() -> None:
    sources = load_sources()
    api_sources = [s for s in sources if s.get("type") == "api"]

    hn_source = next((s for s in api_sources if s.get("hn_item_fetch")), None)
    regular_sources = [
        s for s in api_sources
        if not s.get("hn_item_fetch") and not s.get("hn_id_list")
    ]

    for source in regular_sources:
        fetch_api_source(source)

    if hn_source:
        hn_headers = hn_source.get("headers", {}).copy()
        hn_ssl_verify = hn_source.get("ssl_verify", True)
        hn_fetcher = GeneralApiFetcher(
            base_url=hn_source["base_url"],
            headers=hn_headers,
            timeout=15,
            ssl_verify=hn_ssl_verify,
        )
        fetch_hacker_news(hn_fetcher)


if __name__ == "__main__":
    main()
