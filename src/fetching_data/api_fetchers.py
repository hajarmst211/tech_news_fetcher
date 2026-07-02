import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

import trafilatura

from general_api_fetcher import GeneralApiFetcher

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sources.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

HN_TOP_STORIES_TO_FETCH = 10


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


def _extract_urls_from_notes(notes: str) -> list[str]:
    if not notes:
        return []
    urls = []
    for item in notes.split(";"):
        item = item.strip()
        if not item:
            continue
        if item.startswith("BOD"):
            continue
        found = None
        for part in item.split():
            if part.startswith("http://") or part.startswith("https://"):
                found = part
                break
        if not found and (item.startswith("http://") or item.startswith("https://")):
            found = item
        if found and "/directives/" not in found:
            urls.append(found)
    return urls


def _fetch_url_content(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_links=False, include_images=False, include_tables=False)
        return text.strip() if text else None
    except Exception:
        return None


def _transform_cisa_to_articles(data: dict) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    vulnerabilities = data.get("vulnerabilities", [])

    articles = []
    for vuln in vulnerabilities:
        if vuln.get("dateAdded") != today:
            continue

        cve_id = vuln.get("cveID", "")
        vuln_name = vuln.get("vulnerabilityName") or vuln.get("product", "")
        title = f"{cve_id} — {vuln_name}"

        short_desc = vuln.get("shortDescription", "")
        required_action = vuln.get("requiredAction", "")
        notes = vuln.get("notes", "")

        extracted = _extract_urls_from_notes(notes)
        vendor_url = extracted[0] if extracted else None
        nvd_url = next((u for u in extracted if "nvd.nist.gov" in u), None)

        url = vendor_url or nvd_url or f"https://nvd.nist.gov/vuln/detail/{cve_id}"

        parts = [short_desc]
        if required_action:
            parts.append(f"Required Action: {required_action}")
        if notes:
            refs = notes.replace(" ; ", "\n")
            parts.append(f"References:\n{refs}")
        content = "\n\n".join(p for p in parts if p)

        articles.append({
            "title": title,
            "url": url,
            "published": vuln.get("dateAdded", ""),
            "summary": short_desc,
            "content": content,
            "id": cve_id,
            "cveID": cve_id,
            "vulnerabilityName": vuln.get("vulnerabilityName", ""),
            "vendorProject": vuln.get("vendorProject", ""),
            "product": vuln.get("product", ""),
            "shortDescription": short_desc,
            "requiredAction": required_action,
            "notes": notes,
            "vendor_advisory_url": vendor_url,
            "nvd_url": nvd_url,
            "dueDate": vuln.get("dueDate", ""),
            "knownRansomwareCampaignUse": vuln.get("knownRansomwareCampaignUse", ""),
            "cwes": vuln.get("cwes", []),
        })

    return articles


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
        data = fetcher.request(endpoint, params=params)
        if data is not None:
            _print_json_source(data)
            _save_json(data, name)
            print(f"  [OK]   {name} — data saved (JSON)")
        else:
            raw = fetcher.request_raw(endpoint, params=params)
            if raw is None:
                print(f"  [FAIL] {name} — request returned no data")
                return
            entries = _xml_entries_to_dicts(raw)
            if entries:
                _save_json(entries, name)
                print(f"  [OK]   {name} — {len(entries)} entries saved")
            else:
                print(f"  [WARN] {name} — parsed 0 entries")
    else:
        data = fetcher.request(endpoint, params=params)
        if data is None:
            print(f"  [FAIL] {name} — request returned no data")
            return

        if "CISA" in name:
            articles = _transform_cisa_to_articles(data)
            if not articles:
                print(f"  [SKIP] No new vulnerabilities added today")
                return
            data = articles
            print(f"  Found {len(data)} new vulnerabilities for today")

            for i, article in enumerate(data, 1):
                fetch_url = article.get("vendor_advisory_url") or article.get("nvd_url") or article.get("url")
                if fetch_url:
                    text = _fetch_url_content(fetch_url)
                    if text:
                        article["article_content"] = text
                        print(f"    [{i}/{len(data)}] Fetched content ({len(text)} chars) — {article['title'][:60]}...")
                    else:
                        article["article_content"] = None
                        print(f"    [{i}/{len(data)}] No content fetched — {article['title'][:60]}...")

        _save_json(data, name)
        print(f"  [OK]   {name} — data saved")


def _xml_entries_to_dicts(raw_xml: str) -> list[dict]:
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall(".//atom:entry", ns)
    if not entries:
        entries = root.findall(".//entry")

    result = []
    for entry in entries:
        item = {}
        for field in ("id", "title", "summary", "published", "updated"):
            el = entry.find(f"atom:{field}", ns)
            if el is None:
                el = entry.find(field)
            if el is not None and el.text:
                item[field] = el.text.strip()

        authors = entry.findall("atom:author", ns) or entry.findall("author")
        if authors:
            names = []
            for author in authors:
                name_el = author.find("atom:name", ns) or author.find("name")
                if name_el is not None and name_el.text:
                    names.append(name_el.text.strip())
            if names:
                item["authors"] = names

        links = entry.findall("atom:link", ns) or entry.findall("link")
        for link in links:
            if link.get("rel", "alternate") == "alternate":
                item["link"] = link.get("href", "")
                break
        if "link" not in item and links:
            item["link"] = links[0].get("href", "")

        for link in links:
            if link.get("title") == "pdf":
                item["pdf_url"] = link.get("href", "")
                break

        categories = entry.findall("atom:category", ns) or entry.findall("category")
        if categories:
            item["categories"] = [c.get("term", "") for c in categories if c.get("term")]

        result.append(item)

    return result



def fetch_hacker_news(fetcher: GeneralApiFetcher) -> None:
    print(f"\n{'='*60}")
    print("  Hacker News — Top Stories Details")
    print(f"{'='*60}")

    ids_data = fetcher.request("/v0/newstories.json", params={"print": "pretty"})
    if not ids_data or not isinstance(ids_data, list):
        print(f"  [FAIL] Hacker News — could not fetch story IDs")
        return

    _save_json(ids_data, "Hacker News (New Stories)")
    print(f"  [OK]   Hacker News (New Stories) — {len(ids_data)} IDs")

    print(f"  Fetching top {HN_TOP_STORIES_TO_FETCH} story details...")
    story_ids = ids_data[:HN_TOP_STORIES_TO_FETCH]

    items = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        fut_to_id = {
            executor.submit(fetcher.request, f"/v0/item/{sid}.json"): sid
            for sid in story_ids
        }
        for future in as_completed(fut_to_id):
            sid = fut_to_id[future]
            item = future.result()
            if item and isinstance(item, dict):
                items.append(item)
                title = item.get("title", "N/A")
                url = item.get("url", f"https://news.ycombinator.com/item?id={sid}")
                print(f"  [{sid}] {title}")
                print(f"         {url}")
            else:
                print(f"  [WARN] Hacker News — failed to fetch story {sid}")

    if items:
        _save_json(items, "Hacker News (Item Detail)")
        print(f"  [OK]   Hacker News (Item Detail) — {len(items)}/{HN_TOP_STORIES_TO_FETCH} stories saved")
    else:
        print(f"  [FAIL] Hacker News — no story details fetched")


def main() -> None:
    sources = load_sources()
    api_sources = [s for s in sources if s.get("type") == "api"]

    hn_source = next((s for s in api_sources if s.get("hn_item_fetch")), None)
    regular_sources = [
        s for s in api_sources
        if not s.get("hn_item_fetch") and not s.get("hn_id_list")
    ]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_api_source, s) for s in regular_sources]
        for future in as_completed(futures):
            future.result()

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
