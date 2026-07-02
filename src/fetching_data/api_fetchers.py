import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

import trafilatura
from playwright.sync_api import sync_playwright

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


def _fetch_url_content(url: str) -> str | None:
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

        if source.get("fetch_full_content") and isinstance(data, list):
            print(f"  Enriching {len(data)} articles with full content...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                fut_to_article = {}
                for article in data:
                    article_id = article.get("id")
                    if article_id:
                        fut = executor.submit(fetcher.request, f"/api/articles/{article_id}")
                        fut_to_article[fut] = (article, article_id)
                for future in as_completed(fut_to_article):
                    article, article_id = fut_to_article[future]
                    detail = future.result()
                    if detail and isinstance(detail, dict) and "body_markdown" in detail:
                        article["content"] = detail["body_markdown"]
                        article["body_markdown"] = detail["body_markdown"]
                        print(f"    Fetched body_markdown ({len(detail['body_markdown'])} chars) for article {article_id}")
                    else:
                        print(f"    [WARN] Failed to fetch detail for article {article_id}")

            print(f"  Fetching comments for {len(data)} articles...")
            all_comments = {}
            for article in data:
                article_id = article.get("id")
                if not article_id:
                    continue
                time.sleep(0.5)
                comments = fetcher.request("/api/comments", params={"a_id": article_id})
                if comments and isinstance(comments, list) and len(comments) > 0:
                    all_comments[str(article_id)] = comments
                    print(f"    Fetched {len(comments)} comments for article {article_id}")

            if all_comments:
                for article_comments in all_comments.values():
                    for comment in article_comments:
                        comment.pop("user", None)
                        comment.pop("children", None)
                _save_json(all_comments, f"{name} (Comments)")
                print(f"  [OK]   {name} — comments saved ({len(all_comments)} articles with comments)")
            else:
                print(f"  [WARN] {name} — no comments found")

            for article in data:
                user = article.get("user")
                if user and isinstance(user, dict):
                    article["author"] = user.get("username") or user.get("name")
                else:
                    article.pop("author", None)

                cover = article.get("cover_image")
                social = article.get("social_image")
                article["image_url"] = cover or social or None

            fields_to_remove = [
                "user", "organization", "flare_tag",
                "readable_publish_date", "path", "slug",
                "public_reactions_count", "published_timestamp",
                "language", "subforem_id", "cover_image", "social_image",
                "canonical_url", "created_at", "edited_at",
                "crossposted_at", "posted_at", "last_comment_at",
                "reading_time_minutes", "tags", "body_markdown",
                "type_of", "collection_id", "positive_reactions_count",
                "published_at",
            ]
            for article in data:
                for field in fields_to_remove:
                    article.pop(field, None)

        if source.get("github_enrich") and isinstance(data, dict):
            data = _enrich_github_release(data, fetcher, name)

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



def _parse_last_page_from_link(link_header: str | None) -> int | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="last"' in part:
            match = re.search(r'[?&]page=(\d+)', part)
            if match:
                return int(match.group(1))
    return None


def _enrich_github_release(data: dict, fetcher: GeneralApiFetcher, name: str) -> list[dict]:
    items = data.get("items", [])
    if not items:
        print(f"  [WARN] {name} — no repositories in search results")
        return []

    print(f"  Enriching {len(items)} repositories...")
    enriched = []
    for item in items:
        full_name = item.get("full_name", "")
        if "/" not in full_name:
            continue
        owner, repo = full_name.split("/", 1)

        for field in ("id", "node_id", "url", "owner", "private", "fork", "size", "score", "visibility", "default_branch", "pushed_at", "homepage", "updated_at", "topics", "mirror_url", "has_downloads", "has_issues", "has_pages", "has_projects", "has_wiki", "is_template", "archived", "disabled", "allow_forking", "forks", "open_issues", "watchers", "watchers_count", "forks_count", "open_issues_count", "ssh_url", "clone_url", "svn_url", "git_url"):
            item.pop(field, None)

        item["repo_url"] = f"https://github.com/{owner}/{repo}"

        readme = fetcher.request_raw(
            f"/repos/{owner}/{repo}/readme",
            extra_headers={"Accept": "application/vnd.github.v3.raw"}
        )
        if readme:
            item["readme"] = readme

        languages = fetcher.request(f"/repos/{owner}/{repo}/languages")
        if languages and isinstance(languages, dict):
            total = sum(languages.values())
            if total > 0:
                item["languages"] = {
                    lang: round(count / total * 100, 2)
                    for lang, count in languages.items()
                }

        contrib_resp = fetcher.request_response(
            f"/repos/{owner}/{repo}/contributors",
            params={"per_page": 1, "anon": "true"}
        )
        if contrib_resp is not None:
            link_header = contrib_resp.headers.get("Link")
            last_page = _parse_last_page_from_link(link_header)
            if last_page is not None:
                item["contributor_count"] = last_page
            else:
                body = contrib_resp.json()
                item["contributor_count"] = len(body) if isinstance(body, list) else 0

        enriched.append(item)

    print(f"  [OK]   {name} — {len(enriched)} repositories enriched")
    return enriched





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
        for item in items:
            item.pop("id", None)
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
