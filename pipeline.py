#!/usr/bin/env python3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fetching_data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "cleaning"))

from src.fetching_data.api_fetchers import main as fetch_api
from src.fetching_data.rss_fetcher import main as fetch_rss
from src.fetching_data.article_content_fetcher import main as fetch_content
from src.cleaning.process_and_store import process


def main() -> None:
    t0 = time.time()

    print("=== Step 1: Fetching API sources ===")
    fetch_api()

    print("\n=== Step 2: Fetching RSS sources ===")
    fetch_rss()

    print("\n=== Step 3: Fetching article content ===")
    fetch_content()

    print("\n=== Step 4: Processing articles (dedup → clean → summarize → topics → sentiment → insert) ===")
    process()

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
