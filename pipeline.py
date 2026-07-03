#!/usr/bin/env python3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "fetching_data"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "cleaning"))

from api_fetchers import main as fetch_api
from rss_fetcher import main as fetch_rss
from general_cleaning import clean_all


def main() -> None:
    t0 = time.time()

    print("=== Step 1: Fetching API sources ===")
    fetch_api()

    print("\n=== Step 2: Fetching RSS sources ===")
    fetch_rss()

    print("\n=== Step 3: Cleaning and loading into database ===")
    clean_all()

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
