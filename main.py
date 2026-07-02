#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str], cwd: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print(f"  [STDERR] {result.stderr}")
    if result.returncode != 0:
        print(f"  [ERROR] exited with code {result.returncode}")
    print(f"  [{label} finished in {elapsed:.1f}s]")


def main() -> None:
    t0 = time.time()

    run_step(
        "Fetching API sources",
        [sys.executable, "api_fetchers.py"],
        str(PROJECT_ROOT / "src" / "fetching_data"),
    )

    run_step(
        "Fetching RSS sources",
        [sys.executable, "rss_fetcher.py"],
        str(PROJECT_ROOT / "src" / "fetching_data"),
    )

    run_step(
        "Fetching article content",
        [sys.executable, "article_content_fetcher.py"],
        str(PROJECT_ROOT / "src" / "fetching_data"),
    )

    run_step(
        "Cleaning data",
        [sys.executable, "general_cleaning.py"],
        str(PROJECT_ROOT / "src" / "cleaning"),
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
