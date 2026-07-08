#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "data" / "pipeline.log"


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
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w")
    log.write(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    old_stdout = sys.stdout
    sys.stdout = log
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

    sys.stdout = old_stdout
    log.close()
    print(f"Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
