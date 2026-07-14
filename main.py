import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_FILE = PROJECT_ROOT / "data" / "pipeline.log"


def run_step(label: str, cmd: list[str], cwd: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  [DEBUG] Command: {' '.join(cmd)}")
    print(f"  [DEBUG] Working Directory: {cwd}")
    
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        print(result.stdout)
        if result.stderr:
            print(f"  [STDERR]\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"  [FOLLOW-UP REQUIRED] {label} failed with exit code {result.returncode}")
            return False
            
        print(f"  [{label} finished in {elapsed:.1f}s]")
        return True
    except Exception as e:
        print(f"  [DEBUG] Execution threw an exception:")
        traceback.print_exc(file=sys.stdout)
        print(f"  [FOLLOW-UP REQUIRED] {label} failed due to process exception: {e}")
        return False


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    log.write(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    old_stdout = sys.stdout
    sys.stdout = log
    t0 = time.time()

    steps = [
        (
            "Fetching API sources",
            [sys.executable, "api_fetchers.py"],
            str(PROJECT_ROOT / "src" / "fetching_data"),
        ),
        (
            "Fetching RSS sources",
            [sys.executable, "rss_fetcher.py"],
            str(PROJECT_ROOT / "src" / "fetching_data"),
        ),
        (
            "Fetching article content",
            [sys.executable, "article_content_fetcher.py"],
            str(PROJECT_ROOT / "src" / "fetching_data"),
        ),
        (
            "Cleaning data",
            [sys.executable, "general_cleaning.py"],
            str(PROJECT_ROOT / "src" / "cleaning"),
        ),
        (
            "Generating summaries",
            [sys.executable, "summarizer.py"],
            str(PROJECT_ROOT / "src" / "functionalities" / "summary"),
        ),
        (
            "Extracting themes",
            [sys.executable, "theme_extractor.py"],
            str(PROJECT_ROOT / "src" / "functionalities" / "theme_reconginition"),
        ),
        (
            "Analyzing comment sentiment",
            [sys.executable, "analysis.py"],
            str(PROJECT_ROOT / "src" / "functionalities" / "sentiment_analysis"),
        ),
    ]

    failed_steps = []

    for label, cmd, cwd in steps:
        success = run_step(label, cmd, cwd)
        if not success:
            failed_steps.append(label)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print(f"{'='*60}")

    if failed_steps:
        print("\n  [FOLLOW-UP SUMMARY]")
        print("  The following steps failed and require investigation:")
        for step in failed_steps:
            print(f"    - {step}")
    else:
        print("\n  [STATUS] All steps completed successfully.")

    sys.stdout = old_stdout
    log.close()
    print(f"Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()