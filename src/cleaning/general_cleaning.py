import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GeneralCleaner:
    def __init__(self, date_str: str | None = None):
        self.date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        self.raw_dir = PROJECT_ROOT / "data" / "raw" / self.date_str
        self.cleaned_dir = PROJECT_ROOT / "data" / "cleaned" / self.date_str

    def clean_all(self) -> None:
        if not self.raw_dir.exists():
            print(f"No raw data directory found: {self.raw_dir}")
            return

        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(self.raw_dir.glob("*.json"))
        if not files:
            print(f"No JSON files found in {self.raw_dir}")
            return

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._clean_file, f): f for f in files}
            for future in as_completed(futures):
                f = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [ERROR] Failed to clean {f.name}: {e}")

    def _clean_file(self, filepath: Path) -> None:
        print(f"\n  Cleaning: {filepath.name}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned = self._clean_value(data)

        out_path = self.cleaned_dir / filepath.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        print(f"  [SAVED] {out_path}")

    def _clean_value(self, value):
        if isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean_value(item) for item in value]
        if isinstance(value, str):
            return self._strip_html(value)
        return value

    @staticmethod
    def _strip_html(text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


if __name__ == "__main__":
    cleaner = GeneralCleaner()
    cleaner.clean_all()
