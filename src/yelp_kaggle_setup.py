from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from config import INTERIM_DIR, RAW_DIR, YELP_RAW_DIR, ensure_dirs

KAGGLE_DATASET = "yelp-dataset/yelp-dataset"
EXPECTED_FILES = ["business.json", "review.json", "tip.json", "checkin.json"]


def kaggle_credentials_available() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_creds = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    return kaggle_json.exists() or bool(env_creds)


def validate_yelp_files(yelp_dir: Path = YELP_RAW_DIR) -> pd.DataFrame:
    ensure_dirs()
    rows = []
    for name in EXPECTED_FILES:
        path = yelp_dir / name
        rows.append(
            {
                "file": name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1_000_000, 2) if path.exists() else 0,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(INTERIM_DIR / "yelp_file_status.csv", index=False)
    print(out.to_string(index=False))
    return out


def _find_yelp_files(root: Path) -> dict[str, Path]:
    found = {}
    for path in root.rglob("*.json"):
        if path.name in EXPECTED_FILES:
            found[path.name] = path
        elif path.name.startswith("yelp_academic_dataset_"):
            short_name = path.name.replace("yelp_academic_dataset_", "")
            if short_name in EXPECTED_FILES:
                found[short_name] = path
    return found


def download_yelp_from_kaggle(force: bool = False) -> bool:
    ensure_dirs()
    YELP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    status = validate_yelp_files(YELP_RAW_DIR)
    if status["exists"].all() and not force:
        print("Yelp Open Dataset files already exist. Skipping Kaggle download.")
        return True
    if not kaggle_credentials_available():
        print(
            "Kaggle credentials are missing. Add ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY, "
            "then rerun `python src/run_pipeline.py --setup-yelp`."
        )
        return False
    if shutil.which("kaggle") is None:
        print("The Kaggle CLI is not installed. Run `pip install kaggle` or install project requirements.")
        return False

    download_dir = RAW_DIR / "kaggle_yelp_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(download_dir),
        "--unzip",
    ]
    print("Downloading Yelp Open Dataset from Kaggle...")
    subprocess.run(cmd, check=True)

    found = _find_yelp_files(download_dir)
    if not found:
        for archive in download_dir.glob("*.zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(download_dir)
        found = _find_yelp_files(download_dir)

    missing = [name for name in EXPECTED_FILES if name not in found]
    if missing:
        print(f"Downloaded files, but these expected Yelp files were not found: {missing}")
        return False

    for name, source in found.items():
        target = YELP_RAW_DIR / name
        if target.exists() and not force:
            continue
        shutil.copy2(source, target)
        print(f"Prepared {target}")

    final_status = validate_yelp_files(YELP_RAW_DIR)
    return bool(final_status["exists"].all())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Redownload/copy Yelp files even if outputs exist.")
    args = parser.parse_args()
    ok = download_yelp_from_kaggle(force=args.force)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
