from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import PROCESSED_DIR, YELP_RAW_DIR, ensure_dirs
from text_preprocessing import preprocess_text


REVIEW_COLUMNS = ["review_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"]
TIP_COLUMNS = ["user_id", "business_id", "text", "date", "compliment_count"]


def _load_business_ids(path: Path) -> set[str]:
    if not path.exists():
        print(f"Missing {path}. Run filter_vancouver.py first.")
        return set()
    df = pd.read_csv(path, usecols=["business_id"])
    return set(df["business_id"].dropna().astype(str))


def clean_yelp_reviews(yelp_dir: Path = YELP_RAW_DIR, chunksize: int = 100_000, min_chars: int = 30) -> pd.DataFrame:
    ensure_dirs()
    business_ids = _load_business_ids(PROCESSED_DIR / "vancouver_places_yelp.csv")
    review_path = yelp_dir / "review.json"
    if not business_ids or not review_path.exists():
        print("No Vancouver business IDs or review.json found. Writing empty reviews file.")
        out = pd.DataFrame(columns=REVIEW_COLUMNS + ["clean_text"])
        out.to_csv(PROCESSED_DIR / "vancouver_reviews_clean.csv", index=False)
        return out

    parts = []
    for chunk in tqdm(pd.read_json(review_path, lines=True, chunksize=chunksize), desc="Cleaning reviews"):
        filtered = chunk[chunk["business_id"].isin(business_ids)][REVIEW_COLUMNS].copy()
        filtered = filtered.drop_duplicates(subset=["review_id"])
        filtered = filtered[filtered["text"].fillna("").str.len() >= min_chars]
        filtered["clean_text"] = filtered["text"].map(preprocess_text)
        parts.append(filtered)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=REVIEW_COLUMNS + ["clean_text"])
    out = out.drop_duplicates(subset=["review_id"])
    out.to_csv(PROCESSED_DIR / "vancouver_reviews_clean.csv", index=False)
    print(f"Saved {len(out):,} cleaned Vancouver Yelp reviews.")
    return out


def clean_yelp_tips(yelp_dir: Path = YELP_RAW_DIR, chunksize: int = 100_000, min_chars: int = 15) -> pd.DataFrame:
    business_ids = _load_business_ids(PROCESSED_DIR / "vancouver_places_yelp.csv")
    tip_path = yelp_dir / "tip.json"
    if not business_ids or not tip_path.exists():
        print("No Vancouver business IDs or tip.json found. Writing empty tips file.")
        out = pd.DataFrame(columns=["business_id", "text", "date", "compliment_count", "clean_text"])
        out.to_csv(PROCESSED_DIR / "vancouver_tips_clean.csv", index=False)
        return out

    parts = []
    for chunk in tqdm(pd.read_json(tip_path, lines=True, chunksize=chunksize), desc="Cleaning tips"):
        cols = [c for c in TIP_COLUMNS if c in chunk.columns]
        filtered = chunk[chunk["business_id"].isin(business_ids)][cols].copy()
        filtered = filtered.drop(columns=["user_id"], errors="ignore")
        filtered = filtered[filtered["text"].fillna("").str.len() >= min_chars]
        filtered["clean_text"] = filtered["text"].map(preprocess_text)
        parts.append(filtered)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = out.drop_duplicates(subset=["business_id", "text", "date"])
    out.to_csv(PROCESSED_DIR / "vancouver_tips_clean.csv", index=False)
    print(f"Saved {len(out):,} cleaned Vancouver Yelp tips.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp-dir", type=Path, default=YELP_RAW_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    clean_yelp_reviews(args.yelp_dir, args.chunksize)
    clean_yelp_tips(args.yelp_dir, args.chunksize)


if __name__ == "__main__":
    main()

