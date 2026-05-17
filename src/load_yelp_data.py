from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import INTERIM_DIR, YELP_RAW_DIR, ensure_dirs


def read_jsonl_chunks(path: Path, chunksize: int = 100_000):
    return pd.read_json(path, lines=True, chunksize=chunksize)


def inspect_businesses(yelp_dir: Path = YELP_RAW_DIR, chunksize: int = 100_000) -> pd.DataFrame:
    ensure_dirs()
    business_path = yelp_dir / "business.json"
    if not business_path.exists():
        print(f"Missing {business_path}. Download the Yelp Open Dataset and place files in data/raw/yelp/.")
        return pd.DataFrame()

    samples = []
    city_counts = {}
    total = 0
    vancouver_food = 0

    for chunk in tqdm(read_jsonl_chunks(business_path, chunksize), desc="Reading businesses"):
        total += len(chunk)
        samples.append(chunk.head(100))
        counts = chunk["city"].fillna("Unknown").value_counts()
        for city, count in counts.items():
            city_counts[city] = city_counts.get(city, 0) + int(count)

        is_vancouver = chunk["city"].fillna("").str.lower().eq("vancouver")
        categories = chunk.get("categories", pd.Series("", index=chunk.index)).fillna("").str.lower()
        is_food = categories.str.contains(
            "restaurants|food|cafes|coffee & tea|bakeries|desserts|bubble tea|fast food",
            regex=True,
        )
        vancouver_food += int((is_vancouver & is_food).sum())

    sample_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    city_counts_df = (
        pd.DataFrame({"city": list(city_counts.keys()), "business_count": list(city_counts.values())})
        .sort_values("business_count", ascending=False)
        .reset_index(drop=True)
    )

    sample_df.to_csv(INTERIM_DIR / "yelp_businesses_sample.csv", index=False)
    city_counts_df.to_csv(INTERIM_DIR / "yelp_city_counts.csv", index=False)

    has_vancouver = "Vancouver" in city_counts
    print(f"Businesses: {total:,}")
    print("Top cities:")
    print(city_counts_df.head(15).to_string(index=False))
    print(f"Vancouver available: {has_vancouver}")
    print(f"Vancouver food/cafe businesses: {vancouver_food:,}")
    if not has_vancouver or vancouver_food < 50:
        print(
            "WARNING: Vancouver Yelp Open Dataset coverage appears limited. "
            "Use Yelp as a prototype review-text source and rely on OSM, Vancouver Open Data, "
            "and optional Reddit for Vancouver-specific coverage."
        )
    pd.DataFrame(
        [
            {
                "total_businesses": total,
                "vancouver_available": has_vancouver,
                "vancouver_food_businesses": vancouver_food,
            }
        ]
    ).to_csv(INTERIM_DIR / "yelp_vancouver_coverage_summary.csv", index=False)
    return city_counts_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp-dir", type=Path, default=YELP_RAW_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    inspect_businesses(args.yelp_dir, args.chunksize)


if __name__ == "__main__":
    main()
