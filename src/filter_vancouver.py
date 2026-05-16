from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None

from config import FOOD_CATEGORIES, PROCESSED_DIR, VANCOUVER_CENTER, YELP_RAW_DIR, ensure_dirs


KEEP_COLUMNS = [
    "business_id",
    "name",
    "address",
    "city",
    "state",
    "postal_code",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "is_open",
    "attributes",
    "categories",
    "hours",
]


def is_near_vancouver(lat: float, lon: float, radius_km: float = 35.0) -> bool:
    if pd.isna(lat) or pd.isna(lon):
        return False
    if geodesic is not None:
        return geodesic(VANCOUVER_CENTER, (lat, lon)).km <= radius_km
    lat1, lon1 = map(math.radians, VANCOUVER_CENTER)
    lat2, lon2 = map(math.radians, (lat, lon))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return (6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))) <= radius_km


def has_food_category(categories: object) -> bool:
    text = str(categories or "").lower()
    return any(cat in text for cat in FOOD_CATEGORIES)


def filter_vancouver_businesses(yelp_dir: Path = YELP_RAW_DIR, chunksize: int = 100_000) -> pd.DataFrame:
    ensure_dirs()
    path = yelp_dir / "business.json"
    if not path.exists():
        print(f"Missing {path}. Run after downloading the Yelp Open Dataset.")
        out = pd.DataFrame(columns=KEEP_COLUMNS)
        out.to_csv(PROCESSED_DIR / "vancouver_places_yelp.csv", index=False)
        return out

    parts = []
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize):
        city_match = chunk["city"].fillna("").str.lower().eq("vancouver")
        coord_match = chunk.apply(lambda r: is_near_vancouver(r.get("latitude"), r.get("longitude")), axis=1)
        food_match = chunk["categories"].apply(has_food_category)
        filtered = chunk[(city_match | coord_match) & food_match].copy()
        parts.append(filtered[[c for c in KEEP_COLUMNS if c in filtered.columns]])

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP_COLUMNS)
    out = out.drop_duplicates(subset=["business_id"])
    out.to_csv(PROCESSED_DIR / "vancouver_places_yelp.csv", index=False)
    print(f"Saved {len(out):,} Vancouver food/cafe Yelp businesses.")
    if len(out) < 50:
        print("WARNING: Few Vancouver Yelp businesses found. Keep the pipeline runnable with other local sources.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yelp-dir", type=Path, default=YELP_RAW_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    filter_vancouver_businesses(args.yelp_dir, args.chunksize)


if __name__ == "__main__":
    main()
