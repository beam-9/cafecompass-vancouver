from __future__ import annotations

import pandas as pd
import requests

from config import PROCESSED_DIR, ensure_dirs

BUSINESS_LICENCES_URL = (
    "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/business-licences/exports/json"
)
FOOD_VENDORS_URL = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/food-vendors/exports/json"


def _download_json(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    data = response.json()
    return pd.json_normalize(data)


def collect_business_licences() -> pd.DataFrame:
    ensure_dirs()
    df = _download_json(BUSINESS_LICENCES_URL)
    food_mask = df.astype(str).agg(" ".join, axis=1).str.contains(
        "restaurant|cafe|coffee|food|bakery|tea|catering|refreshment", case=False, na=False
    )
    out = df[food_mask].copy()
    out.to_csv(PROCESSED_DIR / "vancouver_business_licences.csv", index=False)
    print(f"Saved {len(out):,} Vancouver food-related business licences.")
    return out


def collect_food_vendors() -> pd.DataFrame:
    ensure_dirs()
    df = _download_json(FOOD_VENDORS_URL)
    df.to_csv(PROCESSED_DIR / "vancouver_food_vendors.csv", index=False)
    print(f"Saved {len(df):,} Vancouver food vendors.")
    return df


def main() -> None:
    for collector in [collect_business_licences, collect_food_vendors]:
        try:
            collector()
        except requests.RequestException as exc:
            print(f"{collector.__name__} failed: {exc}")


if __name__ == "__main__":
    main()

