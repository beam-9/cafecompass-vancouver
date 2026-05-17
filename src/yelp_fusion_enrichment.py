from __future__ import annotations

import os
import time

import pandas as pd
import requests
from rapidfuzz import fuzz

from config import PROCESSED_DIR, ensure_dirs

SEARCH_URL = "https://api.yelp.com/v3/businesses/search"


def _load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return os.getenv("YELP_FUSION_API_KEY")


def enrich_with_yelp_fusion(limit: int = 500, min_match_score: int = 90, sleep_s: float = 0.2) -> pd.DataFrame:
    ensure_dirs()
    key = _load_api_key()
    out_path = PROCESSED_DIR / "yelp_fusion_enrichment.csv"
    if not key:
        out = pd.DataFrame(columns=["place_id", "yelp_fusion_id", "rating", "review_count", "price", "url", "match_score"])
        out.to_csv(out_path, index=False)
        print("Missing YELP_FUSION_API_KEY. Add it to .env or your shell to enrich ratings/review counts.")
        return out

    master_path = PROCESSED_DIR / "place_master.csv"
    if not master_path.exists():
        raise FileNotFoundError("Missing place_master.csv. Run local metadata pipeline first.")
    master = pd.read_csv(master_path).dropna(subset=["place_id", "name", "latitude", "longitude"]).head(limit)
    headers = {"Authorization": f"Bearer {key}"}
    rows = []
    for _, place in master.iterrows():
        params = {
            "term": place["name"],
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "radius": 200,
            "limit": 3,
            "categories": "restaurants,food,cafes",
        }
        try:
            response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=20)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Yelp Fusion lookup failed for {place['name']}: {exc}")
            continue
        businesses = response.json().get("businesses", [])
        best = None
        best_score = 0
        for business in businesses:
            score = fuzz.token_set_ratio(str(place["name"]), business.get("name", ""))
            if score > best_score:
                best = business
                best_score = score
        if best and best_score >= min_match_score:
            rows.append(
                {
                    "place_id": place["place_id"],
                    "yelp_fusion_id": best.get("id"),
                    "rating": best.get("rating"),
                    "review_count": best.get("review_count"),
                    "price": best.get("price"),
                    "url": best.get("url"),
                    "match_score": best_score,
                }
            )
        time.sleep(sleep_s)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved Yelp Fusion enrichment for {len(out):,} places.")
    return out


if __name__ == "__main__":
    enrich_with_yelp_fusion()

