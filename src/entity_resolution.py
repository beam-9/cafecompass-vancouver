from __future__ import annotations

import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FuzzFallback:
        @staticmethod
        def token_set_ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FuzzFallback()

from config import PROCESSED_DIR, ensure_dirs

SUFFIX_RE = re.compile(r"\b(cafe|coffee|restaurant|restaurants|ltd|inc|limited|corp|co|the)\b")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
MASTER_COLUMNS = [
    "place_id",
    "yelp_business_id",
    "osm_id",
    "city_business_id",
    "foursquare_id",
    "name",
    "normalized_name",
    "latitude",
    "longitude",
    "address",
    "categories",
    "cuisine",
    "source_flags",
    "stars",
    "review_count",
    "is_open",
    "hours",
    "official_business_type",
    "yelp_osm_match_score",
    "yelp_osm_match_distance_m",
    "match_quality",
]


def normalize_name(name: object) -> str:
    text = "" if name is None else str(name).lower()
    text = PUNCT_RE.sub(" ", text)
    text = SUFFIX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def distance_meters(a: tuple[float, float], b: tuple[float, float]) -> float:
    if any(pd.isna(v) for v in [*a, *b]):
        return float("inf")
    if geodesic is not None:
        return geodesic(a, b).meters
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_place_master() -> pd.DataFrame:
    ensure_dirs()
    yelp = _read(PROCESSED_DIR / "vancouver_places_yelp.csv")
    osm = _read(PROCESSED_DIR / "vancouver_osm_food_places.csv")
    licences = _read(PROCESSED_DIR / "vancouver_business_licences.csv")

    records = []
    for _, row in yelp.iterrows():
        records.append(
            {
                "place_id": f"pl_{len(records)+1:06d}",
                "yelp_business_id": row.get("business_id"),
                "osm_id": np.nan,
                "city_business_id": np.nan,
                "foursquare_id": np.nan,
                "name": row.get("name"),
                "normalized_name": normalize_name(row.get("name")),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "address": row.get("address"),
                "categories": row.get("categories"),
                "cuisine": np.nan,
                "source_flags": "yelp",
                "stars": row.get("stars"),
                "review_count": row.get("review_count"),
                "is_open": row.get("is_open"),
                "hours": row.get("hours"),
                "official_business_type": np.nan,
                "yelp_osm_match_score": np.nan,
                "yelp_osm_match_distance_m": np.nan,
                "match_quality": "yelp_only",
            }
        )

    master = pd.DataFrame(records, columns=MASTER_COLUMNS)
    if master.empty and not osm.empty:
        master = pd.DataFrame(columns=MASTER_COLUMNS)

    for _, osm_row in osm.iterrows():
        norm = normalize_name(osm_row.get("name"))
        matched_idx = None
        for idx, master_row in master.iterrows():
            sim = fuzz.token_set_ratio(norm, master_row.get("normalized_name", ""))
            dist = distance_meters(
                (osm_row.get("latitude"), osm_row.get("longitude")),
                (master_row.get("latitude"), master_row.get("longitude")),
            )
            if (sim >= 85 and dist <= 150) or (sim >= 92 and dist <= 300):
                matched_idx = idx
                break
        if matched_idx is None:
            master.loc[len(master)] = {
                "place_id": f"pl_{len(master)+1:06d}",
                "yelp_business_id": np.nan,
                "osm_id": osm_row.get("osm_id"),
                "city_business_id": np.nan,
                "foursquare_id": np.nan,
                "name": osm_row.get("name"),
                "normalized_name": norm,
                "latitude": osm_row.get("latitude"),
                "longitude": osm_row.get("longitude"),
                "address": osm_row.get("address"),
                "categories": osm_row.get("amenity"),
                "cuisine": osm_row.get("cuisine"),
                "source_flags": "osm",
                "stars": np.nan,
                "review_count": np.nan,
                "is_open": np.nan,
                "hours": osm_row.get("opening_hours"),
                "official_business_type": np.nan,
                "yelp_osm_match_score": np.nan,
                "yelp_osm_match_distance_m": np.nan,
                "match_quality": "osm_only",
            }
        else:
            sim = fuzz.token_set_ratio(norm, master.loc[matched_idx, "normalized_name"])
            dist = distance_meters(
                (osm_row.get("latitude"), osm_row.get("longitude")),
                (master.loc[matched_idx, "latitude"], master.loc[matched_idx, "longitude"]),
            )
            master.loc[matched_idx, "osm_id"] = osm_row.get("osm_id")
            master.loc[matched_idx, "source_flags"] = f"{master.loc[matched_idx, 'source_flags']}|osm"
            master.loc[matched_idx, "yelp_osm_match_score"] = sim
            master.loc[matched_idx, "yelp_osm_match_distance_m"] = dist
            master.loc[matched_idx, "match_quality"] = "high" if sim >= 92 and dist <= 150 else "medium"
            if pd.isna(master.loc[matched_idx, "cuisine"]):
                master.loc[matched_idx, "cuisine"] = osm_row.get("cuisine")

    if not licences.empty and not master.empty:
        name_col = next((c for c in licences.columns if "business" in c.lower() and "name" in c.lower()), None)
        type_col = next((c for c in licences.columns if "business" in c.lower() and "type" in c.lower()), None)
        if name_col:
            for idx, row in master.iterrows():
                candidates = licences[name_col].dropna().astype(str)
                if candidates.empty:
                    continue
                scores = candidates.map(lambda name: fuzz.token_set_ratio(row["normalized_name"], normalize_name(name)))
                best_idx = scores.idxmax()
                if scores.loc[best_idx] >= 90:
                    master.loc[idx, "city_business_id"] = best_idx
                    master.loc[idx, "source_flags"] = f"{master.loc[idx, 'source_flags']}|city"
                    if type_col:
                        master.loc[idx, "official_business_type"] = licences.loc[best_idx, type_col]

    master.to_csv(PROCESSED_DIR / "place_master.csv", index=False)
    print(f"Saved {len(master):,} master places.")
    return master


if __name__ == "__main__":
    build_place_master()
