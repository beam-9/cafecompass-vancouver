from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import requests

from config import PROCESSED_DIR, ensure_dirs

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY = """
[out:json][timeout:60];
area["name"="Vancouver"]["boundary"="administrative"]->.searchArea;
(
  node["amenity"~"cafe|restaurant|fast_food|food_court"](area.searchArea);
  way["amenity"~"cafe|restaurant|fast_food|food_court"](area.searchArea);
  relation["amenity"~"cafe|restaurant|fast_food|food_court"](area.searchArea);
);
out center tags;
"""


def _address_from_tags(tags: dict[str, Any]) -> str:
    parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:postcode")]
    return " ".join(str(p) for p in parts if p)


def collect_osm_food_places(url: str = OVERPASS_URL) -> pd.DataFrame:
    ensure_dirs()
    response = requests.post(url, data={"data": QUERY}, timeout=90)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    rows = []
    for element in elements:
        tags = element.get("tags", {})
        center = element.get("center", {})
        lat = element.get("lat", center.get("lat"))
        lon = element.get("lon", center.get("lon"))
        rows.append(
            {
                "osm_id": f"{element.get('type')}/{element.get('id')}",
                "name": tags.get("name"),
                "amenity": tags.get("amenity"),
                "cuisine": tags.get("cuisine"),
                "opening_hours": tags.get("opening_hours"),
                "latitude": lat,
                "longitude": lon,
                "website": tags.get("website") or tags.get("contact:website"),
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "address": _address_from_tags(tags),
                "raw_tags": tags,
            }
        )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["name", "latitude", "longitude"], how="any")
    df.to_csv(PROCESSED_DIR / "vancouver_osm_food_places.csv", index=False)
    print(f"Saved {len(df):,} OSM Vancouver food places.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=OVERPASS_URL)
    args = parser.parse_args()
    try:
        collect_osm_food_places(args.url)
    except requests.RequestException as exc:
        print(f"OSM collection failed: {exc}")


if __name__ == "__main__":
    main()

