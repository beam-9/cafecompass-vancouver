from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz

from config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from entity_resolution import distance_meters, normalize_name

INPUT_PATH = RAW_DIR / "community_text.csv"
OUTPUT_PATH = PROCESSED_DIR / "community_place_text.csv"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["source", "source_id", "place_id", "place_name", "match_score", "match_method", "permalink", "text", "text_date"]
    )


def import_community_text(min_name_score: int = 88, max_distance_m: int = 300) -> pd.DataFrame:
    ensure_dirs()
    if not INPUT_PATH.exists():
        out = _empty()
        out.to_csv(OUTPUT_PATH, index=False)
        print(f"Missing {INPUT_PATH}. Copy data/raw/community_text_template.csv to community_text.csv and add rows.")
        return out

    text_df = pd.read_csv(INPUT_PATH)
    master_path = PROCESSED_DIR / "place_master.csv"
    if not master_path.exists():
        out = _empty()
        out.to_csv(OUTPUT_PATH, index=False)
        print("Missing place_master.csv. Run local metadata pipeline first.")
        return out
    master = pd.read_csv(master_path)
    master = master.dropna(subset=["place_id", "name"]).copy()
    master["normalized_name"] = master["name"].map(normalize_name)
    rows = []
    for idx, row in text_df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        place_id = row.get("place_id")
        matched = None
        method = "provided_place_id"
        score = 100
        if pd.notna(place_id) and str(place_id).strip():
            candidates = master[master["place_id"].astype(str) == str(place_id).strip()]
            if not candidates.empty:
                matched = candidates.iloc[0]
        if matched is None:
            method = "name_fuzzy"
            place_name = row.get("place_name")
            norm = normalize_name(place_name)
            if not norm:
                continue
            candidates = master.copy()
            candidates["name_score"] = candidates["normalized_name"].map(lambda value: fuzz.token_set_ratio(norm, value))
            if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                candidates["distance_m"] = candidates.apply(
                    lambda place: distance_meters(
                        (float(row["latitude"]), float(row["longitude"])),
                        (place.get("latitude"), place.get("longitude")),
                    ),
                    axis=1,
                )
                candidates = candidates[candidates["distance_m"] <= max_distance_m]
            candidates = candidates.sort_values("name_score", ascending=False)
            if not candidates.empty and candidates.iloc[0]["name_score"] >= min_name_score:
                matched = candidates.iloc[0]
                score = float(matched["name_score"])
        if matched is None:
            continue
        rows.append(
            {
                "source": row.get("source", "manual"),
                "source_id": f"community_{idx}",
                "place_id": matched["place_id"],
                "place_name": matched["name"],
                "match_score": score,
                "match_method": method,
                "permalink": row.get("permalink"),
                "text": text,
                "text_date": row.get("date"),
            }
        )
    out = pd.DataFrame(rows, columns=_empty().columns)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Imported {len(out):,} community text rows across {out['place_id'].nunique() if not out.empty else 0:,} places.")
    return out


if __name__ == "__main__":
    import_community_text()

