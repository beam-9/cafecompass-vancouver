from __future__ import annotations

import re

import pandas as pd
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FuzzFallback:
        @staticmethod
        def partial_ratio(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio() * 100

    fuzz = _FuzzFallback()

from config import PROCESSED_DIR, ensure_dirs
from entity_resolution import normalize_name

GENERIC_NAMES = {"subway", "restaurant", "cafe", "coffee", "tea", "pizza", "sushi", "bakery", "bar", "kitchen"}


def _source_id(row: pd.Series, idx: int) -> str:
    if pd.notna(row.get("comment_id")):
        return str(row["comment_id"])
    if pd.notna(row.get("post_id")):
        return str(row["post_id"])
    return f"reddit_{idx}"


def _candidate_places(master: pd.DataFrame) -> pd.DataFrame:
    places = master.dropna(subset=["place_id", "name"]).copy()
    places["normalized_name"] = places["name"].map(normalize_name)
    places = places[places["normalized_name"].str.len() >= 5]
    places = places[~places["normalized_name"].isin(GENERIC_NAMES)]
    return places.drop_duplicates(subset=["place_id"])


def link_reddit_to_places(min_score: int = 92, max_matches_per_text: int = 3) -> pd.DataFrame:
    ensure_dirs()
    reddit_path = PROCESSED_DIR / "reddit_vancouver_food_discussions.csv"
    master_path = PROCESSED_DIR / "place_master.csv"
    columns = ["source", "source_id", "post_id", "comment_id", "place_id", "matched_name", "match_score", "text", "text_date"]
    if not reddit_path.exists() or not master_path.exists():
        out = pd.DataFrame(columns=columns)
        out.to_csv(PROCESSED_DIR / "reddit_place_mentions.csv", index=False)
        print("Missing Reddit discussions or place master. Wrote empty reddit_place_mentions.csv.")
        return out

    reddit = pd.read_csv(reddit_path)
    master = pd.read_csv(master_path)
    places = _candidate_places(master)
    if reddit.empty or places.empty:
        out = pd.DataFrame(columns=columns)
        out.to_csv(PROCESSED_DIR / "reddit_place_mentions.csv", index=False)
        print("No Reddit rows or candidate places available for linking.")
        return out

    rows = []
    candidates = places[["place_id", "name", "normalized_name"]].to_dict("records")
    for idx, row in reddit.iterrows():
        text = " ".join(str(row.get(col, "")) for col in ["title", "body"])
        normalized_text = normalize_name(text)
        if not normalized_text:
            continue
        matches = []
        for place in candidates:
            name = place["normalized_name"]
            if len(name.split()) == 1 and name not in normalized_text:
                continue
            score = 100 if re.search(rf"\b{re.escape(name)}\b", normalized_text) else fuzz.partial_ratio(name, normalized_text)
            if score >= min_score:
                matches.append((score, place))
        for score, place in sorted(matches, key=lambda item: item[0], reverse=True)[:max_matches_per_text]:
            rows.append(
                {
                    "source": "reddit",
                    "source_id": _source_id(row, idx),
                    "post_id": row.get("post_id"),
                    "comment_id": row.get("comment_id"),
                    "place_id": place["place_id"],
                    "matched_name": place["name"],
                    "match_score": score,
                    "text": text,
                    "text_date": row.get("created_utc"),
                }
            )
    out = pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["source_id", "place_id", "text"])
    out.to_csv(PROCESSED_DIR / "reddit_place_mentions.csv", index=False)
    print(f"Saved {len(out):,} Reddit place mentions across {out['place_id'].nunique() if not out.empty else 0:,} places.")
    return out


if __name__ == "__main__":
    link_reddit_to_places()
