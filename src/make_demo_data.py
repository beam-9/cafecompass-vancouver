from __future__ import annotations

import json
import math

import pandas as pd

from aspect_extraction import ASPECT_KEYWORDS, extract_aspects_from_text
from config import ASPECTS, PROCESSED_DIR, PROJECT_ROOT, ensure_dirs
from evaluation import run_evaluation
from feature_engineering import build_recommender_features
from sentiment_scoring import score_text

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def _confidence(text_count: int) -> float:
    return min(1.0, math.log1p(text_count) / math.log1p(100))


def _build_aspect_rows(texts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in texts.iterrows():
        detected = extract_aspects_from_text(row["text"])
        for aspect, snippets in detected.items():
            rows.append(
                {
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "business_id": row.get("business_id"),
                    "place_id": row["place_id"],
                    "aspect": aspect,
                    "mention_count": len(snippets),
                    "evidence_sentence": json.dumps(snippets[:3]),
                    "text_date": row["text_date"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=["source", "source_id", "business_id", "place_id", "aspect", "mention_count", "evidence_sentence", "text_date"],
    )


def _build_profiles(texts: pd.DataFrame, aspect_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for place_id, place_texts in texts.groupby("place_id"):
        text_count = int(place_texts["source_id"].nunique())
        confidence = _confidence(text_count)
        place_aspects = aspect_rows[aspect_rows["place_id"] == place_id]
        row = {"place_id": place_id, "confidence_score": confidence, "total_texts_used": text_count}
        evidence = {}
        for aspect in ASPECTS:
            subset = place_aspects[place_aspects["aspect"] == aspect]
            snippets = []
            sentiments = []
            for value in subset["evidence_sentence"].dropna().head(5):
                loaded = json.loads(value)
                snippets.extend(loaded)
                sentiments.extend(score_text(snippet) for snippet in loaded)
            mention_strength = min(1.0, subset["mention_count"].sum() / max(text_count, 1))
            sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            row[f"{aspect}_score"] = mention_strength * ((sentiment + 1) / 2) * confidence
            evidence[aspect] = snippets[:3]
        row["evidence_snippets_json"] = json.dumps(evidence)
        rows.append(row)
    return pd.DataFrame(rows)


def build_demo_data() -> None:
    ensure_dirs()
    places = pd.read_csv(SAMPLE_DIR / "demo_places.csv")
    texts = pd.read_csv(SAMPLE_DIR / "demo_text.csv")

    master = places.copy()
    master["yelp_business_id"] = pd.NA
    master["osm_id"] = pd.NA
    master["city_business_id"] = pd.NA
    master["foursquare_id"] = pd.NA
    master["normalized_name"] = master["name"].str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    master["address"] = pd.NA
    master["source_flags"] = "sample_demo"
    master["hours"] = pd.NA
    master["official_business_type"] = pd.NA
    master.to_csv(PROCESSED_DIR / "place_master.csv", index=False)

    reviews = texts.rename(columns={"source_id": "review_id"}).copy()
    reviews["stars"] = 5
    reviews["useful"] = 0
    reviews["funny"] = 0
    reviews["cool"] = 0
    reviews["date"] = reviews["text_date"]
    reviews["clean_text"] = reviews["text"].str.lower()
    reviews[
        ["review_id", "business_id", "stars", "useful", "funny", "cool", "text", "date", "clean_text", "place_id"]
    ].to_csv(PROCESSED_DIR / "vancouver_reviews_clean.csv", index=False)
    pd.DataFrame(columns=["business_id", "text", "date", "compliment_count", "clean_text"]).to_csv(
        PROCESSED_DIR / "vancouver_tips_clean.csv", index=False
    )

    aspect_rows = _build_aspect_rows(texts)
    aspect_rows.to_csv(PROCESSED_DIR / "review_aspect_scores.csv", index=False)

    profiles = _build_profiles(texts, aspect_rows)
    profiles.to_csv(PROCESSED_DIR / "place_aspect_profile.csv", index=False)

    build_recommender_features()
    run_evaluation()
    features = pd.read_csv(PROCESSED_DIR / "recommender_features.csv")
    evaluation = pd.read_csv(PROCESSED_DIR / "evaluation_results.csv")
    features.to_csv(SAMPLE_DIR / "demo_recommender_features.csv", index=False)
    evaluation.to_csv(SAMPLE_DIR / "demo_evaluation_results.csv", index=False)
    print("Built demo processed data from data/sample.")


if __name__ == "__main__":
    build_demo_data()
