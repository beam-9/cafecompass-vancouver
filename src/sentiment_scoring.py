from __future__ import annotations

import json
import math

import pandas as pd
from pandas.errors import EmptyDataError

from config import ASPECTS, PROCESSED_DIR, ensure_dirs

POSITIVE = {"good", "great", "excellent", "amazing", "delicious", "fresh", "friendly", "quiet", "cozy", "affordable", "worth", "fast", "authentic", "love", "best"}
NEGATIVE = {"bad", "terrible", "slow", "rude", "bland", "dry", "overpriced", "crowded", "loud", "dirty", "worst", "hate", "expensive"}


def _vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def score_text(text: object) -> float:
    text = "" if text is None else str(text)
    analyzer = _vader()
    if analyzer:
        return float(analyzer.polarity_scores(text)["compound"])
    words = {w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()}
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def build_place_aspect_profile() -> pd.DataFrame:
    ensure_dirs()
    aspect_path = PROCESSED_DIR / "review_aspect_scores.csv"
    master_path = PROCESSED_DIR / "place_master.csv"
    if not aspect_path.exists():
        print("Missing review_aspect_scores.csv. Run aspect_extraction.py first.")
        out = pd.DataFrame()
        out.to_csv(PROCESSED_DIR / "place_aspect_profile.csv", index=False)
        return out

    try:
        aspects = pd.read_csv(aspect_path)
    except EmptyDataError:
        aspects = pd.DataFrame()
    try:
        master = pd.read_csv(master_path) if master_path.exists() else pd.DataFrame()
    except EmptyDataError:
        master = pd.DataFrame()
    if "business_id" in aspects.columns and not master.empty and "yelp_business_id" in master.columns:
        lookup = master.set_index("yelp_business_id")["place_id"].to_dict()
        aspects["place_id"] = aspects["place_id"].fillna(aspects["business_id"].map(lookup))

    aspects = aspects.dropna(subset=["place_id"])
    if aspects.empty:
        out = pd.DataFrame(columns=["place_id", *[f"{a}_score" for a in ASPECTS], "confidence_score", "total_texts_used", "evidence_snippets_json"])
        out.to_csv(PROCESSED_DIR / "place_aspect_profile.csv", index=False)
        return out

    aspects["sentiment"] = aspects["evidence_sentence"].map(score_text)
    rows = []
    for place_id, group in aspects.groupby("place_id"):
        total = int(group["source_id"].nunique())
        confidence = min(1.0, math.log1p(total) / math.log1p(100))
        row = {"place_id": place_id, "confidence_score": confidence, "total_texts_used": total}
        evidence = {}
        for aspect in ASPECTS:
            subset = group[group["aspect"] == aspect]
            raw = subset["mention_count"].sum()
            sentiment = subset["sentiment"].mean() if not subset.empty else 0
            row[f"{aspect}_score"] = float(min(1, raw / max(total, 1)) * (1 + sentiment) / 2 * confidence)
            snippets = []
            for value in subset["evidence_sentence"].head(5):
                try:
                    snippets.extend(json.loads(value))
                except Exception:
                    pass
            evidence[aspect] = snippets[:3]
        row["evidence_snippets_json"] = json.dumps(evidence)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "place_aspect_profile.csv", index=False)
    print(f"Saved {len(out):,} place aspect profiles.")
    return out


if __name__ == "__main__":
    build_place_aspect_profile()
