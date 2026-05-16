from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, ensure_dirs

ASPECT_KEYWORDS = {
    "quiet_study": ["quiet", "study", "studying", "laptop", "laptops", "outlet", "outlets", "wifi", "wi-fi", "work", "working", "seating", "spacious", "calm", "peaceful", "not crowded", "plug", "plugs"],
    "date_night": ["date", "romantic", "cozy", "intimate", "ambience", "ambiance", "lighting", "cocktails", "wine", "cute", "vibe", "special occasion"],
    "cheap_value": ["cheap", "affordable", "value", "worth", "portion", "portions", "student", "budget", "inexpensive", "reasonable", "filling"],
    "hidden_gem": ["hidden gem", "underrated", "local favorite", "hole in the wall", "small place", "tucked away", "not crowded", "gem"],
    "authentic": ["authentic", "traditional", "homemade", "family", "family-run", "real", "legit", "reminds me of", "classic"],
    "service_speed": ["quick", "fast", "slow", "wait", "waiting", "line", "queue", "service", "staff", "friendly", "rude", "attentive"],
    "group_friendly": ["group", "groups", "large table", "big table", "reservation", "reservations", "friends", "family", "share", "sharing"],
    "food_quality": ["delicious", "tasty", "fresh", "flavourful", "flavorful", "bland", "dry", "overcooked", "crispy", "rich", "amazing"],
    "dessert_drinks": ["matcha", "coffee", "latte", "tea", "bubble tea", "cake", "pastry", "pastries", "dessert", "croissant", "sweet"],
    "late_night": ["late night", "open late", "midnight", "24 hour", "after hours"],
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: object) -> list[str]:
    text = "" if text is None else str(text).strip()
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def extract_aspects_from_text(text: object, max_snippets: int = 3) -> dict[str, list[str]]:
    sentences = split_sentences(text)
    found = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        snippets = []
        for sentence in sentences:
            sentence_l = sentence.lower()
            if any(keyword in sentence_l for keyword in keywords):
                snippets.append(sentence)
            if len(snippets) >= max_snippets:
                break
        if snippets:
            found[aspect] = snippets
    return found


def _load_text_sources() -> list[pd.DataFrame]:
    sources = []
    files = [
        ("yelp_review", PROCESSED_DIR / "vancouver_reviews_clean.csv", "review_id", "business_id", "text", "date"),
        ("yelp_tip", PROCESSED_DIR / "vancouver_tips_clean.csv", None, "business_id", "text", "date"),
        ("reddit", PROCESSED_DIR / "reddit_vancouver_food_discussions.csv", None, None, "body", "created_utc"),
    ]
    for source, path, id_col, place_col, text_col, date_col in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or text_col not in df.columns:
            continue
        out = pd.DataFrame(
            {
                "source": source,
                "source_id": df[id_col] if id_col and id_col in df.columns else [f"{source}_{i}" for i in range(len(df))],
                "business_id": df[place_col] if place_col and place_col in df.columns else None,
                "place_id": None,
                "text": df[text_col],
                "text_date": df[date_col] if date_col in df.columns else None,
            }
        )
        sources.append(out)
    return sources


def build_review_aspect_scores() -> pd.DataFrame:
    ensure_dirs()
    columns = ["source", "source_id", "business_id", "place_id", "aspect", "mention_count", "evidence_sentence", "text_date"]
    rows = []
    for df in _load_text_sources():
        for _, row in df.iterrows():
            aspects = extract_aspects_from_text(row["text"])
            for aspect, snippets in aspects.items():
                rows.append(
                    {
                        "source": row["source"],
                        "source_id": row["source_id"],
                        "business_id": row["business_id"],
                        "place_id": row["place_id"],
                        "aspect": aspect,
                        "mention_count": len(snippets),
                        "evidence_sentence": json.dumps(snippets[:3]),
                        "text_date": row["text_date"],
                    }
                )
    out = pd.DataFrame(rows, columns=columns)
    out.to_csv(PROCESSED_DIR / "review_aspect_scores.csv", index=False)
    print(f"Saved {len(out):,} aspect evidence rows.")
    return out


if __name__ == "__main__":
    build_review_aspect_scores()
