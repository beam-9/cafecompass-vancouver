from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import ASPECTS, MODELS_DIR, PROCESSED_DIR

DEFAULT_WEIGHTS = {
    "aspect_match_score": 0.30,
    "semantic_similarity_score": 0.20,
    "distance_score": 0.15,
    "rating_score": 0.00,
    "confidence_score": 0.10,
    "context_score": 0.10,
    "hidden_gem_score": 0.05,
}

EXPERIENCE_PRESETS = {
    "quiet study cafe": {"quiet_study": 1.0, "dessert_drinks": 0.4, "service_speed": 0.2},
    "laptop-friendly cafe": {"quiet_study": 1.0, "dessert_drinks": 0.3},
    "cheap eats": {"cheap_value": 1.0, "food_quality": 0.4},
    "date-night restaurant": {"date_night": 1.0, "food_quality": 0.5, "service_speed": 0.2},
    "hidden gem": {"hidden_gem": 1.0, "food_quality": 0.4, "authentic": 0.3},
    "authentic food": {"authentic": 1.0, "food_quality": 0.4},
    "group-friendly restaurant": {"group_friendly": 1.0, "service_speed": 0.3},
    "quick lunch": {"service_speed": 1.0, "cheap_value": 0.4},
    "dessert/matcha/cafe hopping": {"dessert_drinks": 1.0, "food_quality": 0.4},
    "late-night food": {"late_night": 1.0, "service_speed": 0.3},
}

PREFERENCE_KEYWORDS = {
    "quiet_study": ["quiet", "study", "studying", "laptop", "outlet", "outlets", "wifi", "wi-fi", "work", "working"],
    "date_night": ["date", "romantic", "cozy", "intimate", "ambience", "ambiance", "wine", "cocktail"],
    "cheap_value": ["cheap", "affordable", "value", "budget", "student", "inexpensive", "reasonable"],
    "hidden_gem": ["hidden gem", "underrated", "local favorite", "hole in the wall", "tucked away"],
    "authentic": ["authentic", "traditional", "homemade", "family-run", "legit"],
    "service_speed": ["quick", "fast", "lunch", "service", "line", "queue"],
    "group_friendly": ["group", "groups", "large table", "big table", "reservation", "friends", "family"],
    "food_quality": ["delicious", "tasty", "fresh", "flavourful", "flavorful", "good food"],
    "dessert_drinks": ["dessert", "matcha", "coffee", "latte", "tea", "bubble tea", "cake", "pastry", "croissant"],
    "late_night": ["late night", "open late", "midnight", "24 hour", "after hours"],
}

METADATA_INTENT_TERMS = {
    "quiet study cafe": ["cafe", "coffee", "coffee_shop", "tea", "library"],
    "laptop-friendly cafe": ["cafe", "coffee", "coffee_shop", "tea"],
    "cheap eats": ["fast_food", "food", "cheap", "lunch", "sandwich", "burger", "pizza", "noodle", "bento"],
    "date-night restaurant": ["restaurant", "wine", "bar", "italian", "sushi", "izakaya", "seafood"],
    "hidden gem": ["restaurant", "cafe", "food_court", "ramen", "noodle", "bakery"],
    "authentic food": ["restaurant", "chinese", "japanese", "korean", "vietnamese", "thai", "indian", "mexican"],
    "group-friendly restaurant": ["restaurant", "hot_pot", "bbq", "family", "food_court", "group"],
    "quick lunch": ["fast_food", "sandwich", "burger", "pizza", "bento", "lunch", "food_court"],
    "dessert/matcha/cafe hopping": ["dessert", "matcha", "bubble_tea", "ice_cream", "bakery", "cafe", "coffee_shop", "tea"],
    "late-night food": ["fast_food", "restaurant", "bar", "pizza", "burger", "kebab"],
}


@dataclass
class RecommenderConfig:
    start_lat: float
    start_lon: float
    max_distance_km: float = 8.0
    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def load_features(path=PROCESSED_DIR / "recommender_features.csv") -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def preference_vector(experience: str | dict[str, float]) -> dict[str, float]:
    if isinstance(experience, dict):
        return {a: float(experience.get(a, 0)) for a in ASPECTS}
    lowered = experience.lower()
    vector = {a: 0.0 for a in ASPECTS}
    for label, preset in EXPERIENCE_PRESETS.items():
        if label in lowered or any(part in lowered for part in label.split("/")):
            for aspect, value in preset.items():
                vector[aspect] = max(vector[aspect], value)
    for aspect in ASPECTS:
        if aspect.replace("_", " ") in lowered:
            vector[aspect] = 1.0
        elif any(keyword in lowered for keyword in PREFERENCE_KEYWORDS.get(aspect, [])):
            vector[aspect] = max(vector[aspect], 1.0)
    return vector


def _tokenize(text: object) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9_]+", str(text or "").lower()) if len(token) > 1}


def metadata_match_scores(out: pd.DataFrame, experience: str | dict[str, float], query_text: str | None = None) -> pd.Series:
    if out.empty:
        return pd.Series(dtype=float)
    query = query_text or ("" if isinstance(experience, dict) else str(experience))
    lowered = query.lower()
    desired_terms = set(_tokenize(query))
    if not isinstance(experience, dict):
        for label, terms in METADATA_INTENT_TERMS.items():
            label_parts = set(label.replace("/", " ").split())
            if label in lowered or desired_terms.intersection(label_parts):
                desired_terms.update(_tokenize(" ".join(terms)))
    scores = []
    for _, row in out.iterrows():
        metadata = " ".join(str(row.get(col, "")) for col in ["name", "categories", "cuisine"])
        place_terms = _tokenize(metadata)
        if not desired_terms:
            scores.append(0.0)
            continue
        overlap = len(desired_terms.intersection(place_terms))
        partial = sum(1 for term in desired_terms if term and term in metadata.lower())
        scores.append(min(1.0, (overlap + 0.5 * partial) / max(3, len(desired_terms))))
    return pd.Series(scores, index=out.index)


def _normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    low, high = values.min(), values.max()
    if high == low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def add_distance_scores(df: pd.DataFrame, config: RecommenderConfig) -> pd.DataFrame:
    out = df.copy()
    out["distance_km"] = out.apply(
        lambda r: haversine_km(config.start_lat, config.start_lon, r.get("latitude"), r.get("longitude")), axis=1
    )
    out = out[out["distance_km"] <= config.max_distance_km].copy()
    out["distance_score"] = (1 - out["distance_km"] / config.max_distance_km).clip(lower=0)
    return out


def distance_baseline(df: pd.DataFrame, config: RecommenderConfig, top_k: int = 10) -> pd.DataFrame:
    return add_distance_scores(df, config).sort_values("distance_km").head(top_k)


def rating_popularity_baseline(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["rating_score"] = 0.6 * _normalize(out.get("stars", 0)) + 0.4 * _normalize(np.log1p(pd.to_numeric(out.get("review_count", 0), errors="coerce").fillna(0)))
    return out.sort_values("rating_score", ascending=False).head(top_k)


def aspect_based_recommender(df: pd.DataFrame, experience: str | dict[str, float], top_k: int = 10) -> pd.DataFrame:
    out = df.copy()
    vector = preference_vector(experience)
    weights = np.array([vector[a] for a in ASPECTS])
    if weights.sum() == 0:
        weights = np.ones(len(ASPECTS))
    matrix = out[[f"{a}_score" for a in ASPECTS if f"{a}_score" in out.columns]].fillna(0).to_numpy()
    if matrix.shape[1] != len(ASPECTS):
        matrix = np.zeros((len(out), len(ASPECTS)))
    out["aspect_match_score"] = matrix.dot(weights) / max(weights.sum(), 1e-9)
    return out.sort_values("aspect_match_score", ascending=False).head(top_k)


def _semantic_scores(out: pd.DataFrame, query: str | None) -> pd.Series:
    if not query:
        return pd.Series(np.zeros(len(out)), index=out.index)
    emb_path = MODELS_DIR / "place_embeddings.npy"
    idx_path = MODELS_DIR / "place_embedding_index.csv"
    if not emb_path.exists() or not idx_path.exists():
        return pd.Series(np.zeros(len(out)), index=out.index)
    try:
        from embeddings import embed_query

        embeddings = np.load(emb_path)
        index = pd.read_csv(idx_path)
        query_vec = embed_query(query).reshape(1, -1)
        denom = (np.linalg.norm(query_vec, axis=1, keepdims=True) * np.linalg.norm(embeddings, axis=1, keepdims=True).T) + 1e-9
        sims = (query_vec @ embeddings.T / denom).ravel()
        sim_map = dict(zip(index["place_id"], sims))
        return out["place_id"].map(sim_map).fillna(0)
    except Exception:
        return pd.Series(np.zeros(len(out)), index=out.index)


def hybrid_recommender(
    df: pd.DataFrame,
    experience: str | dict[str, float],
    config: RecommenderConfig,
    query_text: str | None = None,
    cuisine_filter: str | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    out = add_distance_scores(df, config)
    if cuisine_filter:
        text = cuisine_filter.lower()
        out = out[out.get("cuisine", "").fillna("").str.lower().str.contains(text) | out.get("categories", "").fillna("").str.lower().str.contains(text)]
    if out.empty:
        return out

    out = aspect_based_recommender(out, experience, top_k=len(out))
    out["semantic_similarity_score"] = _semantic_scores(out, query_text or str(experience))
    out["metadata_match_score"] = metadata_match_scores(out, experience, query_text)
    out["rating_score"] = 0.6 * _normalize(out.get("stars", 0)) + 0.4 * _normalize(np.log1p(pd.to_numeric(out.get("review_count", 0), errors="coerce").fillna(0)))
    out["context_score"] = out[["aspect_match_score", "metadata_match_score"]].max(axis=1)
    out["hidden_gem_score"] = out.get("hidden_gem_adjusted_score", out.get("hidden_gem_score", 0)).fillna(0)
    out["confidence_score"] = out.get("confidence_score", 0).fillna(0)

    weights = config.weights or DEFAULT_WEIGHTS
    weight_sum = sum(weights.values()) or 1
    out["final_score"] = 0
    for col, weight in weights.items():
        if col in out.columns:
            out["final_score"] += out[col].fillna(0) * (weight / weight_sum)
    breakdown_cols = [c for c in [*DEFAULT_WEIGHTS, "metadata_match_score"] if c in out.columns]
    out["score_breakdown_json"] = out[breakdown_cols].apply(lambda r: json.dumps(r.to_dict()), axis=1)
    return out.sort_values("final_score", ascending=False).head(top_k)
