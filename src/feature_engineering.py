from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from config import ASPECTS, MODELS_DIR, PROCESSED_DIR, ensure_dirs


def _norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
    if len(values) == 0 or np.nanmax(values) == np.nanmin(values):
        return pd.Series(np.zeros(len(values)), index=series.index)
    flat = values.ravel()
    return pd.Series((flat - flat.min()) / (flat.max() - flat.min()), index=series.index)


def build_recommender_features() -> pd.DataFrame:
    ensure_dirs()
    master_path = PROCESSED_DIR / "place_master.csv"
    if not master_path.exists():
        print("Missing place_master.csv. Run entity_resolution.py first.")
        out = pd.DataFrame()
        out.to_csv(PROCESSED_DIR / "recommender_features.csv", index=False)
        return out

    try:
        master = pd.read_csv(master_path)
    except EmptyDataError:
        master = pd.DataFrame(columns=["place_id"])
    profile_path = PROCESSED_DIR / "place_aspect_profile.csv"
    try:
        profile = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame(columns=["place_id"])
    except EmptyDataError:
        profile = pd.DataFrame(columns=["place_id"])
    if "place_id" not in profile.columns:
        profile["place_id"] = pd.Series(dtype=str)
    df = master.merge(profile, on="place_id", how="left")
    fusion_path = PROCESSED_DIR / "yelp_fusion_enrichment.csv"
    if fusion_path.exists():
        try:
            fusion = pd.read_csv(fusion_path)
        except EmptyDataError:
            fusion = pd.DataFrame(columns=["place_id"])
        if not fusion.empty and "place_id" in fusion.columns:
            fusion = fusion.rename(columns={"rating": "fusion_rating", "review_count": "fusion_review_count"})
            df = df.merge(fusion[["place_id", "fusion_rating", "fusion_review_count"]], on="place_id", how="left")
    if "name" in df:
        df = df.dropna(subset=["name"])
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        out = pd.DataFrame(columns=[
            "place_id", "name", "latitude", "longitude", "categories", "cuisine", "stars", "review_count", "is_open",
            *[f"{a}_score" for a in ASPECTS], "confidence_score", "popularity_score",
            "hidden_gem_adjusted_score", "place_embedding_available", "evidence_snippets_json",
        ])
        out.to_csv(PROCESSED_DIR / "recommender_features.csv", index=False)
        print("Saved recommender features for 0 places.")
        return out
    for aspect in ASPECTS:
        col = f"{aspect}_score"
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["confidence_score"] = pd.to_numeric(df.get("confidence_score", 0), errors="coerce").fillna(0)
    if "stars" not in df:
        df["stars"] = np.nan
    if "review_count" not in df:
        df["review_count"] = 0
    if "fusion_rating" in df:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(pd.to_numeric(df["fusion_rating"], errors="coerce"))
    if "fusion_review_count" in df:
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(pd.to_numeric(df["fusion_review_count"], errors="coerce"))
    if "is_open" not in df:
        df["is_open"] = np.nan
    if "evidence_snippets_json" not in df:
        df["evidence_snippets_json"] = "{}"
    df["popularity_score"] = 0.6 * _norm(df.get("stars", 0)) + 0.4 * _norm(np.log1p(pd.to_numeric(df.get("review_count", 0), errors="coerce").fillna(0)))
    df["hidden_gem_adjusted_score"] = df["hidden_gem_score"] * df["confidence_score"] * (1 - (df["popularity_score"] * 0.5))
    index_path = MODELS_DIR / "place_embedding_index.csv"
    if index_path.exists():
        embedding_ids = set(pd.read_csv(index_path)["place_id"].dropna())
        df["place_embedding_available"] = df["place_id"].isin(embedding_ids)
    else:
        df["place_embedding_available"] = False
    cols = [
        "place_id", "name", "latitude", "longitude", "categories", "cuisine", "stars", "review_count", "is_open",
        *[f"{a}_score" for a in ASPECTS], "confidence_score", "popularity_score",
        "hidden_gem_adjusted_score", "place_embedding_available", "evidence_snippets_json",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    out.to_csv(PROCESSED_DIR / "recommender_features.csv", index=False)
    print(f"Saved recommender features for {len(out):,} places.")
    return out


if __name__ == "__main__":
    build_recommender_features()
