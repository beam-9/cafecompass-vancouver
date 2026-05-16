from __future__ import annotations

import pandas as pd

from config import PROCESSED_DIR, UBC_CENTER, VANCOUVER_CENTER, ensure_dirs
from recommenders import RecommenderConfig, distance_baseline, hybrid_recommender, rating_popularity_baseline

SYNTHETIC_QUERIES = [
    ("quiet study cafe near UBC", UBC_CENTER),
    ("cheap eats near Downtown", VANCOUVER_CENTER),
    ("date night restaurant in Kitsilano", (49.2684, -123.1683)),
    ("hidden gem ramen Vancouver", VANCOUVER_CENTER),
    ("group dinner Mount Pleasant", (49.2633, -123.0966)),
    ("dessert spot Vancouver", VANCOUVER_CENTER),
    ("late night food Vancouver", VANCOUVER_CENTER),
]


def precision_at_k(results: pd.DataFrame, aspect: str, k: int = 10) -> float:
    if results.empty or f"{aspect}_score" not in results:
        return 0.0
    top = results.head(k)
    return float((top[f"{aspect}_score"].fillna(0) > 0).mean())


def diversity_at_k(results: pd.DataFrame, column: str, k: int = 10) -> int:
    if results.empty or column not in results:
        return 0
    return int(results.head(k)[column].fillna("unknown").nunique())


def run_evaluation() -> pd.DataFrame:
    ensure_dirs()
    feature_path = PROCESSED_DIR / "recommender_features.csv"
    if not feature_path.exists():
        print("Missing recommender_features.csv. Run feature_engineering.py first.")
        out = pd.DataFrame()
        out.to_csv(PROCESSED_DIR / "evaluation_results.csv", index=False)
        return out

    df = pd.read_csv(feature_path)
    rows = []
    for query, start in SYNTHETIC_QUERIES:
        config = RecommenderConfig(start_lat=start[0], start_lon=start[1], max_distance_km=10)
        hybrid = hybrid_recommender(df, query, config, query_text=query, top_k=10)
        distance = distance_baseline(df, config, top_k=10)
        rating = rating_popularity_baseline(df, top_k=10)
        aspect = next((a for a in ["quiet_study", "cheap_value", "date_night", "hidden_gem", "group_friendly", "dessert_drinks", "late_night"] if a.replace("_", " ") in query), "food_quality")
        for model_name, result in [("hybrid", hybrid), ("distance", distance), ("rating_popularity", rating)]:
            rows.append(
                {
                    "query": query,
                    "model": model_name,
                    "precision_at_10_aspect": precision_at_k(result, aspect),
                    "category_diversity_at_10": diversity_at_k(result, "categories"),
                    "cuisine_diversity_at_10": diversity_at_k(result, "cuisine"),
                    "average_distance_km": result["distance_km"].mean() if "distance_km" in result and not result.empty else None,
                    "coverage": len(result),
                    "explanation_coverage": float(result.get("evidence_snippets_json", pd.Series(dtype=str)).fillna("").ne("").mean()) if not result.empty else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "evaluation_results.csv", index=False)
    print(f"Saved {len(out):,} evaluation rows.")
    return out


if __name__ == "__main__":
    run_evaluation()

