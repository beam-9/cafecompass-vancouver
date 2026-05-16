from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ASPECTS, UBC_CENTER
from recommenders import RecommenderConfig, distance_baseline, hybrid_recommender, metadata_match_scores, preference_vector


def demo_frame() -> pd.DataFrame:
    rows = [
        {
            "place_id": "a",
            "name": "Study Cafe",
            "latitude": UBC_CENTER[0],
            "longitude": UBC_CENTER[1],
            "categories": "Cafes",
            "cuisine": "Coffee",
            "stars": 4.2,
            "review_count": 50,
            "confidence_score": 0.8,
            "hidden_gem_adjusted_score": 0.2,
            "evidence_snippets_json": "{}",
        },
        {
            "place_id": "b",
            "name": "Far Dinner",
            "latitude": 49.2827,
            "longitude": -123.1207,
            "categories": "Restaurants",
            "cuisine": "Italian",
            "stars": 4.9,
            "review_count": 500,
            "confidence_score": 0.7,
            "hidden_gem_adjusted_score": 0.1,
            "evidence_snippets_json": "{}",
        },
    ]
    df = pd.DataFrame(rows)
    for aspect in ASPECTS:
        df[f"{aspect}_score"] = 0.0
    df.loc[0, "quiet_study_score"] = 0.95
    df.loc[1, "date_night_score"] = 0.95
    return df


class RecommenderTests(unittest.TestCase):
    def test_preference_vector_detects_quiet_study(self) -> None:
        vector = preference_vector("quiet laptop friendly cafe near UBC with outlets")
        self.assertGreater(vector["quiet_study"], 0)

    def test_distance_baseline_ranks_nearest_first(self) -> None:
        df = demo_frame()
        config = RecommenderConfig(start_lat=UBC_CENTER[0], start_lon=UBC_CENTER[1], max_distance_km=50)
        result = distance_baseline(df, config, top_k=2)
        self.assertEqual(result.iloc[0]["name"], "Study Cafe")

    def test_hybrid_uses_aspect_match(self) -> None:
        df = demo_frame()
        config = RecommenderConfig(start_lat=UBC_CENTER[0], start_lon=UBC_CENTER[1], max_distance_km=50)
        result = hybrid_recommender(df, "quiet study cafe", config, top_k=2)
        self.assertEqual(result.iloc[0]["name"], "Study Cafe")

    def test_metadata_match_uses_cuisine_and_categories(self) -> None:
        df = demo_frame()
        scores = metadata_match_scores(df, "date-night restaurant", "Italian date night restaurant")
        self.assertGreater(scores.iloc[1], scores.iloc[0])


if __name__ == "__main__":
    unittest.main()
