from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

YELP_RAW_DIR = RAW_DIR / "yelp"
VANCOUVER_CENTER = (49.2827, -123.1207)
UBC_CENTER = (49.2606, -123.2460)

FOOD_CATEGORIES = {
    "restaurants",
    "food",
    "cafes",
    "coffee & tea",
    "bakeries",
    "desserts",
    "bubble tea",
    "japanese",
    "korean",
    "chinese",
    "thai",
    "vietnamese",
    "breakfast & brunch",
    "vegetarian",
    "vegan",
    "fast food",
}

ASPECTS = [
    "quiet_study",
    "date_night",
    "cheap_value",
    "hidden_gem",
    "authentic",
    "service_speed",
    "group_friendly",
    "food_quality",
    "dessert_drinks",
    "late_night",
    "comfort_cozy",
    "crowding_noise",
    "study_work_suitability",
    "value_satisfaction",
    "service_warmth",
    "special_excitement",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
