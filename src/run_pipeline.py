from __future__ import annotations

import argparse

from aspect_extraction import build_review_aspect_scores
from clean_reviews import clean_yelp_reviews, clean_yelp_tips
from entity_resolution import build_place_master
from evaluation import run_evaluation
from feature_engineering import build_recommender_features
from filter_vancouver import filter_vancouver_businesses
from load_yelp_data import inspect_businesses
from make_demo_data import build_demo_data
from osm_collector import collect_osm_food_places
from sentiment_scoring import build_place_aspect_profile
from vancouver_open_data import collect_business_licences, collect_food_vendors


def run_yelp_pipeline() -> None:
    inspect_businesses()
    filter_vancouver_businesses()
    clean_yelp_reviews()
    clean_yelp_tips()
    build_place_master()
    build_review_aspect_scores()
    build_place_aspect_profile()
    build_recommender_features()
    run_evaluation()


def run_local_metadata_pipeline() -> None:
    collect_osm_food_places()
    collect_business_licences()
    collect_food_vendors()
    build_place_master()
    build_recommender_features()
    run_evaluation()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CafeCompass Vancouver pipeline stages.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build processed outputs from the committed synthetic demo dataset.",
    )
    parser.add_argument(
        "--yelp",
        action="store_true",
        help="Run the Yelp Open Dataset path. Requires data/raw/yelp/*.json files.",
    )
    parser.add_argument(
        "--local-metadata",
        action="store_true",
        help="Collect actual Vancouver OSM and City Open Data metadata for the map. No review text required.",
    )
    args = parser.parse_args()

    if args.demo:
        build_demo_data()
        return
    if args.yelp:
        run_yelp_pipeline()
        return
    if args.local_metadata:
        run_local_metadata_pipeline()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
