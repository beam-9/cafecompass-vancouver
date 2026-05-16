from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import ASPECTS, PROCESSED_DIR, UBC_CENTER, VANCOUVER_CENTER
from explainability import explain_recommendation
from recommenders import DEFAULT_WEIGHTS, RecommenderConfig, hybrid_recommender

st.set_page_config(page_title="CafeCompass Vancouver", layout="wide")

LOCATIONS = {
    "UBC": UBC_CENTER,
    "Downtown Vancouver": VANCOUVER_CENTER,
    "Kitsilano": (49.2684, -123.1683),
    "Mount Pleasant": (49.2633, -123.0966),
    "Richmond": (49.1666, -123.1336),
}

EXPERIENCES = [
    "quiet study cafe",
    "laptop-friendly cafe",
    "cheap eats",
    "date-night restaurant",
    "hidden gem",
    "authentic food",
    "group-friendly restaurant",
    "quick lunch",
    "dessert/matcha/cafe hopping",
    "late-night food",
]


def page_frame() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f4ee; color: #20201d; }
        section[data-testid="stSidebar"] { background: #24231f; }
        section[data-testid="stSidebar"] * { color: #f7f4ee; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] { background: #fffaf0; border: 1px solid #d7c8aa; padding: 14px; border-radius: 8px; }
        .cc-note { border-left: 4px solid #0b6b5a; padding: 10px 14px; background: #fffaf0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "recommender_features.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_evaluation() -> pd.DataFrame:
    path = PROCESSED_DIR / "evaluation_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def setup_message() -> None:
    st.info(
        "Processed recommender data is not available yet. Run the data pipeline scripts from the README, "
        "or start with OSM/Vancouver Open Data collection before generating features."
    )


def overview_page() -> None:
    st.title("CafeCompass Vancouver")
    st.subheader("Review-aware cafe and food spot recommendations")
    st.markdown(
        """
        <div class="cc-note">
        I built this project to answer experience-based questions that ratings alone do not capture:
        quiet study spots, laptop-friendly cafes, hidden gems, cheap eats, date-night atmosphere,
        and places where the actual review text supports the recommendation.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "The project avoids scraping Google Maps, Yelp pages, TripAdvisor, or similar review platforms. "
        "It uses permitted/public sources such as the Yelp Open Dataset, Reddit API, OpenStreetMap, "
        "City of Vancouver Open Data, and optional Foursquare OS Places."
    )
    st.write(
        "The app is a portfolio recommender prototype, not a production replacement for live local search. "
        "Its recommendations depend on available text, metadata freshness, and entity matching quality."
    )


def map_page(df: pd.DataFrame) -> None:
    st.title("Explore Vancouver Food Map")
    if df.empty:
        setup_message()
        return
    left, right = st.columns([1, 3])
    with left:
        category = st.text_input("Category or cuisine filter")
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    view = df.copy()
    if category:
        mask = view.get("categories", "").fillna("").str.contains(category, case=False) | view.get("cuisine", "").fillna("").str.contains(category, case=False)
        view = view[mask]
    if "confidence_score" in view:
        view = view[view["confidence_score"].fillna(0) >= min_conf]
    with right:
        fig = px.scatter_mapbox(
            view,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            color="categories" if "categories" in view else None,
            zoom=11,
            height=640,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)


def recommender_page(df: pd.DataFrame) -> None:
    st.title("Cafe/Food Spot Recommender")
    if df.empty:
        setup_message()
        return

    controls, results_col = st.columns([1, 2])
    with controls:
        location_mode = st.selectbox("Starting location", [*LOCATIONS.keys(), "Custom latitude/longitude"])
        if location_mode == "Custom latitude/longitude":
            lat = st.number_input("Latitude", value=VANCOUVER_CENTER[0], format="%.6f")
            lon = st.number_input("Longitude", value=VANCOUVER_CENTER[1], format="%.6f")
        else:
            lat, lon = LOCATIONS[location_mode]
        experience = st.selectbox("Desired experience", EXPERIENCES)
        query = st.text_input("Preference text", value=f"{experience} near {location_mode}")
        cuisine = st.text_input("Cuisine filter")
        max_distance = st.slider("Max distance (km)", 1.0, 25.0, 8.0, 0.5)
        st.caption("Hybrid ranking weights")
        weights = {}
        for key, default in DEFAULT_WEIGHTS.items():
            weights[key] = st.slider(key.replace("_", " "), 0.0, 1.0, float(default), 0.05)

    config = RecommenderConfig(start_lat=lat, start_lon=lon, max_distance_km=max_distance, weights=weights)
    results = hybrid_recommender(df, experience, config, query_text=query, cuisine_filter=cuisine or None, top_k=10)

    with results_col:
        if results.empty:
            st.warning("No matching places found for these filters.")
            return
        st.metric("Recommendations", len(results))
        for _, row in results.iterrows():
            with st.container(border=True):
                st.subheader(row.get("name", "Unnamed place"))
                st.write(f"Match score: {row.get('final_score', 0):.3f} | Distance: {row.get('distance_km', 0):.1f} km")
                explanation = explain_recommendation(row)
                st.write("Recommended because:")
                for reason in explanation["reasons"]:
                    st.write(f"- {reason}")
                st.write("Evidence:")
                for snippet in explanation["evidence"]:
                    st.caption(snippet)
                with st.expander("Score breakdown"):
                    st.json(explanation["score_breakdown"])


def similar_places_page(df: pd.DataFrame) -> None:
    st.title("Similar Places")
    if df.empty:
        setup_message()
        return
    place = st.selectbox("Select a place", df["name"].fillna("Unnamed").tolist())
    selected = df[df["name"].fillna("Unnamed") == place].iloc[0]
    score_cols = [f"{a}_score" for a in ASPECTS if f"{a}_score" in df.columns]
    if not score_cols:
        st.info("Aspect profiles are not available yet.")
        return
    matrix = df[score_cols].fillna(0)
    target = selected[score_cols].fillna(0).astype(float)
    sims = matrix.astype(float).dot(target) / ((matrix.pow(2).sum(axis=1).pow(0.5) * (target.pow(2).sum() ** 0.5)) + 1e-9)
    out = df.assign(similarity=sims).sort_values("similarity", ascending=False).head(11)
    st.dataframe(out[out["name"] != place][["name", "categories", "cuisine", "similarity"]].head(10), use_container_width=True)


def intelligence_page(df: pd.DataFrame) -> None:
    st.title("Review Intelligence")
    if df.empty:
        setup_message()
        return
    cols = st.columns(4)
    highlights = [
        ("Quiet study hotspots", "quiet_study_score"),
        ("Cheap eats hotspots", "cheap_value_score"),
        ("Hidden gem candidates", "hidden_gem_adjusted_score"),
        ("Date-night clusters", "date_night_score"),
    ]
    for col, (title, score_col) in zip(cols, highlights):
        with col:
            st.subheader(title)
            if score_col in df:
                st.dataframe(df.sort_values(score_col, ascending=False)[["name", score_col]].head(8), hide_index=True)
            else:
                st.caption("Not available yet.")


def evaluation_page() -> None:
    st.title("Model Evaluation")
    results = load_evaluation()
    if results.empty:
        st.info("Evaluation results are not available yet. Run `python src/evaluation.py` after building features.")
        return
    st.dataframe(results, use_container_width=True)
    fig = px.bar(results, x="query", y="precision_at_10_aspect", color="model", barmode="group", height=520)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    page_frame()
    df = load_features()
    page = st.sidebar.radio(
        "CafeCompass",
        [
            "Project Overview",
            "Explore Vancouver Food Map",
            "Cafe/Food Spot Recommender",
            "Similar Places",
            "Review Intelligence",
            "Model Evaluation",
        ],
    )
    if page == "Project Overview":
        overview_page()
    elif page == "Explore Vancouver Food Map":
        map_page(df)
    elif page == "Cafe/Food Spot Recommender":
        recommender_page(df)
    elif page == "Similar Places":
        similar_places_page(df)
    elif page == "Review Intelligence":
        intelligence_page(df)
    else:
        evaluation_page()


if __name__ == "__main__":
    main()
