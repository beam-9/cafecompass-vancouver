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
        .stApp p, .stApp li, .stApp label, .stApp span, .stApp div { color: #20201d; }
        section[data-testid="stSidebar"] { background: #24231f; }
        section[data-testid="stSidebar"] * { color: #f7f4ee; }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div { color: #f7f4ee; }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea { color: #20201d; }
        [data-baseweb="tag"],
        [data-baseweb="tag"] *,
        .dark-note,
        .dark-note * { color: #f7f4ee !important; }
        div[data-testid="stAlert"] * { color: #20201d; }
        [data-baseweb="popover"] * { color: #20201d; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] { background: #fffaf0; border: 1px solid #d7c8aa; padding: 14px; border-radius: 8px; }
        .cc-note { border-left: 4px solid #0b6b5a; padding: 10px 14px; background: #fffaf0; }
        .dark-note { background: #24231f; padding: 12px 14px; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_features() -> tuple[pd.DataFrame, str]:
    path = PROCESSED_DIR / "recommender_features.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df, "processed"
    sample_path = ROOT / "data" / "sample" / "demo_recommender_features.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path), "sample"
    return pd.DataFrame(), "missing"


@st.cache_data
def load_evaluation() -> pd.DataFrame:
    path = PROCESSED_DIR / "evaluation_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    sample_path = ROOT / "data" / "sample" / "demo_evaluation_results.csv"
    return pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()


def setup_message() -> None:
    st.info(
        "Processed recommender data is not available yet. Run the data pipeline scripts from the README, "
        "or start with OSM/Vancouver Open Data collection before generating features."
    )


def source_banner(source: str) -> None:
    if source == "sample":
        st.warning(
            "Currently showing the synthetic demo dataset. The actual Vancouver dataset will be added by running "
            "the OSM, Vancouver Open Data, Yelp Open Dataset, and optional Reddit collection pipeline."
        )


def overview_page(source: str) -> None:
    st.title("CafeCompass Vancouver")
    st.subheader("Review-aware cafe and food spot recommendations")
    st.markdown(
        """
        <div class="cc-note">
        As an international student at UBC, I spent most of my first year studying in my dorm
        and did not explore as much of Vancouver as I wanted to. I built CafeCompass Vancouver
        to fix that problem in a data-driven way: instead of only asking what is highly rated nearby,
        I wanted to find places that fit the kind of experience I actually need, like a quiet cafe
        for studying, a cheap meal after class, a hidden gem, or somewhere worth taking friends.
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
        "Technically, this project combines NLP, geospatial filtering, fuzzy entity matching, sentiment scoring, "
        "sentence embeddings, and hybrid recommendation ranking. The pipeline turns messy review or community text "
        "into structured aspect signals such as quiet-study, cheap-value, date-night, hidden-gem, authentic-food, "
        "group-friendly, dessert/drinks, and late-night intent."
    )
    st.write(
        "The recommender compares several approaches: a distance-only baseline, a rating/popularity baseline, "
        "an aspect-profile recommender, and a hybrid model that combines aspect match, semantic similarity, "
        "distance, rating, confidence, context match, and hidden-gem adjustment. I also keep evidence snippets "
        "from the original text so the app can explain why each place was recommended instead of returning a black-box list."
    )
    source_banner(source)


def map_page(df: pd.DataFrame, source: str) -> None:
    st.title("Explore Vancouver Food Map")
    source_banner(source)
    if df.empty:
        setup_message()
        return

    st.write(
        "Use this page to inspect place coverage before trusting the recommender. "
        "Filter by cuisine, category, name, rating, and the experience signal extracted from text."
    )

    left, right = st.columns([1, 3])
    with left:
        search = st.text_input("Search name, cuisine, or category", placeholder="Japanese, Chinese, ramen, cafe")
        cuisine_options = sorted([c for c in df.get("cuisine", pd.Series(dtype=str)).dropna().astype(str).unique() if c])
        cuisine_filter = st.multiselect("Cuisine", cuisine_options)
        category_text = df.get("categories", pd.Series(dtype=str)).fillna("").astype(str)
        category_options = sorted({item.strip() for value in category_text for item in value.split(";") if item.strip()})
        category_filter = st.multiselect("Category", category_options)
        min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
        aspect_label = st.selectbox(
            "Color map by experience signal",
            ["None", "Quiet study", "Cheap value", "Date night", "Hidden gem", "Authentic", "Group friendly", "Dessert/drinks", "Late night"],
        )
        st.caption(
            "Text evidence is not a rating. It estimates how much review/community text supports an experience signal."
        )
    view = df.copy()
    if search:
        mask = pd.Series(False, index=view.index)
        for col in ["name", "categories", "cuisine"]:
            if col in view:
                mask = mask | view[col].fillna("").astype(str).str.contains(search, case=False, regex=False)
        view = view[mask]
    if cuisine_filter and "cuisine" in view:
        view = view[view["cuisine"].isin(cuisine_filter)]
    if category_filter and "categories" in view:
        category_pattern = "|".join(category_filter)
        view = view[view["categories"].fillna("").str.contains(category_pattern, case=False, regex=True)]
    if "stars" in view:
        view = view[pd.to_numeric(view["stars"], errors="coerce").fillna(0) >= min_rating]

    aspect_map = {
        "Quiet study": "quiet_study_score",
        "Cheap value": "cheap_value_score",
        "Date night": "date_night_score",
        "Hidden gem": "hidden_gem_adjusted_score",
        "Authentic": "authentic_score",
        "Group friendly": "group_friendly_score",
        "Dessert/drinks": "dessert_drinks_score",
        "Late night": "late_night_score",
    }
    color_col = aspect_map.get(aspect_label)

    with right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Places shown", len(view))
        metric_cols[1].metric("Cuisines", view["cuisine"].nunique() if "cuisine" in view else 0)
        metric_cols[2].metric("Avg rating", f"{view['stars'].mean():.2f}" if "stars" in view and not view.empty else "N/A")
        evidence = view.get("confidence_score", pd.Series(0, index=view.index)).fillna(0)
        metric_cols[3].metric("With text evidence", int((evidence > 0).sum()))
        if view.empty:
            st.warning("No places match these filters. Clear the search or lower the rating filter.")
            return
        fig = px.scatter_mapbox(
            view,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data=[c for c in ["cuisine", "categories", "stars", "review_count"] if c in view.columns],
            color=color_col if color_col in view.columns else "cuisine",
            zoom=11,
            height=640,
            color_continuous_scale="Tealgrn" if color_col in view.columns else None,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        table_cols = [c for c in ["name", "cuisine", "categories", "stars", "review_count", "confidence_score"] if c in view.columns]
        with st.expander("Show place data used for this map"):
            st.dataframe(view[table_cols].sort_values(["cuisine", "name"]), use_container_width=True, hide_index=True)


def recommender_page(df: pd.DataFrame, source: str) -> None:
    st.title("Cafe/Food Spot Recommender")
    source_banner(source)
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


def similar_places_page(df: pd.DataFrame, source: str) -> None:
    st.title("Similar Places")
    source_banner(source)
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


def intelligence_page(df: pd.DataFrame, source: str) -> None:
    st.title("Review Intelligence")
    source_banner(source)
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
    df, source = load_features()
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
        overview_page(source)
    elif page == "Explore Vancouver Food Map":
        map_page(df, source)
    elif page == "Cafe/Food Spot Recommender":
        recommender_page(df, source)
    elif page == "Similar Places":
        similar_places_page(df, source)
    elif page == "Review Intelligence":
        intelligence_page(df, source)
    else:
        evaluation_page()


if __name__ == "__main__":
    main()
