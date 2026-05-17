from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import PROCESSED_DIR, UBC_CENTER, VANCOUVER_CENTER
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
        [data-baseweb="select"] *,
        [role="listbox"],
        [role="listbox"] *,
        .dark-note,
        .dark-note * { color: #f7f4ee !important; }
        [data-baseweb="select"] > div,
        [role="listbox"],
        [data-baseweb="popover"] [role="option"] { background: #24231f !important; }
        div[data-testid="stAlert"] * { color: #20201d; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] { background: #fffaf0; border: 1px solid #d7c8aa; padding: 14px; border-radius: 8px; }
        .cc-note { border-left: 4px solid #0b6b5a; padding: 10px 14px; background: #fffaf0; }
        .dark-note { background: #24231f; padding: 12px 14px; border-radius: 8px; }
        .slider-panel {
            background: #fff7df;
            border: 1px solid #d6bc7d;
            border-radius: 8px;
            padding: 14px;
            margin-top: 10px;
        }
        .slider-panel p, .slider-panel span, .slider-panel label, .slider-panel div { color: #20201d !important; }
        .rec-card {
            background: #fffaf0;
            border: 1px solid #d9c79d;
            border-left: 5px solid #0b6b5a;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 12px;
            box-shadow: 0 1px 0 rgba(36, 35, 31, 0.08);
        }
        .rec-card * { color: #20201d !important; }
        .rec-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 3px;
        }
        .rec-meta {
            font-size: 0.88rem;
            color: #4f4a3f !important;
            margin-bottom: 8px;
        }
        .score-row {
            display: grid;
            grid-template-columns: 140px 1fr 48px;
            gap: 8px;
            align-items: center;
            margin: 6px 0;
            font-size: 0.86rem;
        }
        .score-bar {
            height: 8px;
            background: #eadfca;
            border-radius: 999px;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: #0b6b5a;
        }
        .pill {
            display: inline-block;
            border: 1px solid #cdbb93;
            background: #f5ead2;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 5px;
            margin-top: 4px;
        }
        .stButton > button {
            background: #ded8ca !important;
            color: #20201d !important;
            border: 1px solid #a89d88 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
        }
        .stButton > button * { color: #20201d !important; }
        .stButton > button:hover {
            background: #cfc6b5 !important;
            color: #20201d !important;
            border-color: #7e725e !important;
        }
        .stSlider [data-baseweb="slider"] div { color: #20201d !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_features(data_version: float) -> tuple[pd.DataFrame, str]:
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
def load_evaluation(data_version: float) -> pd.DataFrame:
    path = PROCESSED_DIR / "evaluation_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    sample_path = ROOT / "data" / "sample" / "demo_evaluation_results.csv"
    return pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()


def processed_data_version() -> float:
    paths = [PROCESSED_DIR / "recommender_features.csv", PROCESSED_DIR / "evaluation_results.csv"]
    existing = [path.stat().st_mtime for path in paths if path.exists()]
    return max(existing, default=0.0)


def setup_message() -> None:
    st.info(
        "Processed recommender data is not available yet. Run the data pipeline scripts from the README, "
        "or start with OSM/Vancouver Open Data collection before generating features."
    )


def source_banner(source: str) -> None:
    if source == "sample":
        st.warning(
            "Currently showing the synthetic demo dataset. Run the local data pipeline to build the Vancouver "
            "place metadata used by the map and recommender."
        )


def _fmt_missing(value: object, fallback: str = "Not available") -> str:
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return html.escape(str(value))


def _score_rows(breakdown: dict[str, float]) -> str:
    labels = {
        "metadata_match_score": "Matches what you asked for",
        "distance_score": "Close to your start",
        "rating_score": "Higher rated",
    }
    rows = []
    for key, label in labels.items():
        raw = float(breakdown.get(key, 0) or 0)
        if raw <= 0:
            continue
        pct = max(0, min(100, raw * 100))
        value = f"{pct:.0f}%"
        rows.append(
            f"""
            <div class="score-row">
              <div>{label}</div>
              <div class="score-bar"><div class="score-fill" style="width:{pct:.0f}%"></div></div>
              <div>{value}</div>
            </div>
            """
        )
    if not rows:
        return "<div style='font-size:0.88rem;'>This place matched through available map metadata.</div>"
    return "\n".join(rows)


def render_recommendation_card(row: pd.Series) -> None:
    explanation = explain_recommendation(row)
    breakdown = explanation.get("score_breakdown", {})
    cuisine = _fmt_missing(row.get("cuisine"))
    categories = _fmt_missing(row.get("categories"))
    distance = row.get("distance_km")
    distance_text = f"{float(distance):.1f} km away" if pd.notna(distance) else "Distance unavailable"
    rating = row.get("stars")
    rating_text = ""
    if pd.notna(rating) and float(rating) > 0:
        rating_text = f" · {'★' * round(float(rating))} {float(rating):.1f}/5"
    score = float(row.get("final_score", 0) or 0)
    match_text = f"{score * 100:.0f}% match" if score <= 1 else f"{score:.2f} match score"
    reason_items = "".join(f"<span class='pill'>{html.escape(reason)}</span>" for reason in explanation.get("reasons", [])[:2])
    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-title">{_fmt_missing(row.get("name"), "Unnamed place")}</div>
          <div class="rec-meta">{distance_text}{rating_text} · {cuisine} · {categories}</div>
          <div><span class="pill">{match_text}</span>{reason_items}</div>
          <details style="margin-top:10px;">
            <summary style="font-weight:700; cursor:pointer;">Why this ranked here</summary>
            <div style="margin-top:8px;">{_score_rows(breakdown)}</div>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metadata_tokens(row: pd.Series) -> set[str]:
    text = " ".join(str(row.get(col, "")) for col in ["name", "categories", "cuisine"])
    return {token for token in re.split(r"[^a-z0-9_]+", text.lower()) if len(token) > 1}


def _normalized_place_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value or "").lower())
    text = re.sub(r"\b(cafe|restaurant|restaurants|coffee|house|inc|ltd|limited)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _simple_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def similar_places(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    out = df.copy()
    target_tokens = _metadata_tokens(selected)
    selected_lat = selected.get("latitude")
    selected_lon = selected.get("longitude")
    scores = []
    basis = []
    for _, row in out.iterrows():
        tokens = _metadata_tokens(row)
        union = target_tokens.union(tokens)
        token_score = len(target_tokens.intersection(tokens)) / len(union) if union else 0.0
        same_cuisine = str(row.get("cuisine", "")).lower() == str(selected.get("cuisine", "")).lower()
        same_category = str(row.get("categories", "")).lower() == str(selected.get("categories", "")).lower()
        distance = _simple_distance_km(selected_lat, selected_lon, row.get("latitude"), row.get("longitude"))
        distance_score = max(0.0, 1 - distance / 10)
        score = 0.55 * token_score + 0.20 * float(same_cuisine) + 0.15 * float(same_category) + 0.10 * distance_score
        scores.append(score)
        reasons = []
        if same_cuisine and pd.notna(row.get("cuisine")):
            reasons.append("same cuisine")
        if same_category and pd.notna(row.get("categories")):
            reasons.append("same place type")
        if token_score > 0:
            reasons.append("similar name/category terms")
        if distance_score > 0:
            reasons.append("nearby")
        basis.append(", ".join(reasons) or "metadata match")
    out["similarity"] = scores
    out["similarity_basis"] = basis
    selected_name = _normalized_place_name(selected.get("name"))
    if "place_id" in out.columns:
        out = out[out["place_id"] != selected.get("place_id")]
    else:
        out = out[out["name"] != selected.get("name")]
    if selected_name:
        out = out[out["name"].map(_normalized_place_name) != selected_name]
    out["normalized_result_name"] = out["name"].map(_normalized_place_name)
    out = out.sort_values(["similarity", "name"], ascending=[False, True])
    out = out.drop_duplicates(subset=["normalized_result_name"], keep="first")
    return out.drop(columns=["normalized_result_name"], errors="ignore")


def overview_page(source: str) -> None:
    st.title("CafeCompass Vancouver")
    st.subheader("A metadata-aware cafe and food spot recommender")
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
        "The app currently focuses on a practical first version: it uses Vancouver place metadata, cuisine tags, "
        "category tags, coordinates, and the preference text a user types in. That means it can answer questions "
        "like where to find Japanese food near UBC, cafes around Kitsilano, or quick food options near Downtown."
    )
    st.write(
        "The technical focus in this version is geospatial filtering, Haversine distance scoring, metadata cleaning, "
        "fuzzy place matching, cuisine/category normalization, and a simple hybrid ranking model. The recommender "
        "compares the user's preference text against place names, cuisine labels, and place types, then balances that "
        "preference fit against distance from the selected starting point."
    )
    st.write(
        "I still designed the repository so it can grow into richer text-based recommendation later, but the app itself "
        "only shows signals that are active and understandable right now. The goal is a coherent portfolio demo that is "
        "honest about its data while still feeling useful for exploring Vancouver."
    )
    source_banner(source)


def map_page(df: pd.DataFrame, source: str) -> None:
    st.title("Explore Vancouver Food Map")
    source_banner(source)
    if df.empty:
        setup_message()
        return

    st.write(
        "Use this page to inspect Vancouver place coverage before using the recommender. "
        "The map is strongest for names, locations, cuisines, and place types."
    )

    left, right = st.columns([1, 3])
    ratings_available = "stars" in df and pd.to_numeric(df["stars"], errors="coerce").notna().any()
    with left:
        search = st.text_input("Search name, cuisine, or category", placeholder="Japanese, Chinese, ramen, cafe")
        cuisine_options = sorted([c for c in df.get("cuisine", pd.Series(dtype=str)).dropna().astype(str).unique() if c])
        cuisine_filter = st.multiselect("Cuisine", cuisine_options)
        category_text = df.get("categories", pd.Series(dtype=str)).fillna("").astype(str)
        category_options = sorted({item.strip() for value in category_text for item in value.split(";") if item.strip()})
        category_filter = st.multiselect("Category", category_options)
        min_rating = 0.0
        if ratings_available:
            min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
        color_options = ["Cuisine", "Place type"]
        if ratings_available:
            color_options.append("Rating")
        color_label = st.selectbox("Color map by", color_options)
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
        "Cuisine": "cuisine",
        "Place type": "categories",
        "Rating": "stars",
    }
    color_col = aspect_map.get(color_label, "cuisine")

    with right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Places shown", len(view))
        metric_cols[1].metric("Cuisines", view["cuisine"].nunique() if "cuisine" in view else 0)
        named_places = view.get("name", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
        cuisine_tagged = view.get("cuisine", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
        metric_cols[2].metric("Named places", int(named_places))
        metric_cols[3].metric("Cuisine-tagged", int(cuisine_tagged))
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
            color_continuous_scale="Tealgrn" if color_col in view.columns and color_col not in {"cuisine", "categories"} else None,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
        table_cols = [c for c in ["name", "cuisine", "categories", "latitude", "longitude", "stars", "review_count"] if c in view.columns]
        display_view = view[table_cols].copy()
        if "stars" in display_view:
            display_view["stars"] = display_view["stars"].fillna("Not available")
        if "review_count" in display_view:
            display_view["review_count"] = display_view["review_count"].fillna("Not available")
        with st.expander("Show place data used for this map"):
            st.dataframe(display_view.sort_values(["cuisine", "name"]), use_container_width=True, hide_index=True)


def recommender_page(df: pd.DataFrame, source: str) -> None:
    st.title("Cafe/Food Spot Recommender")
    source_banner(source)
    if df.empty:
        setup_message()
        return

    controls, results_col = st.columns([1, 2])
    ratings_available = "stars" in df and pd.to_numeric(df["stars"], errors="coerce").gt(0).any()
    with controls:
        with st.form("recommendation_controls"):
            location_mode = st.selectbox("Starting location", [*LOCATIONS.keys(), "Custom latitude/longitude"])
            if location_mode == "Custom latitude/longitude":
                lat = st.number_input("Latitude", value=VANCOUVER_CENTER[0], format="%.6f")
                lon = st.number_input("Longitude", value=VANCOUVER_CENTER[1], format="%.6f")
            else:
                lat, lon = LOCATIONS[location_mode]
            experience = st.selectbox("What are you looking for?", EXPERIENCES)
            details = st.text_input("Optional details", placeholder="ramen, matcha, vegetarian, food court")
            cuisine = st.text_input("Cuisine filter", placeholder="Japanese, Chinese, coffee")
            max_distance = st.slider("How far are you willing to go?", 1.0, 25.0, 8.0, 0.5, format="%.1f km")
            min_rating = 0.0
            if ratings_available:
                min_rating = st.slider("Minimum rating", 1.0, 5.0, 1.0, 0.5, format="%.1f ★")
            st.markdown(
                "<div class='slider-panel'><strong>What should matter most?</strong><br>"
                "<span>These sliders change the order of the recommendations.</span></div>",
                unsafe_allow_html=True,
            )
            preference_weight = st.slider(
                "Match my craving",
                0.0,
                1.0,
                float(DEFAULT_WEIGHTS["metadata_match_score"]),
                0.05,
                help="Higher means places with matching cuisine, category, or name words rank first.",
            )
            distance_weight = st.slider(
                "Stay nearby",
                0.0,
                1.0,
                float(DEFAULT_WEIGHTS["distance_score"]),
                0.05,
                help="Higher means closer places rank first.",
            )
            rating_weight = 0.0
            if ratings_available:
                rating_weight = st.slider(
                    "Prefer higher-rated places",
                    0.0,
                    1.0,
                    0.25,
                    0.05,
                    help="Higher means star ratings matter more in the ranking.",
                )
            submitted = st.form_submit_button("Find recommendations")

    default_params = {
        "lat": LOCATIONS["UBC"][0],
        "lon": LOCATIONS["UBC"][1],
        "location_mode": "UBC",
        "experience": EXPERIENCES[0],
        "details": "",
        "cuisine": "",
        "max_distance": 8.0,
        "min_rating": 0.0,
        "weights": {
            "metadata_match_score": DEFAULT_WEIGHTS["metadata_match_score"],
            "distance_score": DEFAULT_WEIGHTS["distance_score"],
            "rating_score": 0.0,
        },
    }
    if submitted or "recommender_params" not in st.session_state:
        query_parts = [experience, details, f"near {location_mode}"]
        query = " ".join(part for part in query_parts if str(part).strip())
        st.session_state["recommender_params"] = {
            "lat": lat,
            "lon": lon,
            "location_mode": location_mode,
            "experience": experience,
            "query": query,
            "cuisine": cuisine,
            "max_distance": max_distance,
            "min_rating": min_rating,
            "weights": {
                "metadata_match_score": preference_weight,
                "distance_score": distance_weight,
                "rating_score": rating_weight,
            },
        }
    params = {**default_params, **st.session_state.get("recommender_params", {})}
    config = RecommenderConfig(
        start_lat=params["lat"],
        start_lon=params["lon"],
        max_distance_km=params["max_distance"],
        weights=params["weights"],
    )
    features = df.copy()
    if ratings_available and params["min_rating"] > 1.0 and "stars" in features:
        features = features[pd.to_numeric(features["stars"], errors="coerce").fillna(0) >= params["min_rating"]]
    results = hybrid_recommender(
        features,
        params["experience"],
        config,
        query_text=params.get("query"),
        cuisine_filter=params["cuisine"] or None,
        top_k=10,
    )

    with results_col:
        if results.empty:
            st.warning("No matching places found for these filters.")
            return
        st.metric("Recommendations", len(results))
        st.info(
            f"Showing places for: {params['experience']} near {params['location_mode']}. "
            "The ranking uses the controls you selected on the left."
        )
        visible_count = 4
        show_more_key = "show_all_recommendations"
        if show_more_key not in st.session_state:
            st.session_state[show_more_key] = False
        first_results = results.head(visible_count)
        for _, row in first_results.iterrows():
            render_recommendation_card(row)
        if len(results) > visible_count:
            remaining = len(results) - visible_count
            if st.session_state[show_more_key]:
                for _, row in results.iloc[visible_count:].iterrows():
                    render_recommendation_card(row)
                if st.button("Show fewer recommendations"):
                    st.session_state[show_more_key] = False
                    st.rerun()
            elif st.button(f"See {remaining} more recommendations"):
                st.session_state[show_more_key] = True
                st.rerun()


def similar_places_page(df: pd.DataFrame, source: str) -> None:
    st.title("Similar Places")
    source_banner(source)
    if df.empty:
        setup_message()
        return
    st.write(
        "Find places that are similar by cuisine, place type, name/category terms, and distance."
    )
    option_df = df.copy()
    option_df["display_name"] = option_df["name"].fillna("Unnamed").astype(str)
    option_df["normalized_display_name"] = option_df["display_name"].map(_normalized_place_name)
    option_df = option_df.sort_values(["display_name", "cuisine", "categories"], na_position="last")
    option_df = option_df.drop_duplicates(subset=["normalized_display_name"], keep="first")
    options = option_df["display_name"].tolist()
    place = st.selectbox("Select a place", options)
    selected = option_df[option_df["display_name"] == place].iloc[0]
    out = similar_places(df, selected).head(10)
    if out.empty:
        st.info("No similar places found for this selection.")
        return
    st.caption("Similarity is shown as a percentage based on the active data available for this project stage.")
    cards = st.columns(2)
    for i, (_, row) in enumerate(out.iterrows()):
        with cards[i % 2]:
            st.markdown(
                f"""
                <div class="rec-card">
                  <div class="rec-title">{_fmt_missing(row.get("name"), "Unnamed place")}</div>
                  <div class="rec-meta">{_fmt_missing(row.get("cuisine"))} · {_fmt_missing(row.get("categories"))}</div>
                  <span class="pill">{float(row.get("similarity", 0)) * 100:.0f}% similar</span>
                  <span class="pill">{_fmt_missing(row.get("similarity_basis"), "metadata match")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def data_coverage_page(df: pd.DataFrame, source: str) -> None:
    st.title("Data Coverage")
    source_banner(source)
    if df.empty:
        setup_message()
        return
    st.write(
        "This page shows the coverage behind the current app: Vancouver food-place names, coordinates, cuisine tags, "
        "place types, and source metadata."
    )

    rating_values = pd.to_numeric(df.get("stars", pd.Series(dtype=float)), errors="coerce")
    named_places = df.get("name", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
    cuisine_tagged = df.get("cuisine", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Places", f"{len(df):,}")
    metric_cols[1].metric("Cuisines", df["cuisine"].nunique() if "cuisine" in df else 0)
    metric_cols[2].metric("Named places", int(named_places))
    metric_cols[3].metric("Cuisine-tagged", int(cuisine_tagged))

    left, right = st.columns(2)
    with left:
        st.subheader("Top cuisines by place count")
        cuisine_counts = (
            df["cuisine"].fillna("Unknown").astype(str).str.replace("_", " ").value_counts().head(15).reset_index()
            if "cuisine" in df
            else pd.DataFrame(columns=["cuisine", "count"])
        )
        cuisine_counts.columns = ["Cuisine", "Places"]
        st.dataframe(cuisine_counts, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Top place types")
        category_counts = (
            df["categories"].fillna("Unknown").astype(str).str.replace("_", " ").value_counts().head(15).reset_index()
            if "categories" in df
            else pd.DataFrame(columns=["categories", "count"])
        )
        category_counts.columns = ["Place type", "Places"]
        st.dataframe(category_counts, use_container_width=True, hide_index=True)

    with st.expander("Show rating availability"):
        st.write(
            "Some sources include ratings while others only provide place metadata. The recommender does not rely on "
            "ratings in this streamlined version."
        )
        st.metric("Rows with ratings", int(rating_values.notna().sum()))


def evaluation_page() -> None:
    st.title("Ranking Logic")
    results = load_evaluation(processed_data_version())
    if results.empty:
        st.info("Evaluation results are not available yet. Run `python src/evaluation.py` after building features.")
        return

    st.write(
        "This page explains the active ranking model. The current app ranks places by combining preference metadata "
        "fit with distance from the selected starting point."
    )

    st.subheader("What each ranking method means")
    model_cards = [
        (
            "Distance baseline",
            "Ranks places only by how close they are to the selected starting point.",
        ),
        (
            "Preference match",
            "Compares the user's words with place names, cuisine tags, and categories.",
        ),
        (
            "Current hybrid",
            "Combines preference match and distance fit, with sliders controlling the balance.",
        ),
    ]
    model_cols = st.columns(3)
    for col, (title, body) in zip(model_cols, model_cards):
        with col:
            st.markdown(
                f"""
                <div class="rec-card" style="min-height:150px;">
                  <div class="rec-title">{title}</div>
                  <div class="rec-meta">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    numeric = results.copy()
    metric_cols = st.columns(4)
    metric_cols[0].metric("Synthetic queries", numeric["query"].nunique())
    metric_cols[1].metric("Models compared", numeric["model"].nunique())
    metric_cols[2].metric("Avg top-10 coverage", f"{numeric['coverage'].mean():.0f}" if "coverage" in numeric else "N/A")
    metric_cols[3].metric(
        "Avg distance",
        f"{numeric['average_distance_km'].mean():.1f} km" if "average_distance_km" in numeric else "N/A",
    )

    st.subheader("Current ranking sanity checks")
    summary = (
        numeric.groupby("model", as_index=False)
        .agg(
            avg_distance_km=("average_distance_km", "mean"),
            avg_cuisine_diversity=("cuisine_diversity_at_10", "mean"),
            avg_category_diversity=("category_diversity_at_10", "mean"),
            avg_coverage=("coverage", "mean"),
        )
        .fillna("Not available")
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    distance_ready = summary[summary["avg_distance_km"] != "Not available"].copy()
    if not distance_ready.empty:
        fig = px.bar(
            distance_ready,
            x="model",
            y="avg_distance_km",
            title="Average distance in top recommendations",
            labels={"model": "Model", "avg_distance_km": "Average distance (km)"},
            height=380,
        )
        fig.update_layout(margin={"r": 20, "t": 50, "l": 20, "b": 20})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show internal evaluation output"):
        st.dataframe(results, use_container_width=True, hide_index=True)


def main() -> None:
    page_frame()
    df, source = load_features(processed_data_version())
    page = st.sidebar.radio(
        "CafeCompass",
        [
            "Project Overview",
            "Explore Vancouver Food Map",
            "Cafe/Food Spot Recommender",
            "Similar Places",
            "Data Coverage",
            "Ranking Logic",
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
    elif page == "Data Coverage":
        data_coverage_page(df, source)
    else:
        evaluation_page()


if __name__ == "__main__":
    main()
