from __future__ import annotations

import json
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
            "Currently showing the synthetic demo dataset. The actual Vancouver dataset will be added by running "
            "the OSM, Vancouver Open Data, Yelp Open Dataset, and optional Reddit collection pipeline."
        )


def _fmt_missing(value: object, fallback: str = "Not available") -> str:
    if pd.isna(value) or str(value).strip() == "":
        return fallback
    return html.escape(str(value))


def _score_rows(breakdown: dict[str, float]) -> str:
    labels = {
        "aspect_match_score": "Experience match",
        "metadata_match_score": "Cuisine/type match",
        "semantic_similarity_score": "Text similarity",
        "distance_score": "Distance fit",
        "rating_score": "Rating/popularity",
        "confidence_score": "Text confidence",
        "context_score": "Context fit",
        "hidden_gem_score": "Hidden-gem signal",
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
        return "<div style='font-size:0.88rem;'>No active ranking signals are available for this place yet.</div>"
    return "\n".join(rows)


def _real_evidence(snippets: list[str]) -> list[str]:
    unavailable = "No direct text evidence available"
    return [snippet for snippet in snippets if unavailable not in snippet]


def render_recommendation_card(row: pd.Series) -> None:
    explanation = explain_recommendation(row)
    breakdown = explanation.get("score_breakdown", {})
    evidence = _real_evidence(explanation.get("evidence", []))
    cuisine = _fmt_missing(row.get("cuisine"))
    categories = _fmt_missing(row.get("categories"))
    distance = row.get("distance_km")
    distance_text = f"{float(distance):.1f} km away" if pd.notna(distance) else "Distance unavailable"
    score = float(row.get("final_score", 0) or 0)
    match_text = f"{score * 100:.0f}% match" if score <= 1 else f"{score:.2f} match score"
    reason_items = "".join(f"<span class='pill'>{html.escape(reason)}</span>" for reason in explanation.get("reasons", [])[:2])
    evidence_html = ""
    if evidence:
        evidence_html = "<div style='margin-top:10px; font-weight:700;'>Evidence from text</div>" + "".join(
            f"<div style='font-size:0.86rem; margin-top:4px;'>\"{html.escape(snippet)}\"</div>" for snippet in evidence[:2]
        )
    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-title">{_fmt_missing(row.get("name"), "Unnamed place")}</div>
          <div class="rec-meta">{distance_text} · {cuisine} · {categories}</div>
          <div><span class="pill">{match_text}</span>{reason_items}</div>
          <details style="margin-top:10px;">
            <summary style="font-weight:700; cursor:pointer;">Why this ranked here</summary>
            <div style="margin-top:8px;">{_score_rows(breakdown)}</div>
          </details>
          {evidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metadata_tokens(row: pd.Series) -> set[str]:
    text = " ".join(str(row.get(col, "")) for col in ["name", "categories", "cuisine"])
    return {token for token in re.split(r"[^a-z0-9_]+", text.lower()) if len(token) > 1}


def _simple_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def similar_places(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    score_cols = [f"{a}_score" for a in ASPECTS if f"{a}_score" in df.columns]
    has_aspects = bool(score_cols) and (df[score_cols].fillna(0).sum(axis=1) > 0).any()
    out = df.copy()
    if has_aspects:
        matrix = out[score_cols].fillna(0).astype(float)
        target = selected[score_cols].fillna(0).astype(float)
        denom = (matrix.pow(2).sum(axis=1).pow(0.5) * (target.pow(2).sum() ** 0.5)) + 1e-9
        out["similarity"] = matrix.dot(target) / denom
        out["similarity_basis"] = "Review aspect profile"
    else:
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
            basis.append(", ".join(reasons) or "metadata fallback")
        out["similarity"] = scores
        out["similarity_basis"] = basis
    out = out[out["place_id"] != selected.get("place_id")] if "place_id" in out.columns else out[out["name"] != selected.get("name")]
    return out.sort_values(["similarity", "name"], ascending=[False, True])


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
        "The current actual dataset is strongest for names, locations, cuisines, and place types. "
        "Ratings and review-text signals appear only after Yelp Open Dataset or Reddit text is added."
    )

    left, right = st.columns([1, 3])
    ratings_available = "stars" in df and pd.to_numeric(df["stars"], errors="coerce").notna().any()
    text_available = "confidence_score" in df and (pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0) > 0).any()
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
        if text_available:
            color_options.extend(["Text evidence", "Quiet study", "Cheap value", "Date night", "Hidden gem", "Authentic", "Group friendly", "Dessert/drinks", "Late night"])
        color_label = st.selectbox("Color map by", color_options)
        if not ratings_available or not text_available:
            st.caption(
                "Ratings, review counts, confidence, and hidden-gem scores are unavailable for OSM-only places. "
                "They will become meaningful after review/community text is joined to these places."
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
        "Cuisine": "cuisine",
        "Place type": "categories",
        "Rating": "stars",
        "Text evidence": "confidence_score",
        "Quiet study": "quiet_study_score",
        "Cheap value": "cheap_value_score",
        "Date night": "date_night_score",
        "Hidden gem": "hidden_gem_adjusted_score",
        "Authentic": "authentic_score",
        "Group friendly": "group_friendly_score",
        "Dessert/drinks": "dessert_drinks_score",
        "Late night": "late_night_score",
    }
    color_col = aspect_map.get(color_label, "cuisine")

    with right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Places shown", len(view))
        metric_cols[1].metric("Cuisines", view["cuisine"].nunique() if "cuisine" in view else 0)
        rating_values = pd.to_numeric(view.get("stars", pd.Series(dtype=float)), errors="coerce")
        metric_cols[2].metric("Ratings available", int(rating_values.notna().sum()))
        evidence = view.get("confidence_score", pd.Series(0, index=view.index)).fillna(0)
        metric_cols[3].metric("With text evidence", int((evidence > 0).sum()))
        if not ratings_available and not text_available:
            st.info(
                "This map is currently a real place-coverage layer from OSM and City data. "
                "It is useful for exploring where restaurants and cafes are, but it should not be interpreted as ranked quality yet."
            )
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
        table_cols = [c for c in ["name", "cuisine", "categories", "stars", "review_count", "confidence_score"] if c in view.columns]
        display_view = view[table_cols].copy()
        if "stars" in display_view:
            display_view["stars"] = display_view["stars"].fillna("Not available")
        if "review_count" in display_view:
            display_view["review_count"] = display_view["review_count"].fillna("Not available")
        if "confidence_score" in display_view:
            display_view["confidence_score"] = display_view["confidence_score"].map(lambda value: "Review text not linked yet" if pd.isna(value) or float(value) == 0 else f"{float(value):.2f}")
        with st.expander("Show place data used for this map"):
            st.dataframe(display_view.sort_values(["cuisine", "name"]), use_container_width=True, hide_index=True)


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
        st.markdown("<div class='slider-panel'><strong>Ranking weights</strong><br><span>Adjust how much each signal affects the order.</span></div>", unsafe_allow_html=True)
        ratings_available = "stars" in df and pd.to_numeric(df["stars"], errors="coerce").notna().any()
        text_available = "confidence_score" in df and (pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0) > 0).any()
        embeddings_available = "place_embedding_available" in df and df["place_embedding_available"].fillna(False).astype(bool).any()
        weight_labels = {
            "aspect_match_score": "Experience match",
            "semantic_similarity_score": "Text similarity",
            "distance_score": "Distance",
            "rating_score": "Rating/popularity",
            "confidence_score": "Text confidence",
            "context_score": "Context",
            "hidden_gem_score": "Hidden gem",
        }
        weights = {}
        for key, default in DEFAULT_WEIGHTS.items():
            disabled = (
                (key in {"aspect_match_score", "confidence_score", "hidden_gem_score"} and not text_available)
                or (key == "semantic_similarity_score" and not embeddings_available)
                or (key == "rating_score" and not ratings_available)
            )
            weights[key] = st.slider(weight_labels.get(key, key), 0.0, 1.0, 0.0 if disabled else float(default), 0.05, disabled=disabled)
        if not text_available or not ratings_available:
            st.caption("Disabled sliders need review text, embeddings, or ratings that are not linked to the current OSM/City metadata yet.")

    config = RecommenderConfig(start_lat=lat, start_lon=lon, max_distance_km=max_distance, weights=weights)
    results = hybrid_recommender(df, experience, config, query_text=query, cuisine_filter=cuisine or None, top_k=10)

    with results_col:
        if results.empty:
            st.warning("No matching places found for these filters.")
            return
        st.metric("Recommendations", len(results))
        if (pd.to_numeric(results.get("confidence_score", pd.Series(0, index=results.index)), errors="coerce").fillna(0) == 0).all():
            st.info(
                "These recommendations are currently ranked by distance plus cuisine/category/name matches from OSM and City metadata. "
                "Review-text evidence will appear after Yelp Open Dataset or Reddit text is linked to places."
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
        "Find places that are similar by cuisine, place type, name/category terms, and distance. "
        "Once review text is linked, this page can switch to review-aspect similarity."
    )
    options = df["name"].fillna("Unnamed").astype(str).tolist()
    place = st.selectbox("Select a place", options)
    selected = df[df["name"].fillna("Unnamed").astype(str) == place].iloc[0]
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
    results = load_evaluation(processed_data_version())
    if results.empty:
        st.info("Evaluation results are not available yet. Run `python src/evaluation.py` after building features.")
        return
    st.dataframe(results, use_container_width=True)
    fig = px.bar(results, x="query", y="precision_at_10_aspect", color="model", barmode="group", height=520)
    st.plotly_chart(fig, use_container_width=True)


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
