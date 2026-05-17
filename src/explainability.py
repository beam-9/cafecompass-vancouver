from __future__ import annotations

import json

import pandas as pd


def _load_evidence(row: pd.Series) -> dict:
    value = row.get("evidence_snippets_json")
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def explain_recommendation(row: pd.Series, max_reasons: int = 3) -> dict:
    reasons = []
    evidence = _load_evidence(row)
    metadata_score = row.get("metadata_match_score")
    if pd.notna(metadata_score) and float(metadata_score) > 0:
        reasons.append("Matches the cuisine, place type, or preference words in the search.")
    if pd.notna(row.get("distance_km")):
        reasons.append(f"Within {float(row['distance_km']):.1f} km of the selected starting point.")
    cuisine = row.get("cuisine")
    if pd.notna(cuisine) and str(cuisine).strip():
        reasons.append(f"Listed with {str(cuisine).replace('_', ' ')} cuisine metadata.")
    categories = row.get("categories")
    if len(reasons) < max_reasons and pd.notna(categories) and str(categories).strip():
        reasons.append(f"Tagged as {str(categories).replace('_', ' ')}.")
    if not reasons:
        reasons.append("Recommended from available place metadata.")

    snippets = []
    for aspect_snippets in evidence.values():
        if isinstance(aspect_snippets, list):
            snippets.extend(str(s) for s in aspect_snippets if s)
    return {
        "reasons": reasons[:max_reasons],
        "score_breakdown": json.loads(row.get("score_breakdown_json", "{}")) if isinstance(row.get("score_breakdown_json"), str) else {},
        "evidence": snippets[:3],
    }


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["explanation_json"] = out.apply(lambda row: json.dumps(explain_recommendation(row)), axis=1)
    return out
