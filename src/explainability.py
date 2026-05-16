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
    score_pairs = []
    for key, value in row.items():
        if key.endswith("_score") and key not in {"final_score", "distance_score", "rating_score"}:
            try:
                score_pairs.append((key, float(value)))
            except Exception:
                pass
    for key, value in sorted(score_pairs, key=lambda kv: kv[1], reverse=True):
        if value <= 0:
            continue
        label = key.replace("_score", "").replace("_", " ")
        reasons.append(f"Strong {label} signal from available text and metadata.")
        if len(reasons) >= max_reasons:
            break
    if pd.notna(row.get("distance_km")):
        reasons.append(f"Within {float(row['distance_km']):.1f} km of the selected starting point.")
    if not reasons:
        reasons.append("No direct text evidence available; recommendation is based on metadata and similarity.")

    snippets = []
    for aspect_snippets in evidence.values():
        if isinstance(aspect_snippets, list):
            snippets.extend(str(s) for s in aspect_snippets if s)
    return {
        "reasons": reasons[:max_reasons],
        "score_breakdown": json.loads(row.get("score_breakdown_json", "{}")) if isinstance(row.get("score_breakdown_json"), str) else {},
        "evidence": snippets[:3] or ["No direct text evidence available; recommendation is based on metadata and similarity."],
    }


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["explanation_json"] = out.apply(lambda row: json.dumps(explain_recommendation(row)), axis=1)
    return out

