from __future__ import annotations

import pandas as pd


def clean_place_name(name: object) -> str:
    return " ".join(str(name or "").split()).strip()


def standardize_places(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "name" in out:
        out["name"] = out["name"].map(clean_place_name)
    return out

