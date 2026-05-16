from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_places_map(df: pd.DataFrame, color: str | None = None):
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return px.scatter_mapbox(lat=[], lon=[])
    return px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="name" if "name" in df.columns else None,
        color=color if color in df.columns else None,
        zoom=11,
        height=520,
    ).update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})

