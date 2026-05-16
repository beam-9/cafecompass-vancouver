from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODELS_DIR, PROCESSED_DIR, ensure_dirs

MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build_place_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    ensure_dirs()
    master_path = PROCESSED_DIR / "place_master.csv"
    reviews_path = PROCESSED_DIR / "vancouver_reviews_clean.csv"
    if not master_path.exists() or not reviews_path.exists():
        print("Missing place master or cleaned reviews. Skipping embeddings.")
        index = pd.DataFrame(columns=["place_id", "business_id", "name"])
        np.save(MODELS_DIR / "place_embeddings.npy", np.zeros((0, 384), dtype="float32"))
        index.to_csv(MODELS_DIR / "place_embedding_index.csv", index=False)
        return np.zeros((0, 384), dtype="float32"), index

    master = pd.read_csv(master_path)
    reviews = pd.read_csv(reviews_path)
    lookup = master.set_index("yelp_business_id")["place_id"].to_dict()
    reviews["place_id"] = reviews["business_id"].map(lookup)
    grouped = reviews.dropna(subset=["place_id"]).groupby("place_id")["text"].apply(lambda s: " ".join(s.dropna().astype(str).head(50)))
    if grouped.empty:
        return np.zeros((0, 384), dtype="float32"), pd.DataFrame()

    model = _load_model()
    embeddings = model.encode(grouped.tolist(), show_progress_bar=True, normalize_embeddings=True)
    index = pd.DataFrame({"place_id": grouped.index})
    index = index.merge(master[["place_id", "yelp_business_id", "name"]], on="place_id", how="left")
    np.save(MODELS_DIR / "place_embeddings.npy", embeddings)
    index.to_csv(MODELS_DIR / "place_embedding_index.csv", index=False)
    print(f"Saved embeddings for {len(index):,} places.")
    return embeddings, index


def embed_query(query: str) -> np.ndarray:
    model = _load_model()
    return model.encode([query], normalize_embeddings=True)[0]


if __name__ == "__main__":
    build_place_embeddings()

