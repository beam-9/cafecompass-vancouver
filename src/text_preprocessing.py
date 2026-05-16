from __future__ import annotations

import re
from functools import lru_cache

URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACE_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        return None


def basic_clean(text: object) -> str:
    text = "" if text is None else str(text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_text(text: object, lemmatize: bool = True) -> str:
    cleaned = basic_clean(text)
    if not cleaned or not lemmatize:
        return cleaned
    nlp = _load_spacy()
    if nlp is None:
        return cleaned
    doc = nlp(cleaned)
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip() or token.text
        tokens.append(lemma)
    return " ".join(tokens)

