from __future__ import annotations

from typing import Any

from src.app.core.config import EXAMPLE_COLLECTION_NAME, REFERENCE_COLLECTION_NAME, AppPaths
from src.app.core.openrouter_client import embed_texts
from src.app.core.vector_db import get_collection, get_persistent_client, query_collection


def retrieve_supporting_context(
    paths: AppPaths,
    product_profile: dict[str, Any],
    reference_top_k: int = 8,
    example_top_k: int = 4,
) -> dict[str, Any]:
    query_texts = _build_search_queries(product_profile)
    query_embeddings = embed_texts(query_texts)

    client = get_persistent_client(paths)
    reference_collection = get_collection(client, REFERENCE_COLLECTION_NAME)
    reference_hits = query_collection(reference_collection, query_embeddings, top_k=reference_top_k)

    try:
        example_collection = get_collection(client, EXAMPLE_COLLECTION_NAME)
    except RuntimeError:
        example_hits = []
    else:
        example_hits = query_collection(example_collection, query_embeddings, top_k=example_top_k)

    return {
        "query_texts": query_texts,
        "reference": reference_hits,
        "examples": example_hits,
    }


def _build_search_queries(product_profile: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()

    for item in product_profile.get("search_queries", []):
        cleaned = str(item).strip()
        lowered = cleaned.lower()
        if not cleaned or lowered in seen:
            continue
        seen.add(lowered)
        queries.append(cleaned)

    if not queries:
        for fallback in (product_profile.get("summary", ""), product_profile.get("intended_use", "")):
            cleaned = str(fallback).strip()
            lowered = cleaned.lower()
            if cleaned and lowered not in seen:
                seen.add(lowered)
                queries.append(cleaned)

    return queries
