from __future__ import annotations

from typing import Any


def validate_product_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("product_profile response must be a JSON object.")

    profile = {
        "summary": _as_string(payload.get("summary")),
        "intended_use": _as_string(payload.get("intended_use")),
        "material_or_composition": _as_string(payload.get("material_or_composition")),
        "key_features": _as_string_list(payload.get("key_features")),
        "search_queries": _as_string_list(payload.get("search_queries")),
        "uncertainty": _as_string(payload.get("uncertainty")),
        "missing_information": _as_string_list(payload.get("missing_information")),
    }

    if not profile["summary"]:
        raise RuntimeError("product_profile.summary must not be empty.")
    if not profile["search_queries"]:
        fallback_queries = [item for item in (profile["summary"], profile["intended_use"]) if item]
        if not fallback_queries:
            raise RuntimeError("product_profile.search_queries must not be empty.")
        profile["search_queries"] = fallback_queries

    return profile


def validate_classification_payload(payload: dict[str, Any], top_k: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("classification response must be a JSON object.")

    warnings = _as_string_list(payload.get("warnings"))
    best_match = _validate_candidate(payload.get("best_match"), allow_none=True)
    candidates = [_validate_candidate(item, allow_none=False) for item in payload.get("candidates", []) if item]

    ordered_candidates: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    if best_match is not None:
        ordered_candidates.append(best_match)
        seen_codes.add(best_match["code"])

    for item in candidates:
        if item["code"] in seen_codes:
            continue
        ordered_candidates.append(item)
        seen_codes.add(item["code"])

    ordered_candidates = ordered_candidates[:top_k]

    if best_match is None and ordered_candidates:
        best_match = ordered_candidates[0]

    return {
        "best_match": best_match,
        "candidates": ordered_candidates,
        "warnings": warnings,
    }


def _validate_candidate(payload: Any, allow_none: bool) -> dict[str, Any] | None:
    if payload is None:
        if allow_none:
            return None
        raise RuntimeError("classification candidate must not be null.")
    if not isinstance(payload, dict):
        raise RuntimeError("classification candidate must be an object.")

    code = _as_string(payload.get("code"))
    if not code:
        raise RuntimeError("classification candidate.code must not be empty.")

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "code": code,
        "title_or_label": _as_string(payload.get("title_or_label")),
        "confidence": confidence,
        "reasoning": _as_string(payload.get("reasoning")),
        "evidence_ids": _as_string_list(payload.get("evidence_ids")),
    }


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        cleaned = _as_string(item)
        if cleaned:
            result.append(cleaned)
    return result
