from __future__ import annotations

from typing import Any

from src.app.core.openrouter_client import chat_json
from src.app.schemas.contracts import validate_classification_payload

FINAL_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a TN VED customs classification assistant. "
    "Reference documents are the primary source of truth. "
    "Example cases are secondary hints. "
    "Return one valid JSON object and nothing else."
)


def classify_from_retrieval(
    raw_text: str,
    product_profile: dict[str, Any],
    retrieval_result: dict[str, Any],
    top_k: int,
    input_warnings: list[str] | None = None,
) -> dict[str, Any]:
    clipped_reference = [_serialize_hit_for_prompt(item) for item in retrieval_result["reference"]]
    clipped_examples = [_serialize_hit_for_prompt(item) for item in retrieval_result["examples"]]

    prompt_payload = {
        "raw_user_text": raw_text.strip(),
        "product_profile": product_profile,
        "reference_evidence": clipped_reference,
        "example_evidence": clipped_examples,
        "top_k": top_k,
    }

    llm_payload = chat_json(
        [
            {"role": "system", "content": FINAL_CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "На основе product_profile и retrieval-контекста выбери один лучший код ТН ВЭД и список кандидатов.\n"
                    "Reference evidence имеет больший приоритет, чем example evidence.\n"
                    "Не выдумывай коды при отсутствии опоры в evidence. Если уверенности мало, best_match может быть null.\n"
                    "Верни только JSON с ключами:\n"
                    "{\n"
                    '  "best_match": {\n'
                    '    "code": string,\n'
                    '    "title_or_label": string,\n'
                    '    "confidence": number,\n'
                    '    "reasoning": string,\n'
                    '    "evidence_ids": [string, ...]\n'
                    '  } | null,\n'
                    '  "candidates": [{...}],\n'
                    '  "warnings": [string, ...]\n'
                    "}\n"
                    "candidates должны быть ранжированы по убыванию уверенности.\n\n"
                    f"{prompt_payload}"
                ),
            },
        ],
        temperature=0.1,
    )

    normalized = validate_classification_payload(llm_payload, top_k=top_k)
    warnings = list(normalized["warnings"])
    for warning in input_warnings or []:
        if warning not in warnings:
            warnings.append(warning)
    normalized["warnings"] = warnings
    return normalized


def render_human_summary(result: dict[str, Any]) -> str:
    recommendation_view = select_primary_and_alternatives(result, alternative_limit=2)
    primary = recommendation_view["primary"]
    alternatives = recommendation_view["alternatives"]
    warnings = list(result.get("warnings", []))

    lines: list[str] = []
    if primary:
        title = "Лучший код" if recommendation_view["primary_source"] == "best_match" else "Основной кандидат"
        lines.append(
            f"{title}: {primary['code']} "
            f"({primary.get('title_or_label', 'без названия')}) "
            f"уверенность={primary.get('confidence', 0.0):.2f}"
        )
        if primary.get("reasoning"):
            lines.append(f"Обоснование: {primary['reasoning']}")
    else:
        lines.append("Подходящий код не определён уверенно.")

    if alternatives:
        lines.append("")
        lines.append("Дополнительные варианты для проверки:")
        for index, item in enumerate(alternatives, start=1):
            lines.append(
                f"{index}. {item['code']} - {item.get('title_or_label', 'без названия')} "
                f"(уверенность={item.get('confidence', 0.0):.2f})"
            )

    if warnings:
        lines.append("")
        lines.append("Предупреждения:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines).strip()


def select_primary_and_alternatives(
    result: dict[str, Any],
    alternative_limit: int = 2,
) -> dict[str, Any]:
    best_match = result.get("best_match")
    primary = best_match if isinstance(best_match, dict) and best_match else None
    primary_source = "best_match" if primary else "none"

    unique_candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    if primary is not None:
        primary_key = _candidate_identity(primary)
        if primary_key is not None:
            seen_keys.add(primary_key)

    for candidate in result.get("candidates", []):
        if not isinstance(candidate, dict) or not candidate:
            continue
        identity = _candidate_identity(candidate)
        if identity is None or identity in seen_keys:
            continue
        seen_keys.add(identity)
        unique_candidates.append(candidate)

    if primary is None and unique_candidates:
        primary = unique_candidates.pop(0)
        primary_source = "candidate"

    return {
        "primary": primary,
        "primary_source": primary_source,
        "alternatives": unique_candidates[: max(0, alternative_limit)],
    }


def _serialize_hit_for_prompt(hit: dict[str, Any], text_limit: int = 900) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id"),
        "source_path": hit.get("source_path"),
        "source_kind": hit.get("source_kind"),
        "document_type": hit.get("document_type"),
        "section_context": hit.get("section_context"),
        "mentioned_codes": list(hit.get("mentioned_codes", [])),
        "score": float(hit.get("score", 0.0)),
        "text": _clip_text(hit.get("text", ""), text_limit),
    }


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _candidate_identity(candidate: dict[str, Any]) -> tuple[str, str] | None:
    code = str(candidate.get("code", "")).strip()
    title = str(candidate.get("title_or_label", "")).strip()
    if not code and not title:
        return None
    return code, title
