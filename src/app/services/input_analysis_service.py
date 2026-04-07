from __future__ import annotations

from pathlib import Path
from typing import Any

from src.app.core.openrouter_client import chat_json
from src.app.schemas.contracts import validate_product_profile
from src.app.utils.document_processing import (
    IMAGE_EXTENSIONS,
    encode_image_to_data_url,
    extract_text_from_file,
    try_extract_text_from_image,
)

MAX_ANALYSIS_TEXT_CHARS = 4000

MULTIMODAL_ANALYSIS_SYSTEM_PROMPT = (
    "You analyze multimodal product inputs for TN VED customs retrieval. "
    "Return one valid JSON object and nothing else."
)


def analyze_multimodal_input(raw_text: str, file_paths: list[str]) -> dict[str, Any]:
    input_bundle = collect_input_bundle(raw_text, file_paths)

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Сформируй JSON product_profile по данным пользователя.\n"
                "Верни только JSON объект с ключами:\n"
                "{\n"
                '  "summary": string,\n'
                '  "intended_use": string,\n'
                '  "material_or_composition": string,\n'
                '  "key_features": [string, ...],\n'
                '  "search_queries": [string, ...],\n'
                '  "uncertainty": string,\n'
                '  "missing_information": [string, ...]\n'
                "}\n"
                "search_queries должны быть короткими и полезными для semantic search по нормативным документам.\n"
                "Если данных мало, не выдумывай и явно укажи uncertainty/missing_information.\n\n"
                f"Данные пользователя:\n{input_bundle['analysis_text']}"
            ),
        }
    ]

    for image_item in input_bundle["images"]:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_item["data_url"]},
            }
        )

    profile_payload = chat_json(
        [
            {"role": "system", "content": MULTIMODAL_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    product_profile = validate_product_profile(profile_payload)
    return {
        "product_profile": product_profile,
        "input_bundle": input_bundle,
    }


def collect_input_bundle(raw_text: str, file_paths: list[str]) -> dict[str, Any]:
    warnings: list[str] = []
    analysis_sections: list[str] = []
    images: list[dict[str, Any]] = []
    file_descriptors: list[dict[str, Any]] = []

    cleaned_text = raw_text.strip()
    if cleaned_text:
        analysis_sections.append(f"Пользовательский текст:\n{cleaned_text}")

    for raw_path in file_paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        descriptor = {
            "path": path.resolve().as_posix(),
            "name": path.name,
            "document_type": suffix.lstrip("."),
        }

        if suffix in IMAGE_EXTENSIONS:
            ocr_text, warning = try_extract_text_from_image(path)
            if ocr_text:
                analysis_sections.append(f"OCR изображения {path.name}:\n{_clip_text(ocr_text, MAX_ANALYSIS_TEXT_CHARS)}")
            if warning:
                warnings.append(f"{path.name}: {warning}")
            images.append(
                {
                    "path": path.resolve().as_posix(),
                    "name": path.name,
                    "data_url": encode_image_to_data_url(path),
                    "ocr_text": _clip_text(ocr_text, MAX_ANALYSIS_TEXT_CHARS),
                }
            )
            descriptor["ocr_text"] = _clip_text(ocr_text, MAX_ANALYSIS_TEXT_CHARS)
            file_descriptors.append(descriptor)
            continue

        extracted_text = extract_text_from_file(path)
        clipped_text = _clip_text(extracted_text, MAX_ANALYSIS_TEXT_CHARS)
        if clipped_text:
            analysis_sections.append(f"Текст файла {path.name}:\n{clipped_text}")
        descriptor["extracted_text"] = clipped_text
        file_descriptors.append(descriptor)

    if not analysis_sections and images:
        analysis_sections.append("Пользователь передал только изображения. Оцени их визуально и по OCR, если он есть.")

    return {
        "raw_text": cleaned_text,
        "analysis_text": "\n\n".join(analysis_sections).strip(),
        "images": images,
        "files": file_descriptors,
        "warnings": warnings,
    }


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
