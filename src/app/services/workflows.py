from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.app.core.config import TOP_K_CHOICES, get_app_paths
from src.app.core.openrouter_client import get_openrouter_settings
from src.app.services.classification_service import classify_from_retrieval, render_human_summary
from src.app.services.input_analysis_service import analyze_multimodal_input
from src.app.services.retrieval_service import retrieve_supporting_context
from src.app.services.train_service import (
    build_training_chunks,
    index_training_chunks,
    load_runtime_metadata,
    plan_training_index_update,
    scan_training_documents,
)
from src.app.utils.io_utils import ensure_directory, generate_run_id, log_event, utc_now_iso

ProgressCallback = Callable[[str], None]


def run_train_workflow(
    progress_callback: ProgressCallback | None = None,
    full_rebuild: bool = False,
) -> dict[str, Any]:
    paths = get_app_paths()
    ensure_directory(paths.runtime_dir)
    ensure_directory(paths.vector_db_dir)
    run_id = generate_run_id()
    mode = "train"

    def bootstrap() -> dict[str, Any]:
        settings = get_openrouter_settings(require_chat_model=True)
        return {
            "reference_dir": paths.reference_docs_dir.as_posix(),
            "example_dir": paths.example_docs_dir.as_posix(),
            "embedding_model": settings["embedding_model"],
            "chat_model": settings["chat_model"],
        }

    bootstrap_payload = run_stage(run_id, mode, "bootstrap", bootstrap)
    _emit_progress(
        progress_callback,
        "Запуск индексации: "
        f"embedding_model={bootstrap_payload['embedding_model']}, "
        f"chat_model={bootstrap_payload['chat_model']}",
    )

    scan_payload = run_stage(
        run_id,
        mode,
        "scan_documents",
        lambda: scan_training_documents(paths),
        success_payload=lambda data: {
            "reference_files": len(data["reference_documents"]),
            "example_files": len(data["example_documents"]),
        },
    )
    _emit_progress(
        progress_callback,
        f"Найдено reference={len(scan_payload['reference_documents'])}, "
        f"examples={len(scan_payload['example_documents'])}",
    )

    index_plan = run_stage(
        run_id,
        mode,
        "plan_index_update",
        lambda: plan_training_index_update(
            paths,
            reference_documents=scan_payload["reference_documents"],
            example_documents=scan_payload["example_documents"],
            embedding_model=bootstrap_payload["embedding_model"],
            full_rebuild=full_rebuild,
        ),
        success_payload=lambda data: {
            "index_mode": data["index_mode"],
            "sync_stats": data["sync_stats"],
            "full_rebuild_reason": data.get("full_rebuild_reason"),
        },
    )
    _emit_progress(
        progress_callback,
        f"Выбран режим {index_plan['index_mode']}: "
        f"added={index_plan['sync_stats']['added']}, "
        f"changed={index_plan['sync_stats']['changed']}, "
        f"removed={index_plan['sync_stats']['removed']}, "
        f"unchanged={index_plan['sync_stats']['unchanged']}",
    )
    if index_plan.get("full_rebuild_reason"):
        _emit_progress(progress_callback, f"Причина полного rebuild: {index_plan['full_rebuild_reason']}")

    chunk_payload = run_stage(
        run_id,
        mode,
        "build_chunks",
        lambda: build_training_chunks(
            paths,
            index_plan["reference_documents_to_index"],
            index_plan["example_documents_to_index"],
        ),
        success_payload=lambda data: {
            "reference_chunks": len(data["reference_chunks"]),
            "example_chunks": len(data["example_chunks"]),
        },
    )
    _emit_progress(
        progress_callback,
        f"Подготовлено чанков: reference={len(chunk_payload['reference_chunks'])}, "
        f"examples={len(chunk_payload['example_chunks'])}",
    )

    metadata = run_stage(
        run_id,
        mode,
        "embed_and_index",
        lambda: index_training_chunks(
            paths,
            index_plan=index_plan,
            manifest_records=chunk_payload["manifest_records"],
            reference_chunks=chunk_payload["reference_chunks"],
            example_chunks=chunk_payload["example_chunks"],
        ),
        success_payload=lambda data: data,
    )
    _emit_progress(
        progress_callback,
        f"Индексация завершена успешно: mode={metadata['index_mode']}, "
        f"processed={metadata['sync_stats']['processed_documents']}",
    )
    return metadata


def run_classify_workflow(
    raw_text: str,
    file_paths: list[str] | None = None,
    top_k: int = 4,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    paths = get_app_paths()
    ensure_directory(paths.runtime_dir)
    run_id = generate_run_id()
    mode = "classify"
    effective_file_paths = list(file_paths or [])

    def bootstrap() -> dict[str, Any]:
        if top_k not in TOP_K_CHOICES:
            raise ValueError("--top-k должен быть равен 3 или 4.")
        if not raw_text.strip() and not effective_file_paths:
            raise ValueError("Нужно передать хотя бы один источник запроса: --text или --file.")
        missing_paths = [Path(item) for item in effective_file_paths if not Path(item).exists()]
        if missing_paths:
            missing_list = ", ".join(path.as_posix() for path in missing_paths)
            raise FileNotFoundError(f"Переданные файлы не найдены: {missing_list}")
        settings = get_openrouter_settings(require_chat_model=True)
        return {
            "files_count": len(effective_file_paths),
            "embedding_model": settings["embedding_model"],
            "chat_model": settings["chat_model"],
            "top_k": top_k,
        }

    bootstrap_payload = run_stage(run_id, mode, "bootstrap", bootstrap)
    _emit_progress(
        progress_callback,
        "Запуск классификации: "
        f"embedding_model={bootstrap_payload['embedding_model']}, "
        f"chat_model={bootstrap_payload['chat_model']}, "
        f"top_k={bootstrap_payload['top_k']}",
    )

    metadata = run_stage(
        run_id,
        mode,
        "load_runtime_metadata",
        lambda: load_runtime_metadata(paths),
        success_payload=lambda data: data,
    )
    _emit_progress(
        progress_callback,
        f"Runtime готов: reference_chunks={metadata['reference_chunk_count']}, "
        f"example_chunks={metadata['example_chunk_count']}",
    )

    analysis_result = run_stage(
        run_id,
        mode,
        "multimodal_analysis",
        lambda: analyze_multimodal_input(raw_text, effective_file_paths),
        success_payload=lambda data: {
            "files_count": len(data["input_bundle"]["files"]),
            "images_count": len(data["input_bundle"]["images"]),
            "search_queries": len(data["product_profile"]["search_queries"]),
        },
    )
    _emit_progress(progress_callback, "Мультимодальный product_profile собран.")

    retrieval_result = run_stage(
        run_id,
        mode,
        "retrieve_context",
        lambda: retrieve_supporting_context(paths, analysis_result["product_profile"]),
        success_payload=lambda data: {
            "reference_hits": len(data["reference"]),
            "example_hits": len(data["examples"]),
        },
    )
    _emit_progress(
        progress_callback,
        f"Поднят retrieval-контекст: reference={len(retrieval_result['reference'])}, "
        f"examples={len(retrieval_result['examples'])}",
    )

    classification_result = run_stage(
        run_id,
        mode,
        "final_classification",
        lambda: classify_from_retrieval(
            raw_text=raw_text,
            product_profile=analysis_result["product_profile"],
            retrieval_result=retrieval_result,
            top_k=top_k,
            input_warnings=analysis_result["input_bundle"]["warnings"],
        ),
        success_payload=lambda data: {
            "has_best_match": data["best_match"] is not None,
            "candidates": len(data["candidates"]),
            "warnings": len(data["warnings"]),
        },
    )
    _emit_progress(progress_callback, "Финальная классификация завершена.")

    result = {
        "product_profile": analysis_result["product_profile"],
        "best_match": classification_result["best_match"],
        "candidates": classification_result["candidates"],
        "retrieval": {
            "reference": [_serialize_hit_for_output(item) for item in retrieval_result["reference"]],
            "examples": [_serialize_hit_for_output(item) for item in retrieval_result["examples"]],
        },
        "warnings": classification_result["warnings"],
    }
    summary = render_human_summary(result)
    return {
        "result": result,
        "summary": summary,
        "analysis_result": analysis_result,
        "retrieval_result": retrieval_result,
    }


def run_stage(
    run_id: str,
    mode: str,
    stage: str,
    func: Callable[[], Any],
    success_payload: Callable[[Any], dict[str, Any]] | None = None,
) -> Any:
    paths = get_app_paths()
    started_at = utc_now_iso()
    log_event(
        paths.events_log_path,
        run_id=run_id,
        mode=mode,
        stage=stage,
        status="pending",
        started_at=started_at,
        finished_at=None,
        payload={},
    )
    try:
        result = func()
    except Exception as exc:
        log_event(
            paths.events_log_path,
            run_id=run_id,
            mode=mode,
            stage=stage,
            status="error",
            started_at=started_at,
            finished_at=utc_now_iso(),
            payload={"error": str(exc)},
        )
        raise

    payload = success_payload(result) if success_payload else {}
    log_event(
        paths.events_log_path,
        run_id=run_id,
        mode=mode,
        stage=stage,
        status="success",
        started_at=started_at,
        finished_at=utc_now_iso(),
        payload=payload,
    )
    return result


def _serialize_hit_for_output(hit: dict[str, Any], text_limit: int = 320) -> dict[str, Any]:
    normalized = " ".join(str(hit.get("text", "")).split())
    if len(normalized) > text_limit:
        normalized = normalized[: text_limit - 3].rstrip() + "..."
    return {
        "chunk_id": hit.get("chunk_id"),
        "source_path": hit.get("source_path"),
        "source_kind": hit.get("source_kind"),
        "document_type": hit.get("document_type"),
        "section_context": hit.get("section_context"),
        "mentioned_codes": list(hit.get("mentioned_codes", [])),
        "score": float(hit.get("score", 0.0)),
        "text": normalized,
    }


def _emit_progress(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
