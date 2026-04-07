from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.app.core.config import (
    EXAMPLE_COLLECTION_NAME,
    REFERENCE_COLLECTION_NAME,
    SNAPSHOT_SCHEMA_VERSION,
    AppPaths,
)
from src.app.core.openrouter_client import embed_texts, get_openrouter_settings
from src.app.core.vector_db import (
    add_embeddings,
    delete_embeddings,
    get_collection,
    get_persistent_client,
    list_collection_records,
    reset_collection,
)
from src.app.utils.document_processing import SUPPORTED_EXTENSIONS, build_knowledge_chunks, list_supported_documents
from src.app.utils.io_utils import ensure_directory, file_sha256, format_json, read_json, read_jsonl


def scan_training_documents(paths: AppPaths) -> dict[str, list[dict[str, Any]]]:
    if not paths.reference_docs_dir.exists() or not paths.reference_docs_dir.is_dir():
        raise FileNotFoundError("Не найдена папка docs/reference. Создайте её и добавьте нормативные документы.")

    reference_paths = list_supported_documents(paths.reference_docs_dir)
    if not reference_paths:
        raise FileNotFoundError("В docs/reference нет поддерживаемых документов для индексации.")

    example_paths = list_supported_documents(paths.example_docs_dir)
    return {
        "reference_documents": _scan_document_batch(reference_paths, paths.project_root, source_kind="reference"),
        "example_documents": _scan_document_batch(example_paths, paths.project_root, source_kind="example"),
    }


def plan_training_index_update(
    paths: AppPaths,
    reference_documents: list[dict[str, Any]],
    example_documents: list[dict[str, Any]],
    embedding_model: str,
    full_rebuild: bool = False,
) -> dict[str, Any]:
    current_documents = sorted(reference_documents + example_documents, key=_document_sort_key)
    runtime_state = _inspect_runtime_for_sync(paths, embedding_model)

    if full_rebuild:
        return _build_full_rebuild_plan(current_documents, reason="requested")

    if not runtime_state["can_sync"]:
        return _build_full_rebuild_plan(current_documents, reason=str(runtime_state["reason"]))

    existing_manifest_records = runtime_state["manifest_records"]
    existing_index = {_manifest_key(record): record for record in existing_manifest_records}
    current_index = {_document_key(document): document for document in current_documents}

    current_keys = set(current_index)
    existing_keys = set(existing_index)

    added_keys = sorted(current_keys - existing_keys)
    removed_keys = sorted(existing_keys - current_keys)
    shared_keys = sorted(current_keys & existing_keys)

    changed_keys: list[tuple[str, str]] = []
    unchanged_keys: list[tuple[str, str]] = []
    for key in shared_keys:
        if current_index[key]["file_sha256"] != existing_index[key]["file_sha256"]:
            changed_keys.append(key)
        else:
            unchanged_keys.append(key)

    added_documents = [current_index[key] for key in added_keys]
    changed_documents = [current_index[key] for key in changed_keys]
    records_to_remove = [existing_index[key] for key in removed_keys + changed_keys]
    unchanged_manifest_records = [existing_index[key] for key in unchanged_keys]

    return {
        "index_mode": "sync",
        "full_rebuild_reason": None,
        "current_documents": current_documents,
        "documents_to_index": sorted(added_documents + changed_documents, key=_document_sort_key),
        "reference_documents_to_index": sorted(
            [item for item in added_documents + changed_documents if item["source_kind"] == "reference"],
            key=_document_sort_key,
        ),
        "example_documents_to_index": sorted(
            [item for item in added_documents + changed_documents if item["source_kind"] == "example"],
            key=_document_sort_key,
        ),
        "records_to_remove": records_to_remove,
        "unchanged_manifest_records": unchanged_manifest_records,
        "sync_stats": {
            "added": len(added_documents),
            "changed": len(changed_documents),
            "removed": len(removed_keys),
            "unchanged": len(unchanged_keys),
            "processed_documents": len(added_documents) + len(changed_documents) + len(removed_keys),
        },
    }


def build_training_chunks(
    paths: AppPaths,
    reference_documents: list[dict[str, Any]],
    example_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_manifest, reference_chunks = build_knowledge_chunks(
        [item["path"] for item in reference_documents],
        base_path=paths.project_root,
        source_kind="reference",
    )
    example_manifest, example_chunks = build_knowledge_chunks(
        [item["path"] for item in example_documents],
        base_path=paths.project_root,
        source_kind="example",
    )

    manifest_records = _attach_scan_metadata(reference_manifest + example_manifest, reference_documents + example_documents)
    return {
        "manifest_records": manifest_records,
        "reference_chunks": reference_chunks,
        "example_chunks": example_chunks,
    }


def index_training_chunks(
    paths: AppPaths,
    index_plan: dict[str, Any],
    manifest_records: list[dict[str, Any]],
    reference_chunks: list[dict[str, Any]],
    example_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    ensure_directory(paths.vector_db_dir)
    ensure_directory(paths.runtime_dir)

    client = get_persistent_client(paths)
    if index_plan["index_mode"] == "full_rebuild":
        reference_collection = reset_collection(client, REFERENCE_COLLECTION_NAME)
        example_collection = reset_collection(client, EXAMPLE_COLLECTION_NAME)
    else:
        reference_collection = get_collection(client, REFERENCE_COLLECTION_NAME)
        example_collection = get_collection(client, EXAMPLE_COLLECTION_NAME)
        _delete_manifest_records(reference_collection, example_collection, index_plan["records_to_remove"])

    reference_embeddings = embed_texts([item["text"] for item in reference_chunks]) if reference_chunks else []
    example_embeddings = embed_texts([item["text"] for item in example_chunks]) if example_chunks else []

    if reference_chunks:
        add_embeddings(reference_collection, reference_chunks, reference_embeddings)
    if example_chunks:
        add_embeddings(example_collection, example_chunks, example_embeddings)

    final_manifest_records = _compose_final_manifest(index_plan, manifest_records)
    settings = get_openrouter_settings(require_chat_model=True)
    metadata = {
        "snapshot_schema": SNAPSHOT_SCHEMA_VERSION,
        "vector_db_engine": "chroma",
        "embedding_model": settings["embedding_model"],
        "chat_model": settings["chat_model"],
        "reference_document_count": sum(1 for item in final_manifest_records if item["source_kind"] == "reference"),
        "example_document_count": sum(1 for item in final_manifest_records if item["source_kind"] == "example"),
        "reference_chunk_count": sum(
            int(item.get("chunk_count", 0) or 0) for item in final_manifest_records if item["source_kind"] == "reference"
        ),
        "example_chunk_count": sum(
            int(item.get("chunk_count", 0) or 0) for item in final_manifest_records if item["source_kind"] == "example"
        ),
        "collections": {
            "reference": REFERENCE_COLLECTION_NAME,
            "examples": EXAMPLE_COLLECTION_NAME,
        },
        "index_mode": index_plan["index_mode"],
        "sync_stats": dict(index_plan["sync_stats"]),
    }
    if index_plan.get("full_rebuild_reason"):
        metadata["full_rebuild_reason"] = index_plan["full_rebuild_reason"]

    _write_runtime_snapshot(paths, metadata, final_manifest_records)
    return metadata


def load_indexed_documents_status(paths: AppPaths) -> dict[str, Any]:
    empty_payload = {
        "is_ready": False,
        "message": "Векторная БД ещё не построена. Выполните train, чтобы увидеть список документов.",
        "documents": [],
        "reference_document_count": 0,
        "example_document_count": 0,
        "reference_chunk_count": 0,
        "example_chunk_count": 0,
    }

    if not paths.vector_meta_path.exists() or not paths.manifest_path.exists():
        return empty_payload

    metadata = read_json(paths.vector_meta_path)
    if metadata.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        empty_payload["message"] = "Обнаружены legacy runtime-данные. Выполните train заново."
        return empty_payload

    documents = _read_manifest_documents(read_jsonl(paths.manifest_path))
    reference_document_count = sum(1 for item in documents if item["source_kind"] == "reference")
    example_document_count = sum(1 for item in documents if item["source_kind"] == "example")
    reference_chunk_count = sum(item["chunk_count"] for item in documents if item["source_kind"] == "reference")
    example_chunk_count = sum(item["chunk_count"] for item in documents if item["source_kind"] == "example")

    return {
        "is_ready": True,
        "message": (
            f"В ВБД сейчас {len(documents)} документ(ов): "
            f"reference={reference_document_count}, examples={example_document_count}."
        ),
        "documents": documents,
        "reference_document_count": reference_document_count,
        "example_document_count": example_document_count,
        "reference_chunk_count": reference_chunk_count,
        "example_chunk_count": example_chunk_count,
    }


def load_indexed_chunks_status(paths: AppPaths, source_path: str | None = None) -> dict[str, Any]:
    empty_payload = {
        "is_ready": False,
        "message": "Векторная БД ещё не построена. Выполните train, чтобы открыть инспектор чанков.",
        "chunks": [],
        "chunk_count": 0,
        "document_options": [],
        "selected_source_path": source_path or "",
    }

    if not paths.vector_meta_path.exists() or not paths.manifest_path.exists():
        return empty_payload

    metadata = read_json(paths.vector_meta_path)
    if metadata.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        empty_payload["message"] = "Обнаружены legacy runtime-данные. Выполните train заново."
        return empty_payload

    document_options = [item["path"] for item in _read_manifest_documents(read_jsonl(paths.manifest_path))]
    requested_source_path = (source_path or "").strip()

    client = get_persistent_client(paths)
    chunks = _load_all_indexed_chunks(client)
    if requested_source_path:
        chunks = [item for item in chunks if item["source_path"] == requested_source_path]

    return {
        "is_ready": True,
        "message": (
            f"Загружено чанков: {len(chunks)}"
            + (f" для {requested_source_path}." if requested_source_path else ".")
        ),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "document_options": document_options,
        "selected_source_path": requested_source_path,
    }


def load_runtime_metadata(paths: AppPaths) -> dict[str, Any]:
    if not paths.vector_meta_path.exists():
        raise FileNotFoundError("Не найден data/runtime/vector_meta.json. Сначала выполните train.")

    metadata = read_json(paths.vector_meta_path)
    if metadata.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError(
            "Обнаружены legacy runtime-данные. Выполните train заново, чтобы собрать multimodal RAG индекс."
        )
    return metadata


def _scan_document_batch(document_paths: list[Path], project_root: Path, source_kind: str) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "relative_path": _make_relative_path(path, project_root),
            "source_kind": source_kind,
            "parser": SUPPORTED_EXTENSIONS[path.suffix.lower()],
            "file_sha256": file_sha256(path),
        }
        for path in document_paths
    ]


def _inspect_runtime_for_sync(paths: AppPaths, embedding_model: str) -> dict[str, Any]:
    if not paths.vector_meta_path.exists() or not paths.manifest_path.exists():
        return {"can_sync": False, "reason": "missing_runtime", "manifest_records": []}

    try:
        metadata = read_json(paths.vector_meta_path)
    except Exception:
        return {"can_sync": False, "reason": "invalid_runtime_metadata", "manifest_records": []}

    if metadata.get("snapshot_schema") != SNAPSHOT_SCHEMA_VERSION:
        return {"can_sync": False, "reason": "legacy_runtime_metadata", "manifest_records": []}

    if str(metadata.get("embedding_model", "")).strip() != embedding_model:
        return {"can_sync": False, "reason": "embedding_model_changed", "manifest_records": []}

    try:
        manifest_records = _normalize_manifest_records(read_jsonl(paths.manifest_path))
    except ValueError as exc:
        return {"can_sync": False, "reason": str(exc), "manifest_records": []}

    client = get_persistent_client(paths)
    try:
        get_collection(client, REFERENCE_COLLECTION_NAME)
        get_collection(client, EXAMPLE_COLLECTION_NAME)
    except Exception:
        return {"can_sync": False, "reason": "missing_collection", "manifest_records": manifest_records}

    return {
        "can_sync": True,
        "reason": None,
        "manifest_records": manifest_records,
    }


def _load_all_indexed_chunks(client: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for collection_name in (REFERENCE_COLLECTION_NAME, EXAMPLE_COLLECTION_NAME):
        collection = get_collection(client, collection_name)
        chunks.extend(list_collection_records(collection))

    chunks.sort(
        key=lambda item: (
            str(item.get("source_kind", "")),
            str(item.get("source_path", "")),
            int(item.get("chunk_index", 0) or 0),
        )
    )
    return chunks


def _read_manifest_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("path", "")).strip()
        if not path:
            continue
        documents.append(
            {
                "path": path,
                "source_kind": str(record.get("source_kind", "")).strip() or "unknown",
                "parser": str(record.get("parser", "")).strip() or "unknown",
                "chunk_count": int(record.get("chunk_count", 0) or 0),
            }
        )
    documents.sort(key=lambda item: (item["source_kind"], item["path"]))
    return documents


def _normalize_manifest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        path = str(record.get("path", "")).strip()
        source_kind = str(record.get("source_kind", "")).strip()
        parser = str(record.get("parser", "")).strip()
        file_hash = str(record.get("file_sha256", "")).strip()
        chunk_count = record.get("chunk_count", 0)

        if not path or not source_kind or not parser:
            raise ValueError("invalid_manifest_record")
        if not file_hash:
            raise ValueError("manifest_missing_file_sha256")

        try:
            normalized_chunk_count = int(chunk_count or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_manifest_chunk_count") from exc

        normalized.append(
            {
                "path": path,
                "source_kind": source_kind,
                "parser": parser,
                "chunk_count": normalized_chunk_count,
                "file_sha256": file_hash,
            }
        )
    return sorted(normalized, key=_manifest_sort_key)


def _build_full_rebuild_plan(current_documents: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "index_mode": "full_rebuild",
        "full_rebuild_reason": reason,
        "current_documents": current_documents,
        "documents_to_index": current_documents,
        "reference_documents_to_index": [item for item in current_documents if item["source_kind"] == "reference"],
        "example_documents_to_index": [item for item in current_documents if item["source_kind"] == "example"],
        "records_to_remove": [],
        "unchanged_manifest_records": [],
        "sync_stats": {
            "added": len(current_documents),
            "changed": 0,
            "removed": 0,
            "unchanged": 0,
            "processed_documents": len(current_documents),
        },
    }


def _attach_scan_metadata(
    manifest_records: list[dict[str, Any]],
    scanned_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scan_index = {_document_key(item): item for item in scanned_documents}
    hydrated_records: list[dict[str, Any]] = []
    for record in manifest_records:
        key = _manifest_key(record)
        document = scan_index.get(key)
        if document is None:
            raise KeyError(f"Missing scan metadata for {record['source_kind']}:{record['path']}")
        hydrated = dict(record)
        hydrated["file_sha256"] = document["file_sha256"]
        hydrated_records.append(hydrated)
    return sorted(hydrated_records, key=_manifest_sort_key)


def _delete_manifest_records(reference_collection: Any, example_collection: Any, records: list[dict[str, Any]]) -> None:
    reference_ids: list[str] = []
    example_ids: list[str] = []

    for record in records:
        ids = _build_chunk_ids(record)
        if not ids:
            continue
        if record["source_kind"] == "reference":
            reference_ids.extend(ids)
        elif record["source_kind"] == "example":
            example_ids.extend(ids)

    delete_embeddings(reference_collection, reference_ids)
    delete_embeddings(example_collection, example_ids)


def _build_chunk_ids(record: dict[str, Any]) -> list[str]:
    chunk_count = int(record.get("chunk_count", 0) or 0)
    path = str(record.get("path", "")).strip()
    source_kind = str(record.get("source_kind", "")).strip()
    if not path or not source_kind or chunk_count <= 0:
        return []
    return [f"{source_kind}:{path}::chunk-{index}" for index in range(chunk_count)]


def _compose_final_manifest(index_plan: dict[str, Any], updated_manifest_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if index_plan["index_mode"] == "full_rebuild":
        return sorted(updated_manifest_records, key=_manifest_sort_key)
    return sorted(index_plan["unchanged_manifest_records"] + updated_manifest_records, key=_manifest_sort_key)


def _write_runtime_snapshot(
    paths: AppPaths,
    metadata: dict[str, Any],
    manifest_records: list[dict[str, Any]],
) -> None:
    ensure_directory(paths.runtime_dir)
    manifest_payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest_records)
    metadata_payload = format_json(metadata) + "\n"
    _atomic_write_text(paths.manifest_path, manifest_payload)
    _atomic_write_text(paths.vector_meta_path, metadata_payload)


def _atomic_write_text(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    temp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _make_relative_path(path: Path, base_path: Path) -> str:
    try:
        return path.resolve().relative_to(base_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _document_key(document: dict[str, Any]) -> tuple[str, str]:
    return str(document["source_kind"]), str(document["relative_path"])


def _manifest_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["source_kind"]), str(record["path"])


def _document_sort_key(document: dict[str, Any]) -> tuple[str, str]:
    return str(document["source_kind"]), str(document["relative_path"])


def _manifest_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["source_kind"]), str(record["path"])
