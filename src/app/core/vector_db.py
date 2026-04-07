from __future__ import annotations

import json
from typing import Any

from src.app.core.config import AppPaths


def get_persistent_client(paths: AppPaths) -> Any:
    chromadb = _import_chromadb()
    return chromadb.PersistentClient(path=str(paths.vector_db_dir))


def reset_collection(client: Any, name: str) -> Any:
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def get_collection(client: Any, name: str) -> Any:
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        raise RuntimeError(f"Chroma collection is missing: {name}. Run train first.") from exc


def add_embeddings(collection: Any, records: list[dict[str, Any]], embeddings: list[list[float]], batch_size: int = 128) -> None:
    if len(records) != len(embeddings):
        raise ValueError("Records count must match embeddings count.")

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_embeddings = embeddings[start : start + batch_size]
        collection.add(
            ids=[record["chunk_id"] for record in batch_records],
            documents=[record["text"] for record in batch_records],
            metadatas=[_serialize_metadata(record) for record in batch_records],
            embeddings=batch_embeddings,
        )


def delete_embeddings(collection: Any, ids: list[str], batch_size: int = 256) -> None:
    if not ids:
        return

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        collection.delete(ids=batch_ids)


def list_collection_records(collection: Any) -> list[dict[str, Any]]:
    payload = collection.get(include=["documents", "metadatas"])
    ids = payload.get("ids", [])
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])

    records: list[dict[str, Any]] = []
    for chunk_id, document, metadata in zip(ids, documents, metadatas):
        records.append(_deserialize_hit(chunk_id, document or "", metadata or {}, distance=None))
    return records


def query_collection(collection: Any, query_embeddings: list[list[float]], top_k: int) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    for query_embedding in query_embeddings:
        payload = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            item = _deserialize_hit(chunk_id, document or "", metadata or {}, distance)
            existing = aggregated.get(chunk_id)
            if existing is None or item["score"] > existing["score"]:
                aggregated[chunk_id] = item

    ranked = sorted(aggregated.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _serialize_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": str(record.get("source_path", "")),
        "source_kind": str(record.get("source_kind", "")),
        "document_type": str(record.get("document_type", "")),
        "section_context": str(record.get("section_context", "")),
        "mentioned_codes_json": json.dumps(record.get("mentioned_codes", []), ensure_ascii=False),
        "chunk_index": int(record.get("chunk_index", 0)),
    }


def _deserialize_hit(chunk_id: str, document: str, metadata: dict[str, Any], distance: float | int | None) -> dict[str, Any]:
    score = 0.0
    if distance is not None:
        score = max(0.0, 1.0 - float(distance))

    mentioned_codes: list[str] = []
    raw_codes = metadata.get("mentioned_codes_json", "[]")
    if isinstance(raw_codes, str):
        try:
            parsed = json.loads(raw_codes)
            if isinstance(parsed, list):
                mentioned_codes = [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            mentioned_codes = []

    return {
        "chunk_id": chunk_id,
        "text": document,
        "source_path": str(metadata.get("source_path", "")),
        "source_kind": str(metadata.get("source_kind", "")),
        "document_type": str(metadata.get("document_type", "")),
        "section_context": str(metadata.get("section_context", "")),
        "mentioned_codes": mentioned_codes,
        "chunk_index": int(metadata.get("chunk_index", 0) or 0),
        "score": score,
    }


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("chromadb is not installed. Install dependencies from requirements.txt.") from exc
    return chromadb
