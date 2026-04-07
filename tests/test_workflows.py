from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.core.config import EXAMPLE_COLLECTION_NAME, REFERENCE_COLLECTION_NAME, get_app_paths
from src.app.core.vector_db import get_collection, get_persistent_client
from src.app.services import workflows
from src.app.services.train_service import load_indexed_chunks_status, load_indexed_documents_status


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.records: dict[str, dict[str, object]] = {}
        self.delete_calls: list[list[str]] = []

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]],
        embeddings: list[list[float]],
    ) -> None:
        for item_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            if item_id in self.records:
                raise ValueError(f"Duplicate id: {item_id}")
            self.records[item_id] = {
                "id": item_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def delete(self, ids: list[str] | None = None) -> None:
        batch_ids = list(ids or [])
        self.delete_calls.append(batch_ids)
        for item_id in batch_ids:
            self.records.pop(item_id, None)

    def count(self) -> int:
        return len(self.records)

    def get(self, include: list[str] | None = None) -> dict[str, list[object]]:
        del include
        ordered = sorted(self.records.values(), key=lambda item: str(item["id"]))
        return {
            "ids": [str(item["id"]) for item in ordered],
            "documents": [str(item["document"]) for item in ordered],
            "metadatas": [dict(item["metadata"]) for item in ordered],  # type: ignore[arg-type]
        }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, list[list[object]]]:
        del include
        ids: list[list[str]] = []
        documents: list[list[str]] = []
        metadatas: list[list[dict[str, object]]] = []
        distances: list[list[float]] = []

        for query_embedding in query_embeddings:
            ranked = sorted(
                self.records.values(),
                key=lambda item: _cosine_distance(query_embedding, item["embedding"]),  # type: ignore[arg-type]
            )[:n_results]
            ids.append([str(item["id"]) for item in ranked])
            documents.append([str(item["document"]) for item in ranked])
            metadatas.append([dict(item["metadata"]) for item in ranked])  # type: ignore[arg-type]
            distances.append([_cosine_distance(query_embedding, item["embedding"]) for item in ranked])  # type: ignore[arg-type]

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }


class FakePersistentClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.delete_collection_calls: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.delete_collection_calls.append(name)
        self.collections.pop(name, None)

    def get_or_create_collection(self, name: str, metadata: dict[str, object] | None = None) -> FakeCollection:
        del metadata
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def get_collection(self, name: str) -> FakeCollection:
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]


class FakeChromaModule:
    def __init__(self) -> None:
        self.clients: dict[str, FakePersistentClient] = {}

    def PersistentClient(self, path: str) -> FakePersistentClient:
        if path not in self.clients:
            self.clients[path] = FakePersistentClient()
        return self.clients[path]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    similarity = numerator / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def fake_embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    del batch_size
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        if "лошад" in lowered or "0101210000" in lowered or "племенн" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        elif "мотоцикл" in lowered:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.1, 0.1, 0.9])
    return vectors


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="tnved_multimodal_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.project_root = Path(self.temp_dir)
        self.env_patcher = patch.dict(os.environ, {"TNVED_PROJECT_ROOT": self.temp_dir})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)
        self.fake_chroma_module = FakeChromaModule()
        self.chroma_patcher = patch("src.app.core.vector_db._import_chromadb", return_value=self.fake_chroma_module)
        self.chroma_patcher.start()
        self.addCleanup(self.chroma_patcher.stop)

        paths = get_app_paths()
        paths.reference_docs_dir.mkdir(parents=True, exist_ok=True)
        paths.example_docs_dir.mkdir(parents=True, exist_ok=True)

    def _settings(self, embedding_model: str = "embedding-model", chat_model: str = "vision-model") -> dict[str, str]:
        return {
            "api_key": "test-key",
            "chat_model": chat_model,
            "embedding_model": embedding_model,
            "base_url": "https://example.invalid",
        }

    @contextmanager
    def _train_context(
        self,
        embedding_model: str = "embedding-model",
        chat_model: str = "vision-model",
    ) -> Iterator[None]:
        settings = self._settings(embedding_model=embedding_model, chat_model=chat_model)
        with (
            patch("src.app.services.workflows.get_openrouter_settings", return_value=settings),
            patch("src.app.services.train_service.get_openrouter_settings", return_value=settings),
        ):
            yield

    def _reference_collection(self) -> FakeCollection:
        paths = get_app_paths()
        client = get_persistent_client(paths)
        return get_collection(client, REFERENCE_COLLECTION_NAME)  # type: ignore[return-value]

    def _example_collection(self) -> FakeCollection:
        paths = get_app_paths()
        client = get_persistent_client(paths)
        return get_collection(client, EXAMPLE_COLLECTION_NAME)  # type: ignore[return-value]

    def _delete_collection_call_count(self) -> int:
        paths = get_app_paths()
        client = get_persistent_client(paths)
        return len(client.delete_collection_calls)  # type: ignore[attr-defined]

    def test_train_indexes_reference_and_examples_into_chroma(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади для разведения.",
            encoding="utf-8",
        )
        (paths.example_docs_dir / "case.txt").write_text(
            "Пример подбора ТН ВЭД 0101210000 для чистопородных племенных лошадей.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "full_rebuild")
        self.assertEqual(metadata["sync_stats"]["processed_documents"], 2)
        self.assertEqual(metadata["reference_document_count"], 1)
        self.assertEqual(metadata["example_document_count"], 1)
        self.assertTrue(paths.vector_meta_path.exists())
        self.assertTrue(paths.manifest_path.exists())

        manifest_records = [json.loads(line) for line in paths.manifest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(manifest_records), 2)
        self.assertTrue(all(record.get("file_sha256") for record in manifest_records))
        self.assertGreater(self._reference_collection().count(), 0)
        self.assertGreater(self._example_collection().count(), 0)

    def test_train_requires_docs_reference_and_ignores_legacy_docs_dir(self) -> None:
        legacy_dir = self.project_root / "simple_tnved_app" / "DOCS"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / "legacy.txt").write_text("Код ТН ВЭД 0101210000", encoding="utf-8")

        shutil.rmtree(get_app_paths().reference_docs_dir)

        with patch("src.app.services.workflows.get_openrouter_settings", return_value=self._settings()):
            with self.assertRaises(FileNotFoundError) as context:
                workflows.run_train_workflow()

        self.assertIn("docs/reference", str(context.exception))

    def test_load_indexed_documents_status_reads_current_manifest(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади.",
            encoding="utf-8",
        )
        (paths.example_docs_dir / "case.txt").write_text(
            "Пример подбора ТН ВЭД 0101210000.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        status = load_indexed_documents_status(paths)

        self.assertTrue(status["is_ready"])
        self.assertEqual(status["reference_document_count"], 1)
        self.assertEqual(status["example_document_count"], 1)
        self.assertGreaterEqual(status["reference_chunk_count"], 1)
        self.assertGreaterEqual(status["example_chunk_count"], 1)
        self.assertEqual(
            {item["path"] for item in status["documents"]},
            {"docs/reference/official.txt", "docs/examples/case.txt"},
        )

    def test_load_indexed_chunks_status_reads_chunks_and_supports_filter(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади.",
            encoding="utf-8",
        )
        (paths.example_docs_dir / "case.txt").write_text(
            "Пример подбора ТН ВЭД 0101210000 для племенных лошадей.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        all_chunks = load_indexed_chunks_status(paths)
        reference_chunks = load_indexed_chunks_status(paths, source_path="docs/reference/official.txt")

        self.assertTrue(all_chunks["is_ready"])
        self.assertGreaterEqual(all_chunks["chunk_count"], 2)
        self.assertIn("docs/reference/official.txt", all_chunks["document_options"])
        self.assertIn("docs/examples/case.txt", all_chunks["document_options"])
        self.assertEqual(reference_chunks["selected_source_path"], "docs/reference/official.txt")
        self.assertGreaterEqual(reference_chunks["chunk_count"], 1)
        self.assertTrue(all(chunk["source_path"] == "docs/reference/official.txt" for chunk in reference_chunks["chunks"]))

    def test_load_indexed_chunks_status_accepts_manifest_without_file_sha256(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        paths.manifest_path.write_text(
            json.dumps(
                {
                    "path": "docs/reference/official.txt",
                    "source_kind": "reference",
                    "parser": "txt",
                    "chunk_count": 1,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        status = load_indexed_chunks_status(paths)

        self.assertTrue(status["is_ready"])
        self.assertEqual(status["document_options"], ["docs/reference/official.txt"])
        self.assertGreaterEqual(status["chunk_count"], 1)

    def test_train_sync_adds_new_document_without_resetting_collections(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        delete_calls_before = self._delete_collection_call_count()
        reference_count_before = self._reference_collection().count()

        (paths.reference_docs_dir / "supplement.txt").write_text(
            "Дополнительное пояснение про живых племенных лошадей.",
            encoding="utf-8",
        )
        embed_mock = Mock(side_effect=fake_embed_texts)
        with self._train_context(), patch("src.app.services.train_service.embed_texts", embed_mock):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "sync")
        self.assertEqual(metadata["sync_stats"]["added"], 1)
        self.assertEqual(metadata["sync_stats"]["changed"], 0)
        self.assertEqual(metadata["sync_stats"]["removed"], 0)
        self.assertEqual(metadata["sync_stats"]["unchanged"], 1)
        self.assertEqual(metadata["sync_stats"]["processed_documents"], 1)
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before)
        self.assertEqual(embed_mock.call_count, 1)
        self.assertGreater(self._reference_collection().count(), reference_count_before)

    def test_train_sync_reindexes_changed_document_and_removes_old_chunks(self) -> None:
        paths = get_app_paths()
        document_path = paths.reference_docs_dir / "official.txt"
        document_path.write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        delete_calls_before = self._delete_collection_call_count()
        document_path.write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади для разведения.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "sync")
        self.assertEqual(metadata["sync_stats"]["changed"], 1)
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before)
        reference_collection = self._reference_collection()
        self.assertEqual(reference_collection.count(), 1)
        updated_record = reference_collection.records["reference:docs/reference/official.txt::chunk-0"]
        self.assertIn("Чистопородные", str(updated_record["document"]))
        self.assertTrue(reference_collection.delete_calls)

    def test_train_sync_removes_deleted_document(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )
        removable = paths.example_docs_dir / "case.txt"
        removable.write_text(
            "Пример классификации для живых животных.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        removable.unlink()
        delete_calls_before = self._delete_collection_call_count()
        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "sync")
        self.assertEqual(metadata["sync_stats"]["removed"], 1)
        self.assertEqual(metadata["sync_stats"]["processed_documents"], 1)
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before)
        self.assertEqual(self._example_collection().count(), 0)
        manifest_records = [json.loads(line) for line in paths.manifest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual({record["path"] for record in manifest_records}, {"docs/reference/official.txt"})

    def test_train_sync_no_op_skips_reembedding_and_reset(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        delete_calls_before = self._delete_collection_call_count()
        embed_mock = Mock(side_effect=fake_embed_texts)
        with self._train_context(chat_model="vision-model-v2"), patch("src.app.services.train_service.embed_texts", embed_mock):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "sync")
        self.assertEqual(metadata["sync_stats"]["added"], 0)
        self.assertEqual(metadata["sync_stats"]["changed"], 0)
        self.assertEqual(metadata["sync_stats"]["removed"], 0)
        self.assertEqual(metadata["sync_stats"]["unchanged"], 1)
        self.assertEqual(metadata["sync_stats"]["processed_documents"], 0)
        self.assertEqual(embed_mock.call_count, 0)
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before)
        metadata_payload = json.loads(paths.vector_meta_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata_payload["chat_model"], "vision-model-v2")

    def test_train_falls_back_to_full_rebuild_for_legacy_manifest(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        delete_calls_before = self._delete_collection_call_count()
        paths.manifest_path.write_text(
            json.dumps(
                {
                    "path": "docs/reference/official.txt",
                    "source_kind": "reference",
                    "parser": "txt",
                    "chunk_count": 1,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "full_rebuild")
        self.assertEqual(metadata["full_rebuild_reason"], "manifest_missing_file_sha256")
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before + 2)

    def test_train_falls_back_to_full_rebuild_when_embedding_model_changes(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Племенные лошади.",
            encoding="utf-8",
        )

        with self._train_context(embedding_model="embedding-v1"), patch(
            "src.app.services.train_service.embed_texts",
            side_effect=fake_embed_texts,
        ):
            workflows.run_train_workflow()

        delete_calls_before = self._delete_collection_call_count()
        with self._train_context(embedding_model="embedding-v2"), patch(
            "src.app.services.train_service.embed_texts",
            side_effect=fake_embed_texts,
        ):
            metadata = workflows.run_train_workflow()

        self.assertEqual(metadata["index_mode"], "full_rebuild")
        self.assertEqual(metadata["full_rebuild_reason"], "embedding_model_changed")
        self.assertEqual(self._delete_collection_call_count(), delete_calls_before + 2)

    def test_classify_returns_best_match_after_sync_update(self) -> None:
        paths = get_app_paths()
        (paths.reference_docs_dir / "official.txt").write_text(
            "Код ТН ВЭД 0101210000. Чистопородные племенные лошади для разведения. Живые животные.",
            encoding="utf-8",
        )

        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            workflows.run_train_workflow()

        (paths.example_docs_dir / "case.txt").write_text(
            "Пример подбора ТН ВЭД 0101210000 для чистопородных племенных лошадей.",
            encoding="utf-8",
        )
        with self._train_context(), patch("src.app.services.train_service.embed_texts", side_effect=fake_embed_texts):
            sync_metadata = workflows.run_train_workflow()

        self.assertEqual(sync_metadata["index_mode"], "sync")
        self.assertEqual(sync_metadata["sync_stats"]["added"], 1)

        analysis_payload = {
            "summary": "Чистопородные племенные лошади",
            "intended_use": "Разведение",
            "material_or_composition": "Живые животные",
            "key_features": ["племенные", "чистопородные"],
            "search_queries": ["чистопородные племенные лошади", "лошади для разведения"],
            "uncertainty": "Низкая",
            "missing_information": [],
        }
        classification_payload = {
            "best_match": {
                "code": "0101210000",
                "title_or_label": "Чистопородные племенные лошади",
                "confidence": 0.94,
                "reasoning": "Reference evidence прямо указывает на племенных лошадей.",
                "evidence_ids": ["reference:docs/reference/official.txt::chunk-0"],
            },
            "candidates": [
                {
                    "code": "0101210000",
                    "title_or_label": "Чистопородные племенные лошади",
                    "confidence": 0.94,
                    "reasoning": "Reference evidence прямо указывает на племенных лошадей.",
                    "evidence_ids": ["reference:docs/reference/official.txt::chunk-0"],
                },
                {
                    "code": "0101291000",
                    "title_or_label": "Прочие лошади",
                    "confidence": 0.35,
                    "reasoning": "Возможная альтернатива, если племенное назначение не подтвердится.",
                    "evidence_ids": ["example:docs/examples/case.txt::chunk-0"],
                },
            ],
            "warnings": [],
        }

        with (
            patch("src.app.services.workflows.get_openrouter_settings", return_value=self._settings()),
            patch("src.app.services.retrieval_service.embed_texts", side_effect=fake_embed_texts),
            patch("src.app.services.input_analysis_service.chat_json", return_value=analysis_payload),
            patch("src.app.services.classification_service.chat_json", return_value=classification_payload),
        ):
            payload = workflows.run_classify_workflow(
                raw_text="Племенные чистопородные лошади для разведения",
                top_k=3,
            )

        self.assertEqual(payload["result"]["best_match"]["code"], "0101210000")
        self.assertGreaterEqual(len(payload["result"]["candidates"]), 1)
        self.assertGreaterEqual(len(payload["result"]["retrieval"]["reference"]), 1)
        self.assertIn("Лучший код: 0101210000", payload["summary"])

    def test_classify_rejects_legacy_runtime_metadata(self) -> None:
        paths = get_app_paths()
        paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        paths.vector_meta_path.write_text(json.dumps({"snapshot_schema": "legacy"}, ensure_ascii=False), encoding="utf-8")

        with patch("src.app.services.workflows.get_openrouter_settings", return_value=self._settings()):
            with self.assertRaises(RuntimeError) as context:
                workflows.run_classify_workflow(raw_text="лошади", top_k=3)

        self.assertIn("legacy runtime", str(context.exception))
