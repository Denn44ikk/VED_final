from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TOP_K_CHOICES = {3, 4}
SNAPSHOT_SCHEMA_VERSION = "multimodal-rag-v1"
REFERENCE_COLLECTION_NAME = "reference_chunks"
EXAMPLE_COLLECTION_NAME = "example_chunks"


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    docs_dir: Path
    reference_docs_dir: Path
    example_docs_dir: Path
    data_dir: Path
    vector_db_dir: Path
    runtime_dir: Path
    vector_meta_path: Path
    manifest_path: Path
    events_log_path: Path


def get_project_root() -> Path:
    raw_root = os.getenv("TNVED_PROJECT_ROOT", "").strip()
    if raw_root:
        return Path(raw_root).resolve()
    return Path(__file__).resolve().parents[3]


def get_app_paths() -> AppPaths:
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    data_dir = project_root / "data"
    runtime_dir = data_dir / "runtime"
    return AppPaths(
        project_root=project_root,
        docs_dir=docs_dir,
        reference_docs_dir=docs_dir / "reference",
        example_docs_dir=docs_dir / "examples",
        data_dir=data_dir,
        vector_db_dir=data_dir / "vector_db" / "chroma",
        runtime_dir=runtime_dir,
        vector_meta_path=runtime_dir / "vector_meta.json",
        manifest_path=runtime_dir / "manifest.jsonl",
        events_log_path=runtime_dir / "events.log.jsonl",
    )
