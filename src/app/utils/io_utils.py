from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def generate_run_id() -> str:
    return str(uuid.uuid4())


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(read_text_file(path))


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    path.write_text(format_json(payload) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_event(
    events_path: Path,
    run_id: str,
    mode: str,
    stage: str,
    status: str,
    started_at: str | None,
    finished_at: str | None,
    payload: dict[str, Any] | None = None,
) -> None:
    append_jsonl(
        events_path,
        {
            "run_id": run_id,
            "mode": mode,
            "stage": stage,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "payload": payload or {},
        },
    )


def print_stdout(message: str) -> None:
    _safe_print(message, sys.stdout)


def print_stderr(message: str) -> None:
    _safe_print(message, sys.stderr)


def _safe_print(message: str, stream: Any) -> None:
    try:
        print(message, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        buffer = getattr(stream, "buffer", None)
        payload = (message + "\n").encode(encoding, errors="replace")
        if buffer is not None:
            buffer.write(payload)
            buffer.flush()
            return
        stream.write(payload.decode(encoding, errors="replace"))
        stream.flush()
