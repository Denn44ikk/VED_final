from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
DEFAULT_EMBEDDING_TIMEOUT = 120
DEFAULT_CHAT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 2.0


def get_openrouter_settings(require_chat_model: bool = False) -> dict[str, str]:
    _load_project_env()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    chat_model = os.getenv("OPENROUTER_CHAT_MODEL", "").strip()
    embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

    missing = []
    if not api_key:
        missing.append("OPENROUTER_API_KEY")
    if not embedding_model:
        missing.append("OPENROUTER_EMBEDDING_MODEL")
    if require_chat_model and not chat_model:
        missing.append("OPENROUTER_CHAT_MODEL")

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required OpenRouter environment variables: {joined}")

    return {
        "api_key": api_key,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "base_url": base_url.rstrip("/"),
    }


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []

    settings = get_openrouter_settings()
    requests = _import_requests()
    session = requests.Session()
    all_embeddings: list[list[float]] = []
    effective_batch_size = _get_env_int("OPENROUTER_EMBEDDING_BATCH_SIZE", batch_size)
    effective_batch_size = max(1, effective_batch_size)
    timeout = _get_env_int("OPENROUTER_EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT)

    for start in range(0, len(texts), effective_batch_size):
        batch = texts[start : start + effective_batch_size]
        payload = post_json_with_retries(
            session,
            f"{settings['base_url']}/embeddings",
            headers=build_headers(settings["api_key"]),
            json_payload={
                "model": settings["embedding_model"],
                "input": batch,
            },
            timeout=timeout,
        )
        items = payload.get("data", [])
        embeddings = [item["embedding"] for item in items]
        if len(embeddings) != len(batch):
            raise RuntimeError("OpenRouter embeddings response size does not match the request batch size.")
        all_embeddings.extend(embeddings)

    return all_embeddings


def chat_json(messages: list[dict[str, Any]], temperature: float = 0.1) -> dict[str, Any]:
    settings = get_openrouter_settings(require_chat_model=True)
    requests = _import_requests()
    session = requests.Session()
    timeout = _get_env_int("OPENROUTER_CHAT_TIMEOUT", DEFAULT_CHAT_TIMEOUT)
    payload = post_json_with_retries(
        session,
        f"{settings['base_url']}/chat/completions",
        headers=build_headers(settings["api_key"]),
        json_payload={
            "model": settings["chat_model"],
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices for chat request.")

    content = extract_message_content(choices[0])
    return parse_json_content(content)


def post_json_with_retries(
    session: Any,
    url: str,
    headers: dict[str, str],
    json_payload: dict[str, Any],
    timeout: int,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    attempts = max_attempts or _get_env_int("OPENROUTER_MAX_RETRIES", DEFAULT_MAX_RETRIES)
    attempts = max(1, attempts)
    backoff = float(os.getenv("OPENROUTER_RETRY_BACKOFF_SECONDS", str(DEFAULT_RETRY_BACKOFF)).strip() or DEFAULT_RETRY_BACKOFF)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.post(
                url,
                headers=headers,
                json=json_payload,
                timeout=timeout,
            )
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                time.sleep(backoff * attempt)
                continue
            return parse_json_response(response)
        except Exception as exc:  # pragma: no cover - network retry branches are hard to force
            last_error = exc
            if attempt >= attempts or not _is_retryable_request_exception(exc):
                break
            time.sleep(backoff * attempt)

    if last_error is None:
        raise RuntimeError("OpenRouter request failed for an unknown reason.")
    raise RuntimeError(f"OpenRouter request failed after {attempts} attempts: {last_error}") from last_error


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def parse_json_response(response: Any) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except Exception as exc:
        body = getattr(response, "text", "")
        body = body[:500].strip()
        raise RuntimeError(f"OpenRouter request failed: {body or exc}") from exc

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError("OpenRouter returned a non-JSON response.") from exc


def extract_message_content(choice: dict[str, Any]) -> str:
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    raise RuntimeError("Unsupported LLM message content format.")


def parse_json_content(content: str) -> dict[str, Any]:
    if not content:
        raise RuntimeError("LLM returned an empty response.")

    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("LLM response is not valid JSON.") from None
        parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response JSON must be an object.")
    return parsed


def _load_project_env() -> None:
    candidate_paths = [
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[3] / ".env.local",
        Path(__file__).resolve().parents[3] / "simple_tnved_app" / ".env",
    ]
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        for env_path in existing_paths:
            _load_project_env_fallback(env_path)
        return

    for env_path in existing_paths:
        load_dotenv(env_path, override=False)


def _load_project_env_fallback(env_path: Path) -> None:
    for raw_line in env_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_retryable_request_exception(exc: Exception) -> bool:
    retryable_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection aborted",
        "connection reset",
        "remote disconnected",
        "eof occurred in violation of protocol",
        "ssl",
    )
    message = str(exc).lower()
    return any(marker in message for marker in retryable_markers)


def _import_requests() -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("requests is not installed. Install dependencies from requirements.txt.") from exc
    return requests
