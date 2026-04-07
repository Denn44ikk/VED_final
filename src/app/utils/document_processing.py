from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
DOCX_TECHNICAL_NOISE_MARKERS = (
    "документ предоставлен консультантплюс",
    "www.consultant.ru",
    "дата сохранения",
)
DOCX_CHANGE_LIST_MARKER = "список изменяющих документов"
DOCX_EDITORIAL_NOTE_RE = re.compile(r"^\(\s*(в ред\.|введено|утратило силу)", re.IGNORECASE)
DOCX_HEADER_MARKERS = ("код", "наименование", "ставк", "пошлин", "ед.", "единиц", "обознач")
DOCX_TITLE_PREFIX_RE = re.compile(r"^(?P<prefix>-+)\s*(?P<body>.+)$")
TNVED_CODE_RE = re.compile(r"\b(?:\d{4}\s?\d{2}\s?\d{3}\s?\d|\d{10})\b")


def list_supported_documents(doc_root: Path) -> list[Path]:
    if not doc_root.exists() or not doc_root.is_dir():
        return []

    result = []
    for path in sorted(doc_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append(path)
    return result


def build_knowledge_chunks(
    document_paths: list[Path],
    base_path: Path,
    source_kind: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_entries: list[dict[str, Any]] = []
    knowledge_chunks: list[dict[str, Any]] = []

    for path in document_paths:
        relative_path = _make_relative_path(path, base_path)
        parser_name = SUPPORTED_EXTENSIONS[path.suffix.lower()]
        if parser_name == "docx":
            docx_chunk_size = max(chunk_size, 4000)
            chunks = chunk_docx_blocks(extract_docx_blocks(path), chunk_size=docx_chunk_size, overlap=0)
        else:
            text = extract_text_from_file(path)
            chunks = [{"text": chunk} for chunk in chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)]

        manifest_entries.append(
            {
                "path": relative_path,
                "source_kind": source_kind,
                "parser": parser_name,
                "chunk_count": len(chunks),
            }
        )

        for chunk_index, chunk in enumerate(chunks):
            section_context = _render_section_context(chunk.get("heading_path", []))
            text = chunk["text"]
            record = {
                "chunk_id": f"{source_kind}:{relative_path}::chunk-{chunk_index}",
                "source_path": relative_path,
                "source_kind": source_kind,
                "document_type": path.suffix.lower().lstrip("."),
                "chunk_index": chunk_index,
                "text": text,
                "mentioned_codes": extract_mentioned_codes(text),
                "section_context": section_context,
            }
            knowledge_chunks.append(record)

    return manifest_entries, knowledge_chunks


def extract_mentioned_codes(text: str) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for match in TNVED_CODE_RE.findall(text):
        normalized = "".join(ch for ch in match if ch.isdigit())
        if len(normalized) != 10 or normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def extract_text_from_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return normalize_text(extract_text_from_pdf(path))
    if suffix == ".docx":
        return normalize_text(extract_text_from_docx(path))
    if suffix in IMAGE_EXTENSIONS:
        return normalize_text(extract_text_from_image(path))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def try_extract_text_from_image(path: Path) -> tuple[str, str | None]:
    try:
        return normalize_text(extract_text_from_image(path)), None
    except Exception as exc:
        return "", str(exc)


def encode_image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("PyPDF2 is not installed. Install dependencies from requirements.txt.") from exc

    reader = PdfReader(str(path))
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts)


def extract_text_from_docx(path: Path) -> str:
    return "\n\n".join(block["text"] for block in extract_docx_blocks(path))


def extract_docx_blocks(path: Path) -> list[dict[str, Any]]:
    Document, Paragraph, Table, qn = _import_python_docx()

    document = Document(str(path))
    blocks: list[dict[str, Any]] = []
    block_index = 0
    table_index = 0
    heading_path: list[str] = []
    pending_heading_lines: list[str] = []

    def append_block(payload: dict[str, Any]) -> None:
        nonlocal block_index
        text = payload.get("text", "").strip()
        if not text:
            return
        entry = dict(payload)
        entry["text"] = text
        entry["block_index"] = block_index
        entry["heading_path"] = list(entry.get("heading_path", heading_path))
        blocks.append(entry)
        block_index += 1

    def flush_heading() -> None:
        nonlocal heading_path
        pending_text = normalize_text(" ".join(pending_heading_lines))
        pending_heading_lines.clear()
        if not pending_text or _is_docx_technical_noise(pending_text):
            return
        heading_path = _update_heading_path(heading_path, pending_text)
        append_block(
            {
                "content_kind": "heading",
                "heading_path": list(heading_path),
                "text": pending_text,
            }
        )

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            if _is_docx_heading(paragraph):
                pending_heading_lines.append(text)
                continue
            if pending_heading_lines:
                flush_heading()
            if _is_docx_technical_noise(text):
                continue
            append_block(
                {
                    "content_kind": "paragraph",
                    "heading_path": list(heading_path),
                    "text": _serialize_docx_paragraph(text, heading_path),
                }
            )
            continue

        if child.tag == qn("w:tbl"):
            if pending_heading_lines:
                flush_heading()
            table = Table(child, document)
            table_blocks = _merge_docx_table_row_blocks(_extract_docx_table_blocks(table, heading_path, table_index))
            for table_block in table_blocks:
                append_block(table_block)
            table_index += 1

    if pending_heading_lines:
        flush_heading()

    return blocks


def extract_text_from_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("Pillow and pytesseract must be installed for OCR support.") from exc

    try:
        with Image.open(path) as image:
            return pytesseract.image_to_string(image, lang="rus+eng")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "OCR requires the system Tesseract binary. Install Tesseract and ensure it is available in PATH."
        ) from exc


def chunk_docx_blocks(blocks: list[dict[str, Any]], chunk_size: int = 1200, overlap: int = 0) -> list[dict[str, Any]]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    if not blocks:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(blocks):
        end = start
        current_blocks: list[dict[str, Any]] = []
        current_length = 0

        while end < len(blocks):
            block = blocks[end]
            block_text = block["text"].strip()
            if not block_text:
                end += 1
                continue

            projected = current_length + len(block_text)
            if current_blocks:
                projected += 2
            if current_blocks and projected > chunk_size:
                break

            current_blocks.append(block)
            current_length = projected
            end += 1

        if not current_blocks:
            break

        text = "\n\n".join(block["text"] for block in current_blocks)
        heading_path = _common_heading_path([block.get("heading_path", []) for block in current_blocks])
        chunk = {
            "text": text,
            "heading_path": heading_path,
        }
        chunks.append(chunk)

        if end >= len(blocks):
            break

        next_start = _calculate_block_overlap_start(blocks, start, end, overlap)
        start = next_start if next_start > start else end

    return chunks


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            split_index = cleaned_text.rfind(" ", start, end)
            if split_index > start + (chunk_size // 2):
                end = split_index

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _import_python_docx() -> tuple[Any, Any, Any, Any]:
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover - dependency failure
        raise RuntimeError("python-docx is not installed. Install dependencies from requirements.txt.") from exc
    return Document, Paragraph, Table, qn


def _is_docx_heading(paragraph: Any) -> bool:
    style_name = ""
    if getattr(paragraph, "style", None) is not None and getattr(paragraph.style, "name", None):
        style_name = str(paragraph.style.name).strip().lower()
    if "consplustitle" in style_name:
        return True
    return any(marker in style_name for marker in ("title", "heading"))


def _is_docx_technical_noise(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(marker in lowered for marker in DOCX_TECHNICAL_NOISE_MARKERS)


def _update_heading_path(current_path: list[str], heading_text: str) -> list[str]:
    if heading_text.startswith("РАЗДЕЛ"):
        return [heading_text]
    if heading_text.startswith("ГРУППА"):
        if current_path and current_path[0].startswith("РАЗДЕЛ"):
            return current_path[:1] + [heading_text]
        return [heading_text]
    if current_path and current_path[-1].startswith("ГРУППА"):
        return current_path[:2] + [heading_text]
    if current_path and current_path[-1].startswith("РАЗДЕЛ"):
        return current_path[:1] + [heading_text]
    return [heading_text]


def _serialize_docx_paragraph(text: str, heading_path: list[str]) -> str:
    if not heading_path:
        return text
    return f"Контекст: {_format_heading_context(heading_path)}\n{text}"


def _extract_docx_table_blocks(table: Any, heading_path: list[str], table_index: int) -> list[dict[str, Any]]:
    rows = [_normalize_docx_table_row(row) for row in table.rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []

    table_text = normalize_text(" ".join(value for row in rows for value in row if value))
    lowered_table_text = table_text.lower()
    if DOCX_CHANGE_LIST_MARKER in lowered_table_text:
        return []
    if _is_docx_technical_noise(table_text):
        return []

    header_index = _detect_docx_table_header_row(rows)
    raw_headers = rows[header_index] if header_index is not None else []
    table_layout = _classify_docx_table_layout(raw_headers)
    header_map = _build_docx_header_map(raw_headers) if header_index is not None else []

    blocks: list[dict[str, Any]] = []
    current_note = ""
    current_code = ""
    title_stack: list[str] = []

    for row_index, row in enumerate(rows):
        if header_index is not None and row_index == header_index:
            continue

        nonempty_values = [value for value in row if value]
        if not nonempty_values:
            continue

        if DOCX_CHANGE_LIST_MARKER in " ".join(nonempty_values).lower():
            continue

        repeated_value = _get_repeated_docx_row_value(nonempty_values)
        if repeated_value and _is_docx_editorial_note(repeated_value):
            current_note = repeated_value
            blocks.append(
                {
                    "content_kind": "table_note",
                    "heading_path": list(heading_path),
                    "table_index": table_index,
                    "table_row_range": [row_index, row_index],
                    "text": _serialize_docx_table_note(repeated_value, heading_path),
                }
            )
            continue

        if table_layout == "tariff":
            row_text, current_code = _serialize_docx_tariff_row(
                row=row,
                header_map=header_map,
                heading_path=heading_path,
                title_stack=title_stack,
                current_code=current_code,
                current_note=current_note,
            )
        elif table_layout == "headered":
            row_text = _serialize_docx_headered_row(row, raw_headers, heading_path, current_note)
        else:
            row_text = _serialize_docx_generic_row(nonempty_values, heading_path, current_note)

        if not row_text:
            continue

        blocks.append(
            {
                "content_kind": "table_row",
                "heading_path": list(heading_path),
                "table_index": table_index,
                "table_row_range": [row_index, row_index],
                "text": row_text,
            }
        )

    return blocks


def _normalize_docx_table_row(row: Any) -> list[str]:
    values = []
    for cell in row.cells:
        values.append(normalize_text(cell.text))
    return values


def _detect_docx_table_header_row(rows: list[list[str]]) -> int | None:
    for index, row in enumerate(rows[:5]):
        score = 0
        for value in row:
            lowered = value.lower()
            if any(marker in lowered for marker in DOCX_HEADER_MARKERS):
                score += 1
        if score >= 2:
            return index
    return None


def _classify_docx_table_layout(raw_headers: list[str]) -> str:
    lowered_headers = [header.lower() for header in raw_headers if header]
    has_code = any("код" in header for header in lowered_headers)
    has_title = any("наименование" in header or "позици" in header for header in lowered_headers)
    if has_code and has_title:
        return "tariff"
    if lowered_headers:
        return "headered"
    return "generic"


def _build_docx_header_map(raw_headers: list[str]) -> list[str | None]:
    header_map: list[str | None] = []
    for header in raw_headers:
        lowered = header.lower()
        if "код" in lowered:
            header_map.append("code")
        elif "наименование" in lowered or "позици" in lowered:
            header_map.append("title")
        elif "ставк" in lowered or "пошлин" in lowered:
            header_map.append("duty")
        elif "ед." in lowered or "единиц" in lowered or "обознач" in lowered:
            header_map.append("unit")
        else:
            header_map.append(None)
    return header_map


def _serialize_docx_tariff_row(
    row: list[str],
    header_map: list[str | None],
    heading_path: list[str],
    title_stack: list[str],
    current_code: str,
    current_note: str,
) -> tuple[str, str]:
    row_data: dict[str, str] = {}
    extra_values: list[str] = []

    for index, value in enumerate(row):
        if not value:
            continue
        key = header_map[index] if index < len(header_map) else None
        if key is None:
            extra_values.append(value)
            continue
        if key not in row_data:
            row_data[key] = value
        elif value not in row_data[key]:
            row_data[key] = f"{row_data[key]}; {value}"

    row_code = row_data.get("code", "").strip()
    if row_code:
        current_code = row_code
    display_code = row_code or current_code

    raw_title = row_data.get("title", "").strip()
    title_hierarchy = _update_docx_title_stack(title_stack, raw_title)
    display_title = title_hierarchy or raw_title

    if not display_code and not display_title and not row_data.get("duty") and not row_data.get("unit") and not extra_values:
        return "", current_code

    lines = []
    if heading_path:
        lines.append(f"Контекст: {_format_heading_context(heading_path)}")
    if display_code:
        lines.append(f"Код ТН ВЭД: {display_code}")
    if display_title:
        lines.append(f"Наименование позиции: {display_title}")
    if row_data.get("unit"):
        lines.append(f"Доп. ед. изм.: {row_data['unit']}")
    if row_data.get("duty"):
        lines.append(f"Ставка пошлины: {row_data['duty']}")
    if extra_values:
        lines.append(f"Доп. контекст: {' | '.join(extra_values)}")
    if current_note:
        lines.append(f"Редакционная пометка: {current_note}")

    return "\n".join(lines), current_code


def _serialize_docx_headered_row(
    row: list[str],
    raw_headers: list[str],
    heading_path: list[str],
    current_note: str,
) -> str:
    repeated_value = _get_repeated_docx_row_value([value for value in row if value])
    if repeated_value:
        return _serialize_docx_generic_row([repeated_value], heading_path, current_note)

    lines = []
    if heading_path:
        lines.append(f"Контекст: {_format_heading_context(heading_path)}")

    for index, value in enumerate(row):
        if not value:
            continue
        header = raw_headers[index] if index < len(raw_headers) else ""
        if header:
            lines.append(f"{header}: {value}")
        else:
            lines.append(value)

    if current_note:
        lines.append(f"Редакционная пометка: {current_note}")

    return "\n".join(lines)


def _serialize_docx_generic_row(values: list[str], heading_path: list[str], current_note: str) -> str:
    lines = []
    if heading_path:
        lines.append(f"Контекст: {_format_heading_context(heading_path)}")
    if len(values) == 1:
        lines.append(values[0])
    else:
        lines.append(f"Строка таблицы: {' | '.join(values)}")
    if current_note:
        lines.append(f"Редакционная пометка: {current_note}")
    return "\n".join(lines)


def _serialize_docx_table_note(text: str, heading_path: list[str]) -> str:
    if heading_path:
        return f"Контекст: {_format_heading_context(heading_path)}\nРедакционная пометка таблицы: {text}"
    return f"Редакционная пометка таблицы: {text}"


def _update_docx_title_stack(title_stack: list[str], raw_title: str) -> str:
    title = raw_title.strip()
    if not title:
        return " > ".join(title_stack)

    match = DOCX_TITLE_PREFIX_RE.match(title)
    if match:
        level = len(match.group("prefix"))
        body = normalize_text(match.group("body"))
    else:
        level = 0
        body = normalize_text(title)

    del title_stack[level:]
    title_stack.append(body)
    return " > ".join(title_stack)


def _get_repeated_docx_row_value(values: list[str]) -> str | None:
    unique_values = {value for value in values if value}
    if len(values) >= 2 and len(unique_values) == 1:
        return next(iter(unique_values))
    return None


def _is_docx_editorial_note(text: str) -> bool:
    return bool(DOCX_EDITORIAL_NOTE_RE.match(normalize_text(text)))


def _calculate_block_overlap_start(blocks: list[dict[str, Any]], start: int, end: int, overlap: int) -> int:
    if overlap <= 0 or end - start <= 1:
        return end

    collected = 0
    cursor = end
    while cursor > start + 1 and collected < overlap:
        cursor -= 1
        collected += len(blocks[cursor]["text"]) + 2
    return cursor


def _format_heading_context(heading_path: list[str], limit: int = 2) -> str:
    if not heading_path:
        return ""
    return " > ".join(heading_path[-limit:])


def _merge_docx_table_row_blocks(blocks: list[dict[str, Any]], target_size: int = 1800) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    buffer: dict[str, Any] | None = None

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer is not None:
            merged.append(buffer)
            buffer = None

    for block in blocks:
        if block.get("content_kind") != "table_row":
            flush_buffer()
            merged.append(block)
            continue

        if buffer is None:
            buffer = dict(block)
            continue

        next_text = block["text"]
        context_prefix = "Контекст: " + _format_heading_context(block.get("heading_path", [])) + "\n"
        if buffer.get("heading_path") == block.get("heading_path") and next_text.startswith(context_prefix):
            next_text = next_text[len(context_prefix):]

        projected = len(buffer["text"]) + 2 + len(next_text)
        if buffer.get("table_index") == block.get("table_index") and projected <= target_size:
            buffer["text"] = buffer["text"] + "\n\n" + next_text
            if buffer.get("table_row_range") and block.get("table_row_range"):
                buffer["table_row_range"] = [buffer["table_row_range"][0], block["table_row_range"][1]]
            continue

        flush_buffer()
        buffer = dict(block)

    flush_buffer()
    return merged


def _common_heading_path(paths: list[list[str]]) -> list[str]:
    nonempty_paths = [list(path) for path in paths if path]
    if not nonempty_paths:
        return []

    prefix = nonempty_paths[0]
    for path in nonempty_paths[1:]:
        limit = min(len(prefix), len(path))
        index = 0
        while index < limit and prefix[index] == path[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _make_relative_path(path: Path, base_path: Path) -> str:
    try:
        return path.resolve().relative_to(base_path.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _render_section_context(heading_path: list[str]) -> str:
    if not heading_path:
        return ""
    return " > ".join(heading_path)
