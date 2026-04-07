# Архитектура

## Multimodal RAG для ТН ВЭД

Основной рабочий путь проекта переносится в `src/app/` по правилам `AGENTS.md`.
Исторический код в `simple_tnved_app/` остаётся только как legacy-слой до отдельной зачистки и не считается основным entrypoint.

### Поток данных

`user input -> multimodal analysis -> vector retrieval -> final LLM classification`

1. Пользователь передаёт текст, изображения и дополнительные файлы.
2. Локальный слой извлекает текст из `txt/pdf/docx`, а для изображений делает OCR как best-effort enrichment.
3. Мультимодальная LLM получает:
   - исходный пользовательский текст;
   - извлечённый текст файлов;
   - OCR-текст;
   - сами изображения.
4. Модель возвращает структурированный `product_profile`, который нормализует запрос.
5. Векторный поиск выполняется по embedded Chroma:
   - primary retrieval по `docs/reference/`;
   - secondary retrieval по `docs/examples/`.
6. Финальная LLM получает `product_profile` и retrieval-контекст и возвращает:
   - `best_match`;
   - `candidates`;
   - `warnings`;
   - привязку к evidence.

### Структура приложения

- `src/app/main.py` — CLI entrypoint.
- `src/app/core/` — конфигурация, OpenRouter client, Chroma persistence, runtime paths.
- `src/app/services/` — train workflow, multimodal input analysis, retrieval, final classification.
- `src/app/schemas/` — JSON contracts и валидация `product_profile` и итогового ответа.
- `src/app/utils/` — file extraction, OCR, chunking, regex metadata, JSON helpers.
- `src/app/ui/desktop_app.py` — `tkinter` GUI.
- `tests/` — тесты, зеркалирующие новую структуру.

### Корпуса знаний

- `docs/reference/` — официальный и нормативный корпус, основной источник retrieval-контекста.
- `docs/examples/` — примеры подбора ТН ВЭД, secondary signal.
- Пользовательские входные файлы не попадают в persistent knowledge base.

### Хранилища

- Embedded vector DB: Chroma в `data/vector_db/chroma/`.
- Runtime-логи и metadata: `data/runtime/`.
- Старые `simple_tnved_app/runtime/*` и JSONL-индексы считаются legacy и не используются в новом main flow.

### Ограничения текущей итерации

- Источник правды для retrieval — chunk'и документов и примеров, а не заранее извлечённый конечный каталог кодов.
- Examples не должны переопределять приоритет reference corpus.
- Основной сценарий требует `OPENROUTER_CHAT_MODEL`, совместимую с multimodal input.
