# TN VED Multimodal RAG

Локальное приложение для подбора кодов ТН ВЭД по тексту, фотографиям и дополнительным файлам пользователя.

Проект построен как multimodal RAG pipeline:

1. пользователь передаёт описание товара, фотографии, сканы, спецификации или другие файлы;
2. мультимодальная LLM нормализует входные данные в структурированный `product_profile`;
3. приложение ищет релевантный контекст в локальной embedded vector DB;
4. финальная LLM выбирает лучший код и список кандидатов на основе найденных нормативных фрагментов и примеров.

Приложение не опирается на заранее фиксированный "конечный каталог" кодов в кодовой базе. Источник правды для поиска и рассуждения - документы, которые вы положили в knowledge base.

## Что умеет приложение

- принимать свободный текст пользователя;
- принимать один или несколько файлов через CLI или GUI;
- анализировать изображения как двумя путями:
  - через OCR как best-effort enrichment;
  - через vision-capable chat model как полноценный multimodal input;
- строить локальную векторную базу знаний на embedded Chroma;
- хранить reference-корпус и example-корпус раздельно;
- использовать reference-документы как основной источник, а examples как вторичный сигнал;
- возвращать:
  - `product_profile`;
  - `best_match`;
  - `candidates`;
  - `retrieval.reference`;
  - `retrieval.examples`;
  - `warnings`.

## Текущая архитектура

Основной рабочий код расположен в `src/app/` и соответствует правилам из `AGENTS.md`.

Высокоуровневый поток данных:

`user input -> multimodal analysis -> vector retrieval -> final LLM classification`

### Шаг 1. Построение knowledge base

Команда `train`:

- читает документы из `docs/reference/` и `docs/examples/`;
- по умолчанию синхронизирует индекс: добавляет новые документы, переиндексирует изменённые и удаляет исчезнувшие;
- извлекает из них текст;
- режет документы на чанки;
- извлекает metadata для каждого чанка;
- строит embeddings через `OPENROUTER_EMBEDDING_MODEL`;
- сохраняет чанки в локальную embedded vector DB на Chroma;
- поддерживает явный полный rebuild через `train --full-rebuild`.

Две коллекции хранятся отдельно:

- `reference_chunks` - нормативные и официальные документы;
- `example_chunks` - примеры и кейсы.

### Шаг 2. Нормализация пользовательского запроса

Команда `classify`:

- принимает `--text`;
- принимает один или несколько `--file`;
- локально извлекает текст из `txt/pdf/docx`;
- пытается сделать OCR для изображений;
- передаёт текст, OCR и сами изображения в vision-capable chat model;
- получает на выходе структурированный `product_profile`.

`product_profile` содержит:

- `summary`;
- `intended_use`;
- `material_or_composition`;
- `key_features`;
- `search_queries`;
- `uncertainty`;
- `missing_information`.

Для retrieval используются именно `search_queries`, а не сырой пользовательский текст.

### Шаг 3. Vector retrieval

Приложение:

- строит embedding для `search_queries`;
- ищет ближайшие чанки в `reference_chunks`;
- отдельно ищет ближайшие чанки в `example_chunks`;
- объединяет результат для финального reasoning.

Reference retrieval всегда считается более надёжным сигналом, чем examples.

### Шаг 4. Финальная классификация

Финальная LLM получает:

- исходный текст пользователя;
- `product_profile`;
- reference evidence;
- example evidence;
- число требуемых кандидатов `top_k`.

На выходе она возвращает:

- `best_match`;
- `candidates`;
- `warnings`.

Если уверенности недостаточно, `best_match` может быть `null`, а предупреждения остаются в `warnings`.

## Структура репозитория

```text
src/
  app/
    main.py
    core/
    schemas/
    services/
    ui/
    utils/
tests/
docs/
  architecture.md
  reference/
  examples/
data/
  runtime/
  vector_db/
scripts/
simple_tnved_app/
  main.py
  .env                  # может использоваться как legacy fallback
PLAN.md
AGENTS.md
README.md
requirements.txt
.env.example
```

### Назначение каталогов

- `src/app/main.py` - основной CLI entrypoint.
- `src/app/core/` - конфигурация, OpenRouter client, работа с Chroma.
- `src/app/services/` - orchestration и бизнес-логика.
- `src/app/schemas/` - валидация JSON-контрактов.
- `src/app/utils/` - извлечение текста, OCR, chunking, служебные helpers.
- `src/app/ui/desktop_app.py` - единый русскоязычный GUI на `tkinter`.
- `tests/` - unit и workflow tests.
- `docs/reference/` - основной корпус нормативных документов.
- `docs/examples/` - вторичный корпус примеров.
- `data/vector_db/chroma/` - persistent storage Chroma.
- `data/runtime/` - runtime metadata, manifest и event log.
- `simple_tnved_app/main.py` - compatibility wrapper для старого пути запуска.

## Требования

Минимально:

- Python 3.12;
- доступ к OpenRouter API;
- embedding model в OpenRouter;
- vision-capable chat model в OpenRouter.

Для OCR рекомендуется дополнительно:

- установленный системный Tesseract;
- `tesseract` должен быть доступен в `PATH`.

Без Tesseract приложение всё равно может работать с изображениями через vision model, но локальный OCR будет недоступен и добавит предупреждение в `warnings`.

## Установка

Примеры ниже даны для PowerShell из корня проекта.

### 1. Создайте и активируйте виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Установите зависимости

```powershell
pip install -r .\requirements.txt
```

### 3. Подготовьте `.env`

Скопируйте шаблон:

```powershell
Copy-Item .\.env.example .\.env
```

Минимальный набор переменных:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_CHAT_MODEL=your_vision_capable_chat_model
OPENROUTER_EMBEDDING_MODEL=your_embedding_model
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Важно:

- `OPENROUTER_CHAT_MODEL` обязателен;
- chat model должна уметь работать с изображениями;
- embedding model используется и при `train`, и при `classify`.

Приложение в первую очередь читает корневой `.env`, но для совместимости также умеет подхватывать `simple_tnved_app/.env`, если он уже существует.

## Подготовка базы знаний

### `docs/reference/` - обязательно

Сюда нужно класть основной нормативный корпус:

- решения ЕЭК;
- пояснения;
- выдержки из классификационных документов;
- внутренние нормативные справки, если вы сознательно используете их как reference.

### `docs/examples/` - опционально

Сюда можно класть:

- примеры уже выполненного подбора ТН ВЭД;
- внутренние кейсы;
- примеры пояснительных писем;
- пользовательские учебные материалы.

Examples участвуют в retrieval как secondary signal и не должны переопределять reference corpus.

### Поддерживаемые форматы документов

Для knowledge base и пользовательских файлов поддерживаются:

- `.txt`
- `.pdf`
- `.docx`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`
- `.tiff`
- `.tif`

### Как индексация работает внутри

При `train` для каждого документа строятся чанки с metadata:

- `chunk_id`
- `source_path`
- `source_kind`
- `document_type`
- `chunk_index`
- `text`
- `mentioned_codes`
- `section_context`

`mentioned_codes` извлекается регэкспом из текста чанка и используется только как metadata, а не как отдельный "каталог кодов".

## Команда `train`

Синхронизация векторной базы по умолчанию:

```powershell
python .\src\app\main.py train
```

Полная пересборка индекса:

```powershell
python .\src\app\main.py train --full-rebuild
```

Или через compatibility wrapper:

```powershell
python .\simple_tnved_app\main.py train
```

Что делает команда:

1. проверяет наличие `docs/reference/`;
2. находит поддерживаемые документы в `docs/reference/` и `docs/examples/`;
3. сравнивает текущие файлы с `data/runtime/manifest.jsonl`;
4. в обычном режиме добавляет новые документы, переиндексирует изменённые и удаляет исчезнувшие;
5. при `--full-rebuild` или несовместимом runtime полностью пересоздаёт Chroma collections;
6. извлекает текст только для документов, которые нужно индексировать в текущем запуске;
7. строит чанки, вызывает embeddings API и обновляет metadata в `data/runtime/`.

### Что создаётся после `train`

В `data/runtime/`:

- `vector_meta.json` - информация о текущем runtime snapshot;
- `manifest.jsonl` - перечень проиндексированных документов и их `file_sha256`;
- `events.log.jsonl` - лог этапов `train` и `classify`.

В `data/vector_db/chroma/`:

- файлы embedded Chroma database.

## Команда `classify`

### Базовый пример по тексту

```powershell
python .\src\app\main.py classify --text "Племенные чистопородные лошади для разведения" --top-k 4
```

### Пример с несколькими файлами

```powershell
python .\src\app\main.py classify `
  --text "Нужно определить код ТН ВЭД для товара" `
  --file ".\samples\photo1.jpg" `
  --file ".\samples\specification.pdf" `
  --file ".\samples\description.docx" `
  --top-k 4
```

### Параметры

- `--text` - свободное текстовое описание.
- `--file` - путь к файлу; параметр можно указывать несколько раз.
- `--top-k` - сколько кандидатов вернуть, только `3` или `4`.

### Что делает команда

1. проверяет, что runtime уже построен через `train`;
2. собирает входной пакет из текста и файлов;
3. извлекает текст из `txt/pdf/docx`;
4. для изображений:
   - делает OCR, если это возможно;
   - кодирует изображение в base64 data URL;
5. отправляет всё это в мультимодальную LLM;
6. получает `product_profile`;
7. ищет релевантные чанки в Chroma;
8. отправляет `product_profile` и retrieval context в финальную LLM;
9. возвращает структурированный JSON и короткое текстовое summary.

## Формат результата `classify`

Выходной JSON имеет вид:

```json
{
  "product_profile": {
    "summary": "string",
    "intended_use": "string",
    "material_or_composition": "string",
    "key_features": ["string"],
    "search_queries": ["string"],
    "uncertainty": "string",
    "missing_information": ["string"]
  },
  "best_match": {
    "code": "string",
    "title_or_label": "string",
    "confidence": 0.0,
    "reasoning": "string",
    "evidence_ids": ["string"]
  },
  "candidates": [
    {
      "code": "string",
      "title_or_label": "string",
      "confidence": 0.0,
      "reasoning": "string",
      "evidence_ids": ["string"]
    }
  ],
  "retrieval": {
    "reference": [
      {
        "chunk_id": "string",
        "source_path": "string",
        "source_kind": "reference",
        "document_type": "string",
        "section_context": "string",
        "mentioned_codes": ["string"],
        "score": 0.0,
        "text": "string"
      }
    ],
    "examples": [
      {
        "chunk_id": "string",
        "source_path": "string",
        "source_kind": "example",
        "document_type": "string",
        "section_context": "string",
        "mentioned_codes": ["string"],
        "score": 0.0,
        "text": "string"
      }
    ]
  },
  "warnings": ["string"]
}
```

### Смысл основных полей

- `product_profile` - нормализованное понимание товара моделью.
- `best_match` - лучший найденный код, если модель смогла выбрать его уверенно.
- `candidates` - ранжированный список альтернатив длиной до `top_k`.
- `retrieval.reference` - фрагменты нормативных источников.
- `retrieval.examples` - фрагменты примеров.
- `warnings` - всё, что может снижать уверенность:
  - плохой OCR;
  - мало входных данных;
  - слабый retrieval;
  - противоречивый контекст.

## GUI

Запуск:

```powershell
python .\src\app\main.py gui
```

GUI реализован на `tkinter`, полностью переведён на русский язык и использует те же сервисы, что и CLI.

В интерфейсе есть:

- одно основное окно без вкладок;
- блок ввода описания товара и пользовательских файлов;
- кнопка запуска классификации с фиксированным сценарием "основной код + до 2 альтернатив";
- три карточки результата: основная рекомендация, альтернатива 1, альтернатива 2;
- отдельные области для предупреждений, `product_profile`, нормативных фрагментов, примеров и технического JSON;
- блок индексации knowledge base в том же окне;
- кнопка инспектора чанков для проверки того, как документы реально разрезаны в текущей ВБД.

## Runtime-файлы и что они значат

### `data/runtime/vector_meta.json`

Содержит metadata текущего индекса:

- версия snapshot schema;
- используемую embedding model;
- используемую chat model;
- режим последней индексации и статистику sync/rebuild;
- число reference-документов и example-документов;
- число чанков в каждой коллекции;
- имя коллекций.

### `data/runtime/manifest.jsonl`

Содержит информацию о том, какие документы были реально проиндексированы:

- относительный путь;
- `source_kind`;
- парсер;
- число чанков;
- `file_sha256`, по которому `train` определяет, изменился ли документ.

### `data/runtime/events.log.jsonl`

Содержит лог этапов:

- `bootstrap`;
- `scan_documents`;
- `build_chunks`;
- `embed_and_index`;
- `multimodal_analysis`;
- `retrieve_context`;
- `final_classification`.

Этот файл удобен для диагностики пайплайна без включения отдельного debug-mode.

## Тесты

Запуск всего набора:

```powershell
python -m unittest discover tests
```

Что проверяется:

- индексация `docs/reference` и `docs/examples` в отдельные Chroma collections;
- запрет на использование `simple_tnved_app/DOCS` как knowledge-base root;
- мультимодальный analysis-step;
- формат `product_profile`;
- workflow orchestration для `train` и `classify`;
- CLI-параметры, включая `--top-k`.

## Важные ограничения

- качество результата напрямую зависит от качества `docs/reference/`;
- examples помогают, но не должны заменять нормативный корпус;
- приложение не сохраняет пользовательские upload-файлы в persistent knowledge base;
- OCR не гарантирует высокое качество на плохих изображениях;
- финальный ответ зависит от качества retrieval и от выбранной chat model;
- без `train` команда `classify` не работает.

## Рекомендации по наполнению корпуса

- В `docs/reference/` кладите только то, что вы готовы считать опорным источником.
- Не смешивайте reference и examples в одной папке.
- Для примеров держите отдельные документы с хорошим описанием товара и обоснованием кода.
- Если документ очень большой и плохо структурирован, лучше сохранить его в `docx` или `pdf` с читаемым текстовым слоем.
- Если у вас есть изображения из официальных материалов, используйте их только если от OCR или vision-анализа есть реальная польза.

## Troubleshooting

### Ошибка: `Missing required OpenRouter environment variables`

Проверьте:

- существует ли корневой `.env`;
- заполнены ли `OPENROUTER_API_KEY`, `OPENROUTER_CHAT_MODEL`, `OPENROUTER_EMBEDDING_MODEL`.

### Ошибка: не найден `docs/reference`

Создайте папку:

```powershell
New-Item -ItemType Directory -Force .\docs\reference
```

И положите туда хотя бы один поддерживаемый документ.

### Ошибка OCR / Tesseract

Это означает, что Python-библиотека `pytesseract` установлена, но системный `tesseract` недоступен.

Что делать:

- установить Tesseract OCR;
- добавить его в `PATH`;
- перезапустить терминал.

Если Tesseract не установлен, изображения всё ещё могут быть проанализированы самой vision model, но локальный OCR-слой не сработает.

### Ошибка legacy runtime

Если приложение пишет про legacy runtime, значит текущий `data/runtime/vector_meta.json` не соответствует актуальной схеме.

Решение:

```powershell
python .\src\app\main.py train
```

### Проблемы с Chroma на Windows

В проекте зафиксирована ветка `chromadb>=1.0,<2.0`.

Это сделано потому, что старые ветки Chroma в этой среде требовали локальную сборку `chroma-hnswlib`, что на Windows/Python 3.12 приводило к проблемам без MSVC Build Tools.

## Legacy и совместимость

Старый путь `simple_tnved_app/main.py` оставлен только как wrapper на новый entrypoint.

Это означает:

- основной рабочий путь - `src/app/main.py`;
- старый monolithic pipeline больше не используется;
- старые JSONL snapshots старого поколения удалены из main flow;
- пользовательский `simple_tnved_app/.env` может временно использоваться как fallback, если вы ещё не перенесли настройки в корневой `.env`.

## Полезные команды

Установка:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\requirements.txt
```

Индексация:

```powershell
python .\src\app\main.py train
```

Полный rebuild:

```powershell
python .\src\app\main.py train --full-rebuild
```

Классификация:

```powershell
python .\src\app\main.py classify --text "описание товара" --top-k 4
```

GUI:

```powershell
python .\src\app\main.py gui
```

Тесты:

```powershell
python -m unittest discover tests
```

## Дополнительные документы

- архитектурное описание: `docs/architecture.md`
- рабочий план проекта: `PLAN.md`
- правила работы агента: `AGENTS.md`
