# PLAN.md

## Goal

- Перевести проект на AGENTS-совместимый multimodal RAG для ТН ВЭД: мигрировать основной app-flow в `src/app`, принимать текст/фото/файлы, строить `product_profile`, искать контекст в embedded vector DB и возвращать лучший код с кандидатами после финального LLM-анализа.

## Architecture / Structure

- Основной рабочий путь расположен в `src/app/` по структуре `AGENTS.md`; legacy-слой в `simple_tnved_app/` сокращается до минимального compatibility shim.
- Новый слой `src/app/ui/` разрешён как документированное расширение структуры `AGENTS.md` для desktop GUI.
- База знаний строится только из `docs/reference/` и `docs/examples/`; папка `simple_tnved_app/DOCS/` не участвует в новом main flow.
- Persistent vector DB реализуется локально через embedded Chroma в `data/vector_db/chroma/`.
- Runtime-логи и служебные артефакты нового pipeline сохраняются в `data/runtime/`.
- Retrieval идёт по chunk'ам reference/examples, а не по заранее извлечённому конечному каталогу кодов.
- Архитектурные пояснения по новому multimodal RAG документируются в `docs/architecture.md`.

## Backlog

- При необходимости полностью удалить compatibility shim `simple_tnved_app/`, когда обратная совместимость по старому пути запуска больше не понадобится.
- Добрать остаточные каталоги `.venv-1`, `simple_tnved_app/.tmp_test_docs` и `simple_tnved_app/.tmp_test_workflows`, если появится доступ на их удаление.
- При необходимости подготовить упаковку GUI в standalone desktop build.
- Добавить более широкие UI-smoke тесты после стабилизации интерфейса.
- При необходимости отдельно спланировать внешнюю vector DB вместо embedded Chroma.

## In Progress

- Подготавливается GitHub-ready репозиторий: git init, `.gitignore`, чистый набор файлов без секретов и локальных runtime-артефактов.

## Done

- Создан `PLAN.md` и добавлена архитектурная заметка в `docs/architecture.md`.
- Вынесен общий workflow-слой в `simple_tnved_app/app_workflows.py`.
- CLI обновлён: `train` и `classify` используют общий workflow, добавлена команда `gui`.
- Реализован `simple_tnved_app/ui_app.py` на `tkinter` с двумя вкладками, прокруткой и фоновым запуском операций.
- Обновлён `simple_tnved_app/README.md` с инструкцией по запуску GUI.
- Добавлены тесты для workflow-слоя и CLI-парсера.
- Реализован DOCX-центричный каталог ТН ВЭД в `simple_tnved_app/tnved_docx_catalog.py`.
- `train` переведён на извлечение кодов из official `docx`; дополнительные документы индексируются только как evidence.
- `classify` переведён на детерминированный vector `top-k` без обязательного LLM-ранжирования.
- CLI и GUI обновлены под `--top-k` и новый JSON-ответ с `matches`.
- README, `.env.example` и тесты обновлены под DOCX-first сценарий.
- Создан новый основной app-flow в `src/app/` по структуре `AGENTS.md`, включая `core`, `services`, `schemas`, `utils` и `ui`.
- Реализован multimodal query-understanding: текст, OCR и изображения собираются в `product_profile` через vision-capable chat model.
- Реализован новый retrieval по embedded vector DB с раздельными коллекциями `reference_chunks` и `example_chunks`.
- Внедрён локальный persistent Chroma storage в `data/vector_db/chroma/` и новый runtime-слой в `data/runtime/`.
- Новый CLI entrypoint реализован в `src/app/main.py`; `simple_tnved_app/main.py` оставлен как compatibility wrapper на новый flow.
- Добавлены root-тесты в `tests/` для ingestion, multimodal analysis, CLI и end-to-end workflow orchestration.
- Созданы новые knowledge roots `docs/reference/` и `docs/examples/`; текущий официальный документ ЕЭК скопирован в `docs/reference/` для нового main flow.
- Проведена ревизия репозитория: удалены старые runtime snapshots, legacy-модули старого pipeline, старые тесты и `__pycache__`, а `simple_tnved_app/` сокращён до compatibility wrapper.
- Добавлен подробный корневой `README.md` с описанием архитектуры, запуска, knowledge base, train/classify/gui, runtime-артефактов и troubleshooting.
- В GUI вкладки индексации добавлено отдельное окно со списком документов, уже внесённых в текущую векторную БД, с чтением из `data/runtime/manifest.jsonl` и автообновлением после `train`.
- `train` переведён в sync-by-default режим: новые документы добавляются, изменённые переиндексируются, удалённые убираются из ВБД, а полный rebuild вынесен в явный режим `--full-rebuild`.
- `manifest.jsonl` расширен `file_sha256`, train/runtime metadata обновлены под diff-планирование, а CLI/GUI показывают режим индексации и sync-статистику.
- Тесты workflow/CLI расширены сценариями sync, no-op, удаления, fallback в rebuild и проверкой совместимости `classify` после sync.
- На вкладке индексации добавлен инспектор чанков: отдельная кнопка открывает окно с фильтром по документу, списком chunk'ов и детальным просмотром текста/metadata из текущей ВБД.
- Инспектор чанков сделан совместимым со старым `manifest.jsonl`: просмотр chunk'ов больше не требует `file_sha256`, если runtime уже существует.
- GUI переведён на русский язык и собран в одно основное окно без вкладок: сверху ввод и результат классификации, снизу индексация и инспектор базы знаний.
- В GUI результат классификации теперь всегда показывает основной код ТН ВЭД и до двух самостоятельных альтернатив для дополнительной проверки; для CLI/summary добавлен тот же dedupe-helper.
- Восстановлен исходник `src/app/ui/desktop_app.py`, обновлены README и русскоязычные CLI/help тексты, добавлены тесты для выбора основной рекомендации и альтернатив.

## Decisions

- Для первой версии GUI используется стандартный `tkinter` без дополнительных UI-зависимостей.
- Основной рабочий entrypoint переносится в `src/app/main.py`; legacy entrypoints в `simple_tnved_app/` не считаются main flow.
- В ходе cleanup legacy-папка `simple_tnved_app/` сокращается до compatibility wrapper и пользовательского `.env`, а старые runtime/test/doc snapshots удаляются как неиспользуемые новым pipeline.
- Остаточные каталоги `.venv-1` и некоторые старые `.tmp_test_*` не удалены автоматически из-за ограничений доступа; они не участвуют в main flow и считаются техническим мусором до отдельной зачистки.
- Текущий состав ВБД для GUI берётся из `manifest.jsonl`, а не из прямого обхода файлов в `docs/`, чтобы интерфейс показывал именно уже проиндексированные документы.
- Источник правды для бизнес-логики общий для CLI и GUI через сервисы в `src/app/services/`.
- Knowledge base строится из двух корпусов: `docs/reference/` как primary signal и `docs/examples/` как secondary signal.
- Реализация vector DB в этой итерации — локальная embedded Chroma без отдельного сервера.
- Пользовательские входные файлы используются только как query input и не попадают в persistent knowledge base.
- Основной сценарий снова требует `OPENROUTER_CHAT_MODEL`, потому что используется multimodal analysis и финальная LLM-классификация.
- `--top-k` сохраняется и управляет длиной списка кандидатов, а не прямым cosine-only ranking.
- Для совместимости с текущей Windows/Python 3.12 средой используется Chroma 1.x, потому что ветка 0.6.x требует сборку `chroma-hnswlib` и не установилась без MSVC Build Tools.
- Основной retrieval идёт только по chunk'ам knowledge base; коды и итоговые кандидаты определяются финальной LLM по retrieved context.
- Индексация knowledge base переводится в режим sync-by-default; полный rebuild сохраняется как отдельное явное действие для CLI и GUI.
- `manifest.jsonl` хранит `file_sha256` и служит источником diff для sync-индексации; смена embedding model автоматически переводит следующий `train` в полный rebuild.
- GUI-инструмент проверки чанков читает данные напрямую из текущих Chroma collections, а не пересобирает их из файлов knowledge base.
- Desktop GUI стандартизирован как единый русскоязычный экран без вкладок; в нём классификация всегда работает в сценарии "основной код + до 2 альтернатив".
- Для публикации на GitHub в репозиторий не включаются локальные `.env`, `data/vector_db`, `data/runtime`, виртуальные окружения, `__pycache__` и временные тестовые каталоги; knowledge base и исходники остаются в git.

## Next

- При возможности вручную добрать оставшиеся каталоги с ошибками доступа и завершить физическую зачистку workspace.
- Прогнать ручной smoke-test на реальном установленном `chromadb`: обычный sync, добавление нового файла, удаление файла и `--full-rebuild` из CLI/GUI.
- Прогнать ручной smoke-test нового единого GUI: классификация с текстом, классификация с файлами, проверка отображения основного кода и двух альтернатив.
- Прогнать ручной GUI smoke-test инспектора чанков на реальной БД: открытие окна, фильтр по документу, просмотр chunk text и обновление после `train`.
- Отдельно проверить на локальном старом runtime, что инспектор чанков открывается до первого нового `train`, а после нового `train` manifest автоматически обновляется до формата с `file_sha256`.
- После миграции прогнать ручные smoke-сценарии на реальных фото и mixed file inputs.
- Проверить качество retrieval на корпусе reference/examples и при необходимости уточнить chunking и prompts.
- При необходимости обновить пользовательскую документацию и legacy README, чтобы все инструкции в репозитории ссылались на `src/app/main.py`.
