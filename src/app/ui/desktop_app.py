from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.app.core.config import get_app_paths
from src.app.services.classification_service import select_primary_and_alternatives
from src.app.services.train_service import load_indexed_chunks_status, load_indexed_documents_status
from src.app.services.workflows import run_classify_workflow, run_train_workflow

ALL_DOCUMENTS_OPTION = "Все документы"


class DesktopApp:
    def __init__(
        self,
        tk_module: Any,
        ttk_module: Any,
        filedialog_module: Any,
        messagebox_module: Any,
        scrolledtext_module: Any,
    ) -> None:
        self.tk = tk_module
        self.ttk = ttk_module
        self.filedialog = filedialog_module
        self.messagebox = messagebox_module
        self.scrolledtext = scrolledtext_module
        self.paths = get_app_paths()

        self.root = self.tk.Tk()
        self.root.title("Подбор кода ТН ВЭД")
        self.root.geometry("1460x980")
        self.root.minsize(1180, 760)

        self.selected_file_paths: list[str] = []
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.is_busy = False

        self.request_status_var = self.tk.StringVar(
            value="Опишите товар и при необходимости добавьте фото, PDF, DOCX или другие файлы."
        )
        self.result_hint_var = self.tk.StringVar(
            value=(
                "После расчёта вы увидите основной код и до двух альтернатив для дополнительной проверки. "
                "Возможную выгоду по ставкам и льготам нужно подтверждать отдельно."
            )
        )
        self.index_status_var = self.tk.StringVar(value="Проверяю состояние базы знаний...")
        self.index_mode_var = self.tk.StringVar(value="")
        self.chunk_status_var = self.tk.StringVar(value="")
        self.chunk_filter_var = self.tk.StringVar(value=ALL_DOCUMENTS_OPTION)

        self._build_layout()
        self._reset_results()
        self.root.after(150, self._poll_worker_queue)
        self.root.after(0, self._refresh_index_status)

    def run(self) -> int:
        self._append_log("Интерфейс готов. Обычный запуск классификации покажет основной код и две альтернативы.")
        self.root.mainloop()
        return 0

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        shell = self.ttk.Frame(self.root, padding=12)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)

        self.canvas = self.tk.Canvas(shell, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = self.ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.content = self.ttk.Frame(self.canvas, padding=(4, 4, 8, 8))
        self.content_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.columnconfigure(0, weight=1)

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self._build_header()
        self._build_main_sections()

    def _build_header(self) -> None:
        header = self.ttk.Frame(self.content)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        self.ttk.Label(
            header,
            text="Подбор кода ТН ВЭД",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            header,
            text=(
                "Окно объединяет классификацию товара, проверку альтернативных кодов и управление базой знаний. "
                "Основная рекомендация строится на контексте, поднятом из нормативных документов и примеров."
            ),
            wraplength=1220,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _build_main_sections(self) -> None:
        top_row = self.ttk.Frame(self.content)
        top_row.grid(row=1, column=0, sticky="nsew")
        top_row.columnconfigure(0, weight=5)
        top_row.columnconfigure(1, weight=7)

        self._build_query_section(top_row)
        self._build_results_section(top_row)
        self._build_index_section()

    def _build_query_section(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="1. Запрос на классификацию", padding=12)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        self.ttk.Label(
            frame,
            text="Описание товара",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.query_text = self.scrolledtext.ScrolledText(
            frame,
            height=12,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.query_text.grid(row=1, column=0, sticky="nsew", pady=(6, 12))

        files_header = self.ttk.Frame(frame)
        files_header.grid(row=2, column=0, sticky="ew")
        files_header.columnconfigure(0, weight=1)

        self.ttk.Label(
            files_header,
            text="Файлы товара",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        file_buttons = self.ttk.Frame(files_header)
        file_buttons.grid(row=0, column=1, sticky="e")
        self.add_files_button = self.ttk.Button(file_buttons, text="Добавить файлы", command=self._add_files)
        self.add_files_button.grid(row=0, column=0, padx=(0, 6))
        self.remove_files_button = self.ttk.Button(
            file_buttons,
            text="Убрать выбранные",
            command=self._remove_selected_files,
        )
        self.remove_files_button.grid(row=0, column=1, padx=(0, 6))
        self.clear_files_button = self.ttk.Button(file_buttons, text="Очистить список", command=self._clear_files)
        self.clear_files_button.grid(row=0, column=2)

        self.files_listbox = self.tk.Listbox(
            frame,
            selectmode=self.tk.EXTENDED,
            height=8,
            font=("Consolas", 10),
        )
        self.files_listbox.grid(row=3, column=0, sticky="nsew", pady=(6, 12))

        action_row = self.ttk.Frame(frame)
        action_row.grid(row=4, column=0, sticky="ew")
        action_row.columnconfigure(1, weight=1)

        self.classify_button = self.ttk.Button(
            action_row,
            text="Подобрать код ТН ВЭД",
            command=self._start_classification,
        )
        self.classify_button.grid(row=0, column=0, sticky="w", padx=(0, 12))

        self.ttk.Label(
            action_row,
            textvariable=self.request_status_var,
            wraplength=420,
            justify="left",
        ).grid(row=0, column=1, sticky="w")

    def _build_results_section(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="2. Результат и проверка гипотез", padding=12)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        self.ttk.Label(
            frame,
            textvariable=self.result_hint_var,
            wraplength=760,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        cards_row = self.ttk.Frame(frame)
        cards_row.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        for column in range(3):
            cards_row.columnconfigure(column, weight=1)

        self.result_cards = [
            self._create_result_card(cards_row, 0, "Основная рекомендация"),
            self._create_result_card(cards_row, 1, "Альтернатива 1"),
            self._create_result_card(cards_row, 2, "Альтернатива 2"),
        ]

        details = self.ttk.Frame(frame)
        details.grid(row=2, column=0, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)
        details.rowconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)

        warnings_frame = self.ttk.LabelFrame(details, text="Предупреждения", padding=8)
        warnings_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        warnings_frame.columnconfigure(0, weight=1)
        warnings_frame.rowconfigure(0, weight=1)
        self.warnings_text = self.scrolledtext.ScrolledText(
            warnings_frame,
            height=7,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.warnings_text.grid(row=0, column=0, sticky="nsew")

        profile_frame = self.ttk.LabelFrame(details, text="Понятый профиль товара", padding=8)
        profile_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        profile_frame.columnconfigure(0, weight=1)
        profile_frame.rowconfigure(0, weight=1)
        self.profile_text = self.scrolledtext.ScrolledText(
            profile_frame,
            height=7,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.profile_text.grid(row=0, column=0, sticky="nsew")

        reference_frame = self.ttk.LabelFrame(details, text="Опорные нормативные фрагменты", padding=8)
        reference_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        reference_frame.columnconfigure(0, weight=1)
        reference_frame.rowconfigure(0, weight=1)
        self.reference_text = self.scrolledtext.ScrolledText(
            reference_frame,
            height=11,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.reference_text.grid(row=0, column=0, sticky="nsew")

        examples_frame = self.ttk.LabelFrame(details, text="Примеры и кейсы", padding=8)
        examples_frame.grid(row=1, column=1, sticky="nsew")
        examples_frame.columnconfigure(0, weight=1)
        examples_frame.rowconfigure(0, weight=1)
        self.examples_text = self.scrolledtext.ScrolledText(
            examples_frame,
            height=11,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.examples_text.grid(row=0, column=0, sticky="nsew")

        raw_frame = self.ttk.LabelFrame(frame, text="Технический JSON результата", padding=8)
        raw_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        raw_frame.columnconfigure(0, weight=1)
        raw_frame.rowconfigure(0, weight=1)
        self.raw_json_text = self.scrolledtext.ScrolledText(
            raw_frame,
            height=10,
            wrap=self.tk.WORD,
            font=("Consolas", 9),
        )
        self.raw_json_text.grid(row=0, column=0, sticky="nsew")

    def _build_index_section(self) -> None:
        frame = self.ttk.LabelFrame(self.content, text="3. База знаний и индексация", padding=12)
        frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        controls = self.ttk.Frame(frame)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)

        self.sync_button = self.ttk.Button(
            controls,
            text="Синхронизировать базу знаний",
            command=lambda: self._start_training(full_rebuild=False),
        )
        self.sync_button.grid(row=0, column=0, padx=(0, 6))

        self.rebuild_button = self.ttk.Button(
            controls,
            text="Полный rebuild",
            command=lambda: self._start_training(full_rebuild=True),
        )
        self.rebuild_button.grid(row=0, column=1, padx=(0, 6))

        self.inspect_chunks_button = self.ttk.Button(
            controls,
            text="Проверить чанки",
            command=self._open_chunk_inspector_window,
        )
        self.inspect_chunks_button.grid(row=0, column=2, padx=(0, 12))

        self.ttk.Label(
            controls,
            text=(
                "Обычная индексация добавляет новые документы, обновляет изменённые и убирает удалённые. "
                "Полный rebuild используется отдельно."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=3, sticky="w")

        status_frame = self.ttk.Frame(frame)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)

        self.ttk.Label(
            status_frame,
            textvariable=self.index_status_var,
            wraplength=560,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Label(
            status_frame,
            textvariable=self.index_mode_var,
            wraplength=560,
            justify="left",
        ).grid(row=0, column=1, sticky="w")

        body = self.ttk.Frame(frame)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        documents_frame = self.ttk.LabelFrame(body, text="Уже проиндексированные документы", padding=8)
        documents_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        documents_frame.columnconfigure(0, weight=1)
        documents_frame.rowconfigure(0, weight=1)

        self.documents_tree = self.ttk.Treeview(
            documents_frame,
            columns=("source_kind", "parser", "chunk_count"),
            show="tree headings",
            height=10,
        )
        self.documents_tree.heading("#0", text="Путь")
        self.documents_tree.heading("source_kind", text="Тип")
        self.documents_tree.heading("parser", text="Парсер")
        self.documents_tree.heading("chunk_count", text="Чанков")
        self.documents_tree.column("#0", width=420, anchor="w")
        self.documents_tree.column("source_kind", width=150, anchor="w")
        self.documents_tree.column("parser", width=90, anchor="center")
        self.documents_tree.column("chunk_count", width=90, anchor="center")
        self.documents_tree.grid(row=0, column=0, sticky="nsew")

        tree_scrollbar = self.ttk.Scrollbar(documents_frame, orient="vertical", command=self.documents_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.documents_tree.configure(yscrollcommand=tree_scrollbar.set)

        log_frame = self.ttk.LabelFrame(body, text="Журнал действий", padding=8)
        log_frame.grid(row=0, column=1, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = self.scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=self.tk.WORD,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self._set_text(self.log_text, "")

    def _create_result_card(self, parent: Any, column: int, title: str) -> dict[str, Any]:
        frame = self.ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column < 2 else 0)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

        code_var = self.tk.StringVar(value="—")
        title_var = self.tk.StringVar(value="Нет данных")
        confidence_var = self.tk.StringVar(value="Уверенность: —")
        source_var = self.tk.StringVar(value="")
        evidence_var = self.tk.StringVar(value="Опорные чанки: —")

        self.ttk.Label(
            frame,
            textvariable=code_var,
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            frame,
            textvariable=title_var,
            wraplength=250,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 4))
        self.ttk.Label(frame, textvariable=confidence_var).grid(row=2, column=0, sticky="w")
        self.ttk.Label(
            frame,
            textvariable=source_var,
            wraplength=250,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(2, 6))

        reasoning_text = self.scrolledtext.ScrolledText(
            frame,
            height=8,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        reasoning_text.grid(row=4, column=0, sticky="nsew")

        self.ttk.Label(
            frame,
            textvariable=evidence_var,
            wraplength=250,
            justify="left",
        ).grid(row=5, column=0, sticky="w", pady=(6, 0))

        return {
            "code_var": code_var,
            "title_var": title_var,
            "confidence_var": confidence_var,
            "source_var": source_var,
            "evidence_var": evidence_var,
            "reasoning_text": reasoning_text,
        }

    def _reset_results(self) -> None:
        placeholders = [
            ("Основная рекомендация пока не сформирована", "После запуска здесь появится лучший код."),
            ("Дополнительный вариант не найден", "После запуска здесь может появиться первая альтернатива."),
            ("Дополнительный вариант не найден", "После запуска здесь может появиться вторая альтернатива."),
        ]
        for card, (title, source_label) in zip(self.result_cards, placeholders):
            self._fill_result_card(card, None, empty_title=title, source_label=source_label)

        self._set_text(self.profile_text, "Профиль товара ещё не сформирован.")
        self._set_text(self.reference_text, "Нормативные фрагменты пока не загружены.")
        self._set_text(self.examples_text, "Примеры пока не загружены.")
        self._set_text(self.warnings_text, "Предупреждений пока нет.")
        self._set_text(self.raw_json_text, "{}")

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                event, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break

            if event == "progress":
                self._append_log(str(payload))
                continue
            if event == "classify_done":
                self._apply_classification_result(payload)
                continue
            if event == "train_done":
                self._apply_training_result(payload)
                continue
            if event == "error":
                title, message = payload
                self._append_log(f"{title}: {message}")
                self.messagebox.showerror(title, message)
                self.request_status_var.set(message)
                self.index_status_var.set(message)
                continue
            if event == "task_finished":
                self._set_busy(False)

        self.root.after(150, self._poll_worker_queue)

    def _refresh_index_status(self) -> None:
        try:
            status = load_indexed_documents_status(self.paths)
        except Exception as exc:
            self.index_status_var.set(f"Не удалось прочитать состояние ВБД: {exc}")
            self.index_mode_var.set("")
            self._populate_documents_tree([])
            return

        self.index_status_var.set(status["message"])
        self.index_mode_var.set(
            "Чанки в ВБД: "
            f"reference={status['reference_chunk_count']}, "
            f"examples={status['example_chunk_count']}."
        )
        self._populate_documents_tree(status["documents"])

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        current = self.log_text.get("1.0", "end").strip()
        prefix = "" if not current else "\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{prefix}[{timestamp}] {message}")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _start_classification(self) -> None:
        if self.is_busy:
            return

        raw_text = self.query_text.get("1.0", "end").strip()
        file_paths = list(self.selected_file_paths)
        if not raw_text and not file_paths:
            self.messagebox.showwarning(
                "Недостаточно данных",
                "Введите описание товара или добавьте хотя бы один файл.",
            )
            return

        self._reset_results()
        self.request_status_var.set("Классификация выполняется. Собираю профиль товара и контекст из базы знаний...")
        self._append_log(
            f"Старт классификации: текст={'да' if bool(raw_text) else 'нет'}, файлов={len(file_paths)}, top_k=3."
        )
        self._set_busy(True)
        thread = threading.Thread(
            target=self._classify_worker,
            args=(raw_text, file_paths),
            daemon=True,
        )
        thread.start()

    def _classify_worker(self, raw_text: str, file_paths: list[str]) -> None:
        def progress(message: str) -> None:
            self.worker_queue.put(("progress", message))

        try:
            payload = run_classify_workflow(
                raw_text=raw_text,
                file_paths=file_paths,
                top_k=3,
                progress_callback=progress,
            )
            self.worker_queue.put(("classify_done", payload))
        except Exception as exc:
            self.worker_queue.put(("error", ("Ошибка классификации", str(exc))))
        finally:
            self.worker_queue.put(("task_finished", None))

    def _start_training(self, full_rebuild: bool) -> None:
        if self.is_busy:
            return

        if full_rebuild:
            approved = self.messagebox.askyesno(
                "Подтвердите полный rebuild",
                (
                    "Полный rebuild очистит текущие Chroma collections и заново пересоберёт индекс.\n\n"
                    "Продолжить?"
                ),
            )
            if not approved:
                return

        mode_label = "полный rebuild" if full_rebuild else "синхронизация"
        self.index_status_var.set(f"Выполняется {mode_label} базы знаний...")
        self._append_log(f"Старт индексации: режим={mode_label}.")
        self._set_busy(True)
        thread = threading.Thread(
            target=self._train_worker,
            args=(full_rebuild,),
            daemon=True,
        )
        thread.start()

    def _train_worker(self, full_rebuild: bool) -> None:
        def progress(message: str) -> None:
            self.worker_queue.put(("progress", message))

        try:
            metadata = run_train_workflow(progress_callback=progress, full_rebuild=full_rebuild)
            self.worker_queue.put(("train_done", metadata))
        except Exception as exc:
            self.worker_queue.put(("error", ("Ошибка индексации", str(exc))))
        finally:
            self.worker_queue.put(("task_finished", None))

    def _apply_classification_result(self, payload: dict[str, Any]) -> None:
        result = payload["result"]
        recommendation_view = select_primary_and_alternatives(result, alternative_limit=2)
        primary = recommendation_view["primary"]
        alternatives = recommendation_view["alternatives"]
        primary_source = recommendation_view["primary_source"]

        if primary_source == "best_match":
            self.result_hint_var.set(
                "Слева показан лучший код по мнению модели. Два соседних блока содержат альтернативы, "
                "которые стоит дополнительно проверить по описанию товара, ставкам и ограничениям."
            )
        elif primary_source == "candidate":
            self.result_hint_var.set(
                "Модель не смогла уверенно назвать лучший код, поэтому первым показан наиболее сильный кандидат. "
                "Остальные блоки содержат дополнительные варианты для проверки."
            )
        else:
            self.result_hint_var.set(
                "Ни один код не определён уверенно. Проверьте предупреждения, retrieval-контекст и полноту описания товара."
            )

        self._fill_result_card(
            self.result_cards[0],
            primary,
            empty_title="Основная рекомендация пока не сформирована",
            source_label=(
                "Лучший код по мнению модели."
                if primary_source == "best_match"
                else "Наиболее сильный кандидат при недостаточной уверенности."
            ),
        )
        for index in range(2):
            item = alternatives[index] if index < len(alternatives) else None
            self._fill_result_card(
                self.result_cards[index + 1],
                item,
                empty_title="Дополнительный вариант не найден",
                source_label=(
                    "Альтернативная гипотеза для дополнительной проверки."
                    if item
                    else "В этом запуске модель не вернула ещё один самостоятельный вариант."
                ),
            )

        self._set_text(self.profile_text, self._format_product_profile(payload["result"]["product_profile"]))
        self._set_text(self.reference_text, self._format_hits(payload["result"]["retrieval"]["reference"]))
        self._set_text(self.examples_text, self._format_hits(payload["result"]["retrieval"]["examples"]))
        self._set_text(self.warnings_text, self._format_warnings(payload["result"]["warnings"]))
        self._set_text(self.raw_json_text, json.dumps(payload["result"], ensure_ascii=False, indent=2))

        primary_code = primary["code"] if primary else "не определён"
        self.request_status_var.set(f"Классификация завершена. Основной код: {primary_code}.")
        self._append_log(f"Классификация завершена. Основной код: {primary_code}.")

    def _apply_training_result(self, metadata: dict[str, Any]) -> None:
        mode_label = self._format_index_mode(metadata.get("index_mode"))
        sync_stats = metadata.get("sync_stats", {})
        self.index_status_var.set(
            "Индексация завершена: "
            f"{mode_label}. "
            f"reference={metadata.get('reference_document_count', 0)}, "
            f"examples={metadata.get('example_document_count', 0)}."
        )
        self.index_mode_var.set(
            "Изменения: "
            f"added={sync_stats.get('added', 0)}, "
            f"changed={sync_stats.get('changed', 0)}, "
            f"removed={sync_stats.get('removed', 0)}, "
            f"unchanged={sync_stats.get('unchanged', 0)}, "
            f"processed={sync_stats.get('processed_documents', 0)}."
        )
        self._append_log(
            f"Индексация завершена: режим={mode_label}, processed={sync_stats.get('processed_documents', 0)}."
        )
        self._refresh_index_status()
        if hasattr(self, "chunk_window") and self.chunk_window.winfo_exists():
            self._refresh_chunk_inspector_view()

    def _populate_documents_tree(self, documents: list[dict[str, Any]]) -> None:
        for item_id in self.documents_tree.get_children():
            self.documents_tree.delete(item_id)

        for index, item in enumerate(documents):
            self.documents_tree.insert(
                "",
                "end",
                iid=f"doc_{index}",
                text=item["path"],
                values=(
                    self._translate_source_kind(item["source_kind"]),
                    item["parser"],
                    item["chunk_count"],
                ),
            )

    def _open_chunk_inspector_window(self) -> None:
        if hasattr(self, "chunk_window") and self.chunk_window.winfo_exists():
            self.chunk_window.focus_set()
            self._refresh_chunk_inspector_view()
            return

        self.chunk_window = self.tk.Toplevel(self.root)
        self.chunk_window.title("Проверка чанков")
        self.chunk_window.geometry("1240x760")
        self.chunk_window.minsize(980, 620)
        self.chunk_window.columnconfigure(0, weight=1)
        self.chunk_window.rowconfigure(1, weight=1)

        top = self.ttk.Frame(self.chunk_window, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        self.ttk.Label(top, text="Документ").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.chunk_filter_combo = self.ttk.Combobox(
            top,
            textvariable=self.chunk_filter_var,
            state="readonly",
        )
        self.chunk_filter_combo.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.chunk_filter_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_chunk_inspector_view())

        refresh_button = self.ttk.Button(top, text="Обновить", command=self._refresh_chunk_inspector_view)
        refresh_button.grid(row=0, column=2, sticky="e")

        self.ttk.Label(
            top,
            textvariable=self.chunk_status_var,
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        body = self.ttk.Frame(self.chunk_window, padding=(12, 0, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        table_frame = self.ttk.LabelFrame(body, text="Список чанков", padding=8)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.chunks_tree = self.ttk.Treeview(
            table_frame,
            columns=("source_kind", "chunk_index", "codes"),
            show="tree headings",
            height=18,
        )
        self.chunks_tree.heading("#0", text="Документ")
        self.chunks_tree.heading("source_kind", text="Тип")
        self.chunks_tree.heading("chunk_index", text="№ чанка")
        self.chunks_tree.heading("codes", text="Упомянутые коды")
        self.chunks_tree.column("#0", width=360, anchor="w")
        self.chunks_tree.column("source_kind", width=120, anchor="w")
        self.chunks_tree.column("chunk_index", width=90, anchor="center")
        self.chunks_tree.column("codes", width=220, anchor="w")
        self.chunks_tree.grid(row=0, column=0, sticky="nsew")
        chunk_scrollbar = self.ttk.Scrollbar(table_frame, orient="vertical", command=self.chunks_tree.yview)
        chunk_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chunks_tree.configure(yscrollcommand=chunk_scrollbar.set)
        self.chunks_tree.bind("<<TreeviewSelect>>", self._show_chunk_details)

        details_frame = self.ttk.Frame(body)
        details_frame.grid(row=0, column=1, sticky="nsew")
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        details_frame.rowconfigure(1, weight=1)

        text_frame = self.ttk.LabelFrame(details_frame, text="Текст чанка", padding=8)
        text_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.chunk_text_widget = self.scrolledtext.ScrolledText(
            text_frame,
            height=12,
            wrap=self.tk.WORD,
            font=("Segoe UI", 10),
        )
        self.chunk_text_widget.grid(row=0, column=0, sticky="nsew")

        metadata_frame = self.ttk.LabelFrame(details_frame, text="Metadata чанка", padding=8)
        metadata_frame.grid(row=1, column=0, sticky="nsew")
        metadata_frame.columnconfigure(0, weight=1)
        metadata_frame.rowconfigure(0, weight=1)
        self.chunk_metadata_widget = self.scrolledtext.ScrolledText(
            metadata_frame,
            height=12,
            wrap=self.tk.WORD,
            font=("Consolas", 9),
        )
        self.chunk_metadata_widget.grid(row=0, column=0, sticky="nsew")

        self.chunk_records: dict[str, dict[str, Any]] = {}
        self._set_text(self.chunk_text_widget, "")
        self._set_text(self.chunk_metadata_widget, "")
        self._refresh_chunk_inspector_view()

    def _refresh_chunk_inspector_view(self) -> None:
        selected_value = self.chunk_filter_var.get().strip()
        source_path = None if selected_value in {"", ALL_DOCUMENTS_OPTION} else selected_value

        try:
            status = load_indexed_chunks_status(self.paths, source_path=source_path)
        except Exception as exc:
            self.chunk_status_var.set(f"Не удалось загрузить чанки: {exc}")
            self._populate_chunk_tree([])
            return

        options = [ALL_DOCUMENTS_OPTION] + list(status["document_options"])
        self.chunk_filter_combo.configure(values=options)
        if source_path and source_path not in status["document_options"]:
            self.chunk_filter_var.set(ALL_DOCUMENTS_OPTION)
        elif source_path:
            self.chunk_filter_var.set(source_path)
        else:
            self.chunk_filter_var.set(ALL_DOCUMENTS_OPTION)

        self.chunk_status_var.set(status["message"])
        self._populate_chunk_tree(status["chunks"])

    def _populate_chunk_tree(self, chunks: list[dict[str, Any]]) -> None:
        if not hasattr(self, "chunks_tree"):
            return

        self.chunk_records = {}
        for item_id in self.chunks_tree.get_children():
            self.chunks_tree.delete(item_id)

        for index, chunk in enumerate(chunks):
            item_id = f"chunk_{index}"
            self.chunk_records[item_id] = chunk
            self.chunks_tree.insert(
                "",
                "end",
                iid=item_id,
                text=chunk.get("source_path", ""),
                values=(
                    self._translate_source_kind(str(chunk.get("source_kind", ""))),
                    chunk.get("chunk_index", 0),
                    ", ".join(chunk.get("mentioned_codes", [])) or "—",
                ),
            )

        if chunks:
            first_item = self.chunks_tree.get_children()[0]
            self.chunks_tree.selection_set(first_item)
            self._show_chunk_details()
        else:
            self._set_text(self.chunk_text_widget, "Чанки не найдены для выбранного фильтра.")
            self._set_text(self.chunk_metadata_widget, "")

    def _show_chunk_details(self, _event: Any | None = None) -> None:
        if not hasattr(self, "chunks_tree"):
            return

        selected_items = self.chunks_tree.selection()
        if not selected_items:
            return

        chunk = self.chunk_records.get(selected_items[0])
        if chunk is None:
            return

        self._set_text(self.chunk_text_widget, chunk.get("text", ""))
        metadata_payload = {
            "chunk_id": chunk.get("chunk_id"),
            "source_path": chunk.get("source_path"),
            "source_kind": chunk.get("source_kind"),
            "document_type": chunk.get("document_type"),
            "section_context": chunk.get("section_context"),
            "mentioned_codes": chunk.get("mentioned_codes", []),
            "chunk_index": chunk.get("chunk_index", 0),
            "score": chunk.get("score", 0.0),
        }
        self._set_text(self.chunk_metadata_widget, json.dumps(metadata_payload, ensure_ascii=False, indent=2))

    def _fill_result_card(
        self,
        card: dict[str, Any],
        item: dict[str, Any] | None,
        empty_title: str,
        source_label: str,
    ) -> None:
        if not item:
            card["code_var"].set("—")
            card["title_var"].set(empty_title)
            card["confidence_var"].set("Уверенность: —")
            card["source_var"].set(source_label)
            card["evidence_var"].set("Опорные чанки: —")
            self._set_text(card["reasoning_text"], "Нет достаточных данных для отдельной рекомендации.")
            return

        card["code_var"].set(str(item.get("code", "—")) or "—")
        card["title_var"].set(str(item.get("title_or_label", "без названия")) or "без названия")
        card["confidence_var"].set(f"Уверенность: {float(item.get('confidence', 0.0)):.2f}")
        card["source_var"].set(source_label)
        evidence_ids = [str(value).strip() for value in item.get("evidence_ids", []) if str(value).strip()]
        card["evidence_var"].set("Опорные чанки: " + (", ".join(evidence_ids) if evidence_ids else "не указаны"))
        self._set_text(card["reasoning_text"], str(item.get("reasoning", "")).strip() or "Обоснование не возвращено.")

    def _add_files(self) -> None:
        file_paths = self.filedialog.askopenfilenames(
            title="Выберите файлы товара",
            initialdir=str(self.paths.project_root),
        )
        if not file_paths:
            return

        known_paths = {Path(item).resolve().as_posix() for item in self.selected_file_paths}
        added_count = 0
        for raw_path in file_paths:
            normalized = Path(raw_path).resolve().as_posix()
            if normalized in known_paths:
                continue
            known_paths.add(normalized)
            self.selected_file_paths.append(normalized)
            added_count += 1

        self._sync_files_listbox()
        self.request_status_var.set(
            f"Файлов в запросе: {len(self.selected_file_paths)}. Добавлено новых: {added_count}."
        )

    def _remove_selected_files(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return

        for index in reversed(selected_indices):
            del self.selected_file_paths[index]

        self._sync_files_listbox()
        self.request_status_var.set(f"Файлов в запросе: {len(self.selected_file_paths)}.")

    def _clear_files(self) -> None:
        self.selected_file_paths.clear()
        self._sync_files_listbox()
        self.request_status_var.set("Список файлов очищен.")

    def _sync_files_listbox(self) -> None:
        self.files_listbox.delete(0, self.tk.END)
        for file_path in self.selected_file_paths:
            self.files_listbox.insert(self.tk.END, file_path)

    def _set_busy(self, is_busy: bool) -> None:
        self.is_busy = is_busy
        state = "disabled" if is_busy else "normal"
        for button in (
            self.classify_button,
            self.add_files_button,
            self.remove_files_button,
            self.clear_files_button,
            self.sync_button,
            self.rebuild_button,
            self.inspect_chunks_button,
        ):
            button.configure(state=state)

    def _format_product_profile(self, profile: dict[str, Any]) -> str:
        key_features = ", ".join(str(item).strip() for item in profile.get("key_features", []) if str(item).strip()) or "—"
        search_queries = ", ".join(
            str(item).strip() for item in profile.get("search_queries", []) if str(item).strip()
        ) or "—"
        missing = ", ".join(
            str(item).strip() for item in profile.get("missing_information", []) if str(item).strip()
        ) or "—"
        return "\n".join(
            [
                f"Краткое описание: {profile.get('summary', '—')}",
                f"Назначение: {profile.get('intended_use', '—')}",
                f"Материал / состав: {profile.get('material_or_composition', '—')}",
                f"Ключевые признаки: {key_features}",
                f"Поисковые запросы: {search_queries}",
                f"Неопределённость: {profile.get('uncertainty', '—')}",
                f"Чего не хватает: {missing}",
            ]
        )

    def _format_hits(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "Подходящие фрагменты не найдены."

        blocks: list[str] = []
        for index, item in enumerate(hits, start=1):
            blocks.append(
                f"{index}. {item.get('source_path', '—')} | score={float(item.get('score', 0.0)):.2f}"
            )
            if item.get("section_context"):
                blocks.append(f"Раздел: {item['section_context']}")
            if item.get("mentioned_codes"):
                blocks.append(f"Упомянутые коды: {', '.join(item['mentioned_codes'])}")
            blocks.append(str(item.get("text", "")).strip() or "Текст фрагмента пуст.")
            blocks.append("")
        return "\n".join(blocks).strip()

    def _format_warnings(self, warnings: list[str]) -> str:
        if not warnings:
            return "Явных предупреждений нет."
        return "\n".join(f"- {warning}" for warning in warnings)

    def _translate_source_kind(self, value: str) -> str:
        return {
            "reference": "Нормативный",
            "example": "Пример",
        }.get(value, value or "неизвестно")

    def _format_index_mode(self, value: Any) -> str:
        return {
            "sync": "синхронизация",
            "full_rebuild": "полный rebuild",
        }.get(str(value), str(value))

    def _set_text(self, widget: Any, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _on_content_configure(self, _event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        self.canvas.itemconfigure(self.content_id, width=event.width)

    def _on_mousewheel(self, event: Any) -> None:
        if self.canvas.winfo_exists():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def launch_gui_app() -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    app = DesktopApp(
        tk_module=tk,
        ttk_module=ttk,
        filedialog_module=filedialog,
        messagebox_module=messagebox,
        scrolledtext_module=scrolledtext,
    )
    return app.run()
