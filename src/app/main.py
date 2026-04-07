from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.app.services.workflows import run_classify_workflow, run_train_workflow
from src.app.ui.desktop_app import launch_gui_app
from src.app.utils.io_utils import format_json, print_stderr, print_stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Локальное приложение для индексации базы знаний и подбора кодов ТН ВЭД по тексту, фото и файлам.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help="Синхронизировать embedded vector DB по docs/reference и docs/examples.",
    )
    train_parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Полностью пересобрать индекс вместо инкрементальной синхронизации.",
    )
    train_parser.set_defaults(handler=handle_train)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Подобрать лучший код и 3-4 кандидата по тексту, фото и дополнительным файлам.",
    )
    classify_parser.add_argument("--text", default="", help="Текстовое описание товара.")
    classify_parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Путь к файлу пользователя. Можно передавать несколько раз.",
    )
    classify_parser.add_argument(
        "--top-k",
        type=int,
        choices=(3, 4),
        default=4,
        help="Сколько кандидатов вернуть: 3 или 4.",
    )
    classify_parser.set_defaults(handler=handle_classify)

    gui_parser = subparsers.add_parser("gui", help="Открыть русскоязычный desktop-интерфейс.")
    gui_parser.set_defaults(handler=handle_gui)
    return parser


def handle_train(args: argparse.Namespace) -> int:
    metadata = run_train_workflow(progress_callback=print_stdout, full_rebuild=bool(args.full_rebuild))
    print_stdout(format_json(metadata))
    return 0


def handle_classify(args: argparse.Namespace) -> int:
    payload = run_classify_workflow(
        raw_text=args.text,
        file_paths=args.file,
        top_k=args.top_k,
        progress_callback=print_stdout,
    )
    print_stdout(format_json(payload["result"]))
    print_stdout("")
    print_stdout(payload["summary"])
    return 0


def handle_gui(args: argparse.Namespace) -> int:
    del args
    return launch_gui_app()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        return 1

    try:
        return args.handler(args)
    except Exception as exc:
        print_stderr(f"Ошибка: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
