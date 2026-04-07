from __future__ import annotations

import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import main


class MainCliParserTests(unittest.TestCase):
    def test_parser_supports_gui_command(self) -> None:
        parser = main.build_parser()
        args = parser.parse_args(["gui"])

        self.assertIs(args.handler, main.handle_gui)

    def test_parser_supports_top_k_for_classify(self) -> None:
        parser = main.build_parser()
        args = parser.parse_args(["classify", "--text", "лошади", "--top-k", "3"])

        self.assertIs(args.handler, main.handle_classify)
        self.assertEqual(args.top_k, 3)

    def test_parser_supports_train_without_full_rebuild(self) -> None:
        parser = main.build_parser()
        args = parser.parse_args(["train"])

        self.assertIs(args.handler, main.handle_train)
        self.assertFalse(args.full_rebuild)

    def test_parser_supports_train_full_rebuild(self) -> None:
        parser = main.build_parser()
        args = parser.parse_args(["train", "--full-rebuild"])

        self.assertIs(args.handler, main.handle_train)
        self.assertTrue(args.full_rebuild)

    def test_parser_rejects_legacy_max_alternates_argument(self) -> None:
        parser = main.build_parser()
        with patch("sys.stderr", new=StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["classify", "--text", "лошади", "--max-alternates", "3"])
