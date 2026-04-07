from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.services.input_analysis_service import analyze_multimodal_input


class InputAnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="tnved_input_analysis_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.env_patcher = patch.dict(os.environ, {"TNVED_PROJECT_ROOT": self.temp_dir})
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

    def test_text_only_query_returns_valid_product_profile(self) -> None:
        payload = {
            "summary": "Маршрутизатор для передачи данных",
            "intended_use": "Домашняя сеть",
            "material_or_composition": "Электронное устройство",
            "key_features": ["wifi", "router"],
            "search_queries": ["маршрутизатор wifi", "устройство передачи данных"],
            "uncertainty": "Средняя",
            "missing_information": ["точная модель"],
        }

        with patch("src.app.services.input_analysis_service.chat_json", return_value=payload):
            result = analyze_multimodal_input("Wi-Fi маршрутизатор для дома", [])

        self.assertEqual(result["product_profile"]["summary"], payload["summary"])
        self.assertEqual(len(result["product_profile"]["search_queries"]), 2)

    def test_image_query_uses_image_url_even_when_ocr_is_unavailable(self) -> None:
        image_path = Path(self.temp_dir) / "photo.png"
        image_path.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
                "0000000C4944415408D763F8FFFF3F0005FE02FEA7A95D2D0000000049454E44AE426082"
            )
        )
        captured_messages: list[dict] = []

        def fake_chat_json(messages, temperature=0.1):
            del temperature
            captured_messages.extend(messages)
            return {
                "summary": "Фото товара",
                "intended_use": "Неизвестно",
                "material_or_composition": "",
                "key_features": ["визуальный анализ"],
                "search_queries": ["товар по фото"],
                "uncertainty": "Высокая",
                "missing_information": ["состав"],
            }

        with (
            patch("src.app.services.input_analysis_service.try_extract_text_from_image", return_value=("", "ocr unavailable")),
            patch("src.app.services.input_analysis_service.chat_json", side_effect=fake_chat_json),
        ):
            analyze_multimodal_input("", [str(image_path)])

        user_message = captured_messages[1]
        content_items = user_message["content"]
        self.assertTrue(any(item.get("type") == "image_url" for item in content_items))
