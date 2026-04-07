from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.services.classification_service import render_human_summary, select_primary_and_alternatives


class ClassificationServiceTests(unittest.TestCase):
    def test_select_primary_and_alternatives_skips_duplicate_best_match(self) -> None:
        result = {
            "best_match": {
                "code": "0101210000",
                "title_or_label": "Чистопородные племенные лошади",
                "confidence": 0.94,
                "reasoning": "Совпадает с reference evidence.",
                "evidence_ids": ["reference:docs/reference/official.txt::chunk-0"],
            },
            "candidates": [
                {
                    "code": "0101210000",
                    "title_or_label": "Чистопородные племенные лошади",
                    "confidence": 0.94,
                    "reasoning": "Совпадает с reference evidence.",
                    "evidence_ids": ["reference:docs/reference/official.txt::chunk-0"],
                },
                {
                    "code": "0101291000",
                    "title_or_label": "Прочие лошади",
                    "confidence": 0.51,
                    "reasoning": "Альтернатива при неподтверждённом племенном назначении.",
                    "evidence_ids": ["example:docs/examples/case.txt::chunk-0"],
                },
                {
                    "code": "0101299000",
                    "title_or_label": "Прочие живые животные",
                    "confidence": 0.24,
                    "reasoning": "Запасная гипотеза.",
                    "evidence_ids": ["example:docs/examples/case-2.txt::chunk-0"],
                },
            ],
            "warnings": [],
        }

        view = select_primary_and_alternatives(result, alternative_limit=2)

        self.assertEqual(view["primary_source"], "best_match")
        self.assertEqual(view["primary"]["code"], "0101210000")
        self.assertEqual([item["code"] for item in view["alternatives"]], ["0101291000", "0101299000"])

    def test_select_primary_and_alternatives_uses_first_candidate_when_best_missing(self) -> None:
        result = {
            "best_match": None,
            "candidates": [
                {
                    "code": "8471300000",
                    "title_or_label": "Портативные вычислительные машины",
                    "confidence": 0.63,
                    "reasoning": "Наиболее сильный кандидат.",
                    "evidence_ids": ["reference:docs/reference/notebooks.txt::chunk-1"],
                },
                {
                    "code": "8471410000",
                    "title_or_label": "Прочие вычислительные машины",
                    "confidence": 0.44,
                    "reasoning": "Проверить, если конфигурация не портативная.",
                    "evidence_ids": ["reference:docs/reference/notebooks.txt::chunk-2"],
                },
            ],
            "warnings": ["Нужно уточнить состав поставки."],
        }

        view = select_primary_and_alternatives(result, alternative_limit=2)

        self.assertEqual(view["primary_source"], "candidate")
        self.assertEqual(view["primary"]["code"], "8471300000")
        self.assertEqual([item["code"] for item in view["alternatives"]], ["8471410000"])

    def test_render_human_summary_uses_russian_warning_label(self) -> None:
        summary = render_human_summary(
            {
                "best_match": None,
                "candidates": [
                    {
                        "code": "8471300000",
                        "title_or_label": "Портативные вычислительные машины",
                        "confidence": 0.63,
                        "reasoning": "Наиболее сильный кандидат.",
                        "evidence_ids": ["reference:docs/reference/notebooks.txt::chunk-1"],
                    }
                ],
                "warnings": ["Нужно уточнить состав поставки."],
            }
        )

        self.assertIn("Основной кандидат: 8471300000", summary)
        self.assertIn("Предупреждения:", summary)


if __name__ == "__main__":
    unittest.main()
