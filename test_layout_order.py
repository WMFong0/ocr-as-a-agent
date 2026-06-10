#!/usr/bin/env python3
"""Regression tests for OCR text layout ordering."""

import unittest

from main import _extract_text_from_result


def _line(text: str, left: int, top: int, right: int, bottom: int) -> dict[str, object]:
    return {
        "text": text,
        "boundingBox": [left, top, right, top, right, bottom, left, bottom],
    }


class LayoutOrderTest(unittest.TestCase):
    def test_multi_column_lines_are_read_top_to_bottom_per_column(self) -> None:
        result = {
            "analyzeResult": {
                "readResults": [
                    {
                        "lines": [
                            _line("Title", 10, 10, 500, 30),
                            _line("A heading", 10, 120, 200, 140),
                            _line("B heading", 250, 120, 440, 140),
                            _line("C heading", 490, 120, 680, 140),
                            _line("A paragraph 1", 10, 155, 200, 175),
                            _line("B paragraph 1", 250, 155, 440, 175),
                            _line("C paragraph 1", 490, 155, 680, 175),
                            _line("A paragraph 2", 10, 180, 200, 200),
                            _line("B paragraph 2", 250, 180, 440, 200),
                            _line("C paragraph 2", 490, 180, 680, 200),
                        ]
                    }
                ]
            }
        }

        self.assertEqual(
            _extract_text_from_result(result).splitlines(),
            [
                "Title",
                "A heading",
                "A paragraph 1",
                "A paragraph 2",
                "B heading",
                "B paragraph 1",
                "B paragraph 2",
                "C heading",
                "C paragraph 1",
                "C paragraph 2",
            ],
        )

    def test_plain_lines_still_work_without_bounding_boxes(self) -> None:
        result = {
            "readResult": {
                "blocks": [
                    {
                        "lines": [
                            {"text": "First line"},
                            {"text": "Second line"},
                        ]
                    }
                ]
            }
        }

        self.assertEqual(_extract_text_from_result(result), "First line\nSecond line")

    def test_full_width_heading_can_precede_columns_without_large_gap(self) -> None:
        result = {
            "analyzeResult": {
                "readResults": [
                    {
                        "lines": [
                            _line("Wide title", 10, 100, 680, 120),
                            _line("Wide subtitle", 10, 128, 680, 148),
                            _line("A heading", 10, 165, 200, 185),
                            _line("B heading", 250, 165, 440, 185),
                            _line("C heading", 490, 165, 680, 185),
                            _line("A body", 10, 190, 200, 210),
                            _line("B body", 250, 190, 440, 210),
                            _line("C body", 490, 190, 680, 210),
                        ]
                    }
                ]
            }
        }

        self.assertEqual(
            _extract_text_from_result(result).splitlines(),
            [
                "Wide title",
                "Wide subtitle",
                "A heading",
                "A body",
                "B heading",
                "B body",
                "C heading",
                "C body",
            ],
        )


if __name__ == "__main__":
    unittest.main()
