"""Unit tests for the SegmentIQ pipeline.

All external dependencies (Ollama, SAM 3, filesystem) are mocked so tests
run without a GPU or network access.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from describer import ImageDescription
from segmenter import Instance, SegmentationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seg_result(query: str, n: int = 1) -> SegmentationResult:
    return SegmentationResult(
        query=query,
        objects_found=n,
        instances=[
            Instance(id=i, confidence=0.9, bounding_box=[0.0, 0.0, 10.0, 10.0], mask="")
            for i in range(1, n + 1)
        ],
    )


# ---------------------------------------------------------------------------
# describer tests
# ---------------------------------------------------------------------------

class TestDescribeImage(unittest.TestCase):
    def test_returns_image_description(self) -> None:
        from describer import describe_image

        mock_response = {
            "message": {
                "content": json.dumps({
                    "description": "A field with animals.",
                    "objects": ["lion", "zebra"],
                })
            }
        }

        with patch("describer.ollama.chat", return_value=mock_response), \
             patch("describer.Path.exists", return_value=True):
            result = describe_image("fake.jpg")

        self.assertEqual(result.description, "A field with animals.")
        self.assertEqual(result.objects, ["lion", "zebra"])

    def test_strips_markdown_fences(self) -> None:
        from describer import describe_image

        content = "```json\n" + json.dumps({
            "description": "A street scene.",
            "objects": ["car", "person"],
        }) + "\n```"

        mock_response = {"message": {"content": content}}

        with patch("describer.ollama.chat", return_value=mock_response), \
             patch("describer.Path.exists", return_value=True):
            result = describe_image("fake.jpg")

        self.assertEqual(result.objects, ["car", "person"])

    def test_missing_image_raises(self) -> None:
        from describer import describe_image
        with self.assertRaises(FileNotFoundError):
            describe_image("nonexistent_image.jpg")


# ---------------------------------------------------------------------------
# pipeline tests
# ---------------------------------------------------------------------------

class TestProcessImage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        # Create a minimal valid JPEG as the "input" image
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (64, 64), color=(100, 100, 100))
        self.image_path = os.path.join(self.tmp, "test.jpg")
        img.save(self.image_path)

    def _run(self, objects: list[str]) -> "pipeline.ImageResult":  # type: ignore[name-defined]
        from pipeline import process_image

        desc = ImageDescription(description="A test image.", objects=objects)
        seg_results = [_make_seg_result(obj) for obj in objects]

        with patch("pipeline.describe_image", return_value=desc), \
             patch("pipeline.segment_image", side_effect=seg_results), \
             patch("pipeline.annotate"):
            return process_image(self.image_path, self.tmp)

    def test_calls_segment_once_per_object(self) -> None:
        from pipeline import process_image

        objects = ["cat", "dog", "bird"]
        desc = ImageDescription(description="Pets.", objects=objects)
        seg_results = [_make_seg_result(obj) for obj in objects]

        with patch("pipeline.describe_image", return_value=desc) as mock_desc, \
             patch("pipeline.segment_image", side_effect=seg_results) as mock_seg, \
             patch("pipeline.annotate"):
            process_image(self.image_path, self.tmp)

        mock_desc.assert_called_once()
        self.assertEqual(mock_seg.call_count, 3)

    def test_metadata_json_written(self) -> None:
        result = self._run(["lion", "zebra"])
        metadata_path = Path(self.tmp) / "test_metadata.json"
        self.assertTrue(metadata_path.exists())
        data = json.loads(metadata_path.read_text())
        self.assertEqual(data["description"], "A test image.")
        self.assertEqual(data["objects"], ["lion", "zebra"])
        self.assertEqual(len(data["segmentation"]), 2)

    def test_result_has_correct_structure(self) -> None:
        result = self._run(["cat"])
        self.assertEqual(result.description, "A test image.")
        self.assertEqual(result.objects, ["cat"])
        self.assertEqual(len(result.segmentation), 1)
        self.assertEqual(result.segmentation[0].query, "cat")

    def test_empty_object_list(self) -> None:
        result = self._run([])
        self.assertEqual(result.objects, [])
        self.assertEqual(result.segmentation, [])


class TestProcessDirectory(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        from PIL import Image as PILImage
        for name in ("a.jpg", "b.jpg", "c.png"):
            img = PILImage.new("RGB", (64, 64), color=(50, 50, 50))
            img.save(os.path.join(self.tmp, name))
        # A non-image file that should be ignored
        Path(os.path.join(self.tmp, "notes.txt")).write_text("ignore me")

    def test_processes_all_images(self) -> None:
        from pipeline import process_directory

        desc = ImageDescription(description="x", objects=["cat"])
        seg = [_make_seg_result("cat")]

        with patch("pipeline.describe_image", return_value=desc), \
             patch("pipeline.segment_image", side_effect=seg * 10), \
             patch("pipeline.annotate"):
            results = process_directory(self.tmp, os.path.join(self.tmp, "out"))

        self.assertEqual(len(results), 3)

    def test_ignores_non_image_files(self) -> None:
        from pipeline import process_directory

        desc = ImageDescription(description="x", objects=[])

        with patch("pipeline.describe_image", return_value=desc), \
             patch("pipeline.segment_image", return_value=_make_seg_result("x")), \
             patch("pipeline.annotate"):
            results = process_directory(self.tmp, os.path.join(self.tmp, "out"))

        paths = [r.image_path for r in results]
        self.assertFalse(any("notes.txt" in p for p in paths))


if __name__ == "__main__":
    unittest.main()
