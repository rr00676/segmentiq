from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from annotator import annotate
from describer import describe_image
from segmenter import SegmentationResult, segment_image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageResult:
    image_path: str
    description: str
    objects: List[str]
    segmentation: List[SegmentationResult] = field(default_factory=list)


def process_image(image_path: str, output_dir: str) -> ImageResult:
    """Run the full Gemma → SAM 3 pipeline on a single image.

    Steps:
        1. Gemma 4 describes the image and returns a list of object types.
        2. SAM 3 segments each object type.
        3. Annotated image saved to output_dir/<stem>_annotated.jpg.
        4. Metadata JSON saved to output_dir/<stem>_metadata.json.

    Args:
        image_path: Path to the input image.
        output_dir: Directory where outputs are saved (created if absent).

    Returns:
        ImageResult with description, object list, and segmentation results.
    """
    src = Path(image_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1 — describe
    desc = describe_image(str(src))

    # Step 2 — segment each object type
    seg_results: List[SegmentationResult] = []
    for obj in desc.objects:
        result = segment_image(str(src), obj)
        seg_results.append(result)

    # Step 3 — annotate
    annotated_path = out / f"{src.stem}_annotated.jpg"
    annotate(str(src), seg_results, str(annotated_path))

    # Step 4 — save metadata
    image_result = ImageResult(
        image_path=str(src),
        description=desc.description,
        objects=desc.objects,
        segmentation=seg_results,
    )
    metadata_path = out / f"{src.stem}_metadata.json"
    metadata_path.write_text(
        json.dumps(dataclasses.asdict(image_result), indent=2),
        encoding="utf-8",
    )

    return image_result


def process_directory(input_dir: str, output_dir: str) -> List[ImageResult]:
    """Run the pipeline on every image in *input_dir*.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory where all outputs are saved.

    Returns:
        List of ImageResult, one per processed image.
    """
    src_dir = Path(input_dir)
    images = [
        p for p in sorted(src_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]

    results: List[ImageResult] = []
    for image_path in images:
        result = process_image(str(image_path), output_dir)
        results.append(result)

    return results
