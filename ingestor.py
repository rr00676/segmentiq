from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from describer import describe_image
from extractor import extract_embedding
from segmenter import MaskData, segment_image
from storage import image_paths, init, save_mask

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class IngestResult:
    image_path: str
    description: str
    objects: List[str]
    masks_saved: int


def _encode_mask_png(segmentation: np.ndarray) -> str:
    """Encode a boolean H×W mask as a base64 PNG string."""
    mask_img = Image.fromarray((segmentation * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ingest_image(image_path: str, db_dir: str = ".") -> IngestResult:
    """Run the full pipeline for one image: describe → segment → embed → store.

    Steps:
      1. Gemma 4 describes the image and returns a list of object labels
      2. SAM 3 segments each label, returning per-instance masks
      3. Each mask crop is embedded with CLIP
      4. Mask + embedding saved to ChromaDB and SQLite

    Already-ingested images are skipped.

    Args:
        image_path: Path to the source image.
        db_dir: Directory where catalog.db and .chromadb/ live.

    Returns:
        IngestResult with description, object list, and mask count.
    """
    init(db_dir)

    abs_path = str(Path(image_path).resolve())

    if abs_path in image_paths():
        print(f"[skip] already ingested: {abs_path}")
        return IngestResult(image_path=abs_path, description="", objects=[], masks_saved=0)

    print(f"[describe] {abs_path}")
    desc = describe_image(abs_path)
    print(f"[objects]  {desc.objects}")

    print(f"[segment]  running SAM 3 for {len(desc.objects)} label(s)...")
    masks: List[MaskData] = segment_image(abs_path, desc.objects)
    print(f"[segment]  {len(masks)} mask(s) found")

    source_image = Image.open(abs_path).convert("RGB")

    for mask in masks:
        x, y, w, h = mask.bbox
        crop = source_image.crop((x, y, x + w, y + h))
        embedding = extract_embedding(crop)
        mask_png_b64 = _encode_mask_png(mask.segmentation)

        save_mask(
            mask_id=mask.id,
            image_path=abs_path,
            label=mask.label,
            bbox=mask.bbox,
            area=mask.area,
            embedding=embedding,
            mask_png_b64=mask_png_b64,
        )

    print(f"[done]     {len(masks)} mask(s) saved")
    return IngestResult(
        image_path=abs_path,
        description=desc.description,
        objects=desc.objects,
        masks_saved=len(masks),
    )


def ingest_directory(input_dir: str, db_dir: str = ".") -> List[IngestResult]:
    """Ingest all images in a directory sequentially."""
    images = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not images:
        print(f"[warn] no images found in {input_dir}")
        return []

    return [ingest_image(str(p), db_dir) for p in images]
