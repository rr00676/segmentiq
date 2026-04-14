from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MIN_AREA_FRACTION = 0.01  # discard masks smaller than 1% of image area

_processor: Optional[Sam3Processor] = None


def _get_processor() -> Sam3Processor:
    global _processor
    if _processor is None:
        model = build_sam3_image_model()
        _processor = Sam3Processor(model)
    return _processor


@dataclass
class MaskData:
    id: str                           # UUID — matches ChromaDB and SQLite
    label: str                        # object type from Gemma (e.g. "zebra")
    bbox: List[float]                 # [x, y, w, h] in pixels
    area: int                         # mask area in pixels
    segmentation: np.ndarray = field(repr=False)  # bool H×W array


def _masks_from_state(state: Dict, label: str, total_pixels: int) -> List[MaskData]:
    """Extract accepted MaskData from a Sam3Processor result state."""
    if "masks" not in state or state["masks"].shape[0] == 0:
        return []

    masks_tensor = state["masks"].squeeze(1)  # (N, H, W)
    results: List[MaskData] = []

    for i in range(masks_tensor.shape[0]):
        seg = masks_tensor[i].cpu().bool().numpy()
        area = int(seg.sum())
        if area / total_pixels < MIN_AREA_FRACTION:
            continue
        ys, xs = np.where(seg)
        bbox = [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        results.append(MaskData(
            id=str(uuid.uuid4()),
            label=label,
            bbox=bbox,
            area=area,
            segmentation=seg,
        ))

    return results


def segment_image(image_path: str, objects: List[str]) -> List[MaskData]:
    """Run SAM 3 text-prompted segmentation for each object label.

    Encodes the image once, then runs a separate text prompt for each label,
    collecting all mask instances that pass the area filter.

    Args:
        image_path: Path to the input image.
        objects: List of object labels from Gemma (e.g. ["lion", "zebra"]).

    Returns:
        List of MaskData across all labels, ordered by area descending.
    """
    if not objects:
        return []

    processor = _get_processor()
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    total_pixels = w * h

    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)

    all_masks: List[MaskData] = []

    for label in objects:
        processor.reset_all_prompts(state)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = processor.set_text_prompt(prompt=label, state=state)
        all_masks.extend(_masks_from_state(result, label, total_pixels))

    all_masks.sort(key=lambda m: m.area, reverse=True)
    return all_masks
