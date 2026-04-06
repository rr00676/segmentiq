from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable TF32 for better performance on Ampere+ (and Blackwell) GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5

_model: Optional[object] = None
_processor: Optional[Sam3Processor] = None


def _get_processor() -> Sam3Processor:
    global _model, _processor
    if _processor is None:
        _model = build_sam3_image_model(device=DEVICE)
        _processor = Sam3Processor(
            _model, device=DEVICE, confidence_threshold=CONFIDENCE_THRESHOLD
        )
    return _processor


@dataclass
class Instance:
    id: int
    confidence: float
    bounding_box: List[float]  # [x, y, w, h] in pixels, top-left origin
    mask: str = ""             # base64-encoded PNG of the binary mask (H x W, mode "L")


@dataclass
class SegmentationResult:
    query: str
    objects_found: int
    instances: List[Instance] = field(default_factory=list)


def segment_image(image_path: str, query: str) -> SegmentationResult:
    """Return all instances of *query* found in the image at *image_path*.

    Args:
        image_path: Absolute or relative path to a local image file.
        query: Natural language description of the object to find (e.g. "dog").

    Returns:
        SegmentationResult with one Instance per detected object, each
        containing a confidence score, bounding box [x, y, w, h], and
        a base64-encoded PNG mask.
    """
    processor = _get_processor()
    image = Image.open(image_path).convert("RGB")

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=query)

    scores: torch.Tensor = state.get("scores", torch.tensor([]))
    boxes: torch.Tensor = state.get("boxes", torch.tensor([]))  # XYXY absolute
    masks: torch.Tensor = state.get("masks", torch.tensor([]))  # (N, 1, H, W) bool

    instances: List[Instance] = []
    for i, (score, box) in enumerate(zip(scores, boxes), start=1):
        x1, y1, x2, y2 = box.tolist()

        mask_b64 = ""
        if masks.numel() > 0 and i - 1 < masks.shape[0]:
            mask_np = masks[i - 1, 0].cpu().bool().numpy()
            mask_img = Image.fromarray((mask_np * 255).astype("uint8"), mode="L")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        instances.append(
            Instance(
                id=i,
                confidence=round(float(score), 4),
                bounding_box=[round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                mask=mask_b64,
            )
        )

    return SegmentationResult(
        query=query,
        objects_found=len(instances),
        instances=instances,
    )
