from __future__ import annotations

import base64
import colorsys
import io
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from segmenter import SegmentationResult

# Visually distinct hues spread evenly around the color wheel
_PALETTE_SIZE = 12


def _palette_color(index: int, alpha: int = 130) -> tuple[int, int, int, int]:
    hue = (index * (1.0 / _PALETTE_SIZE)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


def annotate(
    image_path: str,
    results: List[SegmentationResult],
    output_path: str,
) -> None:
    """Render segmentation mask overlays onto an image and save to *output_path*.

    Each query in *results* gets a distinct color. Instances with no mask data
    fall back to a bounding-box rectangle.

    Args:
        image_path: Path to the original image.
        results: List of SegmentationResult, one per queried object type.
        output_path: Destination path for the annotated image (JPEG or PNG).
    """
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, result in enumerate(results):
        r, g, b, a = _palette_color(idx)

        for inst in result.instances:
            if inst.mask:
                mask_data = base64.b64decode(inst.mask)
                mask_img = (
                    Image.open(io.BytesIO(mask_data))
                    .convert("L")
                    .resize(img.size, Image.Resampling.NEAREST)
                )
                colored = Image.new("RGBA", img.size, (r, g, b, a))
                overlay.paste(colored, mask=mask_img)

                # Place label at top-left corner of the mask region
                m = np.array(mask_img)
                ys, xs = np.where(m > 128)
                lx = int(xs.min()) if len(xs) else int(inst.bounding_box[0])
                ly = int(ys.min()) if len(ys) else int(inst.bounding_box[1])
            else:
                # Fallback: draw bounding box
                x, y, w, h = inst.bounding_box
                draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 255), width=3)
                lx, ly = int(x), int(y)

            text = f"{result.query} {inst.confidence:.0%}"
            tw = len(text) * 7 + 4
            draw.rectangle([lx, ly - 16, lx + tw, ly], fill=(r, g, b, 220))
            draw.text((lx + 2, ly - 15), text, fill=(0, 0, 0, 255))

    composite = Image.alpha_composite(img, overlay).convert("RGB")
    composite.save(output_path, quality=95)
