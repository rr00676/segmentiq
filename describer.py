from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import ollama

MODEL = "gemma4"

_PROMPT = (
    "Analyse this image. Respond with ONLY a JSON object in this exact format, "
    "no markdown, no extra text:\n"
    '{"description": "<one sentence describing the image>", '
    '"objects": ["<object type 1>", "<object type 2>", ...]}\n'
    "The objects list should contain every distinct type of object or living thing "
    "visible in the image, using simple singular nouns (e.g. \"dog\", \"car\", \"tree\")."
)


@dataclass
class ImageDescription:
    description: str
    objects: List[str] = field(default_factory=list)


def describe_image(image_path: str) -> ImageDescription:
    """Use Gemma 4 via Ollama to describe an image and list object types.

    Args:
        image_path: Absolute or relative path to a local image file.

    Returns:
        ImageDescription with a text description and list of object type strings.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": _PROMPT,
                "images": [str(path)],
            }
        ],
    )

    raw = response["message"]["content"].strip()

    # Strip markdown code fences if model wraps the JSON
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)
    return ImageDescription(
        description=data.get("description", ""),
        objects=[str(o).lower().strip() for o in data.get("objects", [])],
    )
