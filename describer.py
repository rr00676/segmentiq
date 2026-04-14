from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import ollama

MODEL = "gemma4"

_PROMPT = (
    "Look at this image carefully. "
    "Return ONLY a JSON object — no markdown, no explanation — in exactly this shape:\n"
    '{"description": "<one sentence>", "objects": ["<noun>", ...]}\n'
    "The objects list must contain every distinct type of object or living thing visible. "
    "Use common lowercase nouns only (e.g. 'dog' not 'retriever', "
    "'rhinoceros' not 'rhino', 'person' not 'man'). "
    "Do not include duplicates. Do not include background elements like 'sky' or 'grass'."
)


@dataclass
class ImageDescription:
    description: str
    objects: List[str] = field(default_factory=list)


def describe_image(image_path: str) -> ImageDescription:
    """Send an image to Gemma 4 via Ollama and return a description + object list.

    Args:
        image_path: Path to the image file.

    Returns:
        ImageDescription with a plain-language description and list of object labels.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the model response cannot be parsed as JSON.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = ollama.chat(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": _PROMPT,
            "images": [str(path)],
        }],
    )

    raw = response["message"]["content"].strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemma response was not valid JSON: {e}\nRaw: {raw}") from e

    return ImageDescription(
        description=data.get("description", ""),
        objects=[str(o).lower().strip() for o in data.get("objects", [])],
    )
