from __future__ import annotations

import dataclasses
import json

from mcp.server.fastmcp import FastMCP

from pipeline import ImageResult, process_directory, process_image

mcp = FastMCP("segmentiq")


@mcp.tool()
def segment_image(image_path: str, output_dir: str) -> str:
    """Run the Gemma → SAM 3 pipeline on a single image.

    Gemma 4 describes the image and identifies object types; SAM 3 segments
    each object. Saves an annotated image and a metadata JSON to output_dir.

    Args:
        image_path: Absolute path to a local image file.
        output_dir: Directory where the annotated image and metadata are saved.

    Returns:
        JSON string with description, detected objects, and segmentation results.
    """
    result: ImageResult = process_image(image_path, output_dir)
    return json.dumps(dataclasses.asdict(result))


@mcp.tool()
def segment_directory(input_dir: str, output_dir: str) -> str:
    """Run the Gemma → SAM 3 pipeline on every image in a directory.

    Processes images sequentially. For each image, saves an annotated JPEG
    and a metadata JSON to output_dir.

    Args:
        input_dir: Directory containing input images (jpg, jpeg, png, webp).
        output_dir: Directory where all outputs are saved (created if absent).

    Returns:
        JSON array summarising results for each processed image.
    """
    results = process_directory(input_dir, output_dir)
    return json.dumps([dataclasses.asdict(r) for r in results])


if __name__ == "__main__":
    mcp.run()
