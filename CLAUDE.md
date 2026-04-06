# SegmentIQ

A Python MCP server that uses Gemma 4 (via Ollama) to identify objects in an image,
then segments each object with SAM 3. Supports single-image and batch (directory) processing.
Results are saved as annotated images and JSON metadata.

## Project structure

- `describer.py` — Ollama/Gemma 4 vision: describe an image and return object types
- `segmenter.py` — SAM 3 core: load model, run inference, return masks + bounding boxes
- `annotator.py` — render mask overlays onto an image and save
- `pipeline.py` — orchestrate describer → segmenter → annotator; single image and batch
- `mcp_server.py` — MCP server exposing `segment_image` and `segment_directory` as tools
- `test_pipeline.py` — unit tests (all mocked, no GPU required)

## Environment

Always use the virtual environment:
```bash
source .venv/Scripts/activate
```

Ollama must be running locally with Gemma 4 pulled:
```bash
ollama run gemma4
```

## Tool contracts

```python
segment_image(image_path: str, output_dir: str) -> str (JSON)
segment_directory(input_dir: str, output_dir: str) -> str (JSON array)
```

**Output per image:**
- `<stem>_annotated.jpg` — original image with colored mask overlays
- `<stem>_metadata.json`:
```json
{
  "image_path": "...",
  "description": "A wildlife scene with several animals.",
  "objects": ["lion", "zebra", "elephant"],
  "segmentation": [
    {
      "query": "lion",
      "objects_found": 1,
      "instances": [
        {"id": 1, "confidence": 0.93, "bounding_box": [x, y, w, h], "mask": "<base64 PNG>"}
      ]
    }
  ]
}
```

## Running checks

```bash
# Type checks
mypy describer.py segmenter.py annotator.py pipeline.py mcp_server.py

# Tests
python -m unittest test_pipeline.py -v
```

## Communication preferences

- Explain concepts clearly — do not assume prior knowledge of computer vision or ML tooling
- Propose what a step will cover and wait for confirmation before making changes
- After each change, explain what happened and why, but don't over-elaborate
- Connect concepts back to how they apply in agentic workflows where relevant
- Answer side questions directly without turning them into a lesson unless asked

## Rules

- Gemma/Ollama logic goes in `describer.py`
- SAM 3 inference logic goes in `segmenter.py`
- Orchestration logic goes in `pipeline.py`
- `mcp_server.py` is a thin wrapper only — no business logic
- Always run mypy and tests before committing
- Update `requirements.txt` with `pip freeze > requirements.txt` when adding packages
- Commit messages should follow the pattern: `Step N: description`
