# Semantic Image Catalog

A Python application that ingests a cache of images, identifies objects via AI-assisted segmentation, and stores metadata to support semantic and keyword search.

## Architecture: Describe-Then-Segment

1. **Description** — Gemma 4 (via Ollama) receives the image and returns a structured JSON response containing a plain-language description and a list of specific object types present (e.g. `["lion", "zebra", "elephant"]`)
2. **Segmentation** — SAM 3 receives each object label as a text prompt and returns masks for all instances of that concept in the image
3. **Feature extraction** — each mask crop is passed through CLIP to produce a normalized embedding
4. **Storage** — embeddings go into ChromaDB; image path, bbox, mask PNG, and object label go into SQLite; IDs match across both
5. **Search UI** — Streamlit frontend converts a text query to a CLIP embedding, queries ChromaDB for nearest matches, groups results by image (one card per photo), and shows a detail view with bounding box and mask overlays per object

## Tech Stack

- **Description**: Gemma 4 via Ollama (`ollama` Python package, local server at localhost:11434)
- **Segmentation**: SAM 3 (`sam3` pip package) — text-prompted via `Sam3Processor.set_text_prompt()`
- **Feature extraction**: CLIP (OpenAI ViT-B/32) via `open-clip-torch`
- **Vector store**: ChromaDB
- **Relational store**: SQLite (stdlib `sqlite3`)
- **Frontend**: Streamlit

## Project Structure

- `describer.py` — Gemma 4 integration; sends image to Ollama, parses JSON response, returns description + object list
- `segmenter.py` — SAM 3 text-prompted segmentation; for a given image and object label, returns all matching mask instances
- `extractor.py` — CLIP feature extraction; accepts a cropped PIL image, returns a normalized 512-dim embedding; also encodes text queries
- `storage.py` — ChromaDB and SQLite initialization and read/write helpers
- `ingestor.py` — orchestrates describer → segmenter → extractor → storage for a single image and for a directory
- `app.py` — Streamlit UI: search bar, one result card per image, detail view with bbox/mask overlay toggles
- `test_ingestor.py` — unit tests (all mocked, no GPU or Ollama required)

## Data Model

### SQLite — `masks` table

| column     | type    | notes |
|------------|---------|-------|
| id         | TEXT    | matches ChromaDB embedding ID |
| image_path | TEXT    | absolute path to source image |
| label      | TEXT    | object type from Gemma (e.g. "zebra") |
| bbox       | TEXT    | JSON `[x, y, w, h]` in pixels |
| area       | INTEGER | mask area in pixels |
| mask_png   | TEXT    | base64-encoded PNG of boolean mask |

### ChromaDB — `segments` collection

Each embedding document:
- `id` — matches SQLite `masks.id`
- `embedding` — 512-dim CLIP vector (float32)
- `metadata` — `{"image_path": "...", "label": "...", "bbox": "[x,y,w,h]"}`

## Gemma 4 Prompt Contract

The prompt instructs Gemma to return only valid JSON in this shape:
```json
{
  "description": "A group of African safari animals...",
  "objects": ["lion", "zebra", "elephant", "giraffe", "tiger", "rhinoceros", "ostrich"]
}
```

Object names should be common lowercase nouns (e.g. "dog" not "retriever", "rhinoceros" not "rhino"). Markdown code fences must be stripped before JSON parsing.

## SAM 3 Segmentation Contract

For each object label from Gemma:
1. Call `Sam3Processor.set_image(image)` once per image (cache the state)
2. Call `Sam3Processor.reset_all_prompts(state)` between labels
3. Call `Sam3Processor.set_text_prompt(state, prompt=label)` + run inference
4. Filter masks where `area / (image_width * image_height) < 0.01`
5. Each surviving mask becomes one SQLite + ChromaDB record with that label

## Environment

Always use the virtual environment:
```bash
source .venv/Scripts/activate
```

Ollama must be running locally:
```bash
ollama serve   # if not already running as a service
ollama pull gemma4
```

## Running Checks

```bash
# Type checks
mypy describer.py segmenter.py extractor.py storage.py ingestor.py app.py

# Tests
python -m unittest test_ingestor.py -v
```

## Rules

- Gemma 4 / Ollama logic goes in `describer.py`
- SAM 3 logic goes in `segmenter.py`
- CLIP logic goes in `extractor.py`
- All DB read/write goes in `storage.py`
- Orchestration goes in `ingestor.py`
- `app.py` is UI only — no segmentation, description, or DB logic inline
- Always run mypy and tests before committing
- Update `requirements.txt` with `pip freeze > requirements.txt` when adding packages
- Commit messages follow the pattern: `Step N: description`

## Communication Preferences

- Explain concepts clearly — do not assume prior knowledge of computer vision or ML tooling
- Propose what a step will cover and wait for confirmation before making changes
- After each change, explain what happened and why, but don't over-elaborate
- Connect concepts back to how they apply in agentic workflows where relevant
- Answer side questions directly without turning them into a lesson unless asked
