# SegmentIQ

A semantic image catalog that lets you search a collection of photos by describing what you're looking for in plain English.

## How it works

Ingestion runs a three-stage pipeline on every image:

1. **Gemma 4** (via Ollama) — looks at the image and identifies every distinct object type present, returning a structured list (e.g. `["lion", "zebra", "giraffe", "elephant", "tiger", "rhinoceros"]`)
2. **SAM 3** — uses each object label as a text prompt to segment all instances of that object, producing tight per-object masks
3. **CLIP** — crops each mask from the original image and encodes it as a 512-dimensional embedding

The embeddings are stored in **ChromaDB** for fast similarity search. The image paths, object labels, bounding boxes, and mask images are stored in **SQLite**. Both databases use the same UUID per mask, so a search result in ChromaDB can be instantly resolved to a specific object in a specific photo.

At search time, the user's text query is encoded by CLIP into the same 512-dim space. ChromaDB finds the closest matching mask embeddings, results are grouped by photo, and the top-ranked images are returned — one card per photo.

## Architecture

```
inputs/
  photo.jpg
      │
      ▼
  describer.py     Gemma 4 (Ollama)  →  ["lion", "zebra", ...]
      │
      ▼
  segmenter.py     SAM 3             →  per-object masks + bboxes
      │
      ▼
  extractor.py     CLIP ViT-B/32     →  512-dim embeddings
      │
      ▼
  storage.py       ChromaDB + SQLite →  persisted catalog
      │
      ▼
  app.py           Streamlit UI      →  search + detail view
```

## Tech stack

| Component | Library |
|-----------|---------|
| Object discovery | Gemma 4 via [Ollama](https://ollama.com) |
| Segmentation | SAM 3 (`sam3` pip package) |
| Feature extraction | CLIP ViT-B/32 via `open-clip-torch` |
| Vector store | ChromaDB |
| Relational store | SQLite (stdlib) |
| Frontend | Streamlit |

## Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 5090)
- [Ollama](https://ollama.com) running locally with Gemma 4 pulled

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ./sam3 --no-deps
pip install iopath ftfy regex einops triton-windows pycocotools timm
pip install open-clip-torch chromadb streamlit transformers accelerate

# Pull Gemma 4 via Ollama
ollama pull gemma4
```

## Usage

### Ingest a directory of images

```python
from ingestor import ingest_directory
results = ingest_directory("inputs/", db_dir=".")
```

Or use the Streamlit UI sidebar.

### Search

```bash
streamlit run app.py
```

Type a description in the search bar (e.g. `"striped animal"`, `"bird"`, `"large grey mammal"`). Results show one card per matching photo, ranked by semantic similarity. Click **View details** to see labeled bounding boxes and mask overlays for every object indexed in that image.

## Project structure

```
describer.py   Gemma 4 integration — image → object list
segmenter.py   SAM 3 text-prompted segmentation — label → masks
extractor.py   CLIP feature extraction — crop → embedding, text → embedding
storage.py     ChromaDB + SQLite read/write helpers
ingestor.py    Pipeline orchestration — image → catalog
app.py         Streamlit search UI
inputs/        Sample input images
```
