from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

CHROMA_COLLECTION = "segments"
DB_FILE = "catalog.db"
CHROMA_DIR = ".chromadb"

_chroma_client: Optional[chromadb.ClientAPI] = None
_chroma_collection: Optional[chromadb.Collection] = None
_sqlite_conn: Optional[sqlite3.Connection] = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init(db_dir: str = ".") -> None:
    """Initialise ChromaDB and SQLite, creating tables/collections if absent."""
    global _chroma_client, _chroma_collection, _sqlite_conn

    base = Path(db_dir)
    base.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=str(base / CHROMA_DIR))
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    _sqlite_conn = sqlite3.connect(str(base / DB_FILE))
    _sqlite_conn.execute("""
        CREATE TABLE IF NOT EXISTS masks (
            id          TEXT PRIMARY KEY,
            image_path  TEXT NOT NULL,
            label       TEXT NOT NULL,
            bbox        TEXT NOT NULL,
            area        INTEGER NOT NULL,
            mask_png    TEXT NOT NULL
        )
    """)
    _sqlite_conn.commit()


def _require_init() -> None:
    if _chroma_collection is None or _sqlite_conn is None:
        raise RuntimeError("Call storage.init() before using storage functions.")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_mask(
    mask_id: str,
    image_path: str,
    label: str,
    bbox: List[float],
    area: int,
    embedding: List[float],
    mask_png_b64: str,
) -> None:
    """Persist a single mask to both databases.

    Args:
        mask_id: UUID string — primary key in both SQLite and ChromaDB.
        image_path: Absolute path to the source image.
        label: Object type from Gemma (e.g. "zebra").
        bbox: Bounding box [x, y, w, h] in pixels.
        area: Mask area in pixels.
        embedding: 512-dim CLIP embedding.
        mask_png_b64: Base64-encoded PNG of the boolean mask.
    """
    _require_init()

    _sqlite_conn.execute(  # type: ignore[union-attr]
        "INSERT OR REPLACE INTO masks (id, image_path, label, bbox, area, mask_png) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (mask_id, image_path, label, json.dumps(bbox), area, mask_png_b64),
    )
    _sqlite_conn.commit()  # type: ignore[union-attr]

    _chroma_collection.upsert(  # type: ignore[union-attr]
        ids=[mask_id],
        embeddings=[embedding],
        metadatas=[{"image_path": image_path, "label": label, "bbox": json.dumps(bbox)}],
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

@dataclass
class ImageSearchResult:
    image_path: str
    best_score: float
    best_label: str     # label of the top-matching mask
    mask_count: int


@dataclass
class MaskRecord:
    mask_id: str
    label: str
    bbox: List[float]
    area: int
    mask_png_b64: str


def search_images(
    query_embedding: List[float],
    n_images: int = 5,
) -> List[ImageSearchResult]:
    """Find images whose masks best match the query, one result per image.

    Args:
        query_embedding: 512-dim CLIP text embedding.
        n_images: Maximum number of unique images to return.

    Returns:
        List of ImageSearchResult ordered by best_score descending.
    """
    _require_init()

    total = _chroma_collection.count()  # type: ignore[union-attr]
    if total == 0:
        return []

    response = _chroma_collection.query(  # type: ignore[union-attr]
        query_embeddings=[query_embedding],
        n_results=total,
        include=["distances", "metadatas"],
    )

    ids = response["ids"][0]
    distances = response["distances"][0]
    metadatas = response["metadatas"][0]

    # Group by image_path — keep best score and its label
    best: Dict[str, tuple[float, str]] = {}
    for distance, meta in zip(distances, metadatas):
        path = meta["image_path"]
        score = round(1.0 - distance, 4)
        if path not in best or score > best[path][0]:
            best[path] = (score, meta.get("label", ""))

    ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:n_images]

    results: List[ImageSearchResult] = []
    for image_path, (score, label) in ranked:
        row = _sqlite_conn.execute(  # type: ignore[union-attr]
            "SELECT COUNT(*) FROM masks WHERE image_path = ?", (image_path,)
        ).fetchone()
        results.append(ImageSearchResult(
            image_path=image_path,
            best_score=score,
            best_label=label,
            mask_count=row[0] if row else 0,
        ))

    return results


def get_image_masks(image_path: str) -> List[MaskRecord]:
    """Return all stored masks for a given image, ordered by area descending."""
    _require_init()

    rows = _sqlite_conn.execute(  # type: ignore[union-attr]
        "SELECT id, label, bbox, area, mask_png FROM masks "
        "WHERE image_path = ? ORDER BY area DESC",
        (image_path,),
    ).fetchall()

    return [
        MaskRecord(
            mask_id=row[0],
            label=row[1],
            bbox=json.loads(row[2]),
            area=row[3],
            mask_png_b64=row[4],
        )
        for row in rows
    ]


def image_paths() -> List[str]:
    """Return distinct image paths in the catalog."""
    _require_init()
    rows = _sqlite_conn.execute(  # type: ignore[union-attr]
        "SELECT DISTINCT image_path FROM masks ORDER BY image_path"
    ).fetchall()
    return [r[0] for r in rows]
