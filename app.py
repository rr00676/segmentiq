from __future__ import annotations

import base64
import colorsys
import io
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import storage
from extractor import extract_text_embedding
from ingestor import ingest_directory
from storage import ImageSearchResult, MaskRecord, get_image_masks, image_paths, search_images

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Semantic Image Catalog",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _palette_color(index: int) -> tuple[int, int, int]:
    hue = (index * (1.0 / 12)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _decode_mask(mask_png_b64: str, target_size: tuple[int, int]) -> np.ndarray:
    data = base64.b64decode(mask_png_b64)
    mask_img = Image.open(io.BytesIO(data)).convert("L").resize(target_size, Image.NEAREST)
    return np.array(mask_img) > 128


def _apply_overlays(
    image: Image.Image,
    masks: List[MaskRecord],
    show_bbox: bool,
    show_masks: bool,
) -> Image.Image:
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, mask_rec in enumerate(masks):
        r, g, b = _palette_color(idx)
        x, y, w, h = [int(v) for v in mask_rec.bbox]

        if show_masks:
            seg = _decode_mask(mask_rec.mask_png_b64, img.size)
            colored = Image.new("RGBA", img.size, (r, g, b, 110))
            overlay.paste(colored, mask=Image.fromarray((seg * 255).astype("uint8"), mode="L"))

        if show_bbox:
            draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 255), width=3)
            # Label above the box
            label = mask_rec.label
            tw = len(label) * 7 + 6
            draw.rectangle([x, y - 18, x + tw, y], fill=(r, g, b, 220))
            draw.text((x + 3, y - 16), label, fill=(0, 0, 0, 255))

    return Image.alpha_composite(img, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Sidebar — ingestion controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Semantic Image Catalog")

    db_dir = st.text_input("Database directory", value=".")
    storage.init(db_dir)

    st.divider()
    st.subheader("Ingest images")
    input_dir = st.text_input("Input directory", placeholder="e.g. inputs/")

    if st.button("Ingest directory", type="primary", disabled=not input_dir):
        if not Path(input_dir).is_dir():
            st.error(f"Directory not found: {input_dir}")
        else:
            with st.spinner("Running Gemma 4 + SAM 3 + CLIP on images…"):
                results = ingest_directory(input_dir, db_dir)
            total_masks = sum(r.masks_saved for r in results)
            st.success(f"Ingested {len(results)} image(s) → {total_masks} masks saved")
            st.session_state.selected_image = None

    st.divider()
    known = image_paths()
    st.caption(f"{len(known)} image(s) in catalog")
    for p in known:
        st.caption(f"• {Path(p).name}")

# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------

if st.session_state.selected_image:
    image_path = st.session_state.selected_image

    if st.button("← Back to search"):
        st.session_state.selected_image = None
        st.rerun()

    st.subheader(Path(image_path).name)

    col_toggles, _ = st.columns([2, 3])
    with col_toggles:
        show_bbox = st.toggle("Show bounding boxes", value=True)
        show_masks = st.toggle("Show mask overlays", value=False)

    masks = get_image_masks(image_path)
    image = Image.open(image_path)
    rendered = _apply_overlays(image, masks, show_bbox, show_masks)
    st.image(rendered, use_container_width=True)

    # Legend
    if masks:
        st.caption("**Detected objects:**  " + "  ·  ".join(
            f":{m.label}" if False else m.label for m in masks
        ))

# ---------------------------------------------------------------------------
# Search view
# ---------------------------------------------------------------------------

else:
    st.header("Search")

    col_query, col_n = st.columns([4, 1])
    with col_query:
        query = st.text_input(
            "Find all photos that contain:",
            placeholder="e.g. a rhinoceros, bird, striped animal",
        )
    with col_n:
        n_images = st.number_input("Max photos", min_value=1, max_value=20, value=5)

    if query:
        with st.spinner("Searching…"):
            embedding = extract_text_embedding(query)
            results: List[ImageSearchResult] = search_images(embedding, n_images=n_images)

        if not results:
            st.info("No results found. Try ingesting some images first.")
        else:
            st.caption(f"{len(results)} photo(s) found")
            cols = st.columns(min(len(results), 3))
            for i, result in enumerate(results):
                with cols[i % 3]:
                    st.image(Image.open(result.image_path), use_container_width=True)
                    st.caption(
                        f"**{Path(result.image_path).name}**  \n"
                        f"Best match: **{result.best_label}** · {result.best_score:.1%}  \n"
                        f"{result.mask_count} object(s) indexed"
                    )
                    if st.button("View details", key=result.image_path):
                        st.session_state.selected_image = result.image_path
                        st.rerun()

    elif not image_paths():
        st.info("No images ingested yet. Add a directory in the sidebar and click **Ingest directory**.")
