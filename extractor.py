from __future__ import annotations

from typing import List

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

_model = None
_preprocess = None


def _get_model():
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model, _preprocess


def extract_embedding(crop: Image.Image) -> List[float]:
    """Generate a normalized CLIP embedding for a cropped image.

    Args:
        crop: A PIL Image of a single segmented object (any size — will be
              resized internally by the CLIP preprocessor).

    Returns:
        A 512-dimensional normalized float list ready for ChromaDB storage.
    """
    model, preprocess = _get_model()
    tensor = preprocess(crop).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)

    return embedding.squeeze(0).cpu().tolist()


def extract_text_embedding(text: str) -> List[float]:
    """Generate a normalized CLIP embedding for a text query.

    Used at search time to convert user input into the same vector space
    as the stored image embeddings.

    Args:
        text: A natural-language search query (e.g. "red chair").

    Returns:
        A 512-dimensional normalized float list.
    """
    model, _ = _get_model()
    tokens = open_clip.tokenize([text])
    if torch.cuda.is_available():
        tokens = tokens.cuda()

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)

    return embedding.squeeze(0).cpu().tolist()
