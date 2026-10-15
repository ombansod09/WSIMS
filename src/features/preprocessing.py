"""Preprocessing and frozen-class helpers shared by WSIMS model code."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = [
    "opossum", "rabbit", "coyote", "cat",
    "squirrel", "raccoon", "dog", "bobcat",
]

def load_classes(classes_path: str | Path) -> list[str]:
    """Load the project's authoritative frozen class order; never invent a fallback."""
    path = Path(classes_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("status") != "FROZEN":
        raise ValueError(
            f"Class mapping is not frozen ({config.get('status')!r}); "
            "training is blocked until the authoritative config/classes.json is frozen."
        )
    names = config.get("class_names")
    mapping = config.get("class_to_index")
    if not isinstance(names, list) or len(names) != 8 or len(set(names)) != 8:
        raise ValueError("Frozen class_names must contain exactly 8 unique class names.")
    if names != EXPECTED_CLASSES:
        raise ValueError(
            "Frozen class order differs from the project handoff. Stop and obtain team approval; "
            "do not silently change class order."
        )
    if not isinstance(mapping, dict) or mapping != {name: i for i, name in enumerate(names)}:
        raise ValueError("class_to_index must exactly match class_names order (indices 0..7).")
    if "empty" in [n.lower() for n in names]:
        raise ValueError("'empty' is not allowed in the 8-class animal classifier.")
    return list(names)

def decode_image(path: str | Path, image_size=IMAGE_SIZE) -> np.ndarray:
    """Decode an image as RGB, resize to (height,width), scale float32 to [0,1]."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.resize((int(image_size[1]), int(image_size[0])), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    expected = (int(image_size[0]), int(image_size[1]), 3)
    if arr.shape != expected:
        raise ValueError(f"Unexpected image shape {arr.shape} for {path}; expected {expected}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite pixel values in {path}")
    return arr

def decode_pil_image(image: Image.Image, image_size=IMAGE_SIZE) -> np.ndarray:
    """Decode a PIL image with exactly the same inference preprocessing as training."""
    im = ImageOps.exif_transpose(image).convert("RGB")
    im = im.resize((int(image_size[1]), int(image_size[0])), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    expected = (int(image_size[0]), int(image_size[1]), 3)
    if arr.shape != expected or not np.isfinite(arr).all():
        raise ValueError(f"Invalid preprocessed image: shape={arr.shape}")
    return arr
