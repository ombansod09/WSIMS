"""MobileNetV2 preprocessing contract for WSIMS.
Input tensors are RGB float32 pixels in [0,255]. The model's Rescaling layer
converts them to [-1,1], equivalent to tf.keras.applications.mobilenet_v2.preprocess_input.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageOps
IMAGE_SIZE = (224, 224)


def image_to_rgb_float(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape != (224, 224, 3) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid RGB image tensor: {arr.shape}")
    return arr
