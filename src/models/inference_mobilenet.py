"""Inference adapter for the frozen WSIMS MobileNetV2 classifier."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.features.mobilenet_preprocessing import image_to_rgb_float
from src.models.mobilenet_model import load_classes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "mobilenetv2" / "mobilenetv2_best.keras"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "config" / "classes.json"


def validate_uploaded_image(data: bytes | bytearray | memoryview) -> Image.Image:
    """Decode and validate uploaded image bytes, returning an EXIF-corrected RGB PIL image."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(bytes(data))) as opened:
            opened.verify()
        with Image.open(io.BytesIO(bytes(data))) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid, readable image.") from exc
    if image.width < 1 or image.height < 1:
        raise ValueError("The image has invalid dimensions.")
    return image


class MobileNetV2Classifier:
    """Small reusable inference API; does not train or modify the checkpoint."""

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        classes_path=DEFAULT_CLASSES_PATH,
        model_version="mobilenetv2-transfer-v1",
    ):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load the trained Keras model.") from exc

        model_path = Path(model_path)
        classes_path = Path(classes_path)
        self.classes = load_classes(str(classes_path))
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained MobileNetV2 checkpoint not found: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path), compile=False)
        output_shape = self.model.output_shape
        if not output_shape or output_shape[-1] != len(self.classes):
            raise ValueError("Model output width does not match the frozen eight-class mapping.")
        self.model_version = model_version

    def predict_image(self, image: Image.Image, top_k=3, include_timing=False):
        """Return the predicted label, confidence, all probabilities, and ordered top-k scores."""
        started = time.perf_counter()
        if not isinstance(image, Image.Image):
            raise TypeError("predict_image expects a PIL.Image.Image instance.")
        arr = image_to_rgb_float(image)
        output = self.model.predict(arr[None, ...], verbose=0)
        scores = np.asarray(output[0] if np.asarray(output).ndim == 2 else output, dtype=np.float64)

        if scores.shape != (len(self.classes),):
            raise ValueError(f"Model must return {len(self.classes)} scores; got shape {scores.shape}.")
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Model returned non-finite or out-of-range probabilities.")
        if not np.isclose(scores.sum(), 1.0, atol=1e-3):
            raise ValueError("Model output is not a probability distribution; refusing to display misleading confidence.")

        k = max(1, min(int(top_k), len(self.classes)))
        order = np.argsort(scores)[::-1][:k]
        best = int(order[0])
        result = {
            "predicted_class": self.classes[best],
            "confidence": float(scores[best]),
            "probabilities": [float(v) for v in scores],
            "top_k": [
                {"class_name": self.classes[int(i)], "score": float(scores[i])}
                for i in order
            ],
            "model_version": self.model_version,
        }
        if include_timing:
            result["inference_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return result
