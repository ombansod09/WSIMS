"""Shared WSIMS inference API for baseline CNN and compatible Keras classifiers."""
from __future__ import annotations
import time
import numpy as np
from PIL import Image
from src.features.preprocessing import decode_pil_image, load_classes

class ImageClassifier:
    def __init__(self, model_path, classes_path="config/classes.json", model_version="baseline-cnn-v1"):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load a trained Keras model.") from exc
        self.classes = load_classes(classes_path)
        self.model = tf.keras.models.load_model(model_path)
        self.model_version = model_version
        if self.model.output_shape[-1] != len(self.classes):
            raise ValueError("Model output size does not match frozen class mapping.")

    def predict_image(self, image: Image.Image, top_k=3, include_timing=False):
        x = np.expand_dims(decode_pil_image(image), axis=0)
        started = time.perf_counter()
        scores = np.asarray(self.model.predict(x, verbose=0)[0], dtype=float)
        elapsed = time.perf_counter() - started
        if scores.shape != (len(self.classes),) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid model output vector: shape={scores.shape}")
        order = np.argsort(scores)[::-1][:max(1,min(int(top_k),len(self.classes)))]
        best = int(order[0])
        result = {
            "predicted_class": self.classes[best],
            "confidence": float(scores[best]),
            "top_k": [{"class_name":self.classes[int(i)],"score":float(scores[i])} for i in order],
            "model_version": self.model_version,
        }
        if include_timing:
            result["inference_seconds"] = float(elapsed)
        return result
