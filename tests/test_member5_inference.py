from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.models.inference_mobilenet import MobileNetV2Classifier, validate_uploaded_image

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["opossum", "rabbit", "coyote", "cat", "squirrel", "raccoon", "dog", "bobcat"]


def png_bytes():
    image = Image.new("RGB", (64, 48), (40, 120, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_class_order_matches_frozen_config():
    config = json.loads((ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN"
    assert config["class_names"] == EXPECTED
    assert config["class_to_index"] == {name: i for i, name in enumerate(EXPECTED)}


def test_valid_image_decodes_to_rgb():
    image = validate_uploaded_image(png_bytes())
    assert image.mode == "RGB"
    assert image.size == (64, 48)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\nbroken"])
def test_invalid_image_rejected(payload):
    with pytest.raises(ValueError):
        validate_uploaded_image(payload)


class FakeModel:
    output_shape = (None, 8)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[.05, .10, .30, .20, .08, .07, .15, .05]], dtype=np.float32)


def test_prediction_shape_and_top_three_order(monkeypatch):
    # Avoid TensorFlow/checkpoint loading while testing the output contract.
    classifier = object.__new__(MobileNetV2Classifier)
    classifier.classes = EXPECTED
    classifier.model = FakeModel()
    classifier.model_version = "test"
    result = classifier.predict_image(Image.new("RGB", (32, 32)), top_k=3)
    assert len(result["probabilities"]) == 8
    assert result["predicted_class"] == "coyote"
    assert [x["class_name"] for x in result["top_k"]] == ["coyote", "cat", "dog"]
    assert [x["score"] for x in result["top_k"]] == sorted(
        [x["score"] for x in result["top_k"]], reverse=True
    )
    assert abs(sum(result["probabilities"]) - 1) < 1e-6


def test_invalid_model_probabilities_rejected():
    classifier = object.__new__(MobileNetV2Classifier)
    classifier.classes = EXPECTED
    classifier.model_version = "test"

    class BadModel:
        def predict(self, batch, verbose=0):
            return np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=float)

    classifier.model = BadModel()
    with pytest.raises(ValueError):
        classifier.predict_image(Image.new("RGB", (32, 32)))
