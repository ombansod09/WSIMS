import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.history.store import PredictionHistory, image_sha256
from src.history.analytics import summarize, daily_counts
from src.models.inference_mobilenet import validate_uploaded_image
from PIL import Image
import io

CLASSES = ["opossum", "rabbit", "coyote", "cat", "squirrel", "raccoon", "dog", "bobcat"]

@pytest.fixture
def store(tmp_path):
    return PredictionHistory(tmp_path / "test.sqlite3")

def sample(**overrides):
    data = dict(predicted_class="dog", confidence=0.875, model_version="test-v1",
                image_bytes=b"image-bytes", top_k=[{"class_name":"dog","score":0.875},
                {"class_name":"bobcat","score":0.075},{"class_name":"coyote","score":0.05}])
    data.update(overrides)
    return data

def test_database_creation_schema_and_record(store):
    assert store.db_path.exists()
    event_id = store.record_prediction(**sample())
    row = store.query_predictions()[0]
    assert row["event_id"] == event_id
    assert row["predicted_class"] == "dog"
    assert row["confidence"] == pytest.approx(.875)
    assert row["image_sha256"] == hashlib.sha256(b"image-bytes").hexdigest()
    assert len(row["top_k"]) == 3
    assert store.count_predictions() == 1

def test_unique_event_ids_and_idempotent_event_key(store):
    one = store.record_prediction(**sample(event_key="click-1"))
    two = store.record_prediction(**sample(event_key="click-1"))
    three = store.record_prediction(**sample(event_key="click-2"))
    assert one == two
    assert one != three
    assert store.count_predictions() == 2

def test_utc_timestamp_and_hash():
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    assert stamp.endswith("Z")
    assert image_sha256(b"abc") == hashlib.sha256(b"abc").hexdigest()

def test_reject_invalid_confidence(store):
    with pytest.raises(ValueError):
        store.record_prediction(**sample(confidence=1.1))

def test_filters_and_class_counts(store):
    store.record_prediction(**sample(predicted_class="dog", event_key="1", timestamp_utc="2026-10-01T12:00:00Z"))
    store.record_prediction(**sample(predicted_class="cat", event_key="2", timestamp_utc="2026-10-02T12:00:00Z"))
    store.record_prediction(**sample(predicted_class="dog", event_key="3", timestamp_utc="2026-10-03T12:00:00Z"))
    assert len(store.query_predictions(predicted_class="cat")) == 1
    assert len(store.query_predictions(start_date="2026-10-02", end_date="2026-10-03")) == 2
    assert store.class_counts() == {"dog": 2, "cat": 1}

def test_empty_analytics_and_missing_classes():
    summary = summarize([], CLASSES)
    assert summary["total"] == 0
    assert list(summary["class_counts"].values()) == [0] * 8
    assert daily_counts([]) == []

def test_daily_counts(store):
    store.record_prediction(**sample(event_key="1", timestamp_utc="2026-10-01T12:00:00Z"))
    store.record_prediction(**sample(event_key="2", timestamp_utc="2026-10-01T13:00:00Z"))
    assert daily_counts(store.query_predictions()) == [{"date":"2026-10-01","predictions":2}]

def test_clear_history_requires_call_and_deletes_records(store):
    store.record_prediction(**sample())
    assert store.count_predictions() == 1
    store.clear()
    assert store.count_predictions() == 0

def test_class_order_matches_frozen_config():
    import json
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "classes.json").read_text(encoding="utf-8"))
    assert config["class_names"] == CLASSES
    assert config["status"] == "FROZEN"

def test_corrupt_image_validation_fails_before_history_logging(store):
    with pytest.raises(ValueError):
        validate_uploaded_image(b"not an image")
    assert store.count_predictions() == 0
