"""SQLite persistence for successful WSIMS predictions.

Confidence and top-k scores are stored as fractions in [0, 1]. Timestamps are
UTC ISO-8601 strings ending in Z. Uploaded image bytes are hashed, never stored.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "artifacts" / "history" / "predictions.sqlite3"

def image_sha256(image_bytes: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 hex digest of the uploaded image bytes."""
    return hashlib.sha256(bytes(image_bytes)).hexdigest()

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

class PredictionHistory:
    """Small SQLite repository. Pass a temporary path to isolate tests."""
    def __init__(self, db_path: str | Path = DEFAULT_HISTORY_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    event_id TEXT PRIMARY KEY,
                    event_key TEXT UNIQUE,
                    timestamp_utc TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    model_version TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    top_k_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class)")

    def record_prediction(self, *, predicted_class: str, confidence: float, model_version: str,
                          image_bytes: bytes, top_k=None, event_key: str | None = None,
                          timestamp_utc: str | None = None) -> str:
        """Insert a successful result once; repeated event_key inserts return the existing ID."""
        if not predicted_class or not model_version:
            raise ValueError("predicted_class and model_version are required")
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be a fraction in [0, 1]")
        normalized_top = []
        for item in (top_k or []):
            score = float(item["score"])
            if not 0 <= score <= 1:
                raise ValueError("top-k scores must be fractions in [0, 1]")
            normalized_top.append({"class_name": str(item["class_name"]), "score": score})
        event_id = str(uuid.uuid4())
        stamp = timestamp_utc or utc_now_iso()
        if stamp.endswith("+00:00"):
            stamp = stamp[:-6] + "Z"
        with self._connect() as conn:
            if event_key:
                existing = conn.execute("SELECT event_id FROM predictions WHERE event_key=?", (event_key,)).fetchone()
                if existing:
                    return str(existing["event_id"])
            conn.execute("""
                INSERT INTO predictions(event_id,event_key,timestamp_utc,predicted_class,confidence,
                                        model_version,image_sha256,top_k_json)
                VALUES(?,?,?,?,?,?,?,?)
            """, (event_id, event_key, stamp, predicted_class, confidence, model_version,
                  image_sha256(image_bytes), json.dumps(normalized_top, separators=(",", ":"))))
        return event_id

    def query_predictions(self, *, start_date=None, end_date=None, predicted_class=None, limit=None):
        """Return newest-first dictionaries; dates accept YYYY-MM-DD strings."""
        clauses, params = [], []
        if start_date:
            clauses.append("timestamp_utc >= ?"); params.append(str(start_date) + "T00:00:00")
        if end_date:
            clauses.append("timestamp_utc < ?")
            from datetime import date, timedelta
            end = date.fromisoformat(str(end_date)) + timedelta(days=1)
            params.append(end.isoformat() + "T00:00:00")
        if predicted_class and predicted_class != "All":
            clauses.append("predicted_class = ?"); params.append(predicted_class)
        sql = "SELECT * FROM predictions"
        if clauses: sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp_utc DESC, event_id DESC"
        if limit is not None:
            sql += " LIMIT ?"; params.append(max(0, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["top_k"] = json.loads(item.pop("top_k_json"))
            result.append(item)
        return result

    def count_predictions(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def class_counts(self, *, start_date=None, end_date=None):
        rows = self.query_predictions(start_date=start_date, end_date=end_date)
        counts = {}
        for row in rows:
            counts[row["predicted_class"]] = counts.get(row["predicted_class"], 0) + 1
        return counts

    def clear(self):
        """Delete prediction records explicitly requested by the caller."""
        with self._connect() as conn:
            conn.execute("DELETE FROM predictions")
