"""Manifest checks kept TensorFlow-free so data contracts can be tested anywhere."""
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd

REQUIRED_COLUMNS = {"image_path", "mapped_class", "split", "file_status"}
OFFICIAL_SPLITS = {"train", "cis_val", "cis_test", "trans_val", "trans_test"}
# Member 1's status is a validation result, not a label/eligibility decision.
READABLE_STATUSES = {"ok", "valid", "readable", "scaled_resolution"}

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_training_manifest(path, classes, require_existing_paths=True):
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Manifest is empty.")
    for col in ("image_path", "mapped_class", "split", "file_status"):
        if df[col].isna().any():
            raise ValueError(f"Manifest has missing values in {col}. Resolve upstream.")
    df["image_path"] = df["image_path"].astype(str)
    df["mapped_class"] = df["mapped_class"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip()
    df["file_status"] = df["file_status"].astype(str).str.strip().str.lower()
    if df["image_path"].duplicated().any():
        n = int(df["image_path"].duplicated(keep=False).sum())
        raise ValueError(f"Manifest has {n} rows with duplicate image_path values; review upstream.")
    unknown_splits = sorted(set(df["split"]) - OFFICIAL_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown/non-official split names: {unknown_splits}")
    unknown_classes = sorted(set(df["mapped_class"]) - set(classes))
    if unknown_classes:
        raise ValueError(
            f"Manifest includes unsupported/unknown mapped_class values {unknown_classes}. "
            "Do not map unsupported categories into the 8 classes; resolve policy upstream."
        )
    bad_status = sorted(set(df["file_status"]) - READABLE_STATUSES)
    if bad_status:
        raise ValueError(
            f"Manifest contains statuses not approved as readable: {bad_status}. "
            "Review these records upstream; no silent dropping is performed."
        )
    if require_existing_paths:
        missing_paths = [p for p in df["image_path"] if not Path(p).is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} manifest image paths do not exist from this working directory. "
                f"Examples: {missing_paths[:5]}. Pass repository-relative or valid absolute paths."
            )
    return df
