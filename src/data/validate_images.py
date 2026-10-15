"""Validate manifest image paths and image readability; report rather than silently drop rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--write-validated-manifest", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, keep_default_na=False)
    required = {"image_path", "file_name", "split", "original_category", "mapped_class"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise SystemExit(f"Manifest missing required columns: {missing_columns}")

    statuses = []
    widths, heights, hashes = [], [], []
    path_counts = Counter(str(Path(x).resolve()) for x in df["image_path"] if str(x).strip())
    duplicate_paths = {p for p, count in path_counts.items() if count > 1}
    hash_to_paths: dict[str, list[str]] = defaultdict(list)

    for _, row in df.iterrows():
        raw_path = str(row["image_path"]).strip()
        path = Path(raw_path) if raw_path else Path("__missing__")
        status = "ok"
        width = height = ""
        file_hash = ""
        if not raw_path or not path.is_file():
            status = "missing_file"
        elif path.suffix.lower() not in SUPPORTED_EXTS:
            status = "unsupported_extension"
        elif str(path.resolve()) in duplicate_paths:
            status = "duplicate_manifest_path"
        else:
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
                    im.convert("RGB").load()
                file_hash = sha256_file(path)
                hash_to_paths[file_hash].append(str(path.resolve()))
            except (UnidentifiedImageError, OSError, ValueError):
                status = "unreadable_image"
        statuses.append(status)
        widths.append(width)
        heights.append(height)
        hashes.append(file_hash)

    df["validation_status"] = statuses
    df["validated_width"] = widths
    df["validated_height"] = heights
    df["sha256"] = hashes

    exact_duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    status_counts = Counter(statuses)
    split_counts = df.groupby(["split", "mapped_class"], dropna=False).size().to_dict()
    report = {
        "manifest": str(args.manifest),
        "total_rows": int(len(df)),
        "status_counts": dict(status_counts),
        "duplicate_content_groups": [
            {"sha256": digest, "paths": paths} for digest, paths in exact_duplicates.items()
        ],
        "split_class_counts": [
            {"split": split, "mapped_class": klass, "count": int(count)}
            for (split, klass), count in split_counts.items()
        ],
        "notes": [
            "Exact duplicate hashes are reported, not automatically removed.",
            "Near-duplicates/consecutive frames require sequence/location-aware review; hashes cannot detect them.",
            "Keep official split boundaries. Do not move images across splits to fix imbalance.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    if args.write_validated_manifest:
        args.write_validated_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.write_validated_manifest, index=False)

    print(f"Validated {len(df)} rows. Status counts: {dict(status_counts)}")
    print(f"Found {len(exact_duplicates)} exact duplicate-content group(s).")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
