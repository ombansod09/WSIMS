"""Build a reproducible manifest from COCO Camera Traps JSON annotations.

Supports one annotation JSON per split. Does not randomly split images.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

EMPTY_NAMES = {"empty", "no animal", "no_animal", "blank", "empty image", "empty_image"}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "images" not in data or "categories" not in data:
        raise ValueError(f"{path} is not a COCO-style annotation JSON (needs images/categories).")
    return data


def category_lookup(data: dict[str, Any]) -> dict[str, str]:
    return {str(c["id"]): str(c["name"]).strip() for c in data.get("categories", [])}


def image_category_map(data: dict[str, Any], categories: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Return image_id -> (category_id, category_name); report ambiguous multi-class images."""
    by_image: dict[str, set[str]] = defaultdict(set)
    for ann in data.get("annotations", []):
        image_id = str(ann.get("image_id"))
        category_id = str(ann.get("category_id"))
        if category_id in categories:
            by_image[image_id].add(category_id)

    result = {}
    for image in data.get("images", []):
        iid = str(image.get("id"))
        found = by_image.get(iid, set())
        if len(found) == 1:
            cid = next(iter(found))
            result[iid] = (cid, categories[cid])
        elif len(found) == 0:
            # Some metadata variants place category_id directly on image records.
            cid = str(image.get("category_id", ""))
            if cid in categories:
                result[iid] = (cid, categories[cid])
            else:
                result[iid] = ("", "unlabeled")
        else:
            result[iid] = ("", "ambiguous_multi_category")
    return result


def parse_annotation_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use split=path, e.g. train=.../train_annotations.json")
    split, path = value.split("=", 1)
    if split not in {"train", "val", "test"}:
        raise argparse.ArgumentTypeError("Split must be train, val, or test.")
    return split, Path(path)


def is_empty(name: str) -> bool:
    return name.strip().lower().replace("-", " ") in EMPTY_NAMES


def resolve_image(images_root: Path, file_name: str) -> Path:
    # Annotation file_name is usually relative to the image root. Avoid silently choosing
    # a different file if the exact path is absent.
    return images_root / file_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images-root", required=True, type=Path)
    p.add_argument("--annotation", required=True, action="append", type=parse_annotation_arg,
                   help="Repeat as split=annotation.json for train, val, and test.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--classes-output", required=True, type=Path)
    p.add_argument("--min-class-images", type=int, default=500)
    p.add_argument("--max-classes", type=int, default=8)
    args = p.parse_args()

    if args.min_class_images < 1 or args.max_classes != 8:
        raise SystemExit("--min-class-images must be positive; this PRD requires --max-classes 8.")
    split_paths = dict(args.annotation)
    if "train" not in split_paths:
        raise SystemExit("A train=... annotation is required to select classes.")
    if len(split_paths) != len(args.annotation):
        raise SystemExit("Each split may be supplied only once.")

    rows: list[dict[str, Any]] = []
    train_counts: Counter[str] = Counter()
    category_names_by_id: dict[str, str] = {}
    image_id_seen: set[tuple[str, str]] = set()

    for split, ann_path in split_paths.items():
        data = load_json(ann_path)
        cats = category_lookup(data)
        for cid, name in cats.items():
            category_names_by_id[cid] = name
        labels = image_category_map(data, cats)
        for image in data["images"]:
            image_id = str(image.get("id", ""))
            file_name = str(image.get("file_name", "")).strip()
            cid, original_category = labels.get(image_id, ("", "unlabeled"))
            loc = image.get("location", image.get("camera_id", image.get("seq_id", "")))
            path = resolve_image(args.images_root, file_name) if file_name else Path("")
            ext = path.suffix.lower() if file_name else ""
            row = {
                "image_id": image_id,
                "image_path": str(path),
                "file_name": file_name,
                "original_category_id": cid,
                "original_category": original_category,
                "mapped_class": "",
                "split": split,
                "location_camera_id": "" if loc is None else str(loc),
                "sequence_id": str(image.get("seq_id", "")),
                "file_status": "not_checked",
                "width": image.get("width", ""),
                "height": image.get("height", ""),
            }
            rows.append(row)
            if split == "train" and original_category not in {"unlabeled", "ambiguous_multi_category"} and not is_empty(original_category):
                train_counts[original_category] += 1

    eligible = [(name, count) for name, count in train_counts.items() if count >= args.min_class_images]
    eligible.sort(key=lambda pair: (-pair[1], pair[0].casefold()))
    if len(eligible) < 8:
        raise SystemExit(
            f"Only {len(eligible)} animal categories have >= {args.min_class_images} training images. "
            "If this matches the PRD fallback condition, rerun with --min-class-images 300 and document team approval."
        )
    selected = [name for name, _ in eligible[:8]]
    class_to_index = {name: i for i, name in enumerate(selected)}
    id_to_name = {cid: name for cid, name in category_names_by_id.items()}

    # Map by original category name, while preserving original category ID/name in the manifest.
    for row in rows:
        if row["original_category"] in class_to_index:
            row["mapped_class"] = row["original_category"]
        elif row["original_category"] == "ambiguous_multi_category":
            row["file_status"] = "ambiguous_multi_category"
        elif row["original_category"] == "unlabeled":
            row["file_status"] = "unlabeled"
        elif is_empty(row["original_category"]):
            row["mapped_class"] = "excluded_empty"
        else:
            row["mapped_class"] = "excluded_other_class"

    # Basic path checks here; detailed decoding/duplicates happen in validate_images.py.
    for row in rows:
        if row["file_status"] != "not_checked":
            continue
        path = Path(row["image_path"]) if row["image_path"] else Path("__missing__")
        if not path.is_file():
            row["file_status"] = "missing_file"
        elif path.suffix.lower() not in SUPPORTED_EXTS:
            row["file_status"] = "unsupported_extension"
        else:
            row["file_status"] = "present_not_decoded"

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.classes_output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    selected_id_map = {}
    for cid, name in id_to_name.items():
        if name in class_to_index:
            selected_id_map[cid] = name
    classes_doc = {
        "dataset": "Caltech Camera Traps CCT20",
        "status": "REVIEW_REQUIRED_BEFORE_FREEZE",
        "class_names": selected,
        "class_to_index": class_to_index,
        "original_category_id_to_class": selected_id_map,
        "selection_rule": {
            "minimum_training_images_per_class": args.min_class_images,
            "maximum_classes": 8,
            "count_source": "training annotation split only",
            "empty_category_excluded": True,
        },
    }
    with args.classes_output.open("w", encoding="utf-8") as f:
        json.dump(classes_doc, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(out)} manifest rows to {args.output}")
    print(f"Candidate classes (review and freeze with team): {selected}")
    print(f"Wrote class config to {args.classes_output}; status remains REVIEW_REQUIRED_BEFORE_FREEZE.")


if __name__ == "__main__":
    main()
