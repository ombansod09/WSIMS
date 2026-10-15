#!/usr/bin/env python3
"""Reproducible metadata-first EDA for the supplied CCT annotations.

Run from repository root:
python scripts/eda_cct.py --metadata-dir data/annotations --output-dir reports/eda --image-root data/images
Metadata directory should contain train_annotations.json, cis_val_annotations.json,
cis_test_annotations.json, trans_val_annotations.json, trans_test_annotations.json.
--image-root is optional; if supplied, files are searched recursively by file_name.
This script does not change classes.json or create new dataset splits.
"""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

SPLITS = ["train", "cis_val", "cis_test", "trans_val", "trans_test"]
FILES = {s: f"{s}_annotations.json" for s in SPLITS}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-dir", required=True, type=Path)
    p.add_argument("--output-dir", default=Path("reports/eda"), type=Path)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--classes-json", type=Path, default=Path("config/classes.json"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {s: json.loads((args.metadata_dir / fn).read_text()) for s, fn in FILES.items()}
    categories = {int(c["id"]): c["name"] for c in data["train"]["categories"]}
    # Ensure all supplied metadata files agree on official category ID/name pairs.
    for split, d in data.items():
        got = {int(c["id"]): c["name"] for c in d["categories"]}
        if got != categories:
            raise ValueError(f"Category definitions differ in {split}; investigate before analysis.")

    summary, count_rows, image_rows, overlap_rows = [], [], [], []
    records_by_split = {}
    for split, d in data.items():
        ann_by_image = defaultdict(set)
        for ann in d.get("annotations", []):
            ann_by_image[ann["image_id"]].add(categories.get(int(ann["category_id"]), f"unknown_{ann['category_id']}"))
        images = d["images"]
        seqs = {im.get("seq_id") for im in images if im.get("seq_id") is not None}
        locs = {im.get("location") for im in images if im.get("location") is not None}
        summary.append({"split": split, "images": len(images), "annotations": len(d.get("annotations", [])),
                        "unique_seq_ids": len(seqs), "unique_locations": len(locs),
                        "duplicate_image_ids": len(images)-len({im.get("id") for im in images}),
                        "duplicate_file_names": len(images)-len({im.get("file_name") for im in images})})
        records_by_split[split] = images
        category_image_counts = Counter()
        for im in images:
            labels = ann_by_image.get(im.get("id"), set())
            for label in labels:
                category_image_counts[label] += 1
            image_rows.append({"split": split, "image_id": im.get("id"), "file_name": im.get("file_name"),
                "width": im.get("width"), "height": im.get("height"),
                "aspect_ratio": im.get("width") / im.get("height") if im.get("width") and im.get("height") else None,
                "seq_id": im.get("seq_id"), "location": im.get("location"),
                "frame_num": im.get("frame_num"), "seq_num_frames": im.get("seq_num_frames"),
                "date_captured": im.get("date_captured"), "categories": "|".join(sorted(labels))})
        for name in categories.values():
            count_rows.append({"split": split, "category": name, "images_with_category": category_image_counts[name]})

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            seq_a = {x.get("seq_id") for x in records_by_split[a] if x.get("seq_id") is not None}
            seq_b = {x.get("seq_id") for x in records_by_split[b] if x.get("seq_id") is not None}
            loc_a = {x.get("location") for x in records_by_split[a] if x.get("location") is not None}
            loc_b = {x.get("location") for x in records_by_split[b] if x.get("location") is not None}
            overlap_rows.append({"split_a": a, "split_b": b, "shared_seq_ids": len(seq_a & seq_b),
                                 "shared_locations": len(loc_a & loc_b)})

    sdf, cdf, idf, odf = map(pd.DataFrame, (summary, count_rows, image_rows, overlap_rows))
    sdf.to_csv(args.output_dir / "split_summary.csv", index=False)
    cdf.to_csv(args.output_dir / "class_counts_by_split.csv", index=False)
    idf.to_csv(args.output_dir / "image_metadata_manifest.csv", index=False)
    odf.to_csv(args.output_dir / "split_overlap_audit.csv", index=False)

    pivot = cdf.pivot(index="category", columns="split", values="images_with_category").fillna(0)
    ax = pivot.sort_values("train", ascending=True).plot(kind="barh", figsize=(11, 8))
    ax.set(title="Images containing each category by provided split", xlabel="Image count (category presence)", ylabel="Category")
    plt.tight_layout(); plt.savefig(args.output_dir / "class_distribution_by_split.png", dpi=160); plt.close()

    valid_dims = idf.dropna(subset=["width", "height"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, group in valid_dims.groupby("split"):
        ax.scatter(group["width"], group["height"], s=8, alpha=.25, label=split)
    ax.set(title="Declared image dimensions by split", xlabel="Metadata width", ylabel="Metadata height")
    ax.legend(); plt.tight_layout(); plt.savefig(args.output_dir / "dimension_scatter.png", dpi=160); plt.close()

    # Optional file integrity and exact byte-duplicate scan. Only checks files that exist under image_root.
    file_rows, hash_to_paths = [], defaultdict(list)
    if args.image_root:
        for row in idf.to_dict("records"):
            candidates = list(args.image_root.rglob(Path(str(row["file_name"])).name))
            path = candidates[0] if candidates else None
            status, digest, actual_w, actual_h = "missing", None, None, None
            if path:
                try:
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        actual_w, actual_h = im.size
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    hash_to_paths[digest].append(str(path))
                    status = "readable"
                except Exception as e:
                    status = "unreadable"
            file_rows.append({**row, "resolved_path": str(path) if path else None, "file_status": status,
                              "sha256": digest, "actual_width": actual_w, "actual_height": actual_h})
        fdf = pd.DataFrame(file_rows)
        fdf.to_csv(args.output_dir / "image_file_validation.csv", index=False)
        pd.DataFrame([{"sha256": h, "paths": "|".join(ps), "duplicate_count": len(ps)}
                      for h, ps in hash_to_paths.items() if len(ps) > 1]).to_csv(
                          args.output_dir / "exact_duplicate_files.csv", index=False)
        # A compact representative sample grid, one image per category where available.
        sample_rows = idf[idf["categories"].fillna("").ne("")].copy()
        sample_rows["primary_category"] = sample_rows["categories"].str.split("|").str[0]
        chosen = sample_rows.drop_duplicates("primary_category").head(16)
        if len(chosen):
            fig, axes = plt.subplots((len(chosen)+3)//4, 4, figsize=(14, 3.3*((len(chosen)+3)//4)))
            axes = list(axes.flat)
            for ax, row in zip(axes, chosen.to_dict("records")):
                candidates = list(args.image_root.rglob(Path(str(row["file_name"])).name))
                if candidates:
                    try:
                        with Image.open(candidates[0]) as im:
                            ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
                        ax.set_title(f"{row['primary_category']} | {row['split']}", fontsize=9)
                    except Exception:
                        ax.set_title(f"Unreadable: {row['file_name']}", fontsize=8)
                ax.axis("off")
            for ax in axes[len(chosen):]: ax.axis("off")
            plt.tight_layout(); plt.savefig(args.output_dir / "representative_samples.png", dpi=140); plt.close()

    print("EDA outputs written to", args.output_dir)
    print(sdf.to_string(index=False))
    print("\\nTrain category presence counts:\\n", cdf[cdf.split == "train"].sort_values("images_with_category", ascending=False).to_string(index=False))
    print("\\nSplit overlap audit:\\n", odf.to_string(index=False))
if __name__ == "__main__":
    main()
