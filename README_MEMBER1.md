# Member 1 deliverables — Dataset & preprocessing

This package implements the dataset/preprocessing starter for WSIMS. It does **not** include CCT20 images or pretend that the final eight classes are known before official metadata is inspected.

## Dataset source
Use the official CCT20 benchmark subset and its official COCO Camera Traps annotations/splits:
https://lila.science/datasets/caltech-camera-traps

Download/extract locally. Do not commit data or model files to Git.

Expected local structure (adapt paths if the official archive differs):
```
data/
  raw/CCT20/images/...
  raw/CCT20/annotations/
    train_annotations.json
    cis_val_annotations.json
    trans_val_annotations.json
config/
  classes.json
```

## Important split decision
Do not randomly split individual images. Keep the official camera/location-aware split. In the example command below, `train` is training, `cis_val` is validation (same/known locations), and `trans_val` is the held-out-location evaluation split. Confirm this mapping with the team and document it before using it as the final test split. Never tune hyperparameters on the test split.

## Commands
From repository root, with Python 3.11:
```
pip install pandas pillow
python src/data/build_manifest.py \
  --images-root data/raw/CCT20/images \
  --annotation train=data/raw/CCT20/annotations/train_annotations.json \
  --annotation val=data/raw/CCT20/annotations/cis_val_annotations.json \
  --annotation test=data/raw/CCT20/annotations/trans_val_annotations.json \
  --output data/manifests/manifest.csv \
  --classes-output config/classes.json \
  --min-class-images 500
python src/data/validate_images.py \
  --manifest data/manifests/manifest.csv \
  --report data/manifests/validation_report.json
```
If fewer than eight animal classes pass the minimum threshold, the PRD permits lowering the threshold to 300; agree as a team and record the change. The script selects the eight most frequent eligible animal categories using **training-split counts**, and writes the mapping file. Review the generated class list and mapping with the team lead before freezing it and starting training.

## Outputs
- `data/manifests/manifest.csv`: one row per metadata image, including source category, mapped class, split, location/camera, file status and image dimensions where readable.
- `config/classes.json`: ordered class list and original category ID/name mapping.
- `data/manifests/validation_report.json`: missing/unreadable/duplicate/unsupported files and counts.

The script reports invalid records instead of silently dropping them. It does not modify image bytes, resize files, or create random splits. RGB conversion and resizing belong in the shared preprocessing helper used consistently by training and inference.

## Git/data safety
Keep `data/raw/`, `data/processed/`, `artifacts/`, `models/`, `logs/`, and `.env` out of Git. Commit scripts and metadata manifests only if they do not expose private data or violate dataset terms; confirm with the team.
