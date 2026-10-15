# WSIMS Member 3 — Baseline CNN

This is the baseline-CNN implementation package for the WSIMS repository. It does **not** include the CCT20 images, the full Member 1 manifest, or a replacement `config/classes.json`.

## Data and governance gates

Before training:
1. Use the project's actual frozen `config/classes.json` (`status: FROZEN`, exactly eight unique `class_names`, and matching `class_to_index`).
2. Generate/review a derived, single-label training manifest with columns `image_path,mapped_class,split,file_status`. `mapped_class` must be one of the eight frozen class names; do not put unsupported labels into the eight-class target.
3. Confirm the team's policy for images with unsupported, multiple, empty, or conflicting categories. This package intentionally does not choose that policy.
4. Confirm the validation/evaluation protocol. `train` and `cis_val` reportedly share 224 sequence IDs, so `cis_val` metrics need that caveat until the team confirms the benchmark interpretation.
5. Resolve the reported `cis_test` coyote/deer duplicate-label discrepancy manually. Do not alter the official split or label based on this package.

The `file_status` field is Member 1's image-validation status, not a training eligibility label. `scaled_resolution` is accepted as a validated/readable status; the training manifest should contain only records explicitly approved for the chosen task. The training code fails on unsupported/missing class labels rather than silently dropping them.

## Input / preprocessing contract

- Pillow RGB decode, EXIF orientation correction, resize to 224×224.
- Custom CNN input: float32 pixels in `[0,1]` (`pixel / 255`).
- Training-only augmentation: horizontal flip and mild rotation inside the model.
- Validation/inference: no random augmentation.
- Output: 8-way softmax in the exact `config/classes.json` order.
- Model artifact: `.keras`; metadata and CSV history saved beside it.
- Shared API: `ImageClassifier(...).predict_image(PIL.Image)` returns `predicted_class`, `confidence`, `top_k`, `model_version`.

`confidence` is the top softmax score, not a calibrated probability.

## Setup (Python 3.11 recommended)

From repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-member3.txt
```

TensorFlow installation must match the operating system/Python version and the team's shared environment. If the repository already has a pinned `requirements.txt`, coordinate dependency changes with Member 6 rather than installing incompatible versions.

## Commands

Run CPU-light contract tests (TensorFlow not required for preprocessing tests):

```powershell
python -m pytest -q tests
```

Run training only after the gates above pass:

```powershell
python -m src.models.train `
  --manifest data/manifests/training_manifest.csv `
  --classes config/classes.json `
  --config config/baseline_cnn.yaml `
  --output-dir artifacts/models/baseline_cnn `
  --validation-split cis_val
```

To train against `trans_val` instead, the team must explicitly approve that protocol. Do not choose the validation split by looking at test results.

Training artifacts:
- `best.keras`: best validation-loss checkpoint.
- `final.keras`: model with best weights restored.
- `training_history.csv`: epoch history.
- `model_metadata.json`: class order, input/preprocessing contract, seed, manifest path/hash, and run configuration.

Evaluation (after protocol approval; do not use test results for tuning):

```powershell
python -m src.models.evaluate `
  --manifest data/manifests/training_manifest.csv `
  --classes config/classes.json `
  --model artifacts/models/baseline_cnn/best.keras `
  --split cis_val `
  --output-dir artifacts/metrics/baseline_cnn
```

## Important current status

The files available in the ChatGPT working environment included a `classes.json` whose status was `NOT_FROZEN` and whose class list was empty. The user-provided handoff reports that the local project has a frozen mapping in the required order, but that local file was not mounted here. Therefore training was not run, and this package does not invent or replace the authoritative mapping. The TensorFlow package is also absent in this environment. No model checkpoint, training history, or measured model metrics are claimed.
