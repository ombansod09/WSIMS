# WSIMS Member 3 — Baseline CNN status report

## Result status
**Source implementation and contract tests are prepared. Training has NOT been run. No measured accuracy/F1, checkpoint, or training history exists yet.**

## Inputs and inconsistencies found
- The supplied PRD requires CCT20, TensorFlow/Keras, 224×224 RGB input, a custom CNN baseline, a frozen eight-class mapping, preserved official splits, and a common inference interface.
- The supplied EDA reports 57,864 image records and 59,941 annotations across five official splits. Training category presence is imbalanced: opossum 2,726 and bobcat 786, about 3.47:1. Counts are image/category presence, not mutually exclusive single-label counts.
- The EDA reports 224 sequence IDs shared between `train` and `cis_val`. Do not change the split assignments; document this when interpreting `cis_val`.
- Member 1's handoff reports `scaled_resolution` for all 57,864 images, eight exact-duplicate groups, no cross-split exact duplicate groups, and one `cis_test` duplicate pair with coyote/deer disagreement. These are handoff-reported findings, not independently recomputed here.
- The actual `classes.json` available in this working environment says `NOT_FROZEN` and has an empty class list. The latest handoff reports that the local Windows project has a frozen mapping with the requested class order, but that local file is not mounted in this environment. The code checks the real config at runtime and does not override it.
- The complete Member 1 manifest and actual CCT20 image root are not available in this working environment, so the manifest columns, image paths, filtered eligible counts, and per-class single-label counts cannot be independently confirmed.
- TensorFlow is not installed in this environment (`ModuleNotFoundError: No module named 'tensorflow'`). The model cannot be trained or loaded here.

## Architecture and settings
- Input: `(224, 224, 3)` RGB float32 scaled to `[0,1]`.
- Three convolutional blocks: 32, 64, 128 filters; two 3×3 convolutions per block; batch normalization and ReLU; max pooling and dropout.
- Global average pooling, dense 128 ReLU, dropout, eight-way softmax.
- Adam, learning rate 0.001, batch size 32, maximum 30 epochs, early stopping on validation loss (patience 5), best-checkpoint saving, CSV history.
- Random seed 42; deterministic TensorFlow ops enabled where available.
- Training-only horizontal flip and mild rotation; no stochastic validation/inference augmentation.
- Class weights are computed from the approved single-label training manifest only.

## Data loader behavior
- Reads the frozen mapping from `config/classes.json` and checks exact order/index consistency.
- Requires `image_path,mapped_class,split,file_status`.
- Accepts `scaled_resolution` as a validation/readability status; this is not an unsupported-class policy.
- Rejects missing labels, unsupported class labels, unknown split names, duplicate image paths, non-approved file statuses, missing paths, and decode failures. It does not silently discard records.
- Streams images in batches via `tf.data.Dataset.from_generator` rather than stacking all images in RAM.
- Does not generate a training manifest from the annotation JSONs because the approved policy for unsupported/multilabel/empty images is not yet confirmed and the actual manifest schema is unavailable.

## Evaluation protocol caveat
The current default validation split is `cis_val`, but it must be approved by the team. Since `train` and `cis_val` reportedly share 224 sequence IDs, validation results may not represent fully sequence-independent performance. `cis_test`/`trans_test` should remain untouched until Member 4/6 confirms the ID/OOD protocol. The evaluation script reports accuracy, macro precision/recall/F1, per-class metrics, and confusion matrix for an explicitly requested official split.

## Files in this package
- `src/features/preprocessing.py`
- `src/models/manifest.py`
- `src/models/baseline_cnn.py`
- `src/models/train.py`
- `src/models/evaluate.py`
- `src/models/inference.py`
- `config/baseline_cnn.yaml`
- `tests/test_preprocessing.py`
- `tests/test_manifest_contract.py`
- `tests/test_classes_config.py`
- `requirements-member3.txt`
- `README_MEMBER3.md`

## Verification performed in this environment
- Pre-change test run failed at collection because TensorFlow is not installed and the old test imported the training module eagerly.
- Updated tests are designed to keep data-contract/preprocessing checks independent of TensorFlow; run them after extracting this package with `python -m pytest -q tests`.
- Training and model inference smoke tests remain blocked by TensorFlow and missing project data/config.
- No duplicate-audit CSV was created.
