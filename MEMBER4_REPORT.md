# WSIMS Member 4 — Transfer Learning Report

## Status

**Implementation package created; model training and numerical evaluation are not yet complete.** No accuracy, precision, recall, F1, duration, or inference latency results are claimed because TensorFlow and the dataset are unavailable in this authoring runtime.

## Project artifacts inspected

- Member 3 baseline package and status report.
- Member 1 validation report, which declares the local project root as `E:\CL1- Mini Project`, image root `eccv_18_all_images_sm`, and frozen classes file `config\classes.json`.
- Member 2 EDA report and split-overlap audit supplied in the handoff.

The local Windows paths are metadata only from this runtime; the actual image files and current validated manifest were not mounted. The exact current manifest headers and annotation eligibility therefore still need confirmation on the user's computer.

## Architecture and preprocessing

- MobileNetV2 convolutional backbone, `include_top=False`, ImageNet weights by default.
- Global average pooling, dropout (0.25), and Dense softmax output with eight units.
- Inputs are RGB float32 `[0,255]`, resized to 224×224. A model-embedded Rescaling layer maps values to `[-1,1]`, mathematically equivalent to TensorFlow's official `mobilenet_v2.preprocess_input`. Embedding preprocessing makes training and saved-model inference consistent.
- Phase 1: frozen backbone, Adam `1e-3`, up to 15 epochs.
- Phase 2: optionally unfreeze last 30 backbone layers while keeping BatchNormalization layers non-trainable, Adam `1e-5`, up to 8 epochs.
- Seed 42, class weights computed from eligible training rows only, early stopping, best-`cis_val`-loss checkpoint, CSV histories.
- A pretrained-weight initialization failure is fatal by default. `--weights none` is allowed only as an explicitly non-pretrained experiment.

## Data and benchmark protocol

- Frozen order: `opossum`, `rabbit`, `coyote`, `cat`, `squirrel`, `raccoon`, `dog`, `bobcat`.
- Training uses only `train`; checkpoint selection uses `cis_val`; evaluation can be run separately on `cis_val`, `cis_test`, `trans_val`, and `trans_test`.
- No official split assignments are changed and no random replacement split is created.
- Member 2 reports 224 shared sequence IDs between `train` and `cis_val`. The team must confirm the intended ID/OOD protocol before interpreting final benchmark metrics. Do not overstate validation independence.
- Unsupported/multi-label target policy must be approved before deriving the training manifest. No unsupported category is mapped to a supported class.

## Verification in this runtime

- `python -m compileall -q src scripts tests`: passed.
- `python -m pytest -q tests/test_mobilenet_contract.py`: **3 passed, 1 skipped**. The skipped test constructs/saves/loads a Keras model and checks the eight-probability output; it skipped because TensorFlow is not installed here.
- TensorFlow import check: failed with `ModuleNotFoundError: No module named 'tensorflow'`.
- ImageNet weights: not checked/downloaded here.
- Training, checkpoint creation, dataset decoding, real evaluation metrics, and latency: not run.

## Outputs when run in the agreed environment

- `artifacts/models/mobilenetv2_best.keras`
- `artifacts/models/mobilenetv2_config.json`
- `artifacts/metrics/mobilenetv2_history.csv`
- `artifacts/metrics/mobilenetv2_finetune_history.csv`
- Per-split metrics JSON, per-class report CSV, and confusion-matrix CSV.

## Remaining blockers

1. Run in Python 3.11 with the team's agreed TensorFlow version; Member 3 reported TensorFlow absent in their authoring runtime too.
2. Verify actual `config/classes.json` in the repository is the frozen version, not an older placeholder.
3. Confirm current validated-manifest columns and approve unsupported/multi-label handling before deriving eligible records.
4. Verify ImageNet weights are available through internet access or Keras cache.
5. Train and evaluate both baseline and MobileNetV2 on the same derived manifest and approved evaluation protocol before making comparative claims.
