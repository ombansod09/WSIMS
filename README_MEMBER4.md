# WSIMS Member 4 — MobileNetV2 Transfer Learning

## Status and evidence boundary

This package implements a MobileNetV2 transfer-learning pipeline and a reproducible evaluation entry point. It does **not** include a trained checkpoint or measured accuracy because training requires the full local CCT20 image root, an approved derived manifest, Python 3.11/TensorFlow, and available ImageNet weights. Do not describe an unexecuted run as trained or pretrained.

The existing Member 3 package was inspected. Its manifest contract is `image_path,mapped_class,split,file_status`, its official split names are retained, and its inference adapter expects a model output of eight scores. MobileNetV2 differs in preprocessing: input is RGB float32 in `[0,255]`, converted *inside the model* to `[-1,1]`, equivalent to `tf.keras.applications.mobilenet_v2.preprocess_input`. This keeps the preprocessing in the saved model graph.

## Files

- `src/models/mobilenet_model.py`: MobileNetV2 with 8-way head, ImageNet weights by default, and selective top-layer fine-tuning.
- `src/features/mobilenet_preprocessing.py`: Pillow RGB decode/resize contract.
- `src/models/inference_mobilenet.py`: prediction adapter returning class, confidence, ordered eight probabilities, top-k and optional latency.
- `scripts/train_mobilenetv2.py`: head training then optional fine-tuning; class weights derived from train only; best-validation-loss checkpoint, early stopping and history CSV.
- `scripts/evaluate_mobilenetv2.py`: named official split evaluation with accuracy, macro metrics, per-class report, confusion matrix and model-call latency.
- `tests/test_mobilenet_contract.py`: preprocessing and class-order smoke tests.

## Data policy gate

Before training, create an **approved derived manifest** with exactly these columns: `image_path,mapped_class,split,file_status`. `mapped_class` must be one of the frozen eight names. Unsupported-only images must be excluded with reasons; multi-label images must not be assigned a class unless the annotation policy explicitly approves that rule. The source manifest and official split assignments must remain unchanged. Use Member 3's `scripts/prepare_training_manifest.py` after agreeing the exact source column names and unsupported/multi-label policy.

Example only (replace column names with verified names from your actual CSV):

```powershell
python scripts/prepare_training_manifest.py `
  --source "E:\CL1- Mini Project\member1_validation_output\validated_manifest.csv" `
  --classes "E:\CL1- Mini Project\config\classes.json" `
  --output "E:\CL1- Mini Project\data\manifests\eligible_manifest.csv" `
  --path-column image_path --class-column mapped_class `
  --split-column split --status-column validation_status
```

Do not run that example unchanged until you confirm the real CSV headers and that the class column is a reviewed single-label target. The source manifest can contain multi-annotation records and a guessed class column would invalidate the experiment.

## Setup and run (Windows PowerShell)

Run from the repository root after copying these files into the shared repo and installing the agreed dependencies in Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-member4.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False))"
pytest -q tests/test_mobilenet_contract.py
python scripts/train_mobilenetv2.py --manifest "data/manifests/eligible_manifest.csv" --classes "config/classes.json"
python scripts/evaluate_mobilenetv2.py --manifest "data/manifests/eligible_manifest.csv" --split cis_val
python scripts/evaluate_mobilenetv2.py --manifest "data/manifests/eligible_manifest.csv" --split cis_test
python scripts/evaluate_mobilenetv2.py --manifest "data/manifests/eligible_manifest.csv" --split trans_val
python scripts/evaluate_mobilenetv2.py --manifest "data/manifests/eligible_manifest.csv" --split trans_test
```

The training command defaults to `weights=imagenet`. If the pretrained weights cannot be downloaded or found in Keras cache, training intentionally stops with an error. `--weights none` is an explicit non-pretrained experiment and must not be reported as ImageNet transfer learning.

## Initial hyperparameters

- Resolution: 224×224 RGB.
- Batch size: 32; seed: 42.
- Head phase: Adam 1e-3, up to 15 epochs.
- Fine-tuning phase: last 30 backbone layers considered, BatchNormalization layers remain frozen; Adam 1e-5, up to 8 epochs.
- Early stopping: validation loss, patience 4 for head and 3 for fine-tuning.
- Checkpoint selection: minimum `cis_val` loss only. Do not use test splits for selection.
- Class weighting: inverse-frequency balanced weights computed from eligible training labels only.

Adjust these only using validation data and record changes. `cis_val` shares 224 sequence IDs with `train` according to Member 2's audit; it is not safe to overstate validation independence. The intended ID/OOD protocol must be confirmed by the team before final claims. The official split records are never reassigned by these scripts.

## Outputs

Training writes `artifacts/models/mobilenetv2_best.keras`, `mobilenetv2_config.json`, and history CSVs under `artifacts/metrics`. Evaluation writes JSON metrics, a per-class CSV, and a confusion-matrix CSV for the selected official split. Actual duration and latency are hardware-dependent and must be reported from the actual run.

## Inference integration

```python
from PIL import Image
from src.models.inference_mobilenet import MobileNetV2Classifier
classifier = MobileNetV2Classifier(
    model_path="artifacts/models/mobilenetv2_best.keras",
    classes_path="config/classes.json",
)
with Image.open("example.jpg") as im:
    result = classifier.predict_image(im, top_k=3, include_timing=True)
print(result)
```

`probabilities` is an eight-value vector in frozen class order. `confidence` is the top softmax score, not a calibrated probability or guarantee. Only supported classes can be predicted; no out-of-scope detector or background class is silently introduced.

## Known limitations / blockers

- The training dataset and current local source manifest were not mounted in the authoring runtime, so real dataset counts, class eligibility, checkpoint creation, and metrics have not been verified here.
- The project PRD calls for Python 3.11. Member 3's report states its authoring runtime was Python 3.13.5 with TensorFlow absent; use a consistent Python 3.11 environment and coordinate the TensorFlow version before integration.
- ImageNet weights require network access or a pre-existing Keras cache.
- Class mapping in the actual repository must be checked against the frozen config; do not replace it with a placeholder config.
- Sequence overlap and unsupported/multi-label annotation policy remain benchmark/data governance issues, not something the model code can solve.
