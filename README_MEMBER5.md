# WSIMS Member 5 — Streamlit prediction interface

## Requirements
- Windows with Python compatible with the TensorFlow version already used by this project
- Existing project `.venv`
- Existing checkpoint at `artifacts/models/mobilenetv2/mobilenetv2_best.keras`

## Install
Open PowerShell and move to the project root:

```powershell
cd "E:\CL1- Mini Project"
.\.venv\Scripts\Activate.ps1
python -m pip show streamlit
```

If Streamlit is not installed:

```powershell
python -m pip install "streamlit>=1.35"
```

Alternatively install all declared dependencies (this may resolve packages, so prefer the single-package command if the existing environment is already configured):

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL in the terminal. Open it in a browser, upload a JPEG/JPG, PNG, or WEBP image, and click **Classify image**.

## Features
- Preview and basic validation of uploaded images
- Cached loading of the existing MobileNetV2 checkpoint
- Predicted class, model score as a percentage, and top-three class scores
- Checks that inference returns eight finite probabilities summing to approximately one
- No training, dataset changes, checkpoint writes, bounding boxes, or prediction-history database

The image is converted to RGB, EXIF orientation is applied, and it is resized to 224×224. Pixel values remain in `[0,255]` because the trained network contains the MobileNetV2 rescaling layer. Do not preprocess a second time.

## Tests

```powershell
python -m pytest tests/test_member5_inference.py -q
python -m pytest -q
```

The new unit tests use a fake model for the prediction contract and do not require TensorFlow checkpoint loading. For a real checkpoint smoke test, run the app and classify a valid image after confirming TensorFlow is installed:

```powershell
python -c "from src.models.inference_mobilenet import MobileNetV2Classifier; c=MobileNetV2Classifier(); print(c.model.output_shape)"
```

## Troubleshooting
- **Streamlit not found:** activate `.venv` and run `python -m pip install "streamlit>=1.35"`.
- **TensorFlow import error:** use the project's working TensorFlow/Python combination; avoid upgrading TensorFlow without coordinating with the team.
- **Checkpoint missing:** ensure `artifacts/models/mobilenetv2/mobilenetv2_best.keras` is present. The app uses project-relative paths.
- **Invalid image:** try re-exporting it as a standard JPEG or PNG.
- **Low confidence or unexpected class:** outputs are not guarantees. This is a whole-image classifier for only the frozen eight classes and may perform poorly on out-of-distribution images.
