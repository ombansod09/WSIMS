# WSIMS Member 6 — Prediction history, analytics, and integration

## Overview
The Streamlit app retains Member 5's MobileNetV2 inference API and adds a local SQLite prediction history plus an analytics tab. History contains successful model outputs only; image bytes are not stored. The SHA-256 digest is retained for traceability. Confidence and top-k scores are fractions from 0 to 1; the UI displays percentages. Timestamps are UTC ISO-8601 strings ending in `Z`.

## Database
Default path: `artifacts/history/predictions.sqlite3` (created automatically). Override it with `WSIMS_HISTORY_DB` using a path relative to the current working directory or an absolute path. The database contains event ID, idempotency key, UTC timestamp, predicted class, confidence, model version, image SHA-256, and top-k JSON. SQLite queries are parameterized. History can be cleared from the dashboard only after checking the explicit confirmation box.

## Windows setup and startup
```powershell
cd "E:\CL1- Mini Project"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If dependencies are already installed in the working `.venv`, avoid unnecessary upgrades to TensorFlow.

## Tests
```powershell
python -m pytest tests/test_member5_inference.py -q
python -m pytest tests/test_member6_history.py -q
python -m pytest -q
```
History tests use temporary SQLite databases and do not write to the real history database or load the model checkpoint.

## Troubleshooting
- **History cannot be opened:** check write permissions for `artifacts/history/` or set `WSIMS_HISTORY_DB`.
- **Model cannot load:** confirm `artifacts/models/mobilenetv2/mobilenetv2_best.keras` exists and the current environment has its expected TensorFlow build.
- **No dashboard data:** classify an image successfully; empty history is expected on first run.
- **Bad image:** corrupt or unsupported image bytes fail validation and are not logged as successful predictions.

## Limitations
- This is whole-image classification, not object detection.
- Counts describe application usage and model predictions, not wildlife abundance or population.
- A history record is created when the user clicks Classify and inference returns a valid payload. A new explicit click is a new event, even if the same image is classified again.
