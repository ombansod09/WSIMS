# WSIMS — Wildlife Species Identification

WSIMS classifies camera-trap images into eight frozen Caltech Camera Traps CCT20 classes using a trained MobileNetV2 model. It includes dataset validation, exploratory analysis, model training/evaluation artifacts, a Streamlit inference UI, persistent local prediction history, and analytics.

## Project modules
- `config/classes.json`: frozen class mapping and class order.
- `src/models/inference_mobilenet.py`: shared MobileNetV2 inference interface; do not duplicate preprocessing or inference in the UI.
- `src/history/`: SQLite persistence and pure analytics helpers.
- `app.py`: Streamlit classification UI and prediction analytics dashboard.
- `artifacts/`: existing checkpoints and evaluation outputs. Do not overwrite these during app use or tests.

## Install and run on Windows
```powershell
cd "E:\CL1- Mini Project"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
The default history database is created at `artifacts/history/predictions.sqlite3`. Set `WSIMS_HISTORY_DB` to override it.

## Testing
```powershell
python -m pytest tests/test_member5_inference.py -q
python -m pytest tests/test_member6_history.py -q
python -m pytest -q
```

## Prediction history and dashboard
Each successful classification can be recorded with a UUID event ID, UTC timestamp, predicted class, confidence fraction, model version, SHA-256 hash of the original uploaded bytes, and top-three predictions. Uploaded images are not stored. The analytics tab includes class counts, a class distribution chart, daily prediction trends, filters, and recent history. Clearing history requires explicit confirmation.

Dashboard counts represent recorded model usage/predictions, not wildlife populations or animal abundance. The model predicts one class for the entire image and does not detect or localize animals. Only the eight frozen classes are supported.

See `README_MEMBER1.md`, `README_MEMBER3.md`, `README_MEMBER4.md`, `README_MEMBER5.md`, and `README_MEMBER6.md` for member-specific details.

# Dashboard
<img width="1821" height="792" alt="Screenshot 2026-10-15 053402" src="https://github.com/user-attachments/assets/bb07bc11-3057-4b96-8329-d292b6feaae3" />
<img width="1815" height="861" alt="Screenshot 2026-10-15 053446" src="https://github.com/user-attachments/assets/071fd7a6-6a71-498b-9aad-e3f1e458f925" />
<img width="1825" height="697" alt="Screenshot 2026-10-15 053558" src="https://github.com/user-attachments/assets/032c20b4-e828-466e-bce8-450b8eb5a8b0" />
<img width="1818" height="830" alt="Screenshot 2026-10-15 053626" src="https://github.com/user-attachments/assets/1f7d1b84-a3ca-4e75-af1d-7886da9e761f" />






