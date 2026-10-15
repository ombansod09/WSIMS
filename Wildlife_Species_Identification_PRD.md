# PRODUCT REQUIREMENTS DOCUMENT (PRD)
## Wildlife Species Identification and Monitoring System

**Version:** 1.0  
**Project type:** Student mini project  
**Team size:** 6 members  
**Status:** Baseline specification

---

## 1. Purpose and project definition

Build one integrated Python application that classifies an uploaded wildlife/camera-trap image into one of eight selected animal classes, shows a model confidence score (not a guarantee of correctness), records prediction events with user consent, and presents aggregate monitoring statistics.

The project demonstrates data preparation, exploratory data analysis (EDA), supervised learning, model evaluation, application development, and a small monitoring dashboard.

**Primary user:** A student, researcher, or conservation volunteer reviewing camera-trap images.

This is an educational prototype, not a wildlife management or safety-critical system. It must not claim to identify every species, estimate population size, or make conservation decisions.

## 2. Scope and fixed decisions

| Decision | Fixed specification |
|---|---|
| Project name | Wildlife Species Identification and Monitoring System (WSIMS) |
| Primary task | Single-image, 8-class animal classification |
| Primary dataset | Caltech Camera Traps (CCT); start with the official CCT20 benchmark subset (approximately 6 GB), not the full approximately 105 GB collection |
| Official dataset page | <https://lila.science/datasets/caltech-camera-traps> |
| Dataset metadata | Use the official image-level annotations and recommended CCT20 train/validation split from the same page. Do not invent labels manually. |
| Class selection | From the official training split, select the eight most frequent non-empty animal categories that each have at least 500 usable images across the available labeled data. Exclude “empty/no animal” from the 8-class classifier. If fewer than eight categories meet the threshold, lower it to 300 and document the change. Freeze the final class list in `config/classes.json` before model training. |
| Split policy | Respect the dataset-provided camera/location-aware train/validation split where available. Create the test split only from held-out locations or the official test split if labels are available. Never randomly distribute adjacent frames from the same camera sequence across train and test. |
| Baseline model | Small custom CNN implemented in TensorFlow/Keras |
| Primary model | MobileNetV2 pretrained on ImageNet, with its classification head replaced for the eight project classes |
| Application | Streamlit web application |
| Storage | CSV append-only prediction log for the prototype; do not store uploaded image bytes by default |
| Advanced feature | A pre-trained object detector is an optional separate feature. Clearly label its supported classes and limitations. Do not confuse detector labels with the eight-class classifier output. |
| Source control | One shared Git repository; `main` must always run |

## 3. Dataset and data governance

### Primary dataset

**Caltech Camera Traps (CCT), hosted by LILA BC**

Official page: <https://lila.science/datasets/caltech-camera-traps>

The dataset page describes approximately 243,100 images from 140 camera locations and 21 animal categories plus an empty category. It provides image-level annotations, bounding-box annotations, and recommended splits. The full image collection is approximately 105 GB; the CCT20 benchmark subset is approximately 6 GB. Check the current download options and file sizes on the official page before downloading.

**Dataset citation:** Credit the Caltech Camera Traps dataset and its associated paper by Sara Beery, Grant Van Horn, and Pietro Perona, “Recognition in Terra Incognita,” ECCV 2018. Follow the dataset page’s license and attribution requirements.

Maintain a `DATASET.md` file containing:
- Source URL
- Download date
- License/terms reviewed
- Category mapping
- Filtering and preprocessing steps

### Fallback dataset — only if approved

North American Camera Trap Images (NACTI): <https://lila.science/datasets/nacti>

This is not the default dataset. The team lead must approve a switch and update this PRD/config before any model work is merged. The full NACTI collection is very large, so do not download all of it.

### Data rules

- Never commit the dataset, trained model binaries, private user uploads, or large generated artifacts to Git.
- Add `data/`, `models/`, `logs/`, and `.env` to `.gitignore`.
- Keep dataset files local and provide reproducible setup/download instructions.
- Verify licenses for any sample images used in slides or the README.

## 4. Goals and non-goals

### Goals

- Reproducibly create a clean, documented 8-class dataset manifest.
- Produce EDA charts explaining class balance, image quality, and split distribution.
- Train a simple CNN baseline and a MobileNetV2 transfer-learning model under comparable conditions.
- Evaluate on a held-out test set and publish metrics, confusion matrix, and limitations.
- Deliver a Streamlit app for image upload, top prediction, confidence score, and prediction history.
- Deliver a dashboard with prediction counts by predicted class and date, based on logged app events.
- Provide a single-command local setup/run path and a clear README.

### Non-goals

- No claims of production-grade accuracy or field validation.
- No real-time CCTV pipeline, GPS tracking, animal population estimation, or automated conservation decisions.
- No retraining the model from user uploads in the app.
- No promise that the model can recognize species outside the selected eight classes.
- No database/server/cloud architecture unless the baseline project is complete and the instructor approves an extension.

## 5. Functional requirements

| ID | Requirement |
|---|---|
| FR-01 Data manifest | Generate a CSV/JSON manifest with `image_path`, `original_category`, `mapped_class`, `split`, location/camera identifier when available, and file status. |
| FR-02 Data validation | Detect missing files, unreadable images, duplicate file paths, unsupported formats, and classes below the threshold. Report counts; do not silently discard records. |
| FR-03 EDA | Create class counts, split counts, sample image grid, image dimension summary, imbalance chart, and a written note about possible location/background bias. |
| FR-04 Training | Train baseline CNN and MobileNetV2 with the same class mapping and fixed train/validation/test manifests. Save model, class order, input size, preprocessing details, and training configuration. |
| FR-05 Evaluation | Report test accuracy, macro precision, macro recall, macro F1, per-class precision/recall/F1, confusion matrix, and inference latency on the team’s stated hardware. |
| FR-06 Prediction UI | Accept JPG/JPEG/PNG; validate that the file can be decoded; show the image, predicted class, confidence score, and top-3 scores. |
| FR-07 Prediction logging | After a user runs a prediction, log `timestamp_utc`, `event_id`, `predicted_class`, `confidence`, `model_version`, and `image_sha256`. Do not log image bytes or original filename by default. |
| FR-08 Monitoring dashboard | Display total predictions, predictions per class, predictions over time, confidence distribution, and downloadable CSV log. Clearly state that these are app usage/prediction counts, not actual wildlife population counts. |
| FR-09 Optional detection | If implemented after core acceptance, run a supported pre-trained detector on the image and display boxes/labels. Keep detector output separate from classifier output and explain unsupported classes. |
| FR-10 Reproducibility | Provide `requirements.txt`, config file, seeds, training/evaluation commands, dataset instructions, and README. |

## 6. Fixed technology stack

| Area | Technology |
|---|---|
| Language | Python 3.11; all members use the same version |
| Data/EDA | pandas, NumPy, Matplotlib, seaborn, Pillow |
| Machine learning | TensorFlow/Keras; MobileNetV2 pretrained on ImageNet; custom CNN baseline |
| Evaluation | scikit-learn metrics and confusion matrix |
| Application | Streamlit |
| Logging | CSV using Python `csv` or pandas; UTC timestamps |
| Testing | pytest for preprocessing, model output contract, logging, and app helper functions |
| Collaboration | Git + GitHub; branches and pull requests; `requirements.txt`; `.gitignore` |
| Optional detector | Ultralytics YOLO only if the team verifies package/model licensing and supported classes before use; stretch goal only |

## 7. Six-person work allocation

### Member 1 — Dataset and preprocessing lead

Own dataset download instructions, annotation parsing, category selection, image validation, duplicate checks, split manifests, `classes.json`, and `DATASET.md`.

**Deliverables:** `data/manifest.csv` and a repeatable preprocessing script.

### Member 2 — EDA and data-quality lead

Consume the frozen manifest; generate EDA figures and an EDA report. Do not change class mapping or splits independently. Coordinate any proposed data-quality rule with Member 1 and the team lead.

**Deliverables:** EDA notebook/report, figures, class distribution and split distribution analysis.

### Member 3 — Baseline CNN lead

Implement and train the custom CNN baseline. Use the shared manifest, class order, image size, preprocessing contract, seed, and evaluation interface.

**Deliverables:** Baseline model code, saved model, training history, and run instructions.

### Member 4 — Transfer learning and evaluation lead

Implement MobileNetV2, run fine-tuning experiments, evaluate both models on the exact same held-out test set, and produce metrics and comparison artifacts.

**Deliverables:** `metrics.json`, confusion matrix, per-class report, model comparison report.

### Member 5 — Streamlit prediction UI lead

Build upload/prediction UI, image validation, top-3 output, confidence display, and integration with the selected model loader. Use the shared `inference.py` API; do not duplicate preprocessing logic.

**Deliverables:** Working prediction interface and UI error handling.

### Member 6 — Logging, dashboard, integration/release lead

Implement event logging and dashboard; maintain the integration branch, dependencies, config, README, tests, and final run instructions. Coordinate merges and acceptance checks. Implement the optional detector only after all core work passes.

**Deliverables:** Prediction log module, dashboard, integrated app, tests, and release documentation.

**Shared responsibility:** All six members review the final demo, test each other’s modules, and contribute to the report/presentation. One person may own integration, but no module is complete until another member can run it from the README.

## 8. Repository structure — use this layout

```text
wildlife-species-monitor/
├── app.py
├── requirements.txt
├── README.md
├── DATASET.md
├── .gitignore
├── config/
│   ├── config.yaml
│   └── classes.json
├── data/
│   ├── raw/                  # local only; never commit
│   ├── processed/            # local only; never commit
│   └── manifests/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data/
│   │   ├── build_manifest.py
│   │   └── validate_images.py
│   ├── features/
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   ├── mobilenet_model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── logging/
│   │   └── prediction_log.py
│   └── dashboard/
│       └── analytics.py
├── artifacts/                 # local only; do not commit large files
│   ├── models/
│   ├── metrics/
│   └── figures/
├── logs/                      # local only; do not commit real user logs
└── tests/
    ├── test_preprocessing.py
    ├── test_inference_contract.py
    └── test_prediction_log.py
```

## 9. Shared interfaces and data contracts

### 9.1 Frozen class configuration

`config/classes.json` is the single source of truth. It must contain:
- The exact ordered list of eight class names
- A mapping from original dataset category IDs/names to the project class names

Once Member 1 and the team lead approve it, no member may change the class order without retraining and re-evaluating all models.

### 9.2 Image preprocessing contract

- Input: RGB image decoded with Pillow.
- Model input size: 224 × 224 pixels.
- Classifier output: vector of length 8 in exactly the order defined by `config/classes.json`.
- MobileNetV2 preprocessing: use TensorFlow’s official `MobileNetV2 preprocess_input` consistently in training and inference.
- Training-only augmentation may include horizontal flip and mild rotation/zoom. Validation, test, and inference must not use random augmentation.
- Inference helper signature:

```python
predict_image(image) -> {
    "predicted_class": str,
    "confidence": float,
    "top_k": [{"class_name": str, "score": float}],
    "model_version": str
}
```

- The confidence value is the model’s softmax score. Do not describe it as a calibrated probability unless calibration is separately tested.

### 9.3 Prediction log schema

`logs/predictions.csv` columns, in this order:

```text
event_id,timestamp_utc,predicted_class,confidence,model_version,image_sha256
```

Write the header once and append one row per successful user-triggered prediction. Do not store image data, user identity, geolocation, or original filename. The dashboard must handle a missing or empty log gracefully.

### 9.4 Model artifact contract

Each selected model artifact must be accompanied by:
- `classes.json`
- `model_version`
- `input_size`
- Preprocessing method
- Training manifest identifier/hash
- Random seed
- Evaluation metrics

Member 6 must be able to load the artifact using `src/models/inference.py` without notebook-only variables.

## 10. Model development protocol

1. Establish the dataset manifest and freeze the class list. No model training before this gate passes.
2. Create a baseline CNN with a fixed seed and early stopping on validation loss.
3. Train MobileNetV2 with the pretrained base initially frozen, then optionally fine-tune the top layers with a low learning rate.
4. Use the same train/validation/test manifests for both models. Do not tune hyperparameters on the test set.
5. Save training curves, best checkpoint, class mapping, and config for each run.
6. Select the demo model using macro F1 and per-class performance, not accuracy alone. Record why the model was selected.
7. Report limitations such as background/location bias, class imbalance, confusing species, and poor performance on images unlike the training data.

## 11. Acceptance criteria / definition of done

- **Data:** Eight classes are frozen; manifest contains no missing image paths; corrupt images are reported; split and class counts are reproducible.
- **EDA:** At least five labeled figures and a short interpretation report are committed.
- **Training:** Baseline CNN and MobileNetV2 both train and save loadable artifacts.
- **Evaluation:** Metrics are calculated on the same untouched test split; per-class metrics and confusion matrix are saved.
- **App:** Valid image upload returns top class, score, and top-3 predictions; invalid files show a useful error rather than a traceback.
- **Logging/dashboard:** Successful predictions append one log row; dashboard reads the log and displays aggregate counts; no uploaded image bytes are stored by default.
- **Integration:** A fresh clone, documented setup, dataset files, and documented commands can launch the app without relying on a member’s notebook state.
- **Documentation:** README includes setup, dataset access, training, evaluation, app launch, limitations, and team contributions.
- **Demo:** Demonstrate one plausible prediction, one ambiguous/low-confidence case, model metrics, and dashboard updates. Do not cherry-pick demo examples as evidence of general accuracy.

## 12. Suggested milestones

1. **Milestone 1 — Agreement:** Team confirms Python version, repository, dataset page, class selection rule, and role ownership.
2. **Milestone 2 — Data freeze:** Manifest, split rules, class mapping, and `DATASET.md` reviewed and merged.
3. **Milestone 3 — Parallel module work:** EDA and both model tracks work from the frozen manifest; UI/logging can use a mocked inference function.
4. **Milestone 4 — Model selection:** Evaluation report compares CNN and MobileNetV2; team selects one artifact for the app.
5. **Milestone 5 — Integration:** Prediction UI, logging, dashboard, tests, and selected model run together.
6. **Milestone 6 — Release:** Clean-clone test, bug fixes, README, final report, slides, and demo.

## 13. Git and collaboration rules

- Use `main` for stable integrated code and one feature branch per task, e.g. `feature/data-manifest` or `feature/mobilenet`.
- Open a pull request for each module; another member must review it before merge.
- Do not commit notebooks with machine-specific absolute paths or large embedded outputs.
- Do not rename public functions, paths, schema columns, class names, or model artifact names without team approval.
- Pin tested dependency versions in `requirements.txt` after a clean environment install. Avoid mixing TensorFlow/Keras major versions between members.
- Each pull request must include how to run/test the change and a sample output or screenshot where relevant.
- The integration lead merges only when imports, tests, and a smoke test pass.

## 14. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Download/storage constraints | Use the CCT20 subset first; download only the required files; keep raw data outside Git. |
| Class imbalance | Plot counts; use class weights or balanced sampling on training data only; report macro F1 and per-class metrics. |
| Location/background leakage | Use the official location-aware split; do not random-split neighboring camera frames across train/test. |
| Model predicts outside classes | Show an explicit notice that the model only supports the selected eight categories; do not claim universal species recognition. |
| Softmax overconfidence | Call the value a confidence score; show top-3 classes and limitations. An optional “uncertain” threshold must be validated. |
| Confusion between detector and classifier | Keep detector results in a separate panel/API and state its label set. |
| Integration conflicts | Freeze contracts early, use one repository layout, add tests, and use pull requests. |
| Privacy | Do not retain uploaded image bytes by default; log only prediction metadata and image hash; make clear the CSV is local prototype storage. |

## 15. Final demo deliverables

- One Git repository with reproducible setup instructions.
- Dataset documentation, final class list, manifest-generation script, and EDA report.
- Baseline CNN and MobileNetV2 training code and saved model artifacts (large artifacts shared separately if necessary).
- Test metrics, confusion matrix, training curves, and model comparison table.
- Streamlit application with image upload, prediction, confidence/top-3 output, event logging, and monitoring dashboard.
- Optional detector panel only if the core application is complete and tested.
- Final report covering problem, data source/licensing, methodology, results, limitations, member contributions, and future work.

## 16. Official references

- **Caltech Camera Traps dataset (primary):** <https://lila.science/datasets/caltech-camera-traps>
- **North American Camera Trap Images (fallback only):** <https://lila.science/datasets/nacti>
- **TensorFlow transfer learning / MobileNetV2 tutorial:** <https://www.tensorflow.org/tutorials/images/transfer_learning>
- **Streamlit file uploader documentation:** <https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader>

---

> **Project governance rule:** If a proposed implementation conflicts with this PRD, do not build a separate interpretation. Raise the conflict in the team discussion, agree on one change, update the PRD/config, and only then continue.
    