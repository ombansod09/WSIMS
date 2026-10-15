"""Streamlit interface for WSIMS wildlife classification and local history."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.models.inference_mobilenet import MobileNetV2Classifier, validate_uploaded_image
from src.history.store import PredictionHistory, DEFAULT_HISTORY_PATH
from src.history.analytics import summarize, daily_counts

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "models" / "mobilenetv2" / "mobilenetv2_best.keras"
CLASSES_PATH = ROOT / "config" / "classes.json"
HISTORY_PATH = Path(os.environ.get("WSIMS_HISTORY_DB", str(DEFAULT_HISTORY_PATH)))
with open(CLASSES_PATH, encoding="utf-8") as _f:
    CLASS_NAMES = json.load(_f)["class_names"]

st.set_page_config(page_title="WSIMS Wildlife Identification", page_icon="🐾", layout="wide")

@st.cache_resource(show_spinner="Loading trained MobileNetV2 model…")
def load_classifier():
    return MobileNetV2Classifier(model_path=str(MODEL_PATH), classes_path=str(CLASSES_PATH))

@st.cache_resource
def load_history(path: str):
    return PredictionHistory(path)

try:
    history = load_history(str(HISTORY_PATH))
except Exception as exc:
    history = None
    st.error(f"Prediction history database is unavailable: {exc}")

st.title("🐾 Wildlife Species Identification")
st.write("Upload a camera-trap image to see which of the project's eight wildlife classes the trained MobileNetV2 model considers most likely.")
st.caption("WSIMS · Caltech Camera Traps CCT20 · Whole-image classification")

classifier_tab, dashboard_tab = st.tabs(["Classify image", "Prediction analytics"])

with classifier_tab:
    uploaded = st.file_uploader("Choose a wildlife image", type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPEG, PNG, and WEBP. The image is resized to 224 × 224 for inference.")
    if uploaded is None:
        st.info("Choose an image to begin.")
    else:
        raw = uploaded.getvalue()
        try:
            image = validate_uploaded_image(raw)
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with right:
                st.subheader("Prediction")
                if st.button("Classify image", type="primary", use_container_width=True, key="classify_button"):
                    try:
                        with st.spinner("Classifying image…"):
                            model = load_classifier()
                            result = model.predict_image(image, top_k=3)
                        if (result.get("predicted_class") not in CLASS_NAMES or
                            not isinstance(result.get("confidence"), (float, int)) or
                            not 0 <= float(result["confidence"]) <= 1 or
                            not result.get("top_k")):
                            raise ValueError("Inference returned an invalid prediction payload.")
                        st.session_state["wsims_last_result"] = {"result": result, "sha": hashlib.sha256(raw).hexdigest()}
                        if history is not None:
                            # A fresh key is created only on the actual button-click rerun.
                            click_key = str(uuid.uuid4())
                            try:
                                history.record_prediction(
                                    predicted_class=result["predicted_class"],
                                    confidence=result["confidence"],
                                    model_version=result.get("model_version", "unknown"),
                                    image_bytes=raw, top_k=result["top_k"], event_key=click_key)
                                st.session_state["wsims_last_event_key"] = click_key
                            except Exception as exc:
                                st.warning(f"Prediction succeeded, but history could not be saved: {exc}")
                    except Exception as exc:
                        st.error(f"Could not run prediction. Check that the trained model file and dependencies are available. Details: {exc}")
                last = st.session_state.get("wsims_last_result")
                if last and last["sha"] == hashlib.sha256(raw).hexdigest():
                    result = last["result"]
                    st.success("Prediction complete")
                    st.markdown("**Most likely species**")
                    st.header(result["predicted_class"].replace("_", " ").title())
                    st.metric("Model confidence", f'{result["confidence"] * 100:.2f}%')
                    st.markdown("**Top 3 predictions**")
                    for item in result["top_k"]:
                        st.write(f'{item["class_name"].title()} — {item["score"] * 100:.2f}%')
                        st.progress(float(item["score"]))
                    st.caption("Scores are model outputs, not guarantees that the prediction is correct.")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"This file could not be read as a valid image. Please upload a non-corrupted JPEG, PNG, or WEBP image. Details: {exc}")
        except Exception as exc:
            st.error(f"Unable to process this upload: {exc}")

with dashboard_tab:
    st.subheader("Prediction analytics")
    st.caption("These statistics describe recorded model usage and predictions, not wildlife population sizes or animal abundance.")
    if history is None:
        st.warning("History is currently unavailable. Check the database path and permissions.")
    else:
        all_rows = history.query_predictions()
        if not all_rows:
            st.info("No prediction history yet. Classify a valid image to populate this dashboard.")
        else:
            from datetime import date, timedelta
            stamps = [r["timestamp_utc"][:10] for r in all_rows if r.get("timestamp_utc")]
            min_day = date.fromisoformat(min(stamps)) if stamps else date.today()
            max_day = date.fromisoformat(max(stamps)) if stamps else date.today()
            f1, f2, f3 = st.columns([1, 1, 1])
            with f1:
                start = st.date_input("From date (UTC)", value=min_day, min_value=min_day, max_value=max_day, key="history_start")
            with f2:
                end = st.date_input("To date (UTC)", value=max_day, min_value=min_day, max_value=max_day, key="history_end")
            with f3:
                selected_class = st.selectbox("Predicted class", ["All"] + CLASS_NAMES, key="history_class")
            if start > end:
                st.warning("Start date must not be after end date.")
                rows = []
            else:
                rows = history.query_predictions(start_date=start.isoformat(), end_date=end.isoformat(),
                                                 predicted_class=selected_class)
            summary = summarize(rows, CLASS_NAMES)
            m1, m2 = st.columns(2)
            m1.metric("Recorded predictions (filtered)", summary["total"])
            m2.metric("Classes represented", sum(1 for n in summary["class_counts"].values() if n))
            st.markdown("**Predictions by class**")
            class_df = pd.DataFrame({"Class": list(summary["class_counts"].keys()),
                                     "Predictions": list(summary["class_counts"].values())})
            st.bar_chart(class_df.set_index("Class"))
            st.markdown("**Prediction trend by UTC date**")
            trend = pd.DataFrame(daily_counts(rows))
            if trend.empty:
                st.info("No predictions match the selected filters.")
            else:
                st.line_chart(trend.set_index("date"))
            display_rows = [{
                "UTC timestamp": r["timestamp_utc"], "Predicted class": r["predicted_class"].title(),
                "Confidence": f'{r["confidence"] * 100:.2f}%', "Model": r["model_version"],
                "Image SHA-256": r["image_sha256"], "Top 3": ", ".join(
                    f'{x["class_name"]} ({x["score"]*100:.2f}%)' for x in r["top_k"])
            } for r in rows]
            st.markdown("**Recent prediction history**")
            st.dataframe(pd.DataFrame(display_rows).head(100), use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("**Clear prediction history**")
        st.warning("This permanently deletes all local prediction-history records. Model files and dataset artifacts are not affected.")
        confirm = st.checkbox("I understand that all prediction-history records will be deleted.", key="confirm_clear_history")
        if st.button("Clear all history", disabled=not confirm, key="clear_history_button"):
            try:
                history.clear()
                st.session_state.pop("wsims_last_event_key", None)
                st.success("Prediction history cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not clear prediction history: {exc}")

st.divider()
st.caption("Disclaimer: This model can be wrong. It classifies the image as a whole and does not locate or detect individual animals. Results may be unreliable for unfamiliar camera locations, poor-quality images, or images outside the training distribution. Only the eight configured project classes are supported.")
