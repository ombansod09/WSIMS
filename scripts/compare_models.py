"""Fair, side-by-side evaluation of the WSIMS baseline CNN and MobileNetV2.

Run from the integrated repository root. Both models use the same manifest rows,
image decode/resize implementation, class order, batch size, and metric code.
The evaluator applies model-specific input scaling because the architectures have
intentionally different input contracts.
"""
from __future__ import annotations
import argparse
import json
import platform
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

# Support execution as `python scripts/compare_models.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--classes', default='config/classes.json')
    p.add_argument('--baseline-model', required=True, help='Member 3 best.keras checkpoint')
    p.add_argument('--mobilenet-model', required=True, help='Member 4 mobilenetv2_best.keras checkpoint')
    p.add_argument('--split', required=True, choices=['cis_val', 'trans_val', 'cis_test', 'trans_test'])
    p.add_argument('--output-dir', default='artifacts/metrics/model_comparison')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accepted-statuses', default='ok,valid,readable,scaled_resolution')
    return p.parse_args()



OFFICIAL_SPLITS = {"train", "cis_val", "cis_test", "trans_val", "trans_test"}


def validate_comparison_manifest(df, classes, accepted_statuses):
    """Validate required manifest fields without depending on Member 3 helper APIs."""
    required = {"image_path", "mapped_class", "split", "file_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    frame = df.copy()
    for column in ("image_path", "mapped_class", "split", "file_status"):
        frame[column] = frame[column].astype("string").str.strip()

    frame["file_status"] = frame["file_status"].str.lower()

    if frame["image_path"].isna().any() or (frame["image_path"] == "").any():
        raise ValueError("Manifest has missing image_path values.")
    if frame["image_path"].duplicated().any():
        raise ValueError("Manifest contains duplicate image_path rows.")
    if frame["mapped_class"].isna().any() or (frame["mapped_class"] == "").any():
        raise ValueError("Manifest contains missing mapped_class values.")
    if frame["split"].isna().any() or (frame["split"] == "").any():
        raise ValueError("Manifest contains missing split values.")
    bad_splits = sorted(set(frame["split"].astype(str)) - OFFICIAL_SPLITS)
    if bad_splits:
        raise ValueError(f"Invalid official split labels: {bad_splits}")

    unknown_classes = sorted(set(frame["mapped_class"].astype(str)) - set(classes))
    if unknown_classes:
        raise ValueError(f"Manifest contains classes outside the frozen class list: {unknown_classes}")

    if frame["file_status"].isna().any() or (frame["file_status"] == "").any():
        raise ValueError("Manifest contains missing file_status values.")
    bad_statuses = sorted(set(frame["file_status"].astype(str)) - set(accepted_statuses))
    if bad_statuses:
        raise ValueError(
            f"Manifest has statuses not explicitly approved: {bad_statuses}; "
            f"accepted={sorted(accepted_statuses)}"
        )

    missing_paths = [p for p in frame["image_path"].astype(str) if not Path(p).is_file()]
    if missing_paths:
        raise ValueError(
            f"{len(missing_paths)} manifest image paths do not exist. "
            f"Examples: {missing_paths[:5]}"
        )
    return frame


def main():
    args = parse_args()
    import tensorflow as tf
    from src.features.preprocessing import load_classes

    classes = load_classes(args.classes)
    accepted = [x.strip().lower() for x in args.accepted_statuses.split(',') if x.strip()]
    df = validate_comparison_manifest(pd.read_csv(args.manifest), classes, accepted)
    subset = df.loc[df['split'] == args.split].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f'No eligible records for split={args.split}')
    counts = subset['mapped_class'].value_counts().reindex(classes, fill_value=0)
    if (counts == 0).any():
        raise ValueError(f'Selected split is missing one or more classes: {counts[counts == 0].index.tolist()}')

    c2i = {name: i for i, name in enumerate(classes)}
    paths = subset['image_path'].astype(str).to_numpy()
    y_true = subset['mapped_class'].map(c2i).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def decode_resize(path):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        # Keep [0,255] here; transform separately for each model below.
        image = tf.image.resize(image, [224, 224], method='bilinear', antialias=True)
        return image

    ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    model_specs = [
        ('baseline_cnn', args.baseline_model, 'rgb_float32_[0,1]'),
        ('mobilenetv2', args.mobilenet_model, 'rgb_float32_[0,255]_then_in_model_to_[-1,1]'),
    ]
    results = {
        'comparison_protocol': {
            'split': args.split, 'n_examples': int(len(subset)), 'batch_size': int(args.batch_size),
            'image_size': [224, 224], 'color_space': 'RGB', 'class_order': classes,
            'manifest': str(Path(args.manifest).resolve()), 'same_manifest_rows_for_both_models': True,
            'official_split_assignments_changed': False,
            'timing_method': 'sum of warmed-up batched model-call wall times; excludes decode/resize and is hardware/runtime dependent',
            'warning': 'cis_val has reported sequence-ID overlap with train. Use cis_val only for model selection; report test splits only after model selection is frozen.'
        },
        'models': {}
    }
    for name, model_path, input_contract in model_specs:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f'Model checkpoint not found: {model_file}')
        model = tf.keras.models.load_model(model_file)
        if int(model.output_shape[-1]) != len(classes):
            raise ValueError(f'{name} output width {model.output_shape[-1]} != {len(classes)} classes')
        if model.count_params() <= 0:
            raise ValueError(f'{name} has no trainable/model parameters')

        preds, timing_sums = [], []
        for batch_index, raw_batch in enumerate(ds):
            x = raw_batch / 255.0 if name == 'baseline_cnn' else raw_batch
            # First batch warms the model and is predicted, but excluded from timing.
            if batch_index == 0:
                prob = model(x, training=False).numpy()
            else:
                start = time.perf_counter()
                prob = model(x, training=False).numpy()
                timing_sums.append(time.perf_counter() - start)
            preds.extend(np.argmax(prob, axis=1).astype(int).tolist())

        # Ensure alignment is exact and compute identical metric definitions.
        y_pred = np.asarray(preds, dtype=np.int32)
        labels = list(range(len(classes)))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        measured_s = float(sum(timing_sums))
        timed_images = max(0, len(subset) - min(args.batch_size, len(subset)))
        # For a one-batch split, run a second timed pass after warm-up.
        if not timing_sums:
            for raw_batch in ds:
                x = raw_batch / 255.0 if name == 'baseline_cnn' else raw_batch
                start = time.perf_counter()
                _ = model(x, training=False).numpy()
                measured_s += time.perf_counter() - start
            timed_images = len(subset)
        report = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results['models'][name] = {
            'checkpoint': str(model_file.resolve()), 'input_contract': input_contract,
            'parameters': int(model.count_params()), 'checkpoint_size_bytes': model_file.stat().st_size,
            'accuracy': float(accuracy_score(y_true, y_pred)), 'macro_precision': float(precision),
            'macro_recall': float(recall), 'macro_f1': float(f1),
            'timed_model_call_seconds': measured_s,
            'timed_images_excluding_warmup_batch': int(timed_images),
            'batch_call_ms_per_image_excluding_warmup': (measured_s * 1000 / timed_images) if timed_images else None,
            'images_per_second_excluding_warmup': (timed_images / measured_s) if measured_s > 0 and timed_images else None,
            'per_class': report, 'confusion_matrix': cm.tolist(),
        }
        del model

    results['environment'] = {'python': platform.python_version(), 'tensorflow': tf.__version__, 'platform': platform.platform()}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json_path = out / f'comparison_{args.split}.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    rows = []
    for name, item in results['models'].items():
        rows.append({k: item.get(k) for k in ['accuracy','macro_precision','macro_recall','macro_f1','parameters','checkpoint_size_bytes','batch_call_ms_per_image_excluding_warmup','images_per_second_excluding_warmup'] } | {'model': name, 'split': args.split, 'n_examples': len(subset)})
        pd.DataFrame(item['confusion_matrix'], index=classes, columns=classes).to_csv(out / f'{name}_{args.split}_confusion_matrix.csv', index=True, index_label='true_class')
        pd.DataFrame(item['per_class']).T.to_csv(out / f'{name}_{args.split}_classification_report.csv')
    pd.DataFrame(rows).to_csv(out / f'comparison_{args.split}.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Full metrics: {json_path.resolve()}')

if __name__ == '__main__':
    main()
