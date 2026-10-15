"""Reproducible baseline CNN training; preserves Member 1's official split labels."""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.preprocessing import decode_image, load_classes
from src.models.manifest import read_training_manifest, sha256_file
from src.models.baseline_cnn import build_baseline_cnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classes", default="config/classes.json")
    parser.add_argument("--config", default="config/baseline_cnn.yaml")
    parser.add_argument("--output-dir", default="artifacts/models/baseline_cnn")
    parser.add_argument("--validation-split", default="cis_val",
                        choices=["cis_val", "trans_val"],
                        help="Must match the team-approved protocol; no random repartitioning.")
    args = parser.parse_args()

    try:
        import tensorflow as tf
        import yaml
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:
        raise SystemExit(f"Training dependency unavailable: {exc}. Install requirements-member3.txt.") from exc

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    classes = load_classes(args.classes)
    seed = int(cfg.get("seed", 42))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    df = read_training_manifest(args.manifest, classes, require_existing_paths=True)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == args.validation_split].copy()
    if train_df.empty or val_df.empty:
        raise ValueError(f"Need nonempty train and {args.validation_split} official split.")
    if set(train_df.image_path) & set(val_df.image_path):
        raise ValueError("Train and validation contain identical image paths.")

    # Validate every class is represented in training; report exact eligible counts.
    counts = train_df["mapped_class"].value_counts().reindex(classes, fill_value=0)
    missing_classes = counts[counts == 0].index.tolist()
    if missing_classes:
        raise ValueError(f"Training manifest has no examples for classes: {missing_classes}")

    def generator(frame):
        for row in frame.itertuples(index=False):
            try:
                x = decode_image(row.image_path, (cfg.get("image_height",224), cfg.get("image_width",224)))
            except Exception as exc:
                raise RuntimeError(f"Image decode failed: {row.image_path}: {exc}") from exc
            yield x, classes.index(row.mapped_class)

    image_shape = (int(cfg.get("image_height",224)), int(cfg.get("image_width",224)), 3)
    batch_size = int(cfg.get("batch_size",32))
    def make_ds(frame, shuffle):
        ds = tf.data.Dataset.from_generator(
            lambda: generator(frame),
            output_signature=(
                tf.TensorSpec(shape=image_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )
        if shuffle:
            ds = ds.shuffle(min(len(frame), 4096), seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    y_train = np.asarray([classes.index(c) for c in train_df["mapped_class"]], dtype=np.int64)
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    model = build_baseline_cnn(len(classes), cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg.get("learning_rate",0.001))),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss",
                                           save_best_only=True, mode="min"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=int(cfg.get("early_stopping_patience",5)),
                                         restore_best_weights=True, mode="min"),
        tf.keras.callbacks.CSVLogger(str(out / "training_history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    started = time.perf_counter()
    history = model.fit(make_ds(train_df, True), validation_data=make_ds(val_df, False),
                        epochs=int(cfg.get("epochs",30)), class_weight=class_weights,
                        callbacks=callbacks, verbose=1)
    duration = time.perf_counter() - started
    model.save(out / "final.keras")
    metadata = {
        "model_version":"baseline-cnn-v1", "architecture":"3 blocks (32,64,128), two 3x3 convs/block, BN/ReLU, max pool/dropout, GAP, Dense, softmax",
        "classes":classes, "input_shape":[image_shape[0],image_shape[1],3],
        "normalization":"float32 pixel values divided by 255.0",
        "augmentation":"model-side RandomFlip(horizontal), RandomRotation; active only during training",
        "seed":seed, "manifest_path":str(Path(args.manifest).resolve()),
        "manifest_sha256":sha256_file(args.manifest), "validation_split":args.validation_split,
        "train_rows":int(len(train_df)), "validation_rows":int(len(val_df)),
        "train_class_counts":{k:int(v) for k,v in counts.items()},
        "class_weights":class_weights, "epochs_completed":len(history.history.get("loss",[])),
        "training_duration_seconds":round(duration,3), "config":cfg,
        "validation_caveat":"train/cis_val reportedly share 224 seq_ids; confirm benchmark protocol before interpreting cis_val results."
    }
    (out/"model_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps({"status":"training_complete","output_dir":str(out),"epochs_completed":metadata["epochs_completed"],
                      "training_duration_seconds":metadata["training_duration_seconds"]},indent=2))

if __name__ == "__main__":
    main()
