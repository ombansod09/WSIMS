from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Train WSIMS MobileNetV2 on a reviewed derived manifest")
    p.add_argument("--manifest", required=True, help="Derived eligible manifest with image_path,mapped_class,split,file_status")
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--output-dir", default="artifacts/models")
    p.add_argument("--history-dir", default="artifacts/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=8)
    p.add_argument("--fine-tune-layers", type=int, default=30)
    p.add_argument("--weights", default="imagenet", help="Use imagenet; use none only if explicitly approved and document non-pretrained run")
    args = p.parse_args()
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from src.models.mobilenet_model import build_model, load_classes, unfreeze_top_layers
    classes = load_classes(args.classes)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    df = pd.read_csv(args.manifest)
    required = {"image_path", "mapped_class", "split", "file_status"}
    if required - set(df.columns): raise ValueError(f"Manifest missing columns: {sorted(required-set(df.columns))}")
    if df.image_path.duplicated().any(): raise ValueError("Duplicate image paths in derived manifest")
    if not set(df.mapped_class).issubset(set(classes)): raise ValueError("Unsupported labels in manifest; refusing to remap")
    official = {"train", "cis_val", "cis_test", "trans_val", "trans_test"}
    if not set(df.split).issubset(official): raise ValueError("Nonofficial split found")
    if not set(df.file_status.astype(str).str.lower()).issubset({"ok", "valid", "readable", "scaled_resolution"}): raise ValueError("Unapproved file_status in manifest")
    for split in ["train", "cis_val"]:
        if split not in set(df.split): raise ValueError(f"Required split missing: {split}")
    for path in df.image_path:
        if not Path(str(path)).is_file(): raise FileNotFoundError(f"Manifest image path not found: {path}")
    train = df[df.split == "train"].copy(); val = df[df.split == "cis_val"].copy()
    if train.empty or val.empty: raise ValueError("Training and cis_val must both contain eligible records")
    class_indices = {name: i for i, name in enumerate(classes)}
    y_train = train.mapped_class.map(class_indices).to_numpy(dtype=np.int32)
    y_val = val.mapped_class.map(class_indices).to_numpy(dtype=np.int32)
    def dataset(frame, shuffle=False):
        paths = frame.image_path.astype(str).to_numpy()
        labels = frame.mapped_class.map(class_indices).to_numpy(dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle: ds = ds.shuffle(min(len(frame), 10000), seed=args.seed, reshuffle_each_iteration=True)
        def load(path, label):
            image = tf.io.read_file(path)
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(tf.cast(image, tf.float32), [224, 224], method="bilinear", antialias=True)
            return image, tf.one_hot(label, depth=8)
        return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds, val_ds = dataset(train, True), dataset(val)
    present = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(present, weights)}
    model = build_model(weights=None if args.weights.lower() == "none" else args.weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    out = Path(args.output_dir); hist_dir = Path(args.history_dir); out.mkdir(parents=True, exist_ok=True); hist_dir.mkdir(parents=True, exist_ok=True)
    best_path = out / "mobilenetv2_best.keras"
    callbacks = [tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
                 tf.keras.callbacks.CSVLogger(hist_dir / "mobilenetv2_history.csv")]
    started = time.perf_counter()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks, class_weight=class_weights, verbose=2)
    phase = "head_only"
    if args.fine_tune_epochs > 0:
        model = tf.keras.models.load_model(best_path)
        # Reattach backbone by stable layer name from the functional model.
        model.backbone = model.get_layer("mobilenetv2_1.00_224")
        unfreeze_top_layers(model, args.fine_tune_layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
        callbacks2 = [tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
                      tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
                      tf.keras.callbacks.CSVLogger(hist_dir / "mobilenetv2_finetune_history.csv")]
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks2, class_weight=class_weights, verbose=2)
        phase = "head_plus_finetune"
    duration = time.perf_counter() - started
    meta = {"model": "MobileNetV2", "weights_argument": args.weights, "pretrained_imagenet": args.weights.lower() != "none",
            "classes": classes, "input_size": [224, 224], "input_contract": "RGB float32 [0,255]; in-model Rescaling to [-1,1] equivalent to mobilenet_v2.preprocess_input",
            "train_split": "train", "selection_split": "cis_val", "official_splits_unchanged": True,
            "train_records": int(len(train)), "validation_records": int(len(val)), "seed": args.seed,
            "batch_size": args.batch_size, "head_epochs_requested": args.head_epochs, "fine_tune_epochs_requested": args.fine_tune_epochs,
            "fine_tune_layers_requested": args.fine_tune_layers, "class_weights": {classes[k]: v for k, v in class_weights.items()},
            "training_wall_seconds": round(duration, 3), "training_phase": phase,
            "warning": "cis_val shares 224 sequence IDs with train per supplied EDA; validate protocol before claiming independent validation."}
    (out / "mobilenetv2_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Best checkpoint: {best_path}")
if __name__ == "__main__": main()
