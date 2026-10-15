"""MobileNetV2 transfer-learning model for the frozen WSIMS eight-class task."""
from __future__ import annotations

EXPECTED_CLASSES = ["opossum", "rabbit", "coyote", "cat", "squirrel", "raccoon", "dog", "bobcat"]


def load_classes(path="config/classes.json"):
    import json
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("status", "").upper() != "FROZEN":
        raise ValueError("config/classes.json must have status=FROZEN")
    names = cfg.get("class_names")
    mapping = cfg.get("class_to_index")
    if names != EXPECTED_CLASSES or mapping != {n: i for i, n in enumerate(EXPECTED_CLASSES)}:
        raise ValueError("Class configuration differs from the frozen WSIMS class order")
    return list(names)


def build_model(num_classes=8, weights="imagenet", dropout=0.25):
    import tensorflow as tf
    if num_classes != 8:
        raise ValueError("WSIMS frozen task requires exactly eight outputs")
    inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="rgb_0_255")
    # Equivalent to MobileNetV2 preprocess_input; kept in-model for identical train/inference.
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=weights
        )
        weight_source = "imagenet" if weights == "imagenet" else str(weights)
    except Exception as exc:
        if weights == "imagenet":
            raise RuntimeError(
                "Could not initialize ImageNet-pretrained MobileNetV2. Check internet/cache and retry; "
                "do not report this run as pretrained. Original error: " + repr(exc)
            ) from exc
        raise
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="wsims_mobilenetv2")
    model.backbone = backbone
    model.wsims_weight_source = weight_source
    return model


def unfreeze_top_layers(model, trainable_layers=30):
    """Unfreeze only the last N backbone layers, keeping BatchNorm frozen."""
    model.backbone.trainable = True
    cutoff = max(0, len(model.backbone.layers) - int(trainable_layers))
    for i, layer in enumerate(model.backbone.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, __import__("tensorflow").keras.layers.BatchNormalization)
    return model
