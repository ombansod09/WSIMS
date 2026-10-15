"""Custom baseline CNN. TensorFlow is imported lazily by training/inference entry points."""
def build_baseline_cnn(num_classes: int = 8, cfg: dict | None = None):
    import tensorflow as tf
    cfg = cfg or {}
    h, w = int(cfg.get("image_height", 224)), int(cfg.get("image_width", 224))
    dropout = float(cfg.get("dropout_rate", 0.30))
    dense_units = int(cfg.get("dense_units", 128))
    rotation = float(cfg.get("rotation_factor", 0.04))
    inputs = tf.keras.Input(shape=(h, w, 3), name="rgb_image")
    x = tf.keras.layers.RandomFlip("horizontal", name="train_random_flip")(inputs)
    x = tf.keras.layers.RandomRotation(rotation, fill_mode="reflect", name="train_random_rotation")(x)
    for block, filters in enumerate((32, 64, 128), start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                   name=f"block{block}_conv1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{block}_bn1")(x)
        x = tf.keras.layers.Activation("relu", name=f"block{block}_relu1")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                   name=f"block{block}_conv2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{block}_bn2")(x)
        x = tf.keras.layers.Activation("relu", name=f"block{block}_relu2")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"block{block}_pool")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"block{block}_dropout")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_scores")(x)
    return tf.keras.Model(inputs, outputs, name="wsims_baseline_cnn")
