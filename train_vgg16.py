#!/usr/bin/env python3
"""
train_vgg16.py — Fine-tune VGG16 for Copy-Move Forgery Detection.

Freezes the first 15 layers of VGG16, adds a custom classification head,
and trains a binary classifier (authentic vs forged).
"""

import os
import sys
import time
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)


def build_vgg16_classifier(freeze_layers: int = 15) -> "tf.keras.Model":
    """
    Build a fine-tunable VGG16 binary classifier.

    Args:
        freeze_layers: Number of base layers to freeze (default 15).

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    tf.random.set_seed(42)

    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Freeze first `freeze_layers` layers
    for layer in base.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base.layers[freeze_layers:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_vgg16(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "dataset",
    epochs: int = 20,
    batch_size: int = 32,
    save_dir: str = "outputs/models",
) -> tuple:
    """
    Train VGG16 on a preprocessed dataset and return predictions.

    Args:
        X: Images, shape (N, 224, 224, 3), float32.
        y: Labels, shape (N,), int.
        dataset_name: Tag used for saving the model.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        save_dir: Where to save the .h5 model.

    Returns:
        (y_true, y_pred, y_pred_proba, history) — test labels, predicted
        labels, prediction probabilities, and Keras History object.
    """
    import tensorflow as tf
    tf.random.set_seed(42)
    from sklearn.model_selection import train_test_split

    logger.info(f"{'='*60}")
    logger.info(f"  Training VGG16 — {dataset_name}")
    logger.info(f"  Samples: {len(X)}   Epochs: {epochs}   Batch: {batch_size}")
    logger.info(f"{'='*60}")

    # VGG16 preprocessing (expects BGR mean-centred)
    X_prep = tf.keras.applications.vgg16.preprocess_input(X * 255.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    model = build_vgg16_classifier()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    elapsed = time.time() - start
    logger.info(f"  Training complete — {elapsed:.1f}s")

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"vgg16_{dataset_name}.h5")
    model.save(model_path)
    logger.info(f"  Model saved → {model_path}")

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return y_test, y_pred, y_pred_proba, history


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VGG16 classifier.")
    parser.add_argument("--images_npy", type=str, required=True)
    parser.add_argument("--labels_npy", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    args = parser.parse_args()

    X = np.load(args.images_npy)
    y = np.load(args.labels_npy)
    y_true, y_pred, _, _ = train_vgg16(
        X, y,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )
    from sklearn.metrics import accuracy_score
    logger.info(f"  Test Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
