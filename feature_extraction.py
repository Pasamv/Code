#!/usr/bin/env python3
"""
feature_extraction.py — CNN Feature Extraction using VGG16.

Uses a pre-trained VGG16 (ImageNet) truncated at the last pooling layer
to produce high-dimensional feature vectors for each image.
"""

import os
import sys
import time
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)

BATCH_SIZE = 32  # process images in batches for memory efficiency


def _build_feature_extractor():
    """
    Build a VGG16-based feature extractor.

    The model is loaded with ImageNet weights and truncated at the last
    MaxPooling layer (block5_pool), producing a 7×7×512 = 25 088-dim vector
    after flattening.

    Returns:
        tf.keras.Model
    """
    import tensorflow as tf
    tf.random.set_seed(42)

    base = tf.keras.applications.VGG16(
        weights="imagenet",
        include_top=False,       # remove classification head
        input_shape=(224, 224, 3),
    )
    # Output of last pooling layer → (batch, 7, 7, 512)
    model = tf.keras.Model(inputs=base.input, outputs=base.output)
    model.trainable = False
    logger.info(f"  Feature extractor output shape: {model.output_shape}")
    return model


def extract_features(
    X: np.ndarray,
    save_path: str | None = None,
    dataset_name: str = "dataset",
) -> np.ndarray:
    """
    Extract CNN features from a batch of preprocessed images.

    Args:
        X: Image array of shape (N, 224, 224, 3), float32 in [0, 1].
        save_path: Directory where .npy files will be saved.
        dataset_name: Tag for file naming.

    Returns:
        features: np.ndarray of shape (N, 25088).
    """
    import tensorflow as tf

    logger.info(f"{'='*60}")
    logger.info(f"  Extracting CNN features — {dataset_name}")
    logger.info(f"  Input shape: {X.shape}   Batch size: {BATCH_SIZE}")
    logger.info(f"{'='*60}")
    start = time.time()

    # VGG16 expects pixel values preprocessed with its own scheme
    X_prep = tf.keras.applications.vgg16.preprocess_input(X * 255.0)

    model = _build_feature_extractor()

    # Extract in batches
    n = X_prep.shape[0]
    all_features = []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="  Extracting features", unit="batch"):
        batch = X_prep[i : i + BATCH_SIZE]
        feats = model.predict(batch, verbose=0)
        # Flatten from (batch, 7, 7, 512) → (batch, 25088)
        feats = feats.reshape(feats.shape[0], -1)
        all_features.append(feats)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    elapsed = time.time() - start
    logger.info(f"  Feature matrix shape : {features.shape}")
    logger.info(f"  Time elapsed         : {elapsed:.1f}s")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        feat_file = os.path.join(save_path, f"{dataset_name}_features.npy")
        np.save(feat_file, features)
        logger.info(f"  Saved features → {feat_file}")

    return features


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract VGG16 CNN features.")
    parser.add_argument("--images_npy", type=str, required=True,
                        help="Path to preprocessed images .npy file.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save extracted features.")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    args = parser.parse_args()

    X = np.load(args.images_npy)
    logger.info(f"Loaded images from {args.images_npy}, shape: {X.shape}")
    extract_features(X, save_path=args.save_dir, dataset_name=args.dataset_name)
