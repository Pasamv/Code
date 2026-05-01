#!/usr/bin/env python3
"""
preprocess.py — Image Preprocessing Module for CMFD Project.

Loads images from a dataset folder, resizes, normalises, applies data
augmentation, and returns NumPy arrays ready for model consumption.
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
IMG_SIZE = (224, 224)


# ── helpers ──────────────────────────────────────────────────────────────────

def _import_cv2():
    """Lazy import of cv2 to provide a clear error message."""
    try:
        import cv2
        return cv2
    except ImportError:
        logger.error("opencv-python is not installed.  Run:  pip install opencv-python")
        sys.exit(1)


def _load_images_from_dir(
    directory: str,
    label: int,
    img_size: tuple = IMG_SIZE,
    max_images: int | None = None,
):
    """
    Load images from *directory*, resize to *img_size*, assign *label*.

    Args:
        directory: Folder containing image files.
        label: Integer label to assign (0 = authentic, 1 = forged).
        img_size: Target (width, height).
        max_images: If set, randomly sample at most this many images.

    Returns:
        images (list[np.ndarray]): list of (H, W, 3) float32 arrays in [0, 1].
        labels (list[int]): list of integer labels.
    """
    cv2 = _import_cv2()
    images, labels = [], []
    files = sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    # Randomly sample if max_images is specified
    if max_images is not None and len(files) > max_images:
        logger.info(f"  Sampling {max_images} of {len(files)} images from {os.path.basename(directory)}/")
        indices = np.random.choice(len(files), size=max_images, replace=False)
        files = [files[i] for i in sorted(indices)]

    for fname in tqdm(files, desc=f"  Loading {'authentic' if label == 0 else 'forged':>10}", unit="img"):
        path = os.path.join(directory, fname)
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"Could not read image: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        images.append(img)
        labels.append(label)
    return images, labels


# ── augmentation ─────────────────────────────────────────────────────────────

def _augment_image(img: np.ndarray) -> list:
    """
    Apply a set of augmentations and return a list of augmented images.

    Augmentations applied:
        1. Horizontal flip
        2. Vertical flip
        3. Random rotation (up to ±20°)
        4. Random zoom (up to 20 %)
        5. Brightness adjustment (±30 %)
    """
    cv2 = _import_cv2()
    augmented = []
    h, w = img.shape[:2]

    # 1. Horizontal flip
    augmented.append(cv2.flip(img, 1))

    # 2. Vertical flip
    augmented.append(cv2.flip(img, 0))

    # 3. Random rotation
    angle = np.random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    augmented.append(rotated)

    # 4. Random zoom (crop + resize)
    zoom_factor = np.random.uniform(0.8, 1.0)
    zh, zw = int(h * zoom_factor), int(w * zoom_factor)
    top = np.random.randint(0, h - zh + 1)
    left = np.random.randint(0, w - zw + 1)
    cropped = img[top:top + zh, left:left + zw]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    augmented.append(zoomed)

    # 5. Brightness adjustment
    factor = np.random.uniform(0.7, 1.3)
    bright = np.clip(img * factor, 0, 1).astype(np.float32)
    augmented.append(bright)

    return augmented


# ── public API ───────────────────────────────────────────────────────────────

def load_and_preprocess(
    dataset_path: str,
    augment: bool = True,
    save_dir: str | None = None,
    dataset_name: str = "dataset",
    max_per_class: int | None = None,
) -> tuple:
    """
    Load, preprocess and optionally augment a binary-class image dataset.

    Expected folder layout::

        dataset_path/
        ├── authentic/   (label = 0)
        └── forged/      (label = 1)

    Args:
        dataset_path: Root folder containing ``authentic/`` and ``forged/`` sub-dirs.
        augment: Whether to apply data augmentation.
        save_dir: If set, save X.npy and y.npy there.
        dataset_name: Human-readable name used in log messages and file names.
        max_per_class: If set, randomly sample at most this many images per class.

    Returns:
        X (np.ndarray): shape (N, 224, 224, 3), dtype float32.
        y (np.ndarray): shape (N,), dtype int32.
    """
    logger.info(f"{'='*60}")
    logger.info(f"  Preprocessing dataset: {dataset_name}")
    logger.info(f"  Path: {dataset_path}")
    logger.info(f"  Augmentation: {'ON' if augment else 'OFF'}")
    logger.info(f"  Max per class: {max_per_class or 'ALL'}")
    logger.info(f"{'='*60}")
    start = time.time()

    auth_dir = os.path.join(dataset_path, "authentic")
    forged_dir = os.path.join(dataset_path, "forged")

    if not os.path.isdir(auth_dir) or not os.path.isdir(forged_dir):
        raise FileNotFoundError(
            f"Expected sub-folders 'authentic/' and 'forged/' inside {dataset_path}"
        )

    images_auth, labels_auth = _load_images_from_dir(auth_dir, label=0, max_images=max_per_class)
    images_forg, labels_forg = _load_images_from_dir(forged_dir, label=1, max_images=max_per_class)

    all_images = images_auth + images_forg
    all_labels = labels_auth + labels_forg

    logger.info(f"  Original — authentic: {len(images_auth)}, forged: {len(images_forg)}")

    if augment:
        aug_images, aug_labels = [], []
        for img, lbl in tqdm(
            zip(all_images, all_labels), total=len(all_images),
            desc="  Augmenting", unit="img",
        ):
            aug_imgs = _augment_image(img)
            aug_images.extend(aug_imgs)
            aug_labels.extend([lbl] * len(aug_imgs))
        all_images.extend(aug_images)
        all_labels.extend(aug_labels)
        logger.info(f"  After augmentation — total images: {len(all_images)}")

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Stats
    elapsed = time.time() - start
    logger.info(f"  Image tensor shape : {X.shape}")
    logger.info(f"  Label array shape  : {y.shape}")
    logger.info(f"  Class distribution : 0 (authentic) = {int((y == 0).sum())}, "
                f"1 (forged) = {int((y == 1).sum())}")
    logger.info(f"  Time elapsed       : {elapsed:.1f}s")

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{dataset_name}_images.npy"), X)
        np.save(os.path.join(save_dir, f"{dataset_name}_labels.npy"), y)
        logger.info(f"  Saved to {save_dir}/{dataset_name}_images.npy / _labels.npy")

    return X, y


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a CMFD dataset.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset folder (must contain authentic/ and forged/).")
    parser.add_argument("--dataset_name", type=str, default="dataset",
                        help="Name tag for the dataset.")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save .npy arrays.")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Max images per class (authentic/forged). Default: load all.")
    args = parser.parse_args()

    load_and_preprocess(
        dataset_path=args.dataset_path,
        augment=not args.no_augment,
        save_dir=args.save_dir,
        dataset_name=args.dataset_name,
        max_per_class=args.max_per_class,
    )
