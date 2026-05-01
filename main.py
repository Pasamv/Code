#!/usr/bin/env python3
"""
main.py — Master Pipeline for Copy-Move Forgery Detection Project.

Orchestrates the full workflow:
  1. Dataset status check
  2. Preprocessing
  3. CNN feature extraction
  4. Model training (VGG16, ViT, AdaBoost)
  5. Evaluation
  6. Visualisation
  7. Final report generation

Usage examples:
  python main.py                              # Run everything
  python main.py --dataset casia              # Only CASIA v2 dataset
  python main.py --model vgg16                # Only train VGG16
  python main.py --skip_extraction            # Reuse saved CNN features
  python main.py --no_augment                 # Skip data augmentation
"""

import os
import sys
import time
import argparse
import logging
import datetime
import numpy as np

# ── project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Reproducibility
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    pass
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass


# ── paths ────────────────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
CM_DIR = os.path.join(PROJECT_ROOT, "outputs", "confusion_matrices")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "all_results.csv")

DATASET_MAP = {
    "casia": ("casia_v2", "CASIA_v2"),
    "comofod": ("comofod", "CoMoFoD"),
    "micc": ("micc_f2000", "MICC_F2000"),
}

MODEL_LIST = ["vgg16", "vit", "adaboost"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CMFD Pipeline — Copy-Move Forgery Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "casia", "comofod", "micc"],
        help="Which dataset to process (default: all).",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "vgg16", "vit", "adaboost"],
        help="Which model to train (default: all).",
    )
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Skip CNN feature extraction if .npy files already exist.",
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable data augmentation during preprocessing.",
    )
    parser.add_argument(
        "--vgg16_epochs", type=int, default=20,
        help="Epochs for VGG16 training (default: 20).",
    )
    parser.add_argument(
        "--vit_epochs", type=int, default=10,
        help="Epochs for ViT training (default: 10).",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=500,
        help="Max images per class (authentic/forged) per dataset (default: 500).\n"
             "Set to 0 to use ALL available images.",
    )
    return parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def _dataset_ready(folder: str) -> bool:
    """Return True if folder contains at least one image in authentic/ and forged/."""
    from dataset_setup import count_images
    auth = count_images(os.path.join(folder, "authentic"))
    forged = count_images(os.path.join(folder, "forged"))
    return auth > 0 and forged > 0


def _step_banner(step: int, title: str):
    print(f"\n{'#'*64}")
    print(f"#  STEP {step}: {title}")
    print(f"{'#'*64}\n")


# ── pipeline steps ───────────────────────────────────────────────────────────

def step_check_datasets(dataset_keys: list) -> list:
    """Check which requested datasets are available. Returns list of ready keys."""
    _step_banner(1, "DATASET STATUS CHECK")
    from dataset_setup import main as check_datasets
    statuses = check_datasets()

    ready = []
    for key in dataset_keys:
        folder_name, display = DATASET_MAP[key]
        if statuses.get(folder_name, {}).get("ready", False):
            ready.append(key)
        else:
            logger.warning(f"  Dataset '{display}' is NOT ready — will be skipped.")
    return ready


def step_preprocess(key: str, augment: bool, max_per_class: int | None = None) -> tuple:
    """Preprocess a single dataset. Returns (X, y)."""
    folder_name, display = DATASET_MAP[key]
    dataset_path = os.path.join(DATASETS_DIR, folder_name)

    _step_banner(2, f"PREPROCESSING — {display}")
    from src.preprocess import load_and_preprocess

    X, y = load_and_preprocess(
        dataset_path=dataset_path,
        augment=False,
        save_dir=FEATURES_DIR,
        dataset_name=folder_name,
        max_per_class=max_per_class,
    )
    return X, y


def step_extract_features(X: np.ndarray, key: str, skip: bool) -> np.ndarray:
    """Extract CNN features (or load from disk if skip=True)."""
    folder_name, display = DATASET_MAP[key]
    feat_file = os.path.join(FEATURES_DIR, f"{folder_name}_features.npy")

    if skip and os.path.exists(feat_file):
        _step_banner(3, f"FEATURE EXTRACTION — {display} (cached)")
        features = np.load(feat_file)
        logger.info(f"  Loaded cached features: {features.shape}")
        return features

    _step_banner(3, f"FEATURE EXTRACTION — {display}")
    from src.feature_extraction import extract_features

    features = extract_features(X, save_path=FEATURES_DIR, dataset_name=folder_name)
    return features


def step_train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    key: str,
    models_to_run: list,
    vgg16_epochs: int,
    vit_epochs: int,
):
    """Train selected models and evaluate each."""
    folder_name, display = DATASET_MAP[key]
    from src.evaluate import evaluate_and_record

    # ── VGG16 ────────────────────────────────────────────────────────────
    if "vgg16" in models_to_run:
        _step_banner(4, f"TRAINING VGG16 — {display}")
        try:
            from src.train_vgg16 import train_vgg16
            y_true, y_pred, _, _ = train_vgg16(
                X, y,
                dataset_name=folder_name,
                epochs=vgg16_epochs,
                save_dir=MODELS_DIR,
            )
            evaluate_and_record(
                y_true, y_pred,
                model_name="VGG16",
                dataset_name=display,
                results_csv=RESULTS_CSV,
                cm_dir=CM_DIR,
            )
        except Exception as e:
            logger.error(f"  VGG16 training failed: {e}")

    # ── ViT ──────────────────────────────────────────────────────────────
    if "vit" in models_to_run:
        _step_banner(5, f"TRAINING ViT — {display}")
        try:
            from src.train_vit import train_vit
            y_true, y_pred, _, _ = train_vit(
                X, y,
                dataset_name=folder_name,
                epochs=vit_epochs,
                save_dir=MODELS_DIR,
            )
            evaluate_and_record(
                y_true, y_pred,
                model_name="ViT",
                dataset_name=display,
                results_csv=RESULTS_CSV,
                cm_dir=CM_DIR,
            )
        except Exception as e:
            logger.error(f"  ViT training failed: {e}")

    # ── AdaBoost ─────────────────────────────────────────────────────────
    if "adaboost" in models_to_run:
        _step_banner(6, f"TRAINING AdaBoost — {display}")
        try:
            from src.train_adaboost import train_adaboost
            y_true, y_pred, _ = train_adaboost(
                features, y,
                dataset_name=folder_name,
                save_dir=MODELS_DIR,
            )
            evaluate_and_record(
                y_true, y_pred,
                model_name="AdaBoost",
                dataset_name=display,
                results_csv=RESULTS_CSV,
                cm_dir=CM_DIR,
            )
        except Exception as e:
            logger.error(f"  AdaBoost training failed: {e}")


def step_visualise():
    """Generate all comparison charts."""
    _step_banner(7, "GENERATING VISUALISATIONS")
    from src.visualize_results import generate_all_charts
    generate_all_charts(results_csv=RESULTS_CSV, save_dir=RESULTS_DIR)


def step_final_report():
    """Print the final summary table and save a text report."""
    _step_banner(8, "FINAL REPORT")
    from src.evaluate import print_results_table
    print_results_table(RESULTS_CSV)

    # Write report file
    import pandas as pd
    if os.path.exists(RESULTS_CSV):
        df = pd.read_csv(RESULTS_CSV)
        report_path = os.path.join(RESULTS_DIR, "final_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("  COPY-MOVE FORGERY DETECTION — FINAL REPORT\n")
            f.write(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            f.write("REFERENCE PAPER:\n")
            f.write("  Kashish Agarwal, \"Combining Deep Learning and Traditional Methods\n")
            f.write("  for Effective Copy-Move Forgery Detection in Digital Images\",\n")
            f.write("  IJMRA, Vol. 8, Issue 06, June 2025.\n")
            f.write("  URL: https://ijmra.in/v8i6/Doc/32.pdf\n\n")

            f.write("MODELS EVALUATED:\n")
            f.write("  1. VGG16 (fine-tuned, transfer learning from ImageNet)\n")
            f.write("  2. Vision Transformer (ViT) (fine-tuned from pretrained)\n")
            f.write("  3. AdaBoost (100 stumps on CNN-extracted features)\n\n")

            f.write("DATASETS:\n")
            for ds in df["Dataset"].unique():
                f.write(f"  • {ds}\n")
            f.write("\n")

            f.write("RESULTS:\n")
            f.write(f"{'Model':<12} {'Dataset':<16} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}\n")
            f.write("-" * 60 + "\n")
            for _, r in df.iterrows():
                f.write(
                    f"{r['Model']:<12} {r['Dataset']:<16} "
                    f"{r['Accuracy']:>6.2f}% {r['Precision']:>6.2f}% "
                    f"{r['Recall']:>6.2f}% {r['F1_Score']:>6.2f}%\n"
                )

            avg_f1 = df.groupby("Model")["F1_Score"].mean()
            best = avg_f1.idxmax()
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"BEST MODEL OVERALL (by avg F1): {best} ({avg_f1[best]:.2f}%)\n")
            f.write("=" * 60 + "\n\n")

            f.write("CONCLUSION:\n")
            f.write(f"  Based on the average F1-Score across all evaluated datasets,\n")
            f.write(f"  {best} achieves the highest overall performance for\n")
            f.write(f"  Copy-Move Forgery Detection.\n")

        logger.info(f"  Report saved → {report_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 64)
    print("  COPY-MOVE FORGERY DETECTION PIPELINE")
    print("  Based on: Agarwal (2025), IJMRA Vol.8 Issue 06")
    print("=" * 64)

    overall_start = time.time()

    # Determine which datasets / models to run
    if args.dataset == "all":
        dataset_keys = list(DATASET_MAP.keys())
    else:
        dataset_keys = [args.dataset]

    if args.model == "all":
        models_to_run = MODEL_LIST
    else:
        models_to_run = [args.model]

    # Step 1 — check datasets
    ready_keys = step_check_datasets(dataset_keys)
    if not ready_keys:
        logger.error("No datasets available. Exiting.")
        sys.exit(1)

    # Steps 2-6 — per dataset
    for key in ready_keys:
        folder_name, display = DATASET_MAP[key]

        # Preprocessing
        images_npy = os.path.join(FEATURES_DIR, f"{folder_name}_images.npy")
        labels_npy = os.path.join(FEATURES_DIR, f"{folder_name}_labels.npy")

        if os.path.exists(images_npy) and os.path.exists(labels_npy) and args.skip_extraction:
            logger.info(f"  Loading cached preprocessed data for {display}")
            X = np.load(images_npy)
            y = np.load(labels_npy)
        else:
            mpc = args.max_per_class if args.max_per_class > 0 else None
            X, y = step_preprocess(key, augment=not args.no_augment, max_per_class=mpc)

        # Feature extraction (needed for AdaBoost)
        features = step_extract_features(X, key, skip=args.skip_extraction)

        # Train & evaluate
        step_train_and_evaluate(
            X, y, features, key,
            models_to_run=models_to_run,
            vgg16_epochs=args.vgg16_epochs,
            vit_epochs=args.vit_epochs,
        )

    # Step 7 — visualise
    step_visualise()

    # Step 8 — report
    step_final_report()

    total_time = time.time() - overall_start
    print(f"\n  ⏱  Total pipeline time: {total_time/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
