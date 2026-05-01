#!/usr/bin/env python3
"""
evaluate.py — Evaluation Module for CMFD Project.

Computes Accuracy, Precision, Recall, F1-Score; generates confusion matrix
heatmaps; and aggregates all results into a single CSV file.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).

    Returns:
        dict with keys: accuracy, precision, recall, f1_score.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    save_dir: str = "outputs/confusion_matrices",
) -> str:
    """
    Generate and save a confusion matrix heatmap.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Authentic", "Forged"],
        yticklabels=["Authentic", "Forged"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name} on {dataset_name}")
    plt.tight_layout()

    fname = f"{model_name}_{dataset_name}_confusion_matrix.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Confusion matrix saved → {path}")
    return path


def evaluate_and_record(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    dataset_name: str,
    results_csv: str = "outputs/results/all_results.csv",
    cm_dir: str = "outputs/confusion_matrices",
) -> dict:
    """
    Full evaluation pipeline: metrics + confusion matrix + CSV append.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: e.g. "VGG16", "ViT", "AdaBoost".
        dataset_name: e.g. "casia_v2", "comofod", "micc_f2000".
        results_csv: Path to the aggregated CSV.
        cm_dir: Directory for confusion matrix images.

    Returns:
        dict of computed metrics.
    """
    metrics = compute_metrics(y_true, y_pred)

    logger.info(f"\n  {'='*50}")
    logger.info(f"  {model_name}  ×  {dataset_name}")
    logger.info(f"  {'='*50}")
    logger.info(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    logger.info(f"  Precision : {metrics['precision']*100:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']*100:.2f}%")
    logger.info(f"  F1-Score  : {metrics['f1_score']*100:.2f}%")

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, model_name, dataset_name, save_dir=cm_dir)

    # Append to CSV
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    row = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": round(metrics["accuracy"] * 100, 2),
        "Precision": round(metrics["precision"] * 100, 2),
        "Recall": round(metrics["recall"] * 100, 2),
        "F1_Score": round(metrics["f1_score"] * 100, 2),
    }
    df_row = pd.DataFrame([row])

    if os.path.exists(results_csv):
        df_existing = pd.read_csv(results_csv)
        # Remove previous entry for same model+dataset if exists
        mask = ~((df_existing["Model"] == model_name) & (df_existing["Dataset"] == dataset_name))
        df_existing = df_existing[mask]
        df = pd.concat([df_existing, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(results_csv, index=False)
    logger.info(f"  Results appended → {results_csv}")

    return metrics


def print_results_table(results_csv: str = "outputs/results/all_results.csv") -> None:
    """Print a nicely formatted results table to the console."""
    if not os.path.exists(results_csv):
        logger.warning(f"No results file found at {results_csv}")
        return

    df = pd.read_csv(results_csv)

    print("\n" + "=" * 78)
    print("       COPY-MOVE FORGERY DETECTION — RESULTS SUMMARY")
    print("=" * 78)
    print(f"{'Model':<12} | {'Dataset':<14} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print(f"{'-'*12}-+-{'-'*14}-+-{'-'*8}-+-{'-'*9}-+-{'-'*6}-+-{'-'*6}")

    for _, r in df.iterrows():
        print(
            f"{r['Model']:<12} | {r['Dataset']:<14} | "
            f"{r['Accuracy']:>7.2f}% | {r['Precision']:>8.2f}% | "
            f"{r['Recall']:>5.2f}% | {r['F1_Score']:>5.2f}%"
        )

    print("=" * 78)

    # Best model by average F1 across all datasets
    avg_f1 = df.groupby("Model")["F1_Score"].mean()
    best_model = avg_f1.idxmax()
    print(f"  BEST MODEL OVERALL (avg F1): {best_model}  ({avg_f1[best_model]:.2f}%)")
    print("=" * 78 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--y_true_npy", type=str, help="Path to true labels .npy")
    parser.add_argument("--y_pred_npy", type=str, help="Path to pred labels .npy")
    parser.add_argument("--model_name", type=str, default="Model")
    parser.add_argument("--dataset_name", type=str, default="Dataset")
    parser.add_argument("--results_csv", type=str, default="outputs/results/all_results.csv")
    parser.add_argument("--print_table", action="store_true",
                        help="Just print the existing results table.")
    args = parser.parse_args()

    if args.print_table:
        print_results_table(args.results_csv)
    else:
        y_true = np.load(args.y_true_npy)
        y_pred = np.load(args.y_pred_npy)
        evaluate_and_record(
            y_true, y_pred,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            results_csv=args.results_csv,
        )
