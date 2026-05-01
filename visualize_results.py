#!/usr/bin/env python3
"""
visualize_results.py — Visualisation module for CMFD Project.

Reads the aggregated results CSV and produces:
  1. Grouped bar charts for Accuracy, Precision, Recall, F1-Score
  2. A heatmap summary table
All charts are saved as PNG files in outputs/results/.
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

# Colour scheme
MODEL_COLORS = {
    "VGG16": "#3b82f6",     # blue
    "ViT": "#f97316",       # orange
    "AdaBoost": "#22c55e",  # green
}


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


def plot_grouped_bar(
    df: pd.DataFrame,
    metric: str,
    save_dir: str,
) -> str:
    """
    Create a grouped bar chart comparing models across datasets for one metric.

    Returns:
        Path to the saved chart image.
    """
    plt, sns = _ensure_matplotlib()

    models = df["Model"].unique()
    datasets = df["Dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = df[(df["Model"] == model) & (df["Dataset"] == ds)]
            vals.append(row[metric].values[0] if len(row) else 0)
        color = MODEL_COLORS.get(model, "#888888")
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, edgecolor="white")
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(f"{metric} (%)", fontsize=12)
    ax.set_title(f"{metric} Comparison across Models and Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{metric.lower()}_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Chart saved → {path}")
    return path


def plot_heatmap_table(df: pd.DataFrame, save_dir: str) -> str:
    """Create a heatmap grid of all metrics."""
    plt, sns = _ensure_matplotlib()

    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    df["Label"] = df["Model"] + " / " + df["Dataset"]

    pivot = df.set_index("Label")[metrics]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6 + 1)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlGnBu",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Score (%)"},
    )
    ax.set_title("Model × Dataset Performance Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "heatmap_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Heatmap saved → {path}")
    return path


def generate_all_charts(
    results_csv: str = "outputs/results/all_results.csv",
    save_dir: str = "outputs/results",
) -> None:
    """
    Read the results CSV and generate all visualisation charts.
    """
    if not os.path.exists(results_csv):
        logger.error(f"Results CSV not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"  Generating visualisations from {results_csv}")
    logger.info(f"{'='*60}")

    for metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
        plot_grouped_bar(df, metric, save_dir)

    plot_heatmap_table(df, save_dir)

    logger.info("  All charts generated successfully.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate result visualisations.")
    parser.add_argument("--results_csv", type=str, default="outputs/results/all_results.csv")
    parser.add_argument("--save_dir", type=str, default="outputs/results")
    args = parser.parse_args()

    generate_all_charts(results_csv=args.results_csv, save_dir=args.save_dir)
