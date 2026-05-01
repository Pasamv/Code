#!/usr/bin/env python3
"""
train_adaboost.py — Train AdaBoost on CNN-extracted feature vectors.

Uses sklearn's AdaBoostClassifier with DecisionTree stumps on top of
VGG16-extracted features (25 088-dim vectors).
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


def train_adaboost(
    features: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "dataset",
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    save_dir: str = "outputs/models",
) -> tuple:
    """
    Train an AdaBoost classifier on pre-extracted CNN features.

    Args:
        features: Feature matrix of shape (N, D).
        y: Labels of shape (N,).
        dataset_name: Tag for saving.
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each tree.
        save_dir: Directory for the .pkl model.

    Returns:
        (y_true, y_pred, y_pred_proba)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    import joblib

    logger.info(f"{'='*60}")
    logger.info(f"  Training AdaBoost — {dataset_name}")
    logger.info(f"  Feature shape: {features.shape}")
    logger.info(f"  n_estimators={n_estimators}  lr={learning_rate}")
    logger.info(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm="SAMME",
        random_state=42,
    )

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(f"  Training complete — {elapsed:.1f}s")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"adaboost_{dataset_name}.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"  Model saved → {model_path}")

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_pred_proba


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AdaBoost classifier.")
    parser.add_argument("--features_npy", type=str, required=True,
                        help="Path to CNN feature .npy file.")
    parser.add_argument("--labels_npy", type=str, required=True,
                        help="Path to labels .npy file.")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    args = parser.parse_args()

    features = np.load(args.features_npy)
    y = np.load(args.labels_npy)
    y_true, y_pred, _ = train_adaboost(
        features, y,
        dataset_name=args.dataset_name,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
    )
    from sklearn.metrics import accuracy_score
    logger.info(f"  Test Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
