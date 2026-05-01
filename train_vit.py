#!/usr/bin/env python3
"""
train_vit.py — Fine-tune a Vision Transformer for Copy-Move Forgery Detection.

Uses a lightweight ViT from the *timm* library (falls back to HuggingFace
transformers if *timm* is unavailable).  Trains a binary classifier on
preprocessed 224×224 images.
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


def _get_device():
    """Determine the best available device (CUDA / MPS / CPU)."""
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("  Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("  GPU not available — using CPU (training will be slower)")
    return device


def _build_vit_timm(num_classes: int = 2):
    """Build a ViT using the timm library."""
    import timm
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
    )
    return model


def _build_vit_hf(num_classes: int = 2):
    """Build a ViT using HuggingFace transformers (fallback)."""
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def _build_vit(num_classes: int = 2):
    """Try timm first, then fall back to HuggingFace."""
    try:
        model = _build_vit_timm(num_classes)
        logger.info("  Using timm ViT (vit_small_patch16_224)")
        return model, "timm"
    except Exception:
        logger.info("  timm not available — falling back to HuggingFace ViT")
        model = _build_vit_hf(num_classes)
        return model, "hf"


# ── Dataset & DataLoader ────────────────────────────────────────────────────

class _ImageDataset:
    """Simple PyTorch dataset wrapping NumPy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        # X: (N, 224, 224, 3) → (N, 3, 224, 224)
        self.X = torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training loop ───────────────────────────────────────────────────────────

def train_vit(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "dataset",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    save_dir: str = "outputs/models",
) -> tuple:
    """
    Fine-tune a Vision Transformer for binary forgery detection.

    Args:
        X: Images, shape (N, 224, 224, 3), float32 in [0,1].
        y: Labels, shape (N,), int  (0 = authentic, 1 = forged).
        dataset_name: Tag for saving.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        save_dir: Where to save the .pt model.

    Returns:
        (y_true, y_pred, y_pred_proba, train_losses)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = _get_device()

    # Reduce batch size on CPU
    if device.type == "cpu":
        batch_size = min(batch_size, 8)
        logger.info(f"  Reduced batch size to {batch_size} for CPU training")

    logger.info(f"{'='*60}")
    logger.info(f"  Training Vision Transformer — {dataset_name}")
    logger.info(f"  Samples: {len(X)}   Epochs: {epochs}   Batch: {batch_size}")
    logger.info(f"{'='*60}")

    # Normalise with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    X_norm = (X - mean) / std

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y,
    )

    train_ds = _ImageDataset(X_train, y_train)
    test_ds = _ImageDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"  Train: {len(train_ds)}  |  Test: {len(test_ds)}")

    # Build model
    model, backend = _build_vit(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── train ────────────────────────────────────────────────────────────────
    train_losses = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if backend == "hf":
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
            else:
                logits = model(imgs)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        train_losses.append(epoch_loss)
        logger.info(
            f"  Epoch {epoch:>2}/{epochs}  loss={epoch_loss:.4f}  "
            f"train_acc={epoch_acc:.2f}%"
        )

    elapsed = time.time() - start
    logger.info(f"  Training complete — {elapsed:.1f}s")

    # ── evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            if backend == "hf":
                logits = model(pixel_values=imgs).logits
            else:
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_pred_proba = np.array(all_probs)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"vit_{dataset_name}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"  Model saved → {model_path}")

    return y_true, y_pred, y_pred_proba, train_losses


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ViT classifier.")
    parser.add_argument("--images_npy", type=str, required=True)
    parser.add_argument("--labels_npy", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="outputs/models")
    args = parser.parse_args()

    X = np.load(args.images_npy)
    y = np.load(args.labels_npy)
    y_true, y_pred, _, _ = train_vit(
        X, y,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )
    from sklearn.metrics import accuracy_score
    logger.info(f"  Test Accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
