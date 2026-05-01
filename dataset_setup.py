#!/usr/bin/env python3
"""
Dataset Setup Script for Copy-Move Forgery Detection Project.

Checks availability of all three datasets (CASIA v2.0, CoMoFoD, MICC-F2000),
provides download instructions, and validates folder contents.
"""

import os
import sys
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

DATASET_CONFIG = {
    "casia_v2": {
        "name": "CASIA v2.0",
        "path": os.path.join(DATASETS_DIR, "casia_v2"),
        "auth_dir": os.path.join(DATASETS_DIR, "casia_v2", "authentic"),
        "forged_dir": os.path.join(DATASETS_DIR, "casia_v2", "forged"),
        "download_urls": [
            "https://github.com/namtpham/casia2groundtruth",
            "https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset",
            "https://huggingface.co/datasets  (search for 'CASIA v2')",
        ],
        "instructions": (
            "1. Download CASIA v2.0 from one of the URLs above.\n"
            "2. Place AUTHENTIC images in:  datasets/casia_v2/authentic/\n"
            "3. Place TAMPERED / FORGED images in:  datasets/casia_v2/forged/\n"
            "   - Authentic images typically start with 'Au_' prefix.\n"
            "   - Tampered images typically start with 'Tp_' prefix.\n"
            "4. Re-run this script to verify."
        ),
    },
    "comofod": {
        "name": "CoMoFoD",
        "path": os.path.join(DATASETS_DIR, "comofod"),
        "auth_dir": os.path.join(DATASETS_DIR, "comofod", "authentic"),
        "forged_dir": os.path.join(DATASETS_DIR, "comofod", "forged"),
        "download_urls": [
            "https://www.vcl.fer.hr/comofod/download.html",
            "https://www.kaggle.com/datasets  (search for 'CoMoFoD')",
        ],
        "instructions": (
            "1. Visit https://www.vcl.fer.hr/comofod/download.html\n"
            "2. Download the 'Small' category (200 image sets, 512x512).\n"
            "3. Original images → datasets/comofod/authentic/\n"
            "4. Forged images   → datasets/comofod/forged/\n"
            "   - Original images have filenames like '001.png'\n"
            "   - Forged images have filenames like '001_F.png'\n"
            "5. Re-run this script to verify."
        ),
    },
    "micc_f2000": {
        "name": "MICC-F2000",
        "path": os.path.join(DATASETS_DIR, "micc_f2000"),
        "auth_dir": os.path.join(DATASETS_DIR, "micc_f2000", "authentic"),
        "forged_dir": os.path.join(DATASETS_DIR, "micc_f2000", "forged"),
        "download_urls": [
            "https://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/",
            "https://www.kaggle.com/datasets  (search for 'MICC-F2000')",
        ],
        "instructions": (
            "1. Visit the MICC lab page above and request the MICC-F2000 dataset.\n"
            "2. The dataset contains 2000 images (1300 original + 700 forged).\n"
            "3. Place original images in:  datasets/micc_f2000/authentic/\n"
            "4. Place forged images in:    datasets/micc_f2000/forged/\n"
            "5. Re-run this script to verify."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def count_images(directory: str) -> int:
    """Count the number of image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    count = 0
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def check_dataset(key: str, config: dict) -> dict:
    """
    Check whether a dataset folder has images.

    Returns a dict with status info.
    """
    auth_count = count_images(config["auth_dir"])
    forged_count = count_images(config["forged_dir"])
    total = auth_count + forged_count
    ready = total > 0 and auth_count > 0 and forged_count > 0
    return {
        "key": key,
        "name": config["name"],
        "authentic": auth_count,
        "forged": forged_count,
        "total": total,
        "ready": ready,
    }


def print_status(status: dict, config: dict) -> None:
    """Pretty-print the status of a single dataset."""
    tag = "\033[92m✔ READY\033[0m" if status["ready"] else "\033[91m✘ MISSING\033[0m"
    print(f"\n{'='*60}")
    print(f"  {status['name']}  —  {tag}")
    print(f"{'='*60}")
    print(f"  Authentic images : {status['authentic']}")
    print(f"  Forged images    : {status['forged']}")
    print(f"  Total images     : {status['total']}")

    if not status["ready"]:
        print(f"\n  📥 Download URLs:")
        for url in config["download_urls"]:
            print(f"     • {url}")
        print(f"\n  📋 Instructions:")
        for line in config["instructions"].split("\n"):
            print(f"     {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    """
    Check all datasets and print a summary.

    Returns:
        dict mapping dataset key → status dict (for programmatic use).
    """
    print("\n" + "=" * 60)
    print("   COPY-MOVE FORGERY DETECTION — DATASET STATUS CHECK")
    print("=" * 60)

    results = {}
    for key, config in DATASET_CONFIG.items():
        status = check_dataset(key, config)
        print_status(status, config)
        results[key] = status

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Dataset':<20} {'Status':<10} {'Auth':>6} {'Forged':>8} {'Total':>7}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*8} {'-'*7}")
    for key, st in results.items():
        tag = "READY" if st["ready"] else "MISSING"
        print(f"  {st['name']:<20} {tag:<10} {st['authentic']:>6} {st['forged']:>8} {st['total']:>7}")

    ready_count = sum(1 for s in results.values() if s["ready"])
    print(f"\n  {ready_count}/{len(results)} datasets ready.\n")

    if ready_count == 0:
        logger.warning(
            "No datasets found. Please download at least one dataset and "
            "place images in the appropriate folders."
        )

    return results


if __name__ == "__main__":
    main()
