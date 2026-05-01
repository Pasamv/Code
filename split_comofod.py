#!/usr/bin/env python3
"""
split_comofod.py — Split the CoMoFoD dataset into authentic/ and forged/ folders.

CoMoFoD naming convention:
    N1_M1          → base images
    N1_M1_M2[N]    → postprocessed images

    M1 marks:
        "O"  → Original image         → goes to  authentic/
        "F"  → Forged image            → goes to  forged/
        "B"  → Binary mask             → SKIPPED (not an image for classification)
        "M"  → Coloured mask           → SKIPPED

    M2 marks (postprocessing applied to both O and F):
        "JC" → JPEG compression        (9 levels: JC1–JC9)
        "IB" → Image blurring          (3 levels: IB1–IB3)
        "NA" → Noise adding            (3 levels: NA1–NA3)
        "BC" → Brightness change       (3 levels: BC1–BC3)
        "CR" → Colour reduction        (3 levels: CR1–CR3)
        "CA" → Contrast adjustments    (3 levels: CA1–CA3)

Usage:
    python split_comofod.py
    python split_comofod.py --source /path/to/CoMoFoD_small_v2
    python split_comofod.py --no_postprocessed   # only base O/F images
"""

import os
import re
import shutil
import argparse
from collections import Counter

# ── defaults ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(BASE_DIR, "datasets", "comofod", "CoMoFoD_small_v2")
DEFAULT_AUTH = os.path.join(BASE_DIR, "datasets", "comofod", "authentic")
DEFAULT_FORGED = os.path.join(BASE_DIR, "datasets", "comofod", "forged")

# Regex to parse CoMoFoD filenames
# Group 1: image number  (e.g. "001" or "01")
# Group 2: mark M1       (O, F, B, M)
# Group 3: optional postprocessing mark M2+level  (e.g. "JC5", "BC3")
PATTERN = re.compile(
    r"^(\d{2,3})_([OFBM])(?:_([A-Z]{2}\d+))?\.(?:png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)


def split_comofod(
    source_dir: str,
    auth_dir: str,
    forged_dir: str,
    include_postprocessed: bool = True,
    copy: bool = False,
) -> dict:
    """
    Split CoMoFoD images into authentic/ and forged/ directories.

    Args:
        source_dir:  Path to the extracted CoMoFoD folder (e.g. CoMoFoD_small_v2/).
        auth_dir:    Destination for original (authentic) images.
        forged_dir:  Destination for forged images.
        include_postprocessed: If True, include postprocessed variants (JC, IB, etc.).
        copy:        If True, copy files; if False, move them (faster, saves disk space).

    Returns:
        dict with counts: authentic, forged, skipped, errors.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)

    transfer = shutil.copy2 if copy else shutil.move
    action_word = "Copying" if copy else "Moving"

    stats = Counter()
    files = sorted(os.listdir(source_dir))

    print(f"\n{'='*60}")
    print(f"  CoMoFoD Dataset Splitter")
    print(f"{'='*60}")
    print(f"  Source       : {source_dir}")
    print(f"  Authentic →  : {auth_dir}")
    print(f"  Forged    →  : {forged_dir}")
    print(f"  Postprocessed: {'INCLUDED' if include_postprocessed else 'EXCLUDED'}")
    print(f"  Action       : {action_word}")
    print(f"  Total files  : {len(files)}")
    print(f"{'='*60}\n")

    for fname in files:
        src_path = os.path.join(source_dir, fname)
        if not os.path.isfile(src_path):
            continue

        match = PATTERN.match(fname)
        if not match:
            stats["skipped_no_match"] += 1
            continue

        img_num, mark, postproc = match.groups()

        # Skip masks (B = binary mask, M = coloured mask)
        if mark in ("B", "M"):
            stats["skipped_mask"] += 1
            continue

        # Skip postprocessed if user opted out
        if postproc and not include_postprocessed:
            stats["skipped_postproc"] += 1
            continue

        # Route to the right folder
        if mark == "O":
            dest_dir = auth_dir
            stats["authentic"] += 1
        elif mark == "F":
            dest_dir = forged_dir
            stats["forged"] += 1
        else:
            stats["skipped_unknown"] += 1
            continue

        try:
            transfer(src_path, os.path.join(dest_dir, fname))
        except Exception as e:
            print(f"  ⚠ Error with {fname}: {e}")
            stats["errors"] += 1

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"  ✔ Authentic (O) images : {stats['authentic']}")
    print(f"  ✔ Forged (F) images    : {stats['forged']}")
    print(f"  ⊘ Masks skipped (B/M)  : {stats['skipped_mask']}")
    if stats["skipped_postproc"]:
        print(f"  ⊘ Postprocessed skipped: {stats['skipped_postproc']}")
    if stats["skipped_no_match"]:
        print(f"  ⊘ Unrecognised files   : {stats['skipped_no_match']}")
    if stats["errors"]:
        print(f"  ⚠ Errors               : {stats['errors']}")
    print(f"{'='*60}\n")

    return dict(stats)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split CoMoFoD dataset into authentic/ and forged/ folders.",
    )
    parser.add_argument(
        "--source", type=str, default=DEFAULT_SOURCE,
        help="Path to extracted CoMoFoD folder (default: datasets/comofod/CoMoFoD_small_v2/).",
    )
    parser.add_argument(
        "--auth_dir", type=str, default=DEFAULT_AUTH,
        help="Destination for authentic images.",
    )
    parser.add_argument(
        "--forged_dir", type=str, default=DEFAULT_FORGED,
        help="Destination for forged images.",
    )
    parser.add_argument(
        "--no_postprocessed", action="store_true",
        help="Exclude postprocessed images (JC, IB, NA, BC, CR, CA); keep only base O/F.",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of moving (uses more disk space but preserves source).",
    )
    args = parser.parse_args()

    split_comofod(
        source_dir=args.source,
        auth_dir=args.auth_dir,
        forged_dir=args.forged_dir,
        include_postprocessed=not args.no_postprocessed,
        copy=args.copy,
    )
