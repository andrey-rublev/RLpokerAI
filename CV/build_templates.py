"""
Build a template database from the annotated card dataset.

For each image in Data/Images/Images/:
  - Derive the card label from the filename (e.g. "10C0.jpg" → "10C")
  - Run detect_and_get_top_left() to produce a 64×96 binary overlay
  - Accumulate overlays per label and compute a mean template
  - Save {LABEL}.npy (float32 array) and {LABEL}.png for all 52 cards

Run this once before using card_matcher.py:
    python build_templates.py
"""
import os
import sys
import re
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_card import detect_and_get_top_left

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE, "Data", "Images", "Images")
TEMPLATES_DIR = os.path.join(BASE, "Data", "templates", "card_overlays")

VALID_LABELS = {
    f"{r}{s}"
    for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    for s in ["S", "H", "D", "C"]
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def parse_label(filename: str) -> str | None:
    """Extract card label from filename like '10C0.jpg' → '10C', 'AS12.jpg' → 'AS'."""
    name = os.path.splitext(os.path.basename(filename))[0].upper()
    label = re.sub(r"\d+$", "", name)
    return label if label in VALID_LABELS else None


def build_templates(
    images_dir: str = IMAGES_DIR,
    templates_dir: str = TEMPLATES_DIR,
    max_per_class: int = 50,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Process dataset images and save mean overlay templates.

    Returns dict of {label: sample_count} for each saved template.
    """
    os.makedirs(templates_dir, exist_ok=True)

    card_overlays: dict[str, list[np.ndarray]] = {}
    processed = skipped = 0

    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    total = len(image_files)

    if verbose:
        print(f"Found {total} images in {images_dir}")

    for i, filename in enumerate(image_files):
        label = parse_label(filename)
        if label is None:
            skipped += 1
            continue

        if max_per_class and len(card_overlays.get(label, [])) >= max_per_class:
            continue

        img_path = os.path.join(images_dir, filename)

        if verbose and i % 200 == 0:
            pct = 100 * i / total
            print(f"  [{pct:5.1f}%] {i}/{total}  processed={processed}  skipped={skipped}")

        try:
            result = detect_and_get_top_left(img_path)
        except Exception:
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        card_overlays.setdefault(label, []).append(result["overlay"])
        processed += 1

    if verbose:
        print(f"\nFinished: processed={processed}, skipped={skipped}")
        print(f"Cards with samples: {len(card_overlays)}/52\n")

    counts = {}
    missing = []

    for label in sorted(VALID_LABELS):
        overlays = card_overlays.get(label, [])
        if not overlays:
            missing.append(label)
            continue

        stack = np.stack(overlays, axis=0).astype(np.float32)
        mean_overlay = np.mean(stack, axis=0)

        np.save(os.path.join(templates_dir, f"{label}.npy"), mean_overlay.astype(np.float32))
        cv2.imwrite(os.path.join(templates_dir, f"{label}.png"), mean_overlay.astype(np.uint8))

        counts[label] = len(overlays)
        if verbose:
            print(f"  {label:4s}  {len(overlays):3d} samples  → saved")

    if missing and verbose:
        print(f"\nMissing templates (no samples): {missing}")

    if verbose:
        print(f"\nSaved {len(counts)}/52 templates to {templates_dir}")

    return counts


if __name__ == "__main__":
    build_templates(verbose=True)
