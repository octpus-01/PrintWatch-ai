"""Generate pretrain_index.csv from the Kaggle dataset directory.

Scans the ImageFolder-structured dataset at DATA_ROOT and produces:
  data/pretrain_index.csv

Columns: image_path, label, split, source_week

Split ratio: 80% train / 20% val (deterministic via hash of image_path).
"""

import csv
import os
import sys
from pathlib import Path

DATA_ROOT = r"C:\Users\PATAT\Downloads\Kaggle_3D_Print_Defect_Dataset"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "pretrain_index.csv"
TRAIN_RATIO = 0.8


def _split_from_path(image_path: str) -> str:
    """Deterministic split based on path hash."""
    h = hash(image_path) & 0xFFFFFFFF
    return "train" if (h / 0xFFFFFFFF) < TRAIN_RATIO else "val"


def scan():
    data_root = Path(DATA_ROOT)
    if not data_root.is_dir():
        print(f"ERROR: Dataset not found at {DATA_ROOT}")
        sys.exit(1)

    rows = []
    class_dirs = sorted(d.name for d in data_root.iterdir() if d.is_dir())
    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    for class_name in class_dirs:
        class_dir = data_root / class_name
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        for f in files:
            split = _split_from_path(str(f))
            rows.append({
                "image_path": str(f.resolve()),
                "label": class_name,
                "split": split,
                "source_week": "kaggle_original",
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "source_week"])
        writer.writeheader()
        writer.writerows(rows)

    train_count = sum(1 for r in rows if r["split"] == "train")
    val_count = sum(1 for r in rows if r["split"] == "val")
    print(f"Generated {OUTPUT_CSV}")
    print(f"  Total: {len(rows)} samples")
    print(f"  Train: {train_count}, Val: {val_count}")
    print(f"  Ratio: {train_count / len(rows):.1%} / {val_count / len(rows):.1%}")


if __name__ == "__main__":
    scan()