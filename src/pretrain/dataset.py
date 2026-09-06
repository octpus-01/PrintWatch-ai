"""Defect dataset loaded from a CSV index.

The CSV index (pretrain_index.csv) has columns:
    image_path, label, split, source_week

Supports:
  - Stratified sampling by label
  - Train/val split based on the 'split' column
  - Smoke mode with a small subset
"""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Subset, WeightedRandomSampler

from .transforms import get_transforms


class DefectDataset(Dataset):
    """Dataset for 3D print defect classification.

    Loads image paths and labels from a CSV index.
    """

    def __init__(self, index_csv: str, split: str = "train",
                 transform=None, smoke: bool = False, smoke_size: int = 100):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        self._load_index(index_csv, split, smoke, smoke_size)
        self.transform = transform

    def _load_index(self, index_csv: str, split: str,
                    smoke: bool, smoke_size: int):
        label_set = set()
        rows = []
        with open(index_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip() != split:
                    continue
                label = row["label"].strip()
                label_set.add(label)
                rows.append((row["image_path"].strip(), label))

        self.classes = sorted(label_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if smoke and len(rows) > smoke_size:
            label_counts = defaultdict(list)
            for idx, (path, label) in enumerate(rows):
                label_counts[label].append(idx)
            sampled = []
            per_class = max(1, smoke_size // len(self.classes))
            for label in self.classes:
                indices = label_counts.get(label, [])
                sampled.extend(indices[:per_class])
            rows = [rows[i] for i in sorted(sampled)]

        self.samples = rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self.class_to_idx[label]
        return image, target

    @property
    def num_classes(self):
        return len(self.classes)

    def get_label_counts(self):
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1
        return dict(counts)

    def make_weighted_sampler(self):
        """Create a WeightedRandomSampler for class-balanced sampling."""
        label_counts = self.get_label_counts()
        weights = []
        for _, label in self.samples:
            weights.append(1.0 / label_counts[label])
        return WeightedRandomSampler(weights, len(weights), replacement=True)


def create_datasets(index_csv: str, img_size: int = 224,
                    smoke: bool = False, smoke_size: int = 100):
    """Create train and validation datasets from a CSV index."""
    train_ds = DefectDataset(
        index_csv, split="train",
        transform=get_transforms(img_size, is_train=True),
        smoke=smoke, smoke_size=smoke_size,
    )
    val_ds = DefectDataset(
        index_csv, split="val",
        transform=get_transforms(img_size, is_train=False),
        smoke=smoke, smoke_size=smoke_size,
    )
    return train_ds, val_ds