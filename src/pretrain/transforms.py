"""Data transforms for the defect detection pretraining pipeline."""

import torchvision.transforms as T
from torchvision.transforms import Compose


def _mean_std():
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def train_transforms(img_size: int = 224):
    return Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(*_mean_std()),
    ])


def val_transforms(img_size: int = 224):
    return Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(*_mean_std()),
    ])


def get_transforms(img_size: int = 224, is_train: bool = True):
    return train_transforms(img_size) if is_train else val_transforms(img_size)


def get_smoke_transforms():
    return train_transforms(64), val_transforms(64)