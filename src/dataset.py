"""Dataset discovery, deterministic splits, and model-specific transforms."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config


Sample = tuple[str, int]


def _list_samples(data_dir: Path) -> tuple[list[str], list[Sample]]:
    class_names = sorted(
        path.name for path in data_dir.iterdir() if path.is_dir()
    )
    samples: list[Sample] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for path in sorted(class_dir.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in config.IMAGE_EXTENSIONS
            ):
                samples.append((str(path), label))
    if not class_names or not samples:
        raise ValueError(f"No images found in {data_dir}")
    return class_names, samples


def _split_samples(
    data_dir: Path, seed: int = config.SEED
) -> tuple[list[str], dict[str, list[Sample]]]:
    class_names, all_samples = _list_samples(data_dir)
    samples_by_label: dict[int, list[str]] = {
        label: [] for label in range(len(class_names))
    }
    for path, label in all_samples:
        samples_by_label[label].append(path)
    split_samples = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for label in range(len(class_names)):
        class_paths = samples_by_label[label]
        rng.shuffle(class_paths)
        train_end = int(0.70 * len(class_paths))
        val_end = int(0.85 * len(class_paths))
        split_samples["train"].extend(
            (path, label) for path in class_paths[:train_end]
        )
        split_samples["val"].extend(
            (path, label) for path in class_paths[train_end:val_end]
        )
        split_samples["test"].extend(
            (path, label) for path in class_paths[val_end:]
        )

    return class_names, split_samples


def get_splits(
    data_dir: Path = config.DATA_DIR,
    manifest_path: Path = config.SPLIT_PATH,
    seed: int = config.SEED,
) -> tuple[list[str], dict[str, list[Sample]]]:
    """Load a stable split manifest or create it once from class folders."""
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        split_samples = {
            split: [
                (_manifest_path(item["path"]), int(item["label"]))
                for item in items
            ]
            for split, items in payload["splits"].items()
        }
        return payload["class_names"], split_samples

    class_names, split_samples = _split_samples(data_dir, seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "class_names": class_names,
        "splits": {
            split: [
                {"path": _relative_manifest_path(path), "label": label}
                for path, label in items
            ]
            for split, items in split_samples.items()
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return class_names, split_samples


def _relative_manifest_path(path: str) -> str:
    """Store paths relative to the repository for portable manifests."""
    return str(Path(path).resolve().relative_to(config.ROOT_DIR))


def _manifest_path(path: str) -> str:
    """Resolve both new relative paths and legacy absolute paths."""
    manifest_path = Path(path)
    if manifest_path.is_absolute():
        return str(manifest_path)
    return str((config.ROOT_DIR / manifest_path).resolve())


def _build_transform(model_name: str, train: bool) -> transforms.Compose:
    is_resnet = model_name == "resnet50"
    mean = config.IMAGENET_MEAN if is_resnet else config.CNN_MEAN
    std = config.IMAGENET_STD if is_resnet else config.CNN_STD
    operations: list[transforms.Transform] = []

    if train:
        operations.extend(
            [
                transforms.RandomResizedCrop(
                    config.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
                ),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                ),
            ]
        )
    else:
        operations.extend(
            [transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)


class GroceryDataset(Dataset):
    """Dataset with train augmentation and deterministic evaluation."""

    def __init__(
        self, samples: Sequence[Sample], model_name: str, train: bool = False
    ):
        self.samples = list(samples)
        self.transform = _build_transform(model_name, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def create_dataloaders(
    model_name: str,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[list[str], dict[str, DataLoader]]:
    class_names, split_samples = get_splits()
    loaders = {
        "train": DataLoader(
            GroceryDataset(split_samples["train"], model_name, train=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            GroceryDataset(split_samples["val"], model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            GroceryDataset(split_samples["test"], model_name),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return class_names, loaders


def get_train_labels() -> torch.Tensor:
    _, split_samples = get_splits()
    return torch.tensor(
        [label for _, label in split_samples["train"]], dtype=torch.long
    )
