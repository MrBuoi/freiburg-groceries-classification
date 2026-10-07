"""Training utilities, metrics, and checkpoint serialization."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError("Every class must have at least one training sample")
    return labels.numel() / (num_classes * counts)


def classification_metrics(
    targets: Iterable[int], predictions: Iterable[int], class_names: list[str]
) -> dict:
    targets = list(targets)
    predictions = list(predictions)
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(class_names)))[0:]
        ).tolist(),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "metadata": metadata,
        },
        path,
    )


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
