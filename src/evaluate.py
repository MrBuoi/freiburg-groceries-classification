"""Evaluate a saved grocery classification checkpoint on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from . import config
from .dataset import create_dataloaders
from .train import make_model, run_epoch
from .utils import classification_metrics, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "deep_cnn", "resnet50"],
        required=True,
    )
    parser.add_argument(
        "--run-name",
        help="Version name used to locate checkpoint and metrics files.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=config.ARTIFACT_DIR)
    args = parser.parse_args()
    run_name = args.run_name or args.model

    checkpoint_path = args.checkpoint or (
        config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    )
    device = config.get_device()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    metadata = checkpoint["metadata"]
    if metadata["model_name"] != args.model:
        raise ValueError("Checkpoint model does not match --model")
    if metadata.get("run_name", run_name) != run_name:
        raise ValueError("Checkpoint run does not match --run-name")
    class_names, loaders = create_dataloaders(args.model)
    model = make_model(
        args.model, len(class_names), pretrained=False, freeze_backbone=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss_function = nn.CrossEntropyLoss()
    test_loss, targets, predictions = run_epoch(
        model, loaders["test"], loss_function, device
    )
    metrics = classification_metrics(targets, predictions, class_names)
    metrics["loss"] = test_loss
    metrics["model"] = args.model
    save_json(args.output_dir / f"{run_name}_metrics.json", metrics)
    print(
        f"Test loss={test_loss:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
