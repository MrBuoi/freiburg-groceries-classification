"""Train one of the three grocery classification models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from . import config
from .dataset import create_dataloaders, get_train_labels
from .model import (
    DeepCNN,
    SimpleCNN,
    build_resnet50,
    count_trainable_parameters,
)
from .utils import (
    class_weights,
    classification_metrics,
    save_checkpoint,
    save_json,
    seed_everything,
)


def make_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name == "deep_cnn":
        return DeepCNN(num_classes)
    if model_name == "resnet50":
        return build_resnet50(num_classes, pretrained, freeze_backbone)
    raise ValueError(f"Unknown model: {model_name}")


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "deep_cnn", "resnet50"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--run-name",
        help="Version name for checkpoint and history files.",
    )
    parser.add_argument(
        "--weights", choices=["default", "none"], default="default"
    )
    parser.add_argument("--output-dir", type=Path, default=config.ARTIFACT_DIR)
    args = parser.parse_args()
    run_name = args.run_name or args.model

    seed_everything(config.SEED)
    class_names, loaders = create_dataloaders(args.model, args.batch_size)
    device = config.get_device()
    pretrained = args.model == "resnet50" and args.weights == "default"
    freeze_backbone = args.model == "resnet50"
    model = make_model(
        args.model, len(class_names), pretrained, freeze_backbone
    ).to(device)
    weights = class_weights(get_train_labels(), len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    learning_rate = (
        config.RESNET_LEARNING_RATE
        if args.model == "resnet50"
        else config.LEARNING_RATE
    )
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )

    checkpoint_path = config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    best_metric = -1.0
    history = []
    for epoch in range(args.epochs):
        if args.model == "resnet50" and epoch == config.RESNET_WARMUP_EPOCHS:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.RESNET_LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )

        train_loss, train_targets, train_predictions = run_epoch(
            model, loaders["train"], loss_function, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_targets, val_predictions = run_epoch(
                model, loaders["val"], loss_function, device
            )
        train_metrics = classification_metrics(
            train_targets, train_predictions, class_names
        )
        val_metrics = classification_metrics(
            val_targets, val_predictions, class_names
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={train_loss:.4f}, "
            f"macro-F1={record['train_macro_f1']:.3f} | "
            f"val loss={val_loss:.4f}, macro-F1={record['val_macro_f1']:.3f}"
        )
        if record["val_macro_f1"] > best_metric:
            best_metric = record["val_macro_f1"]
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch + 1,
                best_metric,
                {
                    "model_name": args.model,
                    "run_name": run_name,
                    "class_names": class_names,
                    "image_size": config.IMAGE_SIZE,
                    "normalization": (
                        "imagenet" if args.model == "resnet50" else "[-1, 1]"
                    ),
                    "seed": config.SEED,
                    "trainable_parameters": count_trainable_parameters(model),
                    "pretrained": pretrained,
                },
            )

    save_json(args.output_dir / f"{run_name}_history.json", history)
    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
