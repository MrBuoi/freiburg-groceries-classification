"""Model 2: Deep CNN kiểu ResNet (ResNet-10) phân loại 25 loại sản phẩm trong bộ Freiburg Groceries (train từ đầu).

Đọc dữ liệu, augmentation và đánh giá dùng lại code của Model 1 (train_simple_cnn.py).
Chạy thử, chỉ xem val:   python youonlycodeonce/train_deep_cnn.py --epochs 50 --checkpoint checkpoints/deep_cnn_try.pt
Chạy bản cuối và test:   python youonlycodeonce/train_deep_cnn.py --eval_test 1 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from train_simple_cnn import (CROP_SCALE, RANDOM_ERASING, ROOT, SPLIT_DIR, GroceryDataset, accuracy,
                              build_transforms, get_device, load_images, load_split, predict)

# cấu hình của Model 2
IMAGE_SIZE = 160
EVAL_SCALE = 1.25  # val/test dùng ảnh lớn hơn ảnh train 1,25 lần (160 -> 200)
WIDTH = 96  # số kênh của tầng 1, mỗi tầng sau gấp đôi: 96, 192, 384, 768
BLOCKS = "1,1,1,1"  # số khối residual ở 4 tầng: 1,1,1,1 là ResNet-10, 2,2,2,2 là ResNet-18
DROPOUT = 0.3
DROP_PATH = 0.1  # xác suất stochastic depth ở khối cuối
EPOCHS = 450
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1


def conv_bn_relu(in_channels, out_channels, stride=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class BlurPool(nn.Module):
    # làm mờ bằng bộ lọc [1,2,1] x [1,2,1] / 16 rồi giữ 1 trong mỗi 2x2 điểm (chống răng cưa khi thu nhỏ)
    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :] / 16
        self.register_buffer("kernel", k.expand(channels, 1, 3, 3).clone(), persistent=False)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), stride=2, groups=x.shape[1])


class ResidualBlock(nn.Module):
    # đầu ra = ReLU(nhánh chính(x) + đường tắt(x))
    def __init__(self, in_channels, out_channels, stride, drop_path=0.0, aa=1):
        super().__init__()
        self.drop_path = drop_path
        if stride > 1 and aa:
            # thu nhỏ bằng Conv bước 1 rồi BlurPool, thay cho Conv bước 2
            first = conv_bn_relu(in_channels, out_channels) + [BlurPool(out_channels)]
        else:
            first = conv_bn_relu(in_channels, out_channels, stride)
        self.body = nn.Sequential(
            *first,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # gamma của BatchNorm cuối bằng 0, nên lúc mới khởi tạo khối chỉ truyền đường tắt
        nn.init.zeros_(self.body[-1].weight)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # khối thu nhỏ: lấy trung bình 2x2 rồi Conv 1x1 cho khớp kích thước với nhánh chính
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.training and self.drop_path > 0:
            # stochastic depth: với mỗi ảnh trong batch, bỏ nhánh chính với xác suất drop_path
            keep = (torch.rand(out.shape[0], 1, 1, 1, device=out.device) >= self.drop_path).to(out.dtype)
            out = out * keep / (1 - self.drop_path)
        return self.relu(out + self.shortcut(x))


def build_model(num_classes, width=WIDTH, blocks=(1, 1, 1, 1), dropout=DROPOUT, drop_path=DROP_PATH, aa=1):
    model = nn.Sequential()
    # stem: 3 Conv 3x3 (Conv đầu bước 2) rồi MaxBlurPool như Model 1, thu nhỏ ảnh 4 lần
    model.add_module("stem", nn.Sequential(
        *conv_bn_relu(3, width // 2, stride=2),
        *conv_bn_relu(width // 2, width // 2),
        *conv_bn_relu(width // 2, width),
        nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(2, stride=1), nn.AvgPool2d(2),
    ))
    in_channels = width
    k = 0  # số thứ tự của khối, để tăng dần xác suất stochastic depth từ 0 lên drop_path
    for i, num_blocks in enumerate(blocks):
        out_channels = width * 2**i
        stage = []
        for j in range(num_blocks):
            stride = 2 if i > 0 and j == 0 else 1  # khối đầu của tầng 2, 3, 4 thu nhỏ ảnh một nửa
            p = drop_path * k / max(1, sum(blocks) - 1)
            stage.append(ResidualBlock(in_channels, out_channels, stride, p, aa))
            in_channels = out_channels
            k += 1
        model.add_module(f"stage{i + 1}", nn.Sequential(*stage))
    model.add_module("head", nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_channels, num_classes),
    ))
    return model


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--blocks", default=BLOCKS, help="số khối residual ở 4 tầng, ví dụ 2,2,2,2")
    parser.add_argument("--aa", type=int, default=1, help="1 = khối thu nhỏ dùng BlurPool, 0 = Conv bước 2")
    parser.add_argument("--drop_path", type=float, default=DROP_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_test", type=int, default=0, help="1 = đánh giá trên tập test ở cuối")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--checkpoint", default="", help="mặc định: checkpoints/deep_cnn_seed<seed>.pt")
    args = parser.parse_args()
    blocks = tuple(int(b) for b in args.blocks.split(","))
    if len(blocks) != 4:
        parser.error("--blocks cần đúng 4 số, ví dụ 2,2,2,2")
    checkpoint_path = Path(args.checkpoint or ROOT / "checkpoints" / f"deep_cnn_seed{args.seed}.pt")
    history_path = checkpoint_path.with_suffix(".history.json")
    config = {
        "image_size": args.image_size, "width": args.width, "blocks": args.blocks, "dropout": DROPOUT,
        "drop_path": args.drop_path, "aa": args.aa, "epochs": args.epochs, "batch_size": args.batch_size, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING, "crop_scale": CROP_SCALE,
        "trivial_augment": 1, "random_erasing": RANDOM_ERASING, "eval_scale": EVAL_SCALE, "tta": 1,
        "seed": args.seed, "num_workers": args.num_workers, "eval_test": args.eval_test, "data_dir": args.data_dir,
        "split_dir": str(SPLIT_DIR), "checkpoint": str(checkpoint_path),
    }
    torch.manual_seed(args.seed)

    # dữ liệu
    class_names, train_samples, val_samples, test_samples = load_split(Path(args.data_dir))
    print(f"Số lớp: {len(class_names)} | train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    device, amp_dtype = get_device()
    print(f"Thiết bị: {device} ({amp_dtype or torch.float32})")
    train_transform, eval_transform = build_transforms(args.image_size, round(args.image_size * EVAL_SCALE))
    train_loader = DataLoader(
        GroceryDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_images, val_labels = load_images(val_samples, eval_transform, device)

    # model, loss, optimizer (tạo model trước khi lấy batch đầu tiên: khởi tạo trọng số và DataLoader
    # dùng chung bộ sinh số ngẫu nhiên, đổi thứ tự thì cùng seed sẽ không ra đúng kết quả như README)
    model = build_model(len(class_names), args.width, blocks, drop_path=args.drop_path, aa=args.aa).to(device)
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # OneCycle: lr tăng dần lên LR rồi giảm dần về gần 0
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=args.epochs, steps_per_epoch=len(train_loader))
    # float16 dễ tràn số khi tính gradient nên cần GradScaler (bfloat16 thì không)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    # train, giữ lại model có val accuracy cao nhất
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_acc, best_epoch, best_state = -1.0, 0, None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.detach() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum()
        seen = len(train_loader) * args.batch_size
        train_loss, train_acc = total_loss.item() / seen, correct.item() / seen

        probs, probs_tta = predict(model, val_images, amp_dtype)
        val_acc, val_acc_tta = accuracy(probs, val_labels), accuracy(probs_tta, val_labels)
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.3f} | train acc {train_acc:.3f} | "
              f"val acc {val_acc:.3f} (TTA {val_acc_tta:.3f}) | {time.time() - start:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_acc": val_acc, "val_acc_tta": val_acc_tta})
        history_path.write_text(json.dumps({"config": config, "epochs": history}, indent=1))

        if val_acc_tta > best_acc:
            best_acc, best_epoch = val_acc_tta, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "class_names": class_names, "config": config}, checkpoint_path)

    last10 = sum(row["val_acc_tta"] for row in history[-10:]) / len(history[-10:])
    print(f"\nVal accuracy tốt nhất: {best_acc:.3f} (epoch {best_epoch}) | trung bình 10 epoch cuối: {last10:.3f} | "
          f"thời gian train: {(time.time() - start) / 60:.1f} phút")
    print(f"Model đã lưu tại: {checkpoint_path}")
    if not args.eval_test:
        print("Chưa đánh giá test set. Khi đã chốt cấu hình, chạy lại với --eval_test 1.")
        return

    # đánh giá trên test set một lần, bằng model tốt nhất theo val
    print("\n=== Kết quả trên test set ===")
    model = build_model(len(class_names), args.width, blocks, drop_path=args.drop_path, aa=args.aa)
    model.load_state_dict(best_state)
    model = model.to(device)
    test_images, test_labels = load_images(test_samples, eval_transform, device)
    probs, probs_tta = predict(model, test_images, amp_dtype)
    test_acc, test_acc_no_tta = accuracy(probs_tta, test_labels), accuracy(probs, test_labels)
    print(f"Test accuracy: {test_acc:.3f} (không TTA: {test_acc_no_tta:.3f})")

    predictions = probs_tta.argmax(dim=1)
    per_class = {}
    print("Accuracy từng lớp:")
    for label, name in enumerate(class_names):
        mask = test_labels == label
        per_class[name] = (predictions[mask] == label).float().mean().item()
        print(f"  {name:13s} {per_class[name]:.3f}  ({mask.sum().item()} ảnh)")
    mean_per_class = sum(per_class.values()) / len(per_class)
    print(f"Trung bình accuracy các lớp: {mean_per_class:.3f}")

    test_result = {"acc": test_acc, "acc_no_tta": test_acc_no_tta, "mean_per_class_acc": mean_per_class,
                   "per_class_acc": per_class}
    history_path.write_text(json.dumps({"config": config, "epochs": history, "test": test_result}, indent=1))
    print(f"Kết quả test đã lưu vào: {history_path}")


if __name__ == "__main__":  # cần có khi num_workers > 0 trên macOS/Windows
    main()
