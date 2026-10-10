"""Model 1: Simple CNN phân loại 25 loại sản phẩm trong bộ Freiburg Groceries (train từ đầu).

Chạy thử, chỉ xem val:   python youonlycodeonce/train_simple_cnn.py --epochs 50 --checkpoint checkpoints/simple_cnn_try.pt
Chạy bản cuối và test:   python youonlycodeonce/train_simple_cnn.py --eval_test 1 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]  # thư mục gốc của repo, chứa data/ và checkpoints/
SPLIT_DIR = Path(__file__).resolve().parent / "splits"

# mean và std của 3 kênh R, G, B, tính trên tập train
MEAN = (0.521, 0.458, 0.391)
STD = (0.259, 0.247, 0.245)

# cấu hình của Model 1
IMAGE_SIZE = 128
EVAL_SCALE = 1.25  # val/test dùng ảnh 160x160, lớn hơn ảnh train 1,25 lần
NUM_BLOCKS = 6
WIDTH = 32  # số filter của khối đầu, mỗi khối sau gấp đôi
DROPOUT = 0.3
EPOCHS = 200
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
CROP_SCALE = 0.35  # RandomResizedCrop cắt một vùng 35-100% diện tích ảnh
RANDOM_ERASING = 0.25


def load_split(data_dir=ROOT / "data", split_dir=SPLIT_DIR):
    # mỗi dòng trong file split có dạng LỚP/TÊN_FILE.png
    lines = {}
    for name in ["train", "val", "test"]:
        lines[name] = (split_dir / f"{name}.txt").read_text().split()
    class_names = sorted(set(line.split("/")[0] for line in lines["train"]))
    samples = {}
    for name in lines:
        samples[name] = [(data_dir / line, class_names.index(line.split("/")[0])) for line in lines[name]]
    return class_names, samples["train"], samples["val"], samples["test"]


def build_transforms(image_size, eval_size):
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(image_size, scale=(CROP_SCALE, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
        v2.RandomErasing(p=RANDOM_ERASING),
    ])
    # val/test chỉ resize và chuẩn hoá
    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((eval_size, eval_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])
    return train_transform, eval_transform


class GroceryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def load_images(samples, transform, device):
    # val/test không có augmentation nên đọc một lần rồi để sẵn trên GPU
    images = []
    for path, _ in samples:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    labels = torch.tensor([label for _, label in samples])
    return torch.stack(images).to(device), labels.to(device)


def build_model(num_classes, num_blocks=NUM_BLOCKS, width=WIDTH, dropout=DROPOUT):
    layers = []
    in_channels = 3
    for i in range(num_blocks):
        out_channels = width * 2**i
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # MaxBlurPool: lấy max 2x2 trượt từng pixel rồi trung bình 2x2 để thu nhỏ một nửa,
            # nên kết quả ít bị đổi khi vật thể lệch đi vài pixel
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(2, stride=1),
            nn.AvgPool2d(2),
        ]
        in_channels = out_channels
    layers += [
        nn.AdaptiveAvgPool2d(1),  # mỗi feature map còn lại 1 số
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_channels, num_classes),
    ]
    return nn.Sequential(*layers)


def get_device():
    # trả về thiết bị và kiểu số 16-bit dùng khi train (nhanh hơn float32)
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        # GPU đời cũ như Colab T4 chạy bfloat16 rất chậm nên dùng float16
        return torch.device("cuda"), torch.bfloat16 if major >= 8 else torch.float16
    return torch.device("cpu"), None


@torch.no_grad()
def predict(model, images, amp_dtype):
    # trả về xác suất của ảnh gốc và xác suất TTA (trung bình với ảnh lật ngang)
    model.eval()
    probs, probs_tta = [], []
    for i in range(0, len(images), 64):
        batch = images[i:i + 64]
        with torch.autocast(batch.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            p = model(batch).float().softmax(dim=1)
            p_flip = model(batch.flip(dims=[3])).float().softmax(dim=1)
        probs.append(p)
        probs_tta.append((p + p_flip) / 2)
    return torch.cat(probs), torch.cat(probs_tta)


def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num_blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_test", type=int, default=0, help="1 = đánh giá trên tập test ở cuối")
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--checkpoint", default="", help="mặc định: checkpoints/simple_cnn_seed<seed>.pt")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint or ROOT / "checkpoints" / f"simple_cnn_seed{args.seed}.pt")
    history_path = checkpoint_path.with_suffix(".history.json")
    config = {
        "image_size": IMAGE_SIZE, "num_blocks": args.num_blocks, "width": WIDTH, "pool": "aa", "dropout": DROPOUT,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING, "crop_scale": CROP_SCALE, "trivial_augment": 1,
        "random_erasing": RANDOM_ERASING, "eval_scale": EVAL_SCALE, "tta": 1, "seed": args.seed,
        "num_workers": args.num_workers, "eval_test": args.eval_test, "data_dir": args.data_dir,
        "split_dir": str(SPLIT_DIR), "checkpoint": str(checkpoint_path),
    }
    torch.manual_seed(args.seed)

    # dữ liệu
    class_names, train_samples, val_samples, test_samples = load_split(Path(args.data_dir))
    print(f"Số lớp: {len(class_names)} | train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    device, amp_dtype = get_device()
    print(f"Thiết bị: {device} ({amp_dtype or torch.float32})")
    train_transform, eval_transform = build_transforms(IMAGE_SIZE, round(IMAGE_SIZE * EVAL_SCALE))
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
    model = build_model(len(class_names), args.num_blocks).to(device)
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
    model = build_model(len(class_names), args.num_blocks)
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
