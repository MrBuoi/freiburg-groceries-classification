"""Model 1 (baseline): Simple CNN cho bộ dữ liệu Freiburg Groceries (25 lớp).

Model được train từ đầu, không dùng trọng số pretrained. Kiến trúc chỉ gồm 6 khối
Conv 3x3 → BatchNorm → ReLU → MaxBlurPool (max pooling chống răng cưa) xếp nối tiếp,
rồi Global Average Pooling, Dropout và một lớp Linear. Accuracy được nâng lên nhờ cách
huấn luyện: augmentation mạnh, label smoothing, lịch learning rate OneCycle, đánh giá ở
ảnh lớn hơn lúc train và TTA (cộng thêm dự đoán của ảnh lật ngang).

Cách chia train/val/test đọc từ youonlycodeonce/splits/*.txt (tạo bởi youonlycodeonce/make_splits.py,
dùng chung cho Model 1 và Model 2). Khi thử cấu hình, chỉ nhìn val; test chỉ đánh giá một lần cho cấu hình
cuối cùng bằng cờ --eval_test 1.

Thử cấu hình (chỉ báo val). Đặt --checkpoint riêng cho mỗi lần thử: mặc định mọi lần chạy cùng
seed đều ghi vào checkpoints/simple_cnn_seed<seed>.pt và .history.json, nên lần thử chạy sau lần
chạy cuối sẽ xoá model và kết quả test của lần đó:
    python youonlycodeonce/train_simple_cnn.py --epochs 50 --num_blocks 5 --checkpoint checkpoints/simple_cnn_try.pt
Chạy cấu hình cuối và đánh giá test:
    python youonlycodeonce/train_simple_cnn.py --eval_test 1
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

ROOT = Path(__file__).resolve().parents[1]  # gốc repo: data/ và checkpoints/ nằm ở đây
SPLIT_DIR = Path(__file__).resolve().parent / "splits"  # youonlycodeonce/splits
MEAN = (0.521, 0.458, 0.391)  # mean và std từng kênh RGB, tính trên tập train
STD = (0.259, 0.247, 0.245)


# 1. Cấu hình mặc định; đổi bất kỳ khoá nào bằng --ten_khoa gia_tri trên dòng lệnh
CONFIG = {
    "image_size": 128,  # lúc train, ảnh gốc 256x256 được cắt/thu về image_size x image_size
    "num_blocks": 6,  # số khối Conv-BN-ReLU-Pool
    "width": 32,  # số filter của khối đầu tiên; mỗi khối sau gấp đôi
    "pool": "aa",  # aa = max pooling chống răng cưa (MaxBlurPool), max = MaxPool thường
    "dropout": 0.3,  # dropout trước lớp Linear cuối
    "epochs": 200,
    "batch_size": 64,
    "lr": 2e-3,  # learning rate cao nhất của lịch OneCycle
    "weight_decay": 0.05,  # weight decay của AdamW
    "label_smoothing": 0.1,  # bớt "tự tin thái quá" của model, giảm overfit
    "crop_scale": 0.35,  # RandomResizedCrop lấy ngẫu nhiên 35–100% diện tích ảnh
    "trivial_augment": 1,  # 1 = bật TrivialAugmentWide (xoay, nghiêng, đổi sáng, tương phản...)
    "random_erasing": 0.25,  # xác suất xoá một vùng chữ nhật ngẫu nhiên trong ảnh
    "eval_scale": 1.25,  # val/test dùng ảnh lớn hơn lúc train 1.25 lần (128 → 160)
    "tta": 1,  # 1 = khi đánh giá, lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang
    "seed": 42,  # seed cho khởi tạo trọng số, augmentation và thứ tự batch
    "num_workers": 8,  # số tiến trình đọc ảnh song song
    "eval_test": 0,  # 1 = đánh giá test set ở cuối; chỉ bật cho cấu hình cuối cùng đã chọn bằng val
    "data_dir": str(ROOT / "data"),
    "split_dir": str(SPLIT_DIR),
    "checkpoint": "",  # để trống = checkpoints/simple_cnn_seed<seed>.pt, mỗi seed một file
}


def parse_config():
    parser = argparse.ArgumentParser(description="Train Simple CNN baseline")
    for key, value in CONFIG.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    return vars(parser.parse_args())


# 2. Đọc cách chia train/val/test dùng chung (mỗi dòng: LỚP/TÊN_FILE.png)
def load_split(data_dir=ROOT / "data", split_dir=SPLIT_DIR):
    """Nhãn là chỉ số của tên lớp theo thứ tự bảng chữ cái, giống nhau ở mọi model.
    Đường dẫn mặc định tính từ vị trí file này, nên gọi từ thư mục nào cũng được."""
    splits = {}
    for name in ("train", "val", "test"):
        split_file = split_dir / f"{name}.txt"
        if not split_file.exists():  # youonlycodeonce/splits/ có sẵn trong repo, không cần tạo lại
            raise FileNotFoundError(
                f"Không thấy {split_file}. Hãy chạy lệnh từ thư mục gốc của repo hoặc sửa --split_dir."
            )
        splits[name] = split_file.read_text(encoding="utf-8").split()
    class_names = sorted({rel.split("/")[0] for rel in splits["train"]})
    label_of = {class_name: label for label, class_name in enumerate(class_names)}
    samples = {}
    for name, lines in splits.items():
        samples[name] = [(data_dir / rel, label_of[rel.split("/")[0]]) for rel in lines]
        missing = [str(path) for path, _ in samples[name] if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Thiếu {len(missing)} ảnh trong {data_dir}, ví dụ {missing[0]}. "
                "Xem mục tải dữ liệu trong README."
            )
    return class_names, samples["train"], samples["val"], samples["test"]


# 3. Tiền xử lý và augmentation
def build_transforms(cfg):
    size = cfg["image_size"]
    train_steps = [
        v2.ToImage(),
        v2.RandomResizedCrop(size, scale=(cfg["crop_scale"], 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
    ]
    if cfg["trivial_augment"]:
        train_steps.append(v2.TrivialAugmentWide())
    train_steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    if cfg["random_erasing"] > 0:
        train_steps.append(v2.RandomErasing(p=cfg["random_erasing"]))

    # val/test: chỉ resize và chuẩn hoá, không augmentation. RandomResizedCrop khiến vật thể
    # lúc train trông to hơn thực tế khoảng 1.2 lần, nên đánh giá ở ảnh lớn hơn cho khớp
    eval_size = round(size * cfg["eval_scale"])
    eval_steps = [
        v2.ToImage(),
        v2.Resize((eval_size, eval_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ]
    return v2.Compose(train_steps), v2.Compose(eval_steps)


class GroceryImages(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def load_eval_set(samples, transform, device):
    """Val/test không có augmentation nên chỉ cần xử lý một lần rồi giữ sẵn trên GPU."""
    images = []
    for path, _ in samples:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    labels = torch.tensor([label for _, label in samples])
    return torch.stack(images).to(device), labels.to(device)


# 4. Model: các khối Conv-BN-ReLU-Pool, số filter gấp đôi sau mỗi khối
def build_model(num_classes, num_blocks, width, dropout, pool="aa"):
    layers, in_channels = [], 3
    for i in range(num_blocks):
        out_channels = width * 2**i
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # chuẩn hoá đầu ra mỗi lớp giúp train ổn định và nhanh hơn
            nn.ReLU(inplace=True),
        ]
        # cả hai cách đều giảm một nửa chiều cao và chiều rộng
        if pool == "aa":
            # MaxBlurPool: lấy max 2x2 trượt từng pixel, rồi lấy trung bình 2x2 để thu nhỏ, nên
            # dự đoán ít bị đổi khi vật thể dịch đi vài pixel. Đệm thêm số 0 ở mép phải và mép
            # dưới không làm đổi giá trị max, vì sau ReLU mọi giá trị đều >= 0
            layers += [nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(2, stride=1), nn.AvgPool2d(2)]
        elif pool == "max":
            layers.append(nn.MaxPool2d(kernel_size=2))
        else:
            raise ValueError(f"pool phải là 'aa' hoặc 'max', không phải {pool!r}")
        in_channels = out_channels
    layers += [
        nn.AdaptiveAvgPool2d(1),  # mỗi feature map → 1 số, nên model nhận được ảnh mọi kích thước
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_channels, num_classes),
    ]
    return nn.Sequential(*layers)


# 5. Dự đoán và tính accuracy
def choose_amp_dtype(device):
    """Kiểu số thực rút gọn dùng khi tính trên GPU; None = tính bằng float32 như bình thường."""
    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda":
        # GPU NVIDIA đời Ampere trở lên (RTX 30xx, A100...) chạy bfloat16 trực tiếp; GPU cũ hơn
        # như Colab T4 chỉ giả lập bfloat16 (rất chậm) nên dùng float16 kèm GradScaler
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return None


@torch.no_grad()
def predict(model, images, amp_dtype, batch_size=64):
    """Trả về xác suất của ảnh gốc và xác suất TTA (trung bình với ảnh lật ngang).
    Batch nhỏ để GPU ít bộ nhớ (4 GB) không bị tràn khi đánh giá ảnh 160x160."""
    model.eval()
    probs, probs_tta = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        with torch.autocast(
            batch.device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None
        ):
            p = model(batch).float().softmax(dim=1)
            p_flip = model(batch.flip(dims=[3])).float().softmax(dim=1)
        probs.append(p)
        probs_tta.append((p + p_flip) / 2)
    return torch.cat(probs), torch.cat(probs_tta)


def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


def cpu_state(model):
    """Bản sao trọng số trên CPU: máy khác (Windows, Colab) mở được mà không cần GPU của Mac."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows: tránh lỗi khi in tiếng Việt ra file/pipe
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cfg = parse_config()
    if not cfg["checkpoint"]:
        cfg["checkpoint"] = str(ROOT / "checkpoints" / f"simple_cnn_seed{cfg['seed']}.pt")
    torch.manual_seed(cfg["seed"])

    class_names, train_samples, val_samples, test_samples = load_split(
        Path(cfg["data_dir"]), Path(cfg["split_dir"])
    )
    print(
        f"Số lớp: {len(class_names)} | train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    amp_dtype = choose_amp_dtype(device)  # trên GPU tính bằng số 16-bit, nhanh hơn ~1.3 lần
    # float16 dễ tràn số khi tính gradient nên cần GradScaler; bfloat16 và float32 thì không
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    print(f"Thiết bị: {device} ({amp_dtype or torch.float32}) | cấu hình: {cfg}")

    train_transform, eval_transform = build_transforms(cfg)
    train_loader = DataLoader(
        GroceryImages(train_samples, train_transform),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_images, val_labels = load_eval_set(val_samples, eval_transform, device)

    num_classes = len(class_names)
    model = build_model(
        num_classes, cfg["num_blocks"], cfg["width"], cfg["dropout"], cfg["pool"]
    ).to(device)
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")

    loss_function = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(  # tăng dần lr rồi giảm dần về gần 0
        optimizer, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_loader)
    )

    # 6. Huấn luyện, lưu model tốt nhất theo val accuracy
    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.with_suffix(".history.json")  # số liệu từng epoch để vẽ biểu đồ
    history = []
    best_val, best_epoch, best_state = -1.0, 0, None
    train_iter = iter(train_loader)  # tạo sau khi khởi tạo model để giữ nguyên thứ tự số ngẫu nhiên
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(
                device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None
            ):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.detach() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum()
        seen = len(train_loader) * cfg["batch_size"]
        train_iter = iter(train_loader)  # các worker chuẩn bị sẵn batch của epoch sau trong lúc đánh giá val

        probs, probs_tta = predict(model, val_images, amp_dtype)
        val_plain, val_tta = accuracy(probs, val_labels), accuracy(probs_tta, val_labels)
        train_loss, train_acc = total_loss.item() / seen, correct.item() / seen
        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | loss {train_loss:.3f} | "
            f"train acc {train_acc:.3f} | val acc {val_plain:.3f} (TTA {val_tta:.3f}) | "
            f"{time.time() - start:.0f}s",
            flush=True,
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_acc": val_plain, "val_acc_tta": val_tta}
        )
        history_path.write_text(json.dumps({"config": cfg, "epochs": history}, indent=1))

        score = val_tta if cfg["tta"] else val_plain
        if score > best_val:
            best_val, best_epoch, best_state = score, epoch, cpu_state(model)
            torch.save(
                {"model": best_state, "class_names": class_names, "config": cfg},
                checkpoint_path,
            )

    key = "val_acc_tta" if cfg["tta"] else "val_acc"
    last10 = sum(row[key] for row in history[-10:]) / len(history[-10:])
    print(
        f"\nVal accuracy tốt nhất: {best_val:.3f} (epoch {best_epoch}) | "
        f"trung bình 10 epoch cuối: {last10:.3f} | thời gian train: {(time.time() - start) / 60:.1f} phút"
    )
    print(f"Model đã lưu tại: {checkpoint_path}")
    if not cfg["eval_test"]:
        print("Chưa đánh giá test set. Khi đã chốt cấu hình, chạy lại với --eval_test 1.")
        return

    # 7. Đánh giá một lần duy nhất trên test set bằng model tốt nhất theo val
    print("\n=== Kết quả trên test set ===")
    best_model = build_model(
        num_classes, cfg["num_blocks"], cfg["width"], cfg["dropout"], cfg["pool"]
    )
    best_model.load_state_dict(best_state)  # trọng số giữ trong bộ nhớ, không đọc lại file
    best_model = best_model.to(device)
    test_images, test_labels = load_eval_set(test_samples, eval_transform, device)
    probs, probs_tta = predict(best_model, test_images, amp_dtype)
    final_probs = probs_tta if cfg["tta"] else probs
    test_acc, test_acc_plain = accuracy(final_probs, test_labels), accuracy(probs, test_labels)
    print(f"Test accuracy: {test_acc:.3f}", end="")
    print(f" (không TTA: {test_acc_plain:.3f})" if cfg["tta"] else "")

    predictions = final_probs.argmax(dim=1)
    per_class = {}
    print("Accuracy từng lớp:")
    for label, class_name in enumerate(class_names):
        mask = test_labels == label
        per_class[class_name] = (predictions[mask] == label).float().mean().item()
        print(f"  {class_name:13s} {per_class[class_name]:.3f}  ({mask.sum().item()} ảnh)")
    mean_per_class = sum(per_class.values()) / len(per_class)
    print(f"Trung bình accuracy các lớp: {mean_per_class:.3f}")

    # lưu kết quả test cùng file lịch sử để tổng hợp nhiều seed về sau
    test_result = {"acc": test_acc, "acc_no_tta": test_acc_plain,
                   "mean_per_class_acc": mean_per_class, "per_class_acc": per_class}
    history_path.write_text(
        json.dumps({"config": cfg, "epochs": history, "test": test_result}, indent=1)
    )
    print(f"Kết quả test đã lưu vào: {history_path}")


if __name__ == "__main__":  # bắt buộc khi num_workers > 0 trên macOS/Windows
    main()
