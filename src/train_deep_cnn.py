"""Model 2: Deep CNN (kiểu ResNet) cho bộ dữ liệu Freiburg Groceries (25 lớp).

Model được train từ đầu, không dùng trọng số pretrained. Khác Model 1 (6 Conv xếp thẳng
hàng), Model 2 có phần đầu (stem) gồm 3 Conv, rồi 4 tầng (stage) gồm các khối residual.
Mỗi khối có hai Conv 3x3 và một đường tắt đưa đầu vào của khối cộng vào đầu ra của hai Conv.
Đường tắt giữ nguyên đầu vào, trừ ở khối đầu của tầng 2, 3, 4: khối này thu nhỏ ảnh một nửa
và gấp đôi số kênh, nên đường tắt lấy trung bình 2x2 rồi qua Conv 1x1 + BN để khớp kích thước
trước khi cộng. Nhờ đường tắt, tín hiệu và gradient đi qua nhiều layer mà không yếu dần.
Ở các khối thu nhỏ này, đường chính cũng chống răng cưa: Conv bước 1, rồi BlurPool (làm mờ rồi
lấy mẫu bước 2) thay cho Conv bước 2, giống cách MaxBlurPool chống răng cưa ở stem và Model 1.

Cấu hình mặc định có bố cục ResNet-10 (1 khối mỗi tầng, 96 → 768 kênh, 11,1 triệu tham số),
được chọn trên val. Làm mạng sâu hơn không giúp khi thử với 64 kênh, ảnh 128 × 128 và
200 epoch: ResNet-18 (2 khối mỗi tầng) ngang ResNet-10. Thêm stochastic depth 0,1 (lúc train,
ngẫu nhiên bỏ đường chính của một số khối) tăng val khoảng 1,2 điểm. Chống răng cưa ở khối thu
nhỏ cùng với độ rộng 96 kênh (thay vì 64) tăng thêm khoảng 2 điểm nữa (6 seed, seed nào cũng
tăng); chỉ chống răng cưa thì khoảng 1,2 điểm. Cách train cũng được chọn trên val: ảnh train
160 × 160 và 450 epoch, thay vì 128 × 128 và 200 epoch như Model 1, tăng val thêm khoảng
2,8 điểm (6 seed, seed nào cũng tăng). Ở ảnh 128 × 128 và 200 epoch, các kiến trúc khác trong
slide môn học (VGG, GoogLeNet, DenseNet, AlexNet, đầu kiểu NiN), train từ đầu, đều kém
ResNet-10 này (xem README).

Model 2 dùng chung với Model 1 (src/train_simple_cnn.py) cách chia train/val/test, cách đọc
ảnh, augmentation, cách đánh giá (TTA, ảnh đánh giá lớn hơn ảnh train 1,25 lần) và công thức
train (AdamW + OneCycle, label smoothing). Hai model khác nhau ở kiến trúc (kể cả stochastic
depth, vốn chỉ dùng được cho khối residual) và ở kích thước ảnh, số epoch: Model 2 train ở
160 × 160 trong 450 epoch (chọn trên val), Model 1 ở 128 × 128 trong 200 epoch (chưa thử
Model 1 với ảnh 160 × 160 và 450 epoch). Khi train cùng cách (128 × 128, 200 epoch), Model 2
hơn Model 1 khoảng 2,4 điểm val; ảnh lớn hơn và train lâu hơn thêm khoảng 2,8 điểm. Một lần
chạy đối chứng cho thấy train lâu hơn cũng giúp Model 1 (300 epoch: +1 điểm val).

Thử cấu hình (chỉ báo val). Đặt --checkpoint riêng cho mỗi lần thử: mặc định mọi lần chạy
cùng seed đều ghi vào checkpoints/deep_cnn_seed<seed>.pt và .history.json, nên lần thử chạy
sau lần chạy cuối sẽ xoá model và kết quả test của lần đó:
    python src/train_deep_cnn.py --epochs 50 --blocks 2,2,2,2 --checkpoint checkpoints/deep_cnn_r18.pt
Chạy cấu hình cuối và đánh giá test:
    python src/train_deep_cnn.py --eval_test 1 --seed 42
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

from train_simple_cnn import (
    ROOT,
    GroceryImages,
    accuracy,
    build_transforms,
    choose_amp_dtype,
    cpu_state,
    load_eval_set,
    load_split,
    predict,
)

# 1. Cấu hình mặc định; đổi bất kỳ khoá nào bằng --ten_khoa gia_tri trên dòng lệnh
CONFIG = {
    "image_size": 160,  # lúc train, ảnh được cắt/thu về image_size x image_size (Model 1: 128)
    "width": 96,  # số kênh của tầng 1; mỗi tầng sau gấp đôi (96, 192, 384, 768)
    "blocks": "1,1,1,1",  # số khối residual của 4 tầng: 1,1,1,1 = bố cục ResNet-10, 2,2,2,2 = ResNet-18
    "dropout": 0.3,  # dropout trước lớp Linear cuối
    "drop_path": 0.1,  # stochastic depth: xác suất bỏ đường chính ở khối cuối, tăng dần từ 0 ở khối đầu
    "aa": 1,  # 1 = khối thu nhỏ của tầng 2-4 chống răng cưa: Conv bước 1 → BN → ReLU → BlurPool bước 2
    "epochs": 450,  # Model 1: 200; với Model 2, 450 epoch cho val cao hơn 200 và 300
    "batch_size": 64,
    "lr": 2e-3,  # learning rate cao nhất của lịch OneCycle
    "weight_decay": 0.05,  # weight decay của AdamW
    "label_smoothing": 0.1,
    "crop_scale": 0.35,  # RandomResizedCrop lấy ngẫu nhiên 35–100% diện tích ảnh
    "trivial_augment": 1,  # 1 = bật TrivialAugmentWide
    "random_erasing": 0.25,  # xác suất xoá một vùng chữ nhật ngẫu nhiên trong ảnh
    "eval_scale": 1.25,  # val/test dùng ảnh lớn hơn lúc train 1.25 lần (160 → 200)
    "tta": 1,  # 1 = khi đánh giá, lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang
    "seed": 42,
    "num_workers": 8,  # số tiến trình đọc ảnh song song
    "eval_test": 0,  # 1 = đánh giá test set ở cuối; chỉ bật cho cấu hình cuối cùng đã chọn bằng val
    "data_dir": str(ROOT / "data"),
    "split_dir": str(ROOT / "splits"),
    "checkpoint": "",  # để trống = checkpoints/deep_cnn_seed<seed>.pt, mỗi seed một file
}


def parse_config():
    parser = argparse.ArgumentParser(description="Train Deep CNN (ResNet)")
    for key, value in CONFIG.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    return vars(parser.parse_args())


# 2. Model
def conv_bn_relu(in_channels, out_channels, stride=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class BlurPool(nn.Module):
    """Làm mờ bằng bộ lọc cố định [1,2,1]x[1,2,1]/16 rồi lấy mẫu bước 2 (Zhang 2019, "BlurPool").

    Không có tham số học. Lọc bớt chi tiết quá mịn trước khi chỉ giữ 1/4 số điểm, nên dự đoán ít đổi
    khi vật thể dịch đi vài pixel (giống MaxBlurPool ở stem). Cạnh n cho ra ceil(n / 2), khớp với
    đường tắt AvgPool2d(2, ceil_mode=True).
    """

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :] / 16
        # persistent=False: bộ lọc cố định, không lưu vào checkpoint
        self.register_buffer("kernel", k.expand(channels, 1, 3, 3).clone(), persistent=False)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel.to(x.dtype), stride=2, groups=x.shape[1])


class ResidualBlock(nn.Module):
    """Khối residual: đầu ra = ReLU(nhánh chính(x) + đường tắt(x)).

    Nhánh chính là Conv-BN-ReLU-Conv-BN. Đường tắt giữ nguyên x; riêng khối đầu của tầng
    2, 3, 4 (thu nhỏ ảnh và tăng số kênh) thì đường tắt lấy trung bình 2x2 để thu nhỏ và
    dùng Conv 1x1 để đổi số kênh, cho khớp kích thước với nhánh chính trước khi cộng; nhánh
    chính của khối này thu nhỏ bằng BlurPool sau Conv-BN-ReLU đầu tiên (khi aa = 1).
    """

    def __init__(self, in_channels, out_channels, stride, drop_path=0.0, aa=1):
        super().__init__()
        self.drop_path = drop_path
        if stride > 1 and aa:
            # chống răng cưa: Conv bước 1 (giữ cỡ ảnh) → BN → ReLU → BlurPool thu nhỏ một nửa,
            # thay cho Conv bước 2 vốn chỉ tính đầu ra ở một trong mỗi 2 x 2 vị trí mà không làm mờ trước
            first = [*conv_bn_relu(in_channels, out_channels), BlurPool(out_channels)]
        else:
            first = conv_bn_relu(in_channels, out_channels, stride)
        self.body = nn.Sequential(
            *first,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # γ của BatchNorm cuối bằng 0: lúc mới khởi tạo nhánh chính ra 0, khối chỉ truyền
        # đường tắt, nên mạng sâu bắt đầu như một mạng nông rồi mới dần dùng thêm các khối
        nn.init.zeros_(self.body[-1].weight)
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # ceil_mode: với cạnh lẻ (ví dụ 5) vẫn ra 3 như nhánh chính (BlurPool, hoặc Conv bước 2
            # khi aa = 0), không ra 2
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.training and self.drop_path > 0:
            # stochastic depth: với mỗi ảnh trong batch, bỏ hẳn đường chính với xác suất drop_path
            # (khối chỉ còn đường tắt); ảnh giữ lại thì nhân 1 / (1 - drop_path) để trung bình không đổi
            keep = (torch.rand(out.shape[0], 1, 1, 1, device=out.device) >= self.drop_path).to(out.dtype)
            out = out * keep / (1 - self.drop_path)
        return self.relu(out + self.shortcut(x))


def build_model(num_classes, width=96, blocks=(1, 1, 1, 1), dropout=0.3, drop_path=0.1, aa=1):
    model = nn.Sequential()
    # stem: 3 Conv 3x3 (Conv đầu bước 2: 160 → 80), rồi MaxBlurPool như Model 1 (80 → 40)
    model.add_module("stem", nn.Sequential(
        *conv_bn_relu(3, width // 2, stride=2),
        *conv_bn_relu(width // 2, width // 2),
        *conv_bn_relu(width // 2, width),
        nn.ZeroPad2d((0, 1, 0, 1)), nn.MaxPool2d(2, stride=1), nn.AvgPool2d(2),
    ))
    in_channels = width
    total, k = sum(blocks), 0
    for i, num_blocks in enumerate(blocks):
        out_channels = width * 2**i
        stage = []
        for j in range(num_blocks):
            stride = 2 if i > 0 and j == 0 else 1  # khối đầu của tầng 2, 3, 4 thu nhỏ một nửa
            # xác suất bỏ tăng tuyến tính: 0 ở khối đầu, drop_path ở khối cuối
            stage.append(ResidualBlock(in_channels, out_channels, stride, drop_path * k / max(1, total - 1), aa))
            in_channels = out_channels
            k += 1
        model.add_module(f"stage{i + 1}", nn.Sequential(*stage))
    model.add_module("head", nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # mỗi feature map → 1 số
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_channels, num_classes),
    ))
    return model


def parse_blocks(text):
    blocks = tuple(int(v) for v in text.split(","))
    if len(blocks) != 4 or min(blocks) < 1:
        raise ValueError(f"blocks phải gồm 4 số nguyên dương, ví dụ 2,2,2,2; nhận được {text!r}")
    return blocks


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows: tránh lỗi khi in tiếng Việt ra file/pipe
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cfg = parse_config()
    if not cfg["checkpoint"]:
        cfg["checkpoint"] = str(ROOT / "checkpoints" / f"deep_cnn_seed{cfg['seed']}.pt")
    blocks = parse_blocks(cfg["blocks"])
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
    amp_dtype = choose_amp_dtype(device)
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
    model = build_model(num_classes, cfg["width"], blocks, cfg["dropout"], cfg["drop_path"], cfg["aa"]).to(device)
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")

    loss_function = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_loader)
    )

    # 3. Huấn luyện, lưu model tốt nhất theo val accuracy
    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.with_suffix(".history.json")
    history = []
    best_val, best_epoch, best_state = -1.0, 0, None
    train_iter = iter(train_loader)
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

    # 4. Đánh giá một lần duy nhất trên test set bằng model tốt nhất theo val
    print("\n=== Kết quả trên test set ===")
    best_model = build_model(num_classes, cfg["width"], blocks, cfg["dropout"], cfg["drop_path"], cfg["aa"])
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

    test_result = {"acc": test_acc, "acc_no_tta": test_acc_plain,
                   "mean_per_class_acc": mean_per_class, "per_class_acc": per_class}
    history_path.write_text(
        json.dumps({"config": cfg, "epochs": history, "test": test_result}, indent=1)
    )
    print(f"Kết quả test đã lưu vào: {history_path}")


if __name__ == "__main__":  # bắt buộc khi num_workers > 0 trên macOS/Windows
    main()
