"""Simple CNN baseline for the Freiburg Groceries dataset."""

import random  # xáo trộn ảnh trước khi chia tập
from pathlib import Path  # tạo và ghép đường dẫn file/thư mục an toàn

import numpy as np  # xử lý mảng ảnh và xây dựng/huấn luyện mạng
import torch  # xử lý mảng ảnh và xây dựng/huấn luyện mạng
from PIL import Image  # từ Pillow: mở, đổi màu và resize ảnh
from torch import nn  # các lớp mạng, loss và hàm kích hoạt của PyTorch
from torch.utils.data import DataLoader, Dataset  # đóng gói ảnh thành các batch để đưa vào model


# 1. Cấu hình
DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGE_SIZE = 256  # resize mỗi ảnh thành 256×256 để giảm chi phí tính toán
BATCH_SIZE = 32  # model xử lý 32 ảnh mỗi lượt
EPOCHS = 10  # đi qua toàn bộ tập train 10 lần
SEED = 42  # cố định seed để việc chia ảnh và khởi tạo model có thể lặp lại

random.seed(SEED)
torch.manual_seed(SEED)


# 2. Chia ảnh train / validation / test theo từng lớp
class_names = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())  # lấy tên thư mục lớp trong data, sắp xếp để nhãn luôn có thứ tự ổn định.
train_samples = []
val_samples = []
test_samples = []
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for label, class_name in enumerate(class_names):  # duyệt từng lớp và gán cho lớp một số nguyên, ví dụ lớp đầu là 0
    class_dir = DATA_DIR / class_name  # đường dẫn tới thư mục chứa ảnh của lớp hiện tại
    image_paths = sorted(  # sắp xếp để cùng seed luôn tạo cùng split
        path
        for path in class_dir.iterdir()  # duyệt từng file trong thư mục lớp hiện tại
        if path.is_file() and path.suffix.lower() in image_extensions  # chỉ lấy file ảnh có đuôi trong image_extensions, lower cho chắc là không phân biệt hoa thường
    )
    random.shuffle(image_paths)  # xáo trộn ảnh trong từng lớp trước khi chia tập

    train_end = int(0.70 * len(image_paths))  # lấy 70% ảnh lớp đó cho train
    val_end = int(0.85 * len(image_paths))  # lấy 15% ảnh lớp đó cho validation
    
    train_samples.extend(
        (path, label) for path in image_paths[:train_end]  # thêm từng ảnh cùng nhãn vào train_samples
    )
    val_samples.extend(
        (path, label) for path in image_paths[train_end:val_end]
    )
    test_samples.extend(
        (path, label) for path in image_paths[val_end:]
    )
#  Ba lệnh extend(...): thêm từng ảnh cùng nhãn vào train, validation và test. Vì chia riêng bên trong mỗi lớp nên cả ba tập đều có ảnh từ các lớp.
if not class_names or not train_samples or not val_samples or not test_samples:
    raise ValueError(f"Không tìm thấy đủ ảnh trong thư mục: {DATA_DIR}")
print(f"Số lớp: {len(class_names)}")
print(
    f"Số ảnh: train={len(train_samples)}, "
    f"validation={len(val_samples)}, test={len(test_samples)}"
)


# 3. Tiền xử lý ảnh: RGB, resize 64x64, chuẩn hóa pixel về [-1, 1]
class GroceryImages(Dataset):
    def __init__(self, samples, augment=False):  # lưu danh sách ảnh và có augment hay không
        self.samples = samples
        self.augment = augment

    def __len__(self):  # trả về số lượng ảnh trong tập
        return len(self.samples)

    def __getitem__(self, index):  # trả về cặp ảnh và nhãn tại index, lấy một ảnh theo vị trí; đây là phần được gọi mỗi khi DataLoader cần ảnh để huấn luyện hoặc kiểm tra
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.augment:
                width, height = image.size
                crop_size = int(min(width, height) * random.uniform(0.9, 1.0))
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                image = image.crop(
                    (left, top, left + crop_size, top + crop_size)
                )
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )

        image = np.asarray(image, dtype=np.float32) / 255.0  # chuyển pixel từ số nguyên 0–255 thành số thực 0–1
        image = (image - 0.5) / 0.5  # chuẩn hóa pixel về [-1, 1] để mạng học nhanh hơn
        image = torch.from_numpy(image).permute(2, 0, 1)  # chuyển ảnh từ (H, W, C) sang (C, H, W) mà PyTorch mong đợi, chính là định dạng Conv2D mong đợi
        return image, label  # trả ảnh đã xử lý và nhãn đúng của nó


train_loader = DataLoader(
    GroceryImages(train_samples, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    GroceryImages(val_samples), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    GroceryImages(test_samples), batch_size=BATCH_SIZE, shuffle=False
)


# 4. Model 
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),  # nhận ảnh RGB có 3 kênh và học 32 bộ lọc để tìm đặc trưng như cạnh, màu và hoa văn. mỗi bộ lọc nhìn vùng ảnh nhỏ (3x3), và có padding = 1 giữ nguyên chiều rộng/cao sau convolution để không mất thông tin
    nn.ReLU(),  # hàm kích hoạt phi tuyến ReLU giúp mạng học các đặc trưng phức tạp hơn, âm = 0, dương giữ nguyên giá trị
    nn.MaxPool2d(kernel_size=2),  # giảm một nửa chiều rộng/cao của ảnh, giữ lại các đặc trưng quan trọng nhất, giúp giảm kích thước ảnh và giảm số lượng tính toán, tức là ảnh chỉ còn 1/4 số pixel so với trước khi vào MaxPool2d
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.AdaptiveAvgPool2d((1, 1)),  # gộp mỗi feature map thành một giá trị
    nn.Flatten(),
    nn.Linear(32, len(class_names)),  # một điểm số cho mỗi lớp; CrossEntropyLoss nhận logits trực tiếp
)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)

# Dùng class weights vì số ảnh giữa các lớp không cân bằng hoàn toàn
train_labels = torch.tensor([label for _, label in train_samples])
class_counts = torch.bincount(train_labels, minlength=len(class_names)).float()
class_weights = len(train_samples) / (len(class_names) * class_counts)
class_weights = class_weights.to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)  # CrossEntropyLoss tích hợp softmax, nhận logits trực tiếp.
metric_loss_function = nn.CrossEntropyLoss(
    weight=class_weights, reduction="sum"
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # dùng Adam vì nó tự động điều chỉnh learning rate cho từng tham số, giúp mạng hội tụ nhanh hơn so với SGD truyền thống.

print(f"Thiết bị: {device}")
print(model)


# 5. Hàm đánh giá dùng cho validation và test
def evaluate(data_loader):
    model.eval()
    total_loss = 0.0
    total_loss_weight = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():  # không cần tính đạo số khi đánh giá, chỉ cần tính loss và accuracy.
        for images, labels in data_loader:  # lặp qua từng batch ảnh và nhãn trong data_loader
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            total_loss += metric_loss_function(predictions, labels).item()
            total_loss_weight += class_weights[labels].sum().item()
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_images += labels.size(0)

    return total_loss / total_loss_weight, total_correct / total_images


# 6. Huấn luyện và lưu model tốt nhất theo validation accuracy
checkpoint_path = (  # nơi lưu trọng số model tốt nhất theo validation accuracy
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "simple_cnn_baseline_v3.pt"
)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)   # tự tạo thư mục nếu chưa tồn tại
best_val_accuracy = 0.0  # giữ accuracy validation cao nhất đã đạt

for epoch in range(EPOCHS):  # lặp qua từng epoch
    model.train()  # chuyển sang chế huấn luyện
    train_loss = 0.0
    train_loss_weight = 0.0
    train_correct = 0
    train_images = 0

    for images, labels in train_loader:  # gọi từng batch ảnh và nhãn trong train_loader
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # xóa gradient cũ trước khi tính gradient mới, vì PyTorch cộng dồn gradient theo mặc định.
        predictions = model(images)  # chạy forward pass để lấy điểm số dự đoán
        loss = loss_function(predictions, labels)
        loss.backward()  # tính gradient cho các trọng số model theo loss function
        optimizer.step()  # cập nhật trọng số dựa trên gradient và learning rate

        train_loss += metric_loss_function(predictions.detach(), labels).item()
        train_loss_weight += class_weights[labels].sum().item()
        train_correct += (predictions.argmax(dim=1) == labels).sum().item()
        train_images += labels.size(0)

    val_loss, val_accuracy = evaluate(val_loader)  # đánh giá sau mỗi epoch; validation dùng để chọn model, không dùng để cập nhật trọng số
    train_accuracy = train_correct / train_images
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train loss={train_loss / train_loss_weight:.4f}, "
        f"accuracy={train_accuracy:.3f} | "
        f"val loss={val_loss:.4f}, accuracy={val_accuracy:.3f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), checkpoint_path)


# 7. Đánh giá cuối cùng trên test set
model.load_state_dict(torch.load(checkpoint_path, map_location=device))  # nạp lại trọng số tốt nhất theo validation, thay vì mặc định dùng trọng số epoch cuối cùng
test_loss, test_accuracy = evaluate(test_loader)  # đánh giá trên test set
print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.3f}")
print(f"Model đã lưu tại: {checkpoint_path}")
