# Freiburg Groceries Classification

Dự án so sánh ba mô hình phân loại ảnh trên bộ dữ liệu Freiburg Groceries gồm 25 lớp:

- `SimpleCNN`: CNN cơ bản sử dụng convolution, pooling và fully connected layers.
- `DeepCNN`: CNN phức hợp hơn với nhiều convolution blocks, pooling, regularization và fully connected layers.
- `ResNet50`: CNN sử dụng transfer learning/fine-tuning từ ImageNet pretrained weights.

## 1. Chuẩn bị môi trường

Từ thư mục gốc của repository:

```bash
git pull origin main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset cần được đặt trong thư mục `data/`, mỗi thư mục con tương ứng với một class. Dataset và các file trọng số lớn không được commit lên GitHub.

## 2. Vai trò các file chính

- `src/config.py`: seed, batch size, số epoch, learning rate, image size và paths.
- `src/dataset.py`: đọc ảnh, chia train/validation/test theo tỷ lệ 70/15/15 với seed 42, preprocessing và augmentation.
- `src/model.py`: định nghĩa `SimpleCNN`, `DeepCNN` và `ResNet50`.
- `src/utils.py`: class weights, metrics, checkpoint và classification report.
- `src/train.py`: entrypoint huấn luyện model.
- `src/evaluate.py`: đánh giá model trên test set.
- `train_baseline.py`: script baseline lịch sử được giữ lại để tham khảo.

Khi cần thay đổi siêu tham số, chỉnh trong `src/config.py`. Khi cần thay đổi cấu trúc mạng, chỉnh trong `src/model.py`. Khi cần thay đổi quy trình đọc, chia hoặc tiền xử lý dữ liệu, chỉnh trong `src/dataset.py`. Khi cần thay đổi metric, chỉnh trong `src/utils.py`.

## 3. Huấn luyện model

### SimpleCNN

```bash
python -m src.train \
  --model simple_cnn \
  --epochs 20 \
  --run-name simple_cnn_v1
```

### DeepCNN

```bash
python -m src.train \
  --model deep_cnn \
  --epochs 20 \
  --run-name deep_cnn_v1
```

### ResNet50 Fine-Tuning

```bash
python -m src.train \
  --model resnet50 \
  --epochs 10 \
  --weights default \
  --run-name resnet50_v1
```

`--weights default` sử dụng ResNet50 pretrained trên ImageNet. Lần chạy đầu tiên cần internet để tải weights.

### Versioning checkpoint

Mỗi lần train nên đặt một `--run-name` mới để không ghi đè kết quả cũ. Ví dụ:

```bash
python -m src.train --model simple_cnn --epochs 20 --run-name simple_cnn_v1
python -m src.train --model simple_cnn --epochs 20 --run-name simple_cnn_v2
python -m src.train --model resnet50 --epochs 10 --weights default --run-name resnet50_v1
```

Các file tương ứng:

```text
checkpoints/simple_cnn_v1_best.pt
checkpoints/simple_cnn_v2_best.pt
checkpoints/resnet50_v1_best.pt
artifacts/simple_cnn_v1_history.json
artifacts/simple_cnn_v2_history.json
artifacts/resnet50_v1_history.json
```

Nếu không truyền `--run-name`, tên model sẽ được dùng làm tên mặc định và lần chạy sau có thể ghi đè checkpoint cũ.

## 4. Đánh giá trên test set

Đánh giá đúng phiên bản đã train bằng cách dùng cùng `--model` và `--run-name`:

```bash
python -m src.evaluate \
  --model simple_cnn \
  --run-name simple_cnn_v1

python -m src.evaluate \
  --model deep_cnn \
  --run-name deep_cnn_v1

python -m src.evaluate \
  --model resnet50 \
  --run-name resnet50_v1
```

Metrics được lưu trong `artifacts/`:

```text
artifacts/simple_cnn_v1_metrics.json
artifacts/deep_cnn_v1_metrics.json
artifacts/resnet50_v1_metrics.json
```

Mỗi report gồm accuracy, macro-F1, weighted-F1, precision/recall theo class và confusion matrix.

## 5. Chạy notebooks

Khởi động Jupyter từ thư mục gốc:

```bash
jupyter notebook
```

Chạy lần lượt ba notebook:

1. `notebooks/eda_groceries.ipynb`: EDA chi tiết về chất lượng ảnh, kích thước ảnh và phân bố class.
2. `notebooks/01_eda_dataset.ipynb`: kiểm tra class mapping và split manifest dùng chung.
3. `notebooks/02_model_comparison.ipynb`: vẽ loss, macro-F1 và bảng so sánh metrics của ba model.

Notebook `02_model_comparison.ipynb` cần các file history/metrics trong `artifacts/`. Vì vậy hãy train và evaluate các model trước khi mở notebook so sánh.

## 6. Preprocessing và đánh giá

- Ảnh được chuyển sang RGB và resize về `256x256`.
- Augmentation chỉ áp dụng cho tập train.
- Validation/test sử dụng transform deterministic.
- SimpleCNN và DeepCNN sử dụng adaptive average pooling để hạn chế số tham số.
- ResNet50 sử dụng ImageNet normalization.
- Validation được dùng để chọn checkpoint tốt nhất theo macro-F1.
- Test set chỉ dùng ở bước evaluate cuối cùng.

## 7. Kết quả và file sinh ra

Các thư mục sau được ignore để tránh commit dữ liệu và artifact lớn:

```text
data/
checkpoints/
artifacts/
```

Khi clone repository mới, cần chuẩn bị dataset trước và chạy lại quá trình train để tạo checkpoint/metrics tương ứng.