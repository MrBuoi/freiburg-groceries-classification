# Freiburg Groceries Classification (Deep Learning Comparison)

Dự án so sánh hiệu quả giữa 3 kiến trúc Deep Learning từ đơn giản đến phức tạp trên bộ dữ liệu Freiburg Groceries (25 nhóm sản phẩm):
1. **Model 1**: Simple CNN (Baseline), file `youonlycodeonce/train_simple_cnn.py`
2. **Model 2**: Deep Convolutional Neural Network (Deep CNN), file `youonlycodeonce/train_deep_cnn.py`
3. **Model 3**: Transfer Learning (YOLOv5 Classification Backbone)

## Cấu trúc thư mục

```text
data/                    ảnh của bộ dữ liệu (không commit, xem mục tải dữ liệu)
notebooks/               notebook EDA và so sánh model của pipeline trong src/
src/                     pipeline chung của nhóm (python -m src.train): SimpleCNN, DeepCNN, ResNet50
youonlycodeonce/
  train_simple_cnn.py    Model 1
  train_deep_cnn.py      Model 2
  make_splits.py         tạo cách chia train/val/test theo nhóm
  splits/                train.txt, val.txt, test.txt, groups.csv
checkpoints/, artifacts/ model đã train, history và metrics (không commit)
```

Pipeline trong `src/` và hai model trong `youonlycodeonce/` chia dữ liệu khác nhau, nên kết quả của hai nơi không so trực tiếp được với nhau (xem "Lưu ý khi so sánh các model").

## Hướng dẫn cài đặt & Chạy dự án cho nhóm

### 1. Clone repository

```bash
git clone https://github.com/MrBuoi/freiburg-groceries-classification.git
cd freiburg-groceries-classification
```

### 2. Tạo môi trường Python

Dùng Python 3.10–3.13, khuyên dùng 3.12 (TensorFlow chưa hỗ trợ Python 3.14).

```bash
conda create -n freiburg python=3.12 -y
conda activate freiburg
```

**Windows/Linux có GPU NVIDIA:** cài PyTorch bản CUDA trước. Nếu bỏ qua bước này, pip sẽ cài bản chỉ chạy CPU và train mất nhiều giờ.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"   # phải in ra True
```

- Bản `cu126` chạy được trên hầu hết GPU NVIDIA. Chỉ đổi sang `cu130` khi driver từ 580 trở lên **và** GPU từ GTX 16xx / RTX 20xx trở lên. GPU GTX 10xx, MX hoặc cũ hơn phải dùng `cu126`.
- Nếu in ra `False`: chạy `pip uninstall -y torch torchvision` rồi cài lại bằng lệnh trên. Vẫn `False` thì cập nhật driver NVIDIA (kiểm tra bằng `nvidia-smi`).

**macOS:** cần máy chip Apple Silicon (M1 trở lên) và Python bản arm64, tức `python -c "import platform; print(platform.machine())"` phải in ra `arm64`. PyTorch mới không còn bản cho Mac chip Intel, nên với máy đó hãy dùng Google Colab.

macOS và máy chỉ có CPU bỏ qua bước cài PyTorch bản CUDA. Sau đó cài các thư viện còn lại:

```bash
pip install -r requirements.txt
```

Trong VSCode, chọn interpreter của môi trường này (Ctrl/Cmd+Shift+P → "Python: Select Interpreter" → `freiburg`), nếu không VSCode sẽ báo thiếu thư viện.

### 3. Tải dữ liệu (~540 MB)

Giải nén thẳng 25 thư mục lớp vào `data/`. File nén có sẵn một thư mục `images/` bên trong, `--strip-components=1` bỏ thư mục này đi:

```bash
curl -L -o freiburg.tar.gz http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz
tar -xzf freiburg.tar.gz -C data --strip-components=1
rm freiburg.tar.gz
```

Trên Windows PowerShell, gõ `curl.exe` thay cho `curl`. Sau bước này `data/` phải có đúng 25 thư mục (`BEANS/` … `WATER/`) với tổng cộng 4.947 ảnh.

## Pipeline chung trong `src/` (SimpleCNN, DeepCNN, ResNet50)

Phần này do nhóm viết trên nhánh `main`. Cài đặt môi trường như mục trên (hoặc dùng venv: `python -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`).

### Vai trò các file chính

- `src/config.py`: seed, batch size, số epoch, learning rate, image size và paths.
- `src/dataset.py`: đọc ảnh, chia train/validation/test theo tỷ lệ 70/15/15 với seed 42, preprocessing và augmentation.
- `src/model.py`: định nghĩa `SimpleCNN`, `DeepCNN` và `ResNet50`.
- `src/utils.py`: class weights, metrics, checkpoint và classification report.
- `src/train.py`: entrypoint huấn luyện model.
- `src/evaluate.py`: đánh giá model trên test set.

Khi cần thay đổi siêu tham số, chỉnh trong `src/config.py`. Khi cần thay đổi cấu trúc mạng, chỉnh trong `src/model.py`. Khi cần thay đổi quy trình đọc, chia hoặc tiền xử lý dữ liệu, chỉnh trong `src/dataset.py`. Khi cần thay đổi metric, chỉnh trong `src/utils.py`.

### Huấn luyện model

#### SimpleCNN

```bash
python -m src.train \
  --model simple_cnn \
  --epochs 20 \
  --run-name simple_cnn_v1
```

#### DeepCNN

```bash
python -m src.train \
  --model deep_cnn \
  --epochs 20 \
  --run-name deep_cnn_v1
```

#### ResNet50 Fine-Tuning

```bash
python -m src.train \
  --model resnet50 \
  --epochs 10 \
  --weights default \
  --run-name resnet50_v1
```

`--weights default` sử dụng ResNet50 pretrained trên ImageNet. Lần chạy đầu tiên cần internet để tải weights.

#### Versioning checkpoint

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

### Đánh giá trên test set

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

### Chạy notebooks

Khởi động Jupyter từ thư mục gốc:

```bash
jupyter notebook
```

Chạy lần lượt ba notebook:

1. `notebooks/eda_groceries.ipynb`: EDA chi tiết về chất lượng ảnh, kích thước ảnh và phân bố class.
2. `notebooks/01_eda_dataset.ipynb`: kiểm tra class mapping và split manifest dùng chung.
3. `notebooks/02_model_comparison.ipynb`: vẽ loss, macro-F1 và bảng so sánh metrics của ba model.

Notebook `02_model_comparison.ipynb` cần các file history/metrics trong `artifacts/`. Vì vậy hãy train và evaluate các model trước khi mở notebook so sánh.

### Preprocessing và đánh giá

- Ảnh được chuyển sang RGB và resize về `256x256`.
- Augmentation chỉ áp dụng cho tập train.
- Validation/test sử dụng transform deterministic.
- SimpleCNN và DeepCNN sử dụng adaptive average pooling để hạn chế số tham số.
- ResNet50 sử dụng ImageNet normalization.
- Validation được dùng để chọn checkpoint tốt nhất theo macro-F1.
- Test set chỉ dùng ở bước evaluate cuối cùng.

### Kết quả và file sinh ra

Các thư mục sau được ignore để tránh commit dữ liệu và artifact lớn:

```text
data/
checkpoints/
artifacts/
```

Khi clone repository mới, cần chuẩn bị dataset trước và chạy lại quá trình train để tạo checkpoint/metrics tương ứng.

## Cách chia train/val/test theo nhóm (`youonlycodeonce/splits/`)

Ba file `youonlycodeonce/splits/train.txt`, `val.txt`, `test.txt` (3.436 / 760 / 751 ảnh) là cách chia mà Model 1 và Model 2 dùng. Model khác, kể cả pipeline trong `src/`, muốn được chấm trên cùng một test set thì cần đọc cùng các file này. Mỗi dòng là một ảnh, dạng `LỚP/TÊN_FILE.png`.

Bộ dữ liệu có nhiều ảnh chụp lại cùng một kệ hàng từ góc hơi khác. Nếu chia ngẫu nhiên theo từng ảnh, khoảng 20% ảnh test có một ảnh "anh em" cùng cảnh nằm trong train, khiến accuracy cao hơn thực tế khoảng 2 điểm. Vì vậy các ảnh giống nhau được gom thành nhóm rồi chia theo nhóm, 70/15/15 trong từng lớp.

Ngoài ảnh cùng cảnh, cách gom này còn nối các ảnh có cùng mẫu bao bì (cùng logo, cùng hãng) dù chụp ở cảnh khác. Vì vậy cách chia này chặt hơn việc chỉ tách theo cảnh:
- một số mẫu bao bì ở val/test không có trong train;
- ở vài lớp (TEA, HONEY, CAKE, RICE, CORN), gần một nửa trở lên số ảnh val hoặc test thuộc cùng một nhóm, nên accuracy của riêng các lớp này dao động nhiều.

Cách gom nhóm được mô tả trong `youonlycodeonce/make_splits.py`.

- Đọc split trong code Python (từ thư mục nào cũng được, miễn là `sys.path` trỏ tới `youonlycodeonce/`):
  ```python
  import sys; sys.path.insert(0, "youonlycodeonce")   # notebook trong thư mục notebooks/ thì dùng "../youonlycodeonce"
  from train_simple_cnn import load_split

  class_names, train, val, test = load_split()  # list các cặp (đường dẫn ảnh, nhãn)
  ```
- Không sửa tay các file trong `youonlycodeonce/splits/`. Chỉ tạo lại bằng `python youonlycodeonce/make_splits.py` (khoảng 1,5 phút) khi dữ liệu thay đổi. Script tự dừng, không ghi đè, nếu `data/` chưa đủ 25 lớp và 4.947 ảnh.
- Chỉ dùng val để chọn cấu hình. Test chỉ đánh giá một lần, cho cấu hình cuối cùng. Riêng Model 2 đã xem test 3 lần (bản đầu, bản 2, bản hiện tại), xem mục Kết quả của Model 2.

## Model 1: Simple CNN (baseline)

`youonlycodeonce/train_simple_cnn.py` train từ đầu một CNN đơn giản, không dùng trọng số pretrained:

- **Kiến trúc:** 6 khối Conv 3×3 → BatchNorm → ReLU → MaxBlurPool, rồi Global Average Pooling, Dropout và một lớp Linear (6,3 triệu tham số). MaxBlurPool là max pooling chống răng cưa: lấy max 2×2 nhưng trượt từng pixel, rồi lấy trung bình 2×2 khi thu nhỏ, nên dự đoán ít bị đổi khi vật thể dịch đi vài pixel.
- **Huấn luyện:**
  - ảnh 128×128, augmentation mạnh (RandomResizedCrop, lật ngang, TrivialAugmentWide, RandomErasing), label smoothing;
  - AdamW với lịch learning rate OneCycle trong 200 epoch;
  - lưu model của epoch có val accuracy cao nhất.
- **Đánh giá:** ảnh được phóng lên 160×160, lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang (TTA).

Thử cấu hình (chỉ báo val accuracy). Nhớ đặt `--checkpoint` riêng cho mỗi lần thử: mặc định mọi lần chạy cùng seed đều ghi vào `checkpoints/simple_cnn_seed<seed>.pt` và `checkpoints/simple_cnn_seed<seed>.history.json`, nên lần thử chạy sau sẽ xoá model và kết quả test của lần chạy cuối.

```bash
python youonlycodeonce/train_simple_cnn.py --epochs 50 --num_blocks 5 --checkpoint checkpoints/simple_cnn_try.pt
```

Chạy cấu hình cuối và đánh giá trên test set, mỗi seed một lần:

```bash
python youonlycodeonce/train_simple_cnn.py --eval_test 1 --seed 42
python youonlycodeonce/train_simple_cnn.py --eval_test 1 --seed 1
python youonlycodeonce/train_simple_cnn.py --eval_test 1 --seed 2
```

- Mọi tham số trong `CONFIG` đổi được bằng `--ten_tham_so gia_tri`.
- Mỗi seed lưu riêng hai file trong `checkpoints/`:
  - `simple_cnn_seed<seed>.pt`: trọng số;
  - `simple_cnn_seed<seed>.history.json`: loss và accuracy từng epoch (để vẽ biểu đồ), cùng kết quả test nếu chạy với `--eval_test 1`.
- Thời gian: khoảng 7 phút mỗi seed trên Apple M5 Max; mỗi lần chạy dùng khoảng 5 GB bộ nhớ.
- Xử lý sự cố:
  - Colab bản miễn phí (2 CPU): thêm `--num_workers 2`.
  - Windows báo lỗi `WinError 1455` hoặc "DataLoader worker exited unexpectedly": thêm `--num_workers 2`.
  - GPU báo "out of memory": thêm `--batch_size 32`. Kết quả sẽ khác một chút.

### Kết quả

Trung bình ± độ lệch chuẩn qua 3 seed (42, 1, 2), cấu hình mặc định:

| | Accuracy (%) |
|---|---|
| Val, có TTA (trung bình 10 epoch cuối) | 87,1 ± 0,1 |
| **Test, có TTA** | **85,0 ± 0,8** |
| Test, không TTA | 84,2 ± 0,8 |
| Test, trung bình accuracy của 25 lớp | 84,0 ± 0,9 |

Test accuracy của từng seed: 85,8 / 85,1 / 84,2. Test thấp hơn val khoảng 2 điểm, một phần vì cấu hình được chọn theo val.

- Các lớp khó nhất (trung bình 3 seed): FLOUR 59%, FISH 67%, VINEGAR 72%, OIL 74%, COFFEE 76%.
- Các lớp dễ nhất: WATER 94%, JUICE 92%, TEA 91%, CORN 91%, TOMATO_SAUCE 91%.
- Mỗi lớp chỉ có 15–56 ảnh test, nên accuracy của từng lớp dao động nhiều giữa các seed.

### Những gì đã thử

Cấu hình được chọn chỉ bằng val accuracy: có TTA, trung bình 10 epoch cuối, rồi trung bình 3 seed. Luật giữ hay bỏ được chốt trước khi chạy để không chọn nhầm theo nhiễu: một thay đổi riêng lẻ phải tăng ít nhất 1 điểm, còn SWA thêm vào MaxBlurPool phải tăng thêm ít nhất 0,5 điểm.

| Thay đổi so với cấu hình gốc (MaxPool thường) | Val accuracy (%) | Chênh lệch | Quyết định |
|---|---|---|---|
| Cấu hình gốc | 86,1 ± 1,5 | | |
| MaxBlurPool | 87,3 ± 1,1 | +1,2 | giữ |
| SWA: 50 epoch cuối giữ learning rate nhỏ cố định, lấy trung bình trọng số | 86,8 ± 1,4 | +0,7 | bỏ |
| MaxBlurPool + SWA | 87,5 ± 1,0 | +1,4 | bỏ, chỉ hơn MaxBlurPool 0,2 |
| Augmentation mạnh hơn (crop 25–100% diện tích, RandomErasing 0,5) | 86,3 ± 0,2 | +0,1 | bỏ |
| CutMix (một nửa số batch) | 85,9 ± 1,5 | −0,2 | bỏ |
| SAM (rho 0,05; thời gian train gấp đôi) | 85,9 ± 1,1 | −0,3 | bỏ |
| Bỏ pooling ở khối cuối | 85,8 ± 0,4 | −0,3 | bỏ |

SWA, CutMix, SAM và bỏ pooling ở khối cuối chạy bằng script thử nghiệm riêng, không có trong repo. Với SWA, val accuracy là của model trung bình. Ở ảnh 160×160, MaxPool thường bỏ mất hàng và cột cuối của feature map ở khối cuối, còn MaxBlurPool thì không. Vì vậy một phần mức tăng của MaxBlurPool có thể đến từ điểm này chứ không chỉ từ việc chống răng cưa.

## Model 2: Deep CNN (ResNet)

`youonlycodeonce/train_deep_cnn.py` train từ đầu một mạng kiểu ResNet, không dùng trọng số pretrained:

- **Kiến trúc:**
  - stem gồm 3 Conv 3×3 (Conv đầu bước 2) và MaxBlurPool, thu nhỏ ảnh 4 lần;
  - 4 tầng, mỗi tầng 1 khối residual (bố cục ResNet-10), số kênh 96 → 192 → 384 → 768;
  - Global Average Pooling, Dropout và một lớp Linear (11,1 triệu tham số).
- **Khối residual:** hai Conv 3×3 và một đường tắt cộng đầu vào vào đầu ra. Ở khối thu nhỏ ảnh (đầu tầng 2, 3, 4), đường tắt dùng AvgPool 2×2 + Conv 1×1, còn đường chính chống răng cưa: Conv 3×3 bước 1 → BatchNorm → ReLU → BlurPool (làm mờ bằng bộ lọc [1, 2, 1] × [1, 2, 1] / 16 rồi lấy mẫu bước 2) thay cho Conv bước 2. γ của BatchNorm cuối mỗi khối khởi tạo bằng 0.
- **Huấn luyện và đánh giá:**
  - dùng chung với Model 1 code đọc dữ liệu, augmentation và đánh giá (TTA, ảnh đánh giá lớn hơn ảnh train 1,25 lần), cùng công thức train: AdamW + OneCycle, label smoothing; như Model 1, lưu model của epoch có val accuracy cao nhất và dùng model đó để chấm test;
  - thêm stochastic depth 0,1: lúc train, ngẫu nhiên bỏ đường chính của một khối với xác suất tăng dần theo độ sâu;
  - ảnh train 160×160 (đánh giá ở 200×200) và 450 epoch, chọn trên val ở vòng 6–7 bên dưới. Model 1 train ở 128×128 (đánh giá ở 160×160) trong 200 epoch.

Thử cấu hình (chỉ báo val accuracy), nhớ đặt `--checkpoint` riêng để không ghi đè model và kết quả test của lần chạy cuối:

```bash
python youonlycodeonce/train_deep_cnn.py --epochs 50 --blocks 2,2,2,2 --checkpoint checkpoints/deep_cnn_r18.pt
```

Chạy cấu hình cuối và đánh giá trên test set, mỗi seed một lần:

```bash
python youonlycodeonce/train_deep_cnn.py --eval_test 1 --seed 42
python youonlycodeonce/train_deep_cnn.py --eval_test 1 --seed 1
python youonlycodeonce/train_deep_cnn.py --eval_test 1 --seed 2
```

- Mỗi seed lưu `checkpoints/deep_cnn_seed<seed>.pt` và `checkpoints/deep_cnn_seed<seed>.history.json` (kèm kết quả test).
- Thời gian: khoảng 38 phút mỗi seed trên Apple M5 Max khi chạy một mình (khoảng 70 phút nếu chạy 2 seed cùng lúc).
- Các bản trước của Model 2 chạy lại được bằng cách thêm các tham số dưới đây vào lệnh ở trên. Nhớ đặt `--checkpoint` riêng để không ghi đè model và kết quả test của bản hiện tại (tên file dưới đây là ví dụ cho seed 42; với seed khác, đổi số trong tên file cho khớp `--seed`):
  - bản 2 (96 kênh, chống răng cưa; ảnh 128×128, 200 epoch như Model 1): `--image_size 128 --epochs 200 --checkpoint checkpoints/deep_cnn_v2_seed42.pt`;
  - bản đầu (64 kênh, không chống răng cưa): `--image_size 128 --epochs 200 --width 64 --aa 0 --checkpoint checkpoints/deep_cnn_v1_seed42.pt`.
- Các lần thử cần code không có trong `youonlycodeonce/train_deep_cnn.py` (khối SE, bỏ đường tắt, MixUp/CutMix, kiến trúc của torchvision, khối NiN, các kiểu stem, nhiễu và làm mờ, cách lấy mẫu, SAM) chạy bằng script thử nghiệm riêng, không có trong repo. Các thay đổi còn lại (số khối, độ rộng, chống răng cưa, stochastic depth, kích thước ảnh, số epoch) chạy lại được bằng tham số dòng lệnh.

### Kết quả

Trung bình ± độ lệch chuẩn qua 3 seed (42, 1, 2). Ba bản của Model 2: bản đầu (64 kênh, không chống răng cưa), bản 2 (96 kênh, chống răng cưa; ảnh 128×128, 200 epoch như Model 1) và bản hiện tại (kiến trúc như bản 2; ảnh 160×160, 450 epoch):

| | Model 1 | Model 2: bản đầu | Model 2: bản 2 | Model 2: hiện tại |
|---|---|---|---|---|
| Ảnh train, số epoch | 128×128, 200 | 128×128, 200 | 128×128, 200 | 160×160, 450 |
| Val, có TTA (trung bình 10 epoch cuối) | 87,1 ± 0,1 | 87,1 ± 0,5 | 89,5 ± 0,4 | 92,3 ± 0,0 |
| **Test, có TTA** | **85,0 ± 0,8** | **85,8 ± 0,7** | **87,2 ± 0,7** | **89,7 ± 0,2** |
| Test, không TTA | 84,2 ± 0,8 | 85,5 ± 0,9 | 87,1 ± 0,5 | 89,2 ± 0,2 |
| Test, trung bình accuracy của 25 lớp | 84,0 ± 0,9 | 84,5 ± 0,7 | 85,5 ± 1,2 | 88,7 ± 0,5 |
| Tham số | 6,3 triệu | 4,9 triệu | 11,1 triệu | 11,1 triệu |

Test accuracy theo từng seed (42 / 1 / 2): bản hiện tại 89,6 / 89,6 / 90,0; bản 2 87,7 / 86,4 / 87,5; bản đầu 86,4 / 85,1 / 85,8.

- So với bản 2 (cùng kiến trúc, chỉ khác ảnh train và số epoch), bản hiện tại tăng 1,9 / 3,2 / 2,5 điểm test, tăng ở cả 3 seed. Trên val, mức tăng là +2,8 (6 seed, seed nào cũng tăng).
- Bản hiện tại hơn Model 1 khoảng 4,7 điểm test. Khi train cùng cách với Model 1 (ảnh 128×128, 200 epoch), Model 2 (bản 2) hơn Model 1 khoảng 2,2 điểm test (val +2,4); khoảng 2,5 điểm còn lại đến từ ảnh lớn hơn và train lâu hơn, vốn chưa được thử cho Model 1 (xem "Lưu ý khi so sánh các model").
- Với 751 ảnh test, 1 điểm chỉ ứng với khoảng 7–8 ảnh, và sai số chuẩn do chọn mẫu ảnh test vào khoảng 1,1 điểm. Độ lệch chuẩn 0,2 giữa các seed chỉ đo dao động do khởi tạo và augmentation.
- Lần chạy cuối bằng code trong repo cho val giống hệt từng epoch với lần thử tổ hợp ở vòng 7, nên val trong bảng (92,3) cũng là val của chính các lần chạy đã dùng để chọn cấu hình và có thể hơi cao. Trên 3 seed mới 6, 7, 8 (chỉ dùng để xác nhận), val là 92,5 ± 0,2. Test thấp hơn val khoảng 2,5 điểm (bản 2: 2,3; Model 1: 2,1).
- Test của Model 2 đã được xem 3 lần (bản đầu, bản 2, bản hiện tại). Mỗi lần chọn cấu hình chỉ dựa trên val, nhưng việc thử tiếp sau mỗi bản được quyết định khi đã thấy test của bản trước, nên test của Model 2 không hoàn toàn "chưa đụng tới" như của Model 1.
- TTA giúp thêm 0,5–0,7 điểm ở từng seed, tức 4–5 ảnh test (89,2% không TTA).
- Các lớp khó nhất (trung bình 3 seed): FLOUR 65%, FISH 75%, OIL 76%, VINEGAR 81%, SUGAR 81%. Các lớp dễ nhất: RICE 97%, MILK 96%, WATER 95%, TOMATO_SAUCE 95%, TEA 94%.

### Những gì đã thử

Chọn cấu hình chỉ bằng val (có TTA, trung bình 10 epoch cuối, rồi trung bình 3 seed). Luật được chốt trước khi chạy:
- vòng 1 lấy kiến trúc có val cao nhất, nếu hai kiến trúc đứng đầu chênh dưới 0,5 điểm thì lấy kiến trúc nhỏ hơn;
- vòng 2 giữ một cách chống overfit nếu tăng ít nhất 0,5 điểm (khối SE cần 1 điểm);
- nếu hai cách cùng đạt thì thử tổ hợp, và chỉ giữ cả hai nếu tổ hợp hơn cách tốt nhất ít nhất 0,5 điểm.

| Vòng | Cấu hình | Tham số | Val accuracy (%) | Quyết định |
|---|---|---|---|---|
| 1 | ResNet-18 (2 khối mỗi tầng), 64 → 512 kênh | 11,2 triệu | 86,7 ± 0,1 | bỏ, ngang ResNet-10 mà lớn hơn |
| 1 | ResNet-10 (1 khối mỗi tầng), 64 → 512 kênh | 4,9 triệu | 86,7 ± 0,4 | giữ |
| 1 | ResNet-26 (2,3,4,3), 48 → 384 kênh | 10,5 triệu | 85,1 ± 0,9 | bỏ |
| 1 | ResNet-18 hẹp, 32 → 256 kênh | 2,8 triệu | 83,4 ± 1,6 | bỏ |
| 2 | ResNet-10 + stochastic depth 0,1 | 4,9 triệu | 87,9 ± 0,5 | giữ (+1,2) |
| 2 | ResNet-10 + MixUp/CutMix (một nửa số batch) | 4,9 triệu | 87,3 ± 0,7 | đạt ngưỡng (+0,6), thử tổ hợp |
| 2 | ResNet-10 + khối SE | 5,0 triệu | 86,7 ± 1,0 | bỏ |
| 3 | Stochastic depth + MixUp/CutMix | 4,9 triệu | 87,1 ± 0,7 | bỏ (−0,8 so với chỉ stochastic depth) |
| phụ | ResNet-10 bỏ hết đường tắt | 4,8 triệu | 86,1 ± 0,4 | chỉ để so sánh: đường tắt giúp +0,6 |

Model 1 trong cùng điều kiện thí nghiệm đạt 87,3 ± 1,1. Các lần thử trong bảng, kể cả lần thử Model 1 này, chạy với `--num_workers 6`, còn lần chạy cuối dùng mặc định `--num_workers 8`, nên dù cùng seed, augmentation ngẫu nhiên vẫn khác: cấu hình được chọn đạt 87,9 ± 0,5 trong bảng nhưng 87,1 ± 0,5 khi chạy lại (cột "Model 2: bản đầu" ở mục Kết quả), ngang Model 1 (87,1 ± 0,1). Vì vậy chênh lệch dưới khoảng 1 điểm, giữa các dòng hay giữa hai model, có thể chỉ là nhiễu. Với 3.436 ảnh train và train từ đầu, mạng sâu hơn không tự động tốt hơn: độ rộng (số kênh) và cách chống overfit quan trọng hơn độ sâu.

#### Vòng 4: nâng cấp sau khi đã có kết quả test của bản đầu

Mọi lần chạy dùng `--num_workers 8` như lần chạy cuối của bản đầu, và chênh lệch so với bản đầu được tính theo từng cặp seed. Với thay đổi không làm đổi số tham số (chống răng cưa, stochastic depth), cùng seed cho khởi tạo, thứ tự ảnh và augmentation giống hệt bản đầu, nên mỗi cặp so sánh đúng cùng điều kiện. Đổi độ rộng hay độ sâu làm đổi số tham số, nên khác cả khởi tạo, thứ tự ảnh và augmentation (model được tạo trước khi DataLoader lấy số ngẫu nhiên); khi đó hai lần chạy chỉ chung số seed. Luật chốt trước khi chạy:
- thay đổi giữ nguyên số tham số phải tăng trung bình ít nhất 0,5 điểm và tăng ở ít nhất 2/3 seed; thay đổi độ rộng hay độ sâu phải tăng trung bình ít nhất 0,8 điểm và tăng ở cả 3 seed;
- nếu hai thay đổi khác loại cùng đạt thì thử tổ hợp của chúng; tổ hợp chỉ được chọn nếu hơn thay đổi đơn tốt nhất ít nhất 0,3 điểm;
- cấu hình được chọn phải được xác nhận thêm trên seed 3, 4, 5 (cùng với bản đầu trên các seed đó): trung bình 6 seed phải tăng ít nhất 0,5 điểm, tăng ở ít nhất 4/6 seed, và trung bình 3 seed mới phải tăng.

| Cấu hình (seed 42, 1, 2) | Tham số | Val accuracy (%) | So với bản đầu | Quyết định |
|---|---|---|---|---|
| Bản đầu: ResNet-10, 64 kênh, stochastic depth 0,1 | 4,9 triệu | 87,1 ± 0,5 | | mốc so sánh |
| + chống răng cưa ở khối thu nhỏ | 4,9 triệu | 88,3 ± 0,5 | +1,2 (3/3 seed) | đạt; trên 6 seed +1,2, cả 6 seed đều tăng |
| Ảnh train 160×160 (đánh giá 200×200) | 4,9 triệu | 87,6 ± 0,7 | +0,4 (2/3) | bỏ |
| Stochastic depth 0,2 | 4,9 triệu | 87,1 ± 0,8 | +0,0 (1/3) | bỏ |
| Stochastic depth 0,3 | 4,9 triệu | 86,5 ± 0,1 | −0,7 (0/3) | bỏ |
| ResNet-18 + stochastic depth 0,2 | 11,2 triệu | 86,9 (seed 42, 1) | −0,5 (0/2) | bỏ |
| 96 kênh, không chống răng cưa | 11,1 triệu | 88,1 ± 1,3 | +0,9 (2/3) | bỏ |
| **96 kênh + chống răng cưa** | 11,1 triệu | **89,5 ± 0,4** | **+2,4 (3/3)** | **giữ**: trên 6 seed +2,0, cả 6 seed đều tăng; hơn chỉ chống răng cưa +0,8 (5/6 seed) |

Do một lỗi (chống răng cưa được bật làm mặc định trong code khi các lần thử khác chưa chạy xong), 4 lần chạy bị kèm chống răng cưa ngoài ý muốn. Lần chạy seed 2 của ResNet-18 + stochastic depth 0,2 không được tính; hai seed còn lại đều không tăng, nên cấu hình này không đạt dù seed 2 ra sao. Ba lần chạy seed 42, 1, 2 của dòng "96 kênh + chống răng cưa" vốn định là 96 kênh không chống răng cưa (dòng "96 kênh, không chống răng cưa" được chạy bù sau đó). Luật chốt trước chỉ cho thử tổ hợp hai thay đổi khi cả hai cùng đạt, mà riêng 96 kênh không đạt; việc vẫn xét tổ hợp này (chỉ cần hơn chỉ chống răng cưa ít nhất 0,3 điểm) được quyết định sau khi đã thấy 3 seed đó. Vì vậy, bằng chứng không phụ thuộc vào lần chọn này chỉ gồm 3 seed mới 3, 4, 5 (trên val: hơn bản đầu +1,6; hơn chỉ chống răng cưa +0,5, tăng ở 2/3 seed) và tập test (hơn bản đầu 1,3–1,7 điểm ở cả 3 seed).

Ở vòng 4, chống răng cưa ở các bước thu nhỏ là thay đổi đáng giá nhất, giống như MaxBlurPool ở Model 1. Hai thay đổi được giữ có hiệu quả gần như cộng dồn: riêng chống răng cưa +1,2, riêng 96 kênh +0,9 (không ổn định giữa các seed), cả hai +2,4. Mạng sâu hơn (ResNet-18) và stochastic depth mạnh hơn không giúp. Ảnh train 160×160 chỉ tăng +0,4 ở bản đầu (chưa đạt ngưỡng), nhưng tăng +1,1 ở vòng 7, trên một nền khác bản đầu cả về độ rộng, chống răng cưa và số epoch; chưa rõ khác biệt nào trong số đó làm ảnh lớn có ích hơn.

#### Vòng 5–7: các kiến trúc trong slide và cải thiện bản 2 (sau khi đã có kết quả test của bản 2)

Cách làm giống vòng 4. Thước đo là val accuracy có TTA, trung bình 10 epoch cuối, so sánh theo từng cặp seed (42, 1, 2) với cấu hình đang giữ, và luật được chốt trước khi chạy. Ở vòng 6–7, cấu hình đạt phải được xác nhận trên 3 seed mới (6, 7, 8), chạy cùng với cấu hình đang giữ trên các seed đó: trung bình 6 seed phải tăng ít nhất 0,5 điểm, tăng ở ít nhất 5/6 seed, và trung bình 3 seed mới phải tăng. Test chỉ chạy cho cấu hình cuối cùng.

**Vòng 5: các kiến trúc trong slide "Modern CNNs"**, train từ đầu theo đúng cách train của bản 2 (cùng dữ liệu, augmentation, AdamW + OneCycle, ảnh 128×128, 200 epoch). Model 2 vốn đã dùng BatchNorm và khối residual (ResNet). Kiến trúc khác nhau nên không ghép cặp theo seed được: một kiến trúc chỉ được chọn nếu hơn bản 2 ít nhất 0,8 điểm và hơn ở cả 3 seed, rồi phải được xác nhận trên seed 3, 4, 5.

| Kiến trúc | Tham số | Val accuracy (%) | So với bản 2 |
|---|---|---|---|
| Bản 2 (ResNet-10) | 11,1 triệu | 89,5 ± 0,4 | |
| Bản 2 + khối NiN | 11,7 triệu | 88,7 ± 0,5 | −0,8 (0/3 seed tăng) |
| VGG-11 có BatchNorm | 9,2 triệu | 88,6 ± 0,6 | −0,9 (1/3) |
| DenseNet-121 | 7,0 triệu | 85,8 ± 0,7 | −3,7 (0/3) |
| GoogLeNet | 5,6 triệu | 84,9 ± 0,5 | −4,6 (0/3) |
| AlexNet (learning rate 3e-4, lần chạy tham chiếu) | 57,1 triệu | 65,4 ± 0,7 | −24,1 (0/3) |

Không kiến trúc nào hơn bản 2, nên giữ ResNet-10.
- Khối NiN (theo ý tưởng Network in Network): Conv 1×1 + BatchNorm + ReLU, chèn ngay trước Global Average Pooling của bản 2.
- VGG, GoogLeNet, DenseNet và AlexNet dùng bản của torchvision, khởi tạo ngẫu nhiên. VGG dùng Global Average Pooling thay cho 2 lớp Linear 4096 chiều.
- Ứng viên AlexNet của vòng 5 dùng learning rate 2e-3 như các model khác và không học được (val 16,0 ± 14,9), có lẽ vì không có BatchNorm, nên bị loại. Dòng AlexNet trong bảng là lần chạy tham chiếu với learning rate 3e-4, thêm ở vòng 6 để có con số so sánh. Cả hai lần chạy đều đánh giá ở 128×128, vì ở 160×160 lớp adaptive pooling của AlexNet báo lỗi trên GPU Apple.

**Vòng 6: cải thiện cách train và phần stem của bản 2.** Mọi thay đổi phải tăng trung bình ít nhất 0,5 điểm, và:
- thay đổi giữ nguyên khởi tạo, thứ tự ảnh và augmentation ngẫu nhiên của từng seed (số epoch, các kiểu stem) phải tăng ở ít nhất 2/3 seed;
- thay đổi làm khác thứ tự ảnh hoặc augmentation ngẫu nhiên (thêm augmentation, cách lấy mẫu) phải tăng ở cả 3 seed.

| Thay đổi so với bản 2 | Val accuracy (%) | So với bản 2 | Quyết định |
|---|---|---|---|
| **Train 300 epoch thay vì 200** | **90,2 ± 0,3** | **+0,7 (2/3)** | **giữ**: trên 6 seed +0,6, tăng ở 5/6 seed |
| MixUp/CutMix (một nửa số batch) | 89,8 ± 0,6 | +0,3 (2/3) | bỏ |
| Lấy mẫu cân bằng lớp một phần (trọng số mỗi ảnh 1/√n, n = số ảnh của lớp) | 89,6 ± 0,8 | +0,1 (2/3) | bỏ |
| Bỏ bước thu nhỏ cuối stem: các tầng chạy ở độ phân giải gấp đôi, thời gian train gấp khoảng 2,5 lần | 89,6 ± 0,0 | +0,1 (2/3) | bỏ |
| Stem chống răng cưa: Conv đầu bước 1, thêm MaxBlurPool ngay sau để thu nhỏ | 88,7 ± 0,4 | −0,8 (0/3) | bỏ |
| Thêm nhiễu Gauss vào ảnh train | 88,4 ± 0,9 | −1,1 (0/3) | bỏ |
| Làm mờ Gauss ảnh train | 88,3 ± 0,4 | −1,2 (0/3) | bỏ |

**Vòng 7: cải thiện tiếp trên nền 300 epoch.** Mọi thay đổi phải tăng trung bình ít nhất 0,5 điểm (vòng 4 dùng 0,8 cho thay đổi độ rộng; với ngưỡng đó, 128 kênh, tăng trung bình 0,76, đã không đạt). 450 epoch và SAM giữ nguyên khởi tạo và augmentation ngẫu nhiên nên cần tăng ở ít nhất 2/3 seed; 128 kênh (khác khởi tạo), ảnh 160×160 và MixUp/CutMix phải tăng ở cả 3 seed. Đổi cỡ ảnh không làm đổi số tham số nên ở vòng 4 được xét theo luật 2/3 seed; luật chốt trước của vòng 7 xếp nó vào nhóm chặt hơn cho chắc, và nó vẫn tăng ở cả 3 seed. Nếu nhiều thay đổi cùng đạt, chỉ thử tổ hợp hai thay đổi tăng nhiều nhất, và chọn tổ hợp nếu nó hơn thay đổi đơn tốt nhất ít nhất 0,3 điểm.

| Thay đổi so với 300 epoch | Tham số | Val accuracy (%) | So với 300 epoch | Quyết định |
|---|---|---|---|---|
| Ảnh train 160×160 (đánh giá 200×200) | 11,1 triệu | 91,3 ± 0,3 | +1,1 (3/3) | đạt, đưa vào tổ hợp |
| Train 450 epoch | 11,1 triệu | 91,2 ± 0,9 | +1,0 (2/3) | đạt, đưa vào tổ hợp |
| 128 kênh (128 → 1024) | 19,7 triệu | 91,0 ± 0,7 | +0,8 (3/3) | đạt, không ghép thêm |
| SAM (rho 0,05; thời gian train gấp khoảng 2 lần) | 11,1 triệu | 90,7 ± 0,1 | +0,5 (3/3) | đạt, không ghép thêm |
| MixUp/CutMix (một nửa số batch) | 11,1 triệu | 90,6 ± 0,3 | +0,4 (3/3) | bỏ |
| **Ảnh 160×160 + 450 epoch** | 11,1 triệu | **92,3 ± 0,0** | **+2,1 (3/3)** | **giữ**: hơn chỉ ảnh 160×160 +1,0; trên 6 seed +2,2 so với 300 epoch, tăng ở cả 6 seed |

- 128 kênh và SAM cũng đạt nhưng không vào cấu hình cuối: luật chốt trước chỉ ghép hai thay đổi tăng nhiều nhất. Một vòng 8 (thêm 128 kênh vào tổ hợp) đã được chốt luật rồi bị huỷ để giới hạn thời gian, trước khi có kết quả tổ hợp. SAM không được đưa vào vòng 8 vì tăng ít nhất mà thời gian train gấp khoảng 2 lần.
- Cả 4 thay đổi đạt đều tăng lượng tính toán, nhưng không phải cứ tăng tính toán là giúp: bỏ bước thu nhỏ cuối stem (vòng 6) làm thời gian train gấp khoảng 2,5 lần mà chỉ +0,1, và ResNet-18 không hơn ResNet-10 (vòng 1 và 4, trên bản đầu). Có vẻ bản 2 chủ yếu thiếu số epoch, độ phân giải của ảnh đầu vào và độ rộng, hơn là cần một kiến trúc khác (vòng 5).
- Train lâu hơn không chỉ giúp Model 2: Model 1 train 300 epoch đạt 88,1 ± 0,2 trên val (+1,0, tăng ở cả 3 seed). Đây là lần chạy đối chứng, chốt trước là không phải ứng viên và chạy sau khi Model 1 đã có kết quả test, nên không dùng để đổi cấu hình Model 1. Chưa thử Model 1 với ảnh 160×160 và 450 epoch.
- Ở bước xác nhận của vòng 7, mốc so sánh trên seed 6, 7, 8 là các lần chạy 300 epoch của vòng 6. Chính các lần chạy này đã giúp 300 epoch được chấp nhận, nên có thể chúng cao hơn bình thường một chút; nếu vậy, việc xác nhận khó hơn chứ không dễ hơn.
- Trong lúc chờ kết quả tổ hợp, các lần xác nhận cho riêng ảnh 160×160 (phòng khi tổ hợp không được chọn) được chạy trước trên seed 6, 7, 8. Khi tổ hợp được chọn, lần chạy seed 7 và 8 bị dừng giữa chừng theo luật chốt trước; seed 6 đã xong (+2,0 so với 300 epoch), chỉ để tham khảo.
- Val đã được dùng để chọn cấu hình qua 7 vòng, nên val của Model 2 có thể cao hơn thực tế. Test không được dùng để chọn cấu hình nên là con số khách quan hơn (xem lưu ý ở mục Kết quả).

## Lưu ý khi so sánh các model

- Model 2 train ở ảnh lớn hơn và lâu hơn Model 1 (160×160 và 450 epoch, so với 128×128 và 200 epoch). Hai giá trị này của Model 2 được chọn trên val ở vòng 6–7; Model 1 thì chưa được thử với ảnh 160×160 và 450 epoch, chỉ có một lần chạy đối chứng 300 epoch (tăng khoảng 1 điểm val, không dùng để đổi cấu hình). Vì vậy chênh lệch giữa hai model gồm cả phần do kích thước ảnh và số epoch, không chỉ do kiến trúc. Để so riêng hai kiến trúc với cùng cách train, xem cột "Model 2: bản 2" (ảnh 128×128, 200 epoch như Model 1): hơn Model 1 khoảng 2,4 điểm val và 2,2 điểm test.
- Test của Model 2 đã được xem 3 lần (bản đầu, bản 2, bản hiện tại), còn test của Model 1 chỉ một lần. Mỗi bản của Model 2 chỉ được chọn bằng val, nhưng việc cải thiện tiếp được quyết định sau khi đã thấy test của bản trước, nên con số test của Model 2 có thể hơi lạc quan hơn.
- Báo cáo trung bình ± độ lệch chuẩn qua ít nhất 3 seed (`--seed`). Với 751 ảnh test, chênh lệch dưới khoảng 2–3 điểm chưa chắc có ý nghĩa.
- Không so trực tiếp với 78,9% của bài báo gốc. Bài báo dùng CaffeNet đã pretrain trên ImageNet và chia 5 fold ngẫu nhiên theo từng ảnh, cách chia này cũng bị rò rỉ ảnh cùng cảnh như mô tả ở trên.
- Pipeline trong `src/` chia ngẫu nhiên 70/15/15 theo từng ảnh (lưu ở `artifacts/split_manifest.json`), không theo nhóm như `youonlycodeonce/splits/`, nên không chung test set với Model 1, Model 2 và có thể bị rò rỉ ảnh cùng cảnh như mô tả ở mục chia dữ liệu.
- Cùng một seed chỉ tái lập đúng kết quả khi cùng thiết bị (MPS/CUDA/CPU), cùng phiên bản thư viện và cùng `--num_workers`.
