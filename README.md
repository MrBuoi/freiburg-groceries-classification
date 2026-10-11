# Freiburg Groceries Classification (Deep Learning Comparison)

Dự án so sánh 3 model trên bộ dữ liệu Freiburg Groceries (25 nhóm sản phẩm, 4.947 ảnh 256×256). Cả 3 model viết bằng Keras:
1. **Model 1**: Simple CNN train từ đầu, file `youonlycodeonce/train_simple_cnn.py`
2. **Model 2**: Deep CNN kiểu ResNet-10 train từ đầu, file `youonlycodeonce/train_deep_cnn.py`
3. **Model 3**: Transfer learning từ ResNet50 đã pretrain trên ImageNet, file `youonlycodeonce/train_resnet50.py`

Ngoài ra có một **baseline**: kiến trúc như Model 1 nhưng không augmentation và train ngắn với early stopping, file `youonlycodeonce/train_baseline.py`. Vì cách train cũng khác, chênh lệch giữa Model 1 và baseline không chỉ do augmentation (xem mục "Baseline và early stopping").

## Cấu trúc thư mục

```text
data/                    ảnh của bộ dữ liệu (không commit, xem mục tải dữ liệu)
notebooks/               notebook của pipeline trong src/; eda_chia_du_lieu.ipynb giải thích cách chia dữ liệu
src/                     pipeline PyTorch chung của nhóm (python -m src.train): SimpleCNN, DeepCNN, ResNet50
youonlycodeonce/
  split_data.py          chép ảnh theo một cách chia vào artifacts/split_data/
  train_simple_cnn.py    Model 1
  train_deep_cnn.py      Model 2
  train_resnet50.py      Model 3
  train_baseline.py      baseline: như Model 1 nhưng không augmentation, train ngắn với early stopping
  make_splits.py         tạo cách chia theo nhóm (splits/)
  splits/                cách chia theo nhóm: train.txt, val.txt, test.txt, groups.csv
  splits_ngaunhien/      cách chia ngẫu nhiên theo từng ảnh, chỉ dùng để so sánh
checkpoints/, artifacts/ model đã train, kết quả và ảnh đã chia (không commit)
```

Pipeline trong `src/` chia dữ liệu khác với ba model trong `youonlycodeonce/`, nên kết quả của hai nơi không so trực tiếp được với nhau (xem "Lưu ý khi so sánh các model").

## Hướng dẫn cài đặt & Chạy dự án cho nhóm

### 1. Clone repository

```bash
git clone https://github.com/MrBuoi/freiburg-groceries-classification.git
cd freiburg-groceries-classification
```

### 2. Tạo môi trường Python

Dùng Python 3.11–3.13, khuyên dùng 3.12 (Keras 3.15 cần Python 3.11 trở lên, TensorFlow chưa hỗ trợ Python 3.14). Môi trường cũ tạo bằng Python 3.10 thì phải tạo lại.

```bash
conda create -n freiburg python=3.12 -y
conda activate freiburg
```

Keras đọc ảnh bằng TensorFlow, còn phần tính toán khi train chạy trên "backend" chọn bằng biến môi trường `KERAS_BACKEND`. Trên máy có GPU, dùng backend PyTorch (`KERAS_BACKEND=torch`):

- **macOS:** cần máy chip Apple Silicon (M1 trở lên) và Python bản arm64, tức `python -c "import platform; print(platform.machine())"` phải in ra `arm64`. TensorFlow trên Mac chỉ chạy CPU, còn PyTorch chạy được trên GPU của Mac, nên luôn chạy với `KERAS_BACKEND=torch`. PyTorch mới không còn bản cho Mac chip Intel, nên với máy đó hãy dùng Google Colab.
- **Windows/Linux có GPU NVIDIA:** cài PyTorch bản CUDA trước, rồi cũng chạy với `KERAS_BACKEND=torch`. Nếu bỏ qua bước này, pip sẽ cài bản chỉ chạy CPU và train mất nhiều giờ.

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  python -c "import torch; print(torch.cuda.is_available())"   # phải in ra True
  ```

  - Bản `cu126` chạy được trên hầu hết GPU NVIDIA. Chỉ đổi sang `cu130` khi driver từ 580 trở lên **và** GPU từ GTX 16xx / RTX 20xx trở lên. GPU GTX 10xx, MX hoặc cũ hơn phải dùng `cu126`.
  - Nếu in ra `False`: chạy `pip uninstall -y torch torchvision` rồi cài lại bằng lệnh trên. Vẫn `False` thì cập nhật driver NVIDIA (kiểm tra bằng `nvidia-smi`).
- **Máy chỉ có CPU:** vẫn chạy được (bỏ `KERAS_BACKEND` để dùng TensorFlow) nhưng rất chậm: Model 2 mất khoảng 17 giờ ngay cả trên CPU 18 nhân của Apple M5 Max.

Sau đó cài các thư viện còn lại:

```bash
pip install -r requirements.txt
```

Trên macOS/Linux, viết `KERAS_BACKEND=torch` ngay trước lệnh chạy, ví dụ `KERAS_BACKEND=torch python youonlycodeonce/train_simple_cnn.py`. Trên Windows PowerShell, chạy `$env:KERAS_BACKEND = "torch"` một lần cho mỗi cửa sổ terminal rồi chạy lệnh như bình thường.

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

## Ba model trong `youonlycodeonce/`

### Chia dữ liệu

Có hai cách chia, mỗi cách gồm ba file danh sách `train.txt`, `val.txt`, `test.txt`, mỗi dòng một ảnh dạng `LỚP/TÊN_FILE.png`:

| Cách chia | Thư mục | Train / val / test | Ảnh test có ảnh cùng nhóm trong train |
|---|---|---|---|
| Theo nhóm (`nhom`, cách chia chính) | `youonlycodeonce/splits/` | 3.436 / 760 / 751 | 0 |
| Ngẫu nhiên theo từng ảnh (`ngaunhien`) | `youonlycodeonce/splits_ngaunhien/` | 3.450 / 741 / 756 | 212 (28%) |

Bộ dữ liệu có nhiều ảnh chụp lại cùng một kệ hàng từ góc hơi khác. Nếu chia ngẫu nhiên theo từng ảnh, nhiều ảnh test có ảnh cùng nhóm (cùng cảnh hoặc cùng mẫu bao bì, xem bên dưới) nằm trong train: 28% với cách chia `ngaunhien`, trung bình khoảng 25% nếu chia ngẫu nhiên lại 1.000 lần với seed khác nhau. Model chỉ cần nhớ ảnh train là đoán đúng những ảnh này, nên accuracy cao hơn thực tế (xem mục Kết quả). Vì vậy cách chia chính gom các ảnh giống nhau thành nhóm rồi chia theo nhóm, 70/15/15 trong từng lớp.

Ngoài ảnh cùng cảnh, cách gom này còn nối các ảnh có cùng mẫu bao bì (cùng logo, cùng hãng) dù chụp ở cảnh khác. Vì vậy cách chia này chặt hơn việc chỉ tách theo cảnh:
- một số mẫu bao bì ở val/test không có trong train;
- ở vài lớp (TEA, HONEY, CAKE, RICE, CORN), gần một nửa trở lên số ảnh val hoặc test thuộc cùng một nhóm, nên accuracy của riêng các lớp này dao động nhiều.

Cách gom nhóm được mô tả trong `youonlycodeonce/make_splits.py`. Notebook `notebooks/eda_chia_du_lieu.ipynb` minh hoạ bằng ảnh và số liệu vì sao chia theo nhóm: ví dụ ảnh gần trùng, kích thước các nhóm, mô phỏng 1.000 lần chia ngẫu nhiên, và rò rỉ ảnh hưởng tới accuracy ra sao.

- Cách gom nhóm dựa vào ngưỡng số điểm khớp giữa hai ảnh, nên vẫn có thể sót vài cặp gần giống (ví dụ NUTS0037 ở test và NUTS0020 ở train). Con số 0 trong bảng trên là theo chính các nhóm này.
- Cách chia `ngaunhien` là cách chia 70/15/15 theo từng ảnh của bạn cùng nhóm, lưu lại thành danh sách để lần nào chạy cũng giống nhau (script gốc lấy file theo thứ tự hệ điều hành trả về, nên mỗi máy chia ra một kiểu). Có 4 cặp ảnh trùng y hệt bị tách sang train và test.
- `split_data.py` chép ảnh trong `data/` theo danh sách vào `artifacts/split_data/<cách chia>/{train,val,test}/<LỚP>/`, là thư mục mà 4 script train (3 model và baseline) đọc. Chạy một lần cho mỗi cách chia, mỗi lần chép khoảng 520 MB; chạy lại thì thư mục cũ bị xoá và chép lại từ đầu:
  ```bash
  python youonlycodeonce/split_data.py --split nhom
  python youonlycodeonce/split_data.py --split ngaunhien
  ```
- Không sửa tay các file danh sách. `splits/` chỉ được tạo lại bằng `python youonlycodeonce/make_splits.py` (khoảng 1,5 phút, cần `scikit-image`) khi dữ liệu thay đổi, sau đó chạy lại `split_data.py --split nhom`. Script tự dừng, không ghi đè, nếu `data/` không có đúng 25 lớp và 4.947 ảnh.
- Chỉ dùng val để chọn cấu hình. Test chỉ đánh giá một lần, cho cấu hình cuối cùng.

### Train và đánh giá

Mỗi script đọc ảnh đã chia, train, giữ lại trọng số của epoch có val accuracy cao nhất, rồi đánh giá trên test ở cuối:

```bash
KERAS_BACKEND=torch python youonlycodeonce/train_simple_cnn.py --split nhom --seed 42
KERAS_BACKEND=torch python youonlycodeonce/train_deep_cnn.py --split nhom --seed 42
KERAS_BACKEND=torch python youonlycodeonce/train_resnet50.py --split nhom --seed 42
KERAS_BACKEND=torch python youonlycodeonce/train_baseline.py --split nhom --seed 42
```

- Tham số: `--split` (`nhom` hoặc `ngaunhien`), `--seed`, `--epochs` (với Model 3 là số epoch của giai đoạn fine-tune, mặc định 20; với baseline là số epoch tối đa, mặc định 100). Baseline có thêm `--lr` (mặc định 3e-3). Các siêu tham số khác viết thẳng trong code.
- Kết quả lưu trong `checkpoints/`, chạy lại cùng model, cách chia và seed sẽ ghi đè:
  - `<model>_<cách chia>_seed<seed>.weights.h5`: trọng số tốt nhất. Keras lưu kèm trạng thái của optimizer nên file nặng gấp khoảng 3 lần model.
  - `<model>_<cách chia>_seed<seed>.json`: val tốt nhất, test có và không TTA (baseline chỉ có test không TTA), macro-F1, accuracy trung bình của 25 lớp, loss và accuracy từng epoch.
- Thời gian trên Apple M5 Max khi chạy một mình: baseline khoảng 13 phút (dừng ở epoch 64), Model 1 khoảng 42 phút, Model 2 khoảng 2 giờ 50 phút, Model 3 khoảng 35 phút. Bộ nhớ: baseline và Model 1 khoảng 7 GB, Model 2 8–12 GB, Model 3 15–17 GB.
- Cả 4 script gọi `gc.collect()` cuối mỗi epoch. Khi chạy bằng PyTorch, Keras để sót tensor của nhiều bước train trong các vòng tham chiếu mà bộ dọn rác tự động của Python dọn không kịp; không gọi `gc.collect()` thì bộ nhớ của Model 3 tăng khoảng 6–7 GB sau mỗi epoch fine-tune.
- Chạy lại cùng seed không cho kết quả giống hệt, vì augmentation chạy song song trong `tf.data`.

### Model 1: Simple CNN

- **Kiến trúc:** 6 khối Conv 3×3 → BatchNorm → ReLU → MaxBlurPool (32 filter ở khối đầu, gấp đôi sau mỗi khối, tới 1.024), rồi Global Average Pooling, Dropout 0,3 và lớp Dense softmax (6,3 triệu tham số). MaxBlurPool là max pooling chống răng cưa: lấy max 2×2 nhưng trượt từng pixel, rồi lấy trung bình 2×2 khi thu nhỏ, nên dự đoán ít bị đổi khi vật thể dịch đi vài pixel.
- **Dữ liệu:** ảnh train 128×128; augmentation gồm phóng to ngẫu nhiên vào giữa ảnh (tới 40%), lật ngang, một phép RandAugment ngẫu nhiên cho mỗi ảnh, và xoá một vùng chữ nhật ở 25% số ảnh (RandomErasing).
- **Huấn luyện:** 200 epoch, batch 64, AdamW (weight decay 0,05), label smoothing 0,1. Learning rate tăng tuyến tính lên 2e-3 trong 30% số bước đầu rồi giảm theo cosine về 0.
- **Đánh giá:** ảnh val/test 160×160, lớn hơn ảnh train 1,25 lần vì ảnh train bị phóng to. Ở test, lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang (TTA).

### Baseline: Model 1 không augmentation

- **Kiến trúc:** như Model 1, chỉ bỏ lớp RandomErasing (lớp này là một phép augmentation, chỉ chạy lúc train).
- **Dữ liệu:** ảnh train, val và test đều chỉ được thu về 128×128. Không augmentation, không TTA.
- **Huấn luyện:** batch 64, AdamW (weight decay 0,05), label smoothing 0,1 như Model 1, nhưng learning rate cố định 3e-3 thay cho warmup + cosine. Val accuracy không tăng trong 5 epoch thì learning rate giảm một nửa (`ReduceLROnPlateau`), không tăng trong 15 epoch thì dừng (`EarlyStopping`), tối đa 100 epoch. Learning rate được chọn bằng val, xem mục "Baseline và early stopping".

### Model 2: Deep CNN (ResNet-10)

- **Kiến trúc:**
  - stem gồm 3 Conv 3×3 (Conv đầu bước 2) và MaxBlurPool, thu nhỏ ảnh 4 lần;
  - 4 tầng, mỗi tầng 1 khối residual, số kênh 96 → 192 → 384 → 768;
  - Global Average Pooling, Dropout 0,3 và lớp Dense softmax (11,1 triệu tham số).
- **Khối residual:** hai Conv 3×3 và một đường tắt cộng đầu vào vào đầu ra. γ của BatchNorm cuối mỗi khối khởi tạo bằng 0, nên lúc đầu khối chỉ truyền đường tắt. Ở khối thu nhỏ ảnh (đầu tầng 2, 3, 4):
  - đường chính chống răng cưa: Conv 3×3 bước 1 → BatchNorm → ReLU → BlurPool (làm mờ bằng bộ lọc [1, 2, 1] × [1, 2, 1] / 16 rồi lấy mẫu bước 2);
  - đường tắt dùng AvgPool 2×2 + Conv 1×1.
- **Stochastic depth:** lúc train, bỏ đường chính của khối với từng ảnh, xác suất tăng dần theo độ sâu tới 0,1 ở tầng cuối.
- **Huấn luyện và đánh giá:** như Model 1, nhưng ảnh train 160×160 (đánh giá ở 200×200) và 450 epoch.

### Model 3: Transfer learning ResNet50

- **Kiến trúc:** ResNet50 của `keras.applications` với trọng số ImageNet, bỏ lớp phân loại cũ, thêm Global Average Pooling, Dropout 0,3 và lớp Dense softmax cho 25 lớp (23,6 triệu tham số).
- **Dữ liệu:** ảnh train 256×256. Augmentation gần giống Model 1 và 2:
  - cắt ngẫu nhiên kiểu `RandomResizedCrop` (35–100% diện tích ảnh, tỉ lệ khung 3/4–4/3, ở vị trí bất kỳ) rồi đưa về 256×256;
  - lật ngang, một phép RandAugment ngẫu nhiên cho mỗi ảnh, và xoá một vùng chữ nhật ở 25% số ảnh.

  Ảnh được chuẩn hoá bằng `preprocess_input` của ResNet50.
- **Huấn luyện:** batch 32, AdamW (weight decay 0,05), label smoothing 0,1, hai giai đoạn:
  - 2 epoch khoá ResNet50, chỉ train lớp Dense với learning rate 1e-3;
  - 20 epoch fine-tune toàn bộ model: learning rate tăng lên 1e-4 trong epoch đầu rồi giảm theo cosine về 0.
- **Đánh giá:** ảnh val/test 320×320, lớn hơn ảnh train 1,25 lần như ở Model 1 và 2. Giữ trọng số của epoch có val accuracy cao nhất trong cả 22 epoch; test có TTA lật ngang.

Công thức ban đầu của bạn cùng nhóm train 2 + 8 epoch:
- learning rate 1e-4 không đổi ở cả hai giai đoạn, weight decay 1e-4;
- augmentation nhẹ hơn: xoay ±15°, dịch ±10%, lật ngang, zoom ±10%, đổi độ sáng;
- không có label smoothing hay Dropout.

Kết quả của công thức ban đầu nằm ở dòng "Model 3 (công thức ban đầu)" trong bảng kết quả, cách chọn công thức mới ở mục "Nâng cấp Model 3".

## Kết quả

Accuracy (%), mỗi cấu hình train một lần với seed 42. Cột "Test" có TTA, trừ baseline (không dùng TTA nên hai cột test như nhau):

| Model | Cách chia | Val tốt nhất (epoch) | Test | Test không TTA | Macro-F1 | Accuracy trung bình 25 lớp |
|---|---|---|---|---|---|---|
| Baseline (không augmentation, early stopping) | theo nhóm | 75,7 (49) | **73,0** | 73,0 | 70,4 | 69,5 |
| Baseline (không augmentation, early stopping) | ngẫu nhiên | 79,8 (84) | 79,9 | 79,9 | 79,4 | 78,9 |
| Model 1 | theo nhóm | 84,9 (179) | **84,6** | 83,4 | 83,7 | 83,1 |
| Model 1 | ngẫu nhiên | 85,8 (174) | 86,1 | 85,2 | 86,4 | 86,0 |
| Model 2 | theo nhóm | 90,5 (369) | **88,9** | 88,0 | 88,6 | 88,0 |
| Model 2 | ngẫu nhiên | 91,0 (406) | 91,7 | 91,7 | 92,3 | 91,7 |
| Model 3 | theo nhóm | 93,0 (14) | **91,3** | 90,8 | 90,4 | 90,0 |
| Model 3 | ngẫu nhiên | 92,3 (15) | 93,0 | 92,9 | 93,1 | 92,9 |
| Model 3 (công thức ban đầu) | theo nhóm | 84,6 (8) | 83,6 | 82,8 | 82,3 | 82,0 |
| Model 3 (công thức ban đầu) | ngẫu nhiên | 87,3 (5) | 88,1 | 87,7 | 88,3 | 88,8 |

So sánh cặp trên cùng 751 ảnh test của cách chia theo nhóm. Khoảng tin cậy 95% tính bằng bootstrap, lấy mẫu lại theo nhóm ảnh gần giống, vì các ảnh cùng nhóm không độc lập với nhau:

| So sánh | Chênh lệch test | Khoảng tin cậy 95% |
|---|---|---|
| Model 1 so với baseline | +11,6 | [+8,4; +14,9] |
| Model 3 so với Model 1 | +6,8 | [+3,9; +9,6] |
| Model 2 so với Model 1 | +4,4 | [+2,0; +6,8] |
| Model 3 so với Model 2 | +2,4 | [−0,3; +5,2] |

Tách test của cách chia ngẫu nhiên thành 212 ảnh có ảnh cùng nhóm trong train ("rò rỉ") và 544 ảnh còn lại:

| Model | Theo nhóm | Ngẫu nhiên: cả tập test | 212 ảnh rò rỉ | 544 ảnh còn lại |
|---|---|---|---|---|
| Baseline | 73,0 | 79,9 | 93,9 | 74,4 |
| Model 1 | 84,6 | 86,1 | 98,1 | 81,4 |
| Model 2 | 88,9 | 91,7 | 99,1 | 88,8 |
| Model 3 | 91,3 | 93,0 | 98,1 | 91,0 |
| Model 3 (công thức ban đầu) | 83,6 | 88,1 | 96,2 | 84,9 |

- **Model 3 cao nhất ở cả hai cách chia.** Nhờ công thức train mới, test trên cách chia theo nhóm tăng từ 83,6 lên 91,3 (xem mục "Nâng cấp Model 3").
- **Thứ tự Model 3 > Model 2 > Model 1 có độ chắc chắn khác nhau.**
  - Model 2 hơn Model 1 và Model 3 hơn Model 1 là rõ ràng: khoảng tin cậy không chứa 0.
  - Model 3 hơn Model 2 2,4 điểm (50 ảnh Model 3 đúng mà Model 2 sai, 32 ảnh ngược lại). Khoảng tin cậy vẫn chứa 0, nên với một lần chạy chưa khẳng định chắc được.
- **Baseline thấp hơn Model 1 11,6 điểm test** (khoảng tin cậy không chứa 0). Baseline có cùng kiến trúc nhưng không augmentation và train ngắn với early stopping. Không có augmentation, model học thuộc ảnh train: train accuracy gần 100% từ epoch 22, còn val cao nhất chỉ 75,7%. Mức chênh 11,6 điểm đến từ augmentation (kèm TTA) cùng với cách train 200 epoch: cùng cách train ngắn thì Model 1 chỉ hơn baseline 1,9 điểm (xem mục "Baseline và early stopping").
- **Với Model 1–3, cách chia ngẫu nhiên cho điểm cao hơn 1,6–2,7 điểm, và phần chênh này đến từ ảnh rò rỉ.**
  - Model 1–3 đoán đúng 98–99% số ảnh rò rỉ, baseline 94%.
  - Trên các ảnh không rò rỉ, accuracy của Model 1–3 ngang hoặc thấp hơn cách chia theo nhóm (chênh từ −3,1 đến −0,2 điểm).
  - Ảnh rò rỉ cộng điểm không đều: so với 544 ảnh còn lại, baseline được cộng 5,4 điểm, Model 1 4,7, Model 2 2,9, Model 3 2,0. Model yếu được cộng nhiều hơn nên khoảng cách giữa các model bị thu hẹp.
  - Với baseline và công thức ban đầu của Model 3, mức chênh lớn hơn: 6,9 và 4,5 điểm, trong đó 5,4 và khoảng 3,2 điểm đến từ ảnh rò rỉ.

  Vì vậy các kết luận của dự án dựa trên cách chia theo nhóm.
- **So với bản PyTorch trước đây.** Bản PyTorch của Model 1 và Model 2 (commit `77331de`, cùng cách chia theo nhóm, trung bình 3 seed) đạt test 85,0 ± 0,8 và 89,7 ± 0,2. Bản Keras với seed 42 thấp hơn 0,5 và 0,8 điểm.

## Nâng cấp Model 3

Log train của công thức ban đầu cho thấy ba vấn đề:
- **Lớp Dense mới học chưa xong trong giai đoạn 1.** Learning rate 1e-4 quá nhỏ nên sau 2 epoch val chỉ 34,5%, trong khi cùng 2 epoch đó với learning rate 1e-3 (công thức A bên dưới) đạt 67%. Giai đoạn 2 vì vậy bắt đầu với một lớp phân loại còn yếu.
- **Val dao động mạnh vì learning rate không đổi.** Train accuracy đã khoảng 99% nhưng val vẫn lên xuống trong khoảng 80–85% giữa các epoch. Epoch "tốt nhất" chỉ là một đỉnh may mắn.
- **Gần như không có regularization.** Weight decay 1e-4 nhân với learning rate 1e-4 cho ra một số nhỏ hơn độ chính xác của float32, nên bị làm tròn về 0 và không có tác dụng. Augmentation nhẹ, không có label smoothing hay Dropout.

Các công thức dưới đây được so sánh chỉ bằng val (seed 42, trung bình 3 epoch cuối), với luật chọn chốt trước khi chạy:

| Công thức | Val, trung bình 3 epoch cuối | Val tốt nhất (epoch) |
|---|---|---|
| Ban đầu: 2 + 8 epoch, learning rate 1e-4 không đổi | 82,4 | 84,6 (8) |
| A: chỉ sửa learning rate (1e-3 ở giai đoạn 1; warmup rồi cosine ở giai đoạn 2) | 87,9 | 88,2 (10) |
| **B: A + 20 epoch fine-tune, augmentation gần giống Model 1 và 2 (nhưng cắt ngẫu nhiên kiểu `RandomResizedCrop` thay cho phóng to vào giữa ảnh), label smoothing, Dropout, weight decay 0,05, đánh giá ở 320×320** | **92,7** | 93,0 (14) |
| BBN: B + giữ BatchNorm của ResNet50 ở chế độ dự đoán khi fine-tune | 92,9 | 93,0 (20) |
| ConvNeXtTiny thay cho ResNet50, công thức B (chỉ để tham khảo) | 93,1 | 93,6 (15) |

- **Chọn B.**
  - Riêng việc sửa learning rate (A) đã tăng 5,5 điểm val.
  - BBN hơn B 0,2 điểm, dưới ngưỡng 0,5 đã chốt trước, nên giữ công thức ít thay đổi hơn.
- **Model 3 vẫn là ResNet50.** ConvNeXtTiny hơn B 0,4 điểm, dưới ngưỡng 1 điểm để đề xuất đổi backbone, nhưng train nhanh gấp khoảng 2 lần trên GPU của Mac (16 so với 33 phút).
- Test chỉ được tính một lần, cho công thức B.
- Mỗi công thức chỉ chạy một seed, nên chênh lệch dưới khoảng 1 điểm giữa B, BBN và ConvNeXtTiny có thể chỉ là nhiễu.

## Baseline và early stopping

Thử một cách train ngắn hơn 200 epoch: learning rate cố định và cao hơn, giảm một nửa khi val không tăng trong 5 epoch, dừng khi val không tăng trong 15 epoch, tối đa 100 epoch. Áp cho baseline và cho Model 1 (Model 1 giữ nguyên augmentation và TTA). So sánh bằng val (seed 42, cách chia theo nhóm), với luật chọn chốt trước khi chạy:
- learning rate chọn giữa 3e-3 và 6e-3: lấy cái có val tốt nhất cao hơn, nếu chênh dưới 0,5 điểm thì lấy 3e-3;
- Model 1 chỉ đổi sang cách train mới nếu val tốt nhất không thấp hơn 84,9 của bản 200 epoch.

| Cấu hình | Learning rate | Số epoch đã chạy | Val tốt nhất (epoch) | Test | Thời gian khi chạy một mình |
|---|---|---|---|---|---|
| **Baseline** | **3e-3** | 64 | **75,7** (49) | 73,0 | khoảng 13 phút |
| Baseline | 6e-3 | 69 | 71,1 (54) | | |
| Model 1, early stopping | **3e-3** | 92 | **78,9** (77) | 74,8 | khoảng 19 phút |
| Model 1, early stopping | 6e-3 | 100 | 78,8 (94) | | |
| **Model 1, 200 epoch (giữ nguyên)** | tăng lên 2e-3 rồi giảm theo cosine | 200 | **84,9** (179) | **84,6** | khoảng 42 phút |

Bản Model 1 early stopping không đưa vào repo. Muốn chạy lại thì chép `train_simple_cnn.py` sang file mới, thay đoạn code từ dòng `# 3. Train` tới trước `# 4. Đánh giá trên tập test` bằng đoạn tương ứng của `train_baseline.py`, thêm tham số `--lr` (mặc định 3e-3), đổi mặc định của `--epochs` thành 100 và đổi tên trong `save_path` (ví dụ `simple_cnn_es_...`); giữ nguyên tên thì lần chạy sẽ ghi đè trọng số và kết quả của Model 1 bản 200 epoch trong `checkpoints/`. Script, log và kết quả của các lượt trong bảng nằm trong `checkpoints/thu_nghiem_early_stopping/`, chỉ có trên máy đã chạy. Test chỉ tính một lần, cho learning rate đã chọn. Test của Model 1 có TTA, của baseline thì không. Thời gian = số epoch × thời gian một epoch đo khi chạy một mình: 12,0 giây với baseline, 12,7 giây với Model 1 (augmentation chạy song song trên CPU nên gần như không làm chậm).

- **Cả hai chọn learning rate 3e-3.** Với 6e-3, val của baseline thấp hơn 4,6 điểm, của Model 1 thấp hơn 0,1 điểm (dưới ngưỡng 0,5).
- **Model 1 giữ cách train 200 epoch.** Early stopping nhanh hơn khoảng 2 lần nhưng val thấp hơn 5,9 điểm, nên không đạt luật đã chốt. Test cũng thấp hơn 9,7 điểm (khoảng tin cậy 95% [+7,2; +12,3], bootstrap theo nhóm như ở mục Kết quả).
- **Early stopping hợp với baseline nhưng không hợp với Model 1.**
  - Không có augmentation, baseline học thuộc ảnh train rất nhanh: train accuracy đạt 99,7% ở epoch 22 và 100% từ epoch 31, trong khi val mới 65,7% ở epoch 22. Val tăng tiếp lên 75,7 ở epoch 49, phần lớn sau các lần giảm learning rate. Sau epoch 49, val không tăng nữa dù learning rate còn giảm thêm 2 lần, nên dừng sớm là hợp lý.
  - Có augmentation, ảnh train mỗi epoch một khác nên Model 1 học chậm hơn nhiều: ở epoch 92 train accuracy mới 90,8%. Val lên xuống giữa các epoch, nhiều lần 5 epoch liền không vượt được mức cao nhất cũ, nên `ReduceLROnPlateau` giảm learning rate 5 lần, xuống dưới 1e-4 từ epoch 88. Learning rate quá nhỏ để model học tiếp, và early stopping dừng khi model chưa học xong.
  - Bản 200 epoch tới epoch 100 cũng mới đạt val tốt nhất 76,8. Phần lớn số điểm còn lại đến trong 100 epoch sau, khi learning rate giảm dần theo cosine về 0 (val tốt nhất 84,9 ở epoch 179).
- **Augmentation cần train lâu mới phát huy tác dụng.** Cùng cách train ngắn, Model 1 chỉ hơn baseline 1,9 điểm test (khoảng tin cậy [−1,7; +5,4], chưa chắc có ý nghĩa), vì tới lúc dừng Model 1 vẫn chưa học xong. Train 200 epoch thì Model 1 hơn baseline 11,6 điểm [+8,4; +14,9]. Baseline chưa được thử train 200 epoch, nên chưa tách được chính xác phần nào của 11,6 điểm là nhờ augmentation.
- Mỗi cấu hình chỉ chạy một seed.

## Cấu hình của Model 1 và Model 2 được chọn thế nào

Hai model này trước đây viết bằng PyTorch và được chọn cấu hình chỉ dựa trên val (trung bình 3 seed, luật giữ hay bỏ chốt trước khi chạy): Model 1 qua một lượt thử từng thay đổi, Model 2 qua 7 vòng thí nghiệm. Bản Keras giữ nguyên cấu hình cuối. Chi tiết từng vòng nằm trong README cũ (`git show 77331de:README.md`). Tóm tắt, val tính so với cấu hình đang giữ ở từng vòng:

| Thay đổi | Model | Val | Quyết định |
|---|---|---|---|
| MaxBlurPool thay cho MaxPool | 1 | +1,2 | giữ |
| SWA, CutMix, SAM, augmentation mạnh hơn, bỏ pooling ở khối cuối | 1 | −0,3 đến +0,7 | bỏ |
| ResNet-10 thay cho ResNet-18 (cùng độ rộng) | 2 | ngang nhau, ít tham số hơn | giữ ResNet-10 |
| Stochastic depth 0,1 | 2 | +1,2 | giữ |
| Chống răng cưa ở khối thu nhỏ (BlurPool) và 96 kênh | 2 | +2,4 | giữ (tổ hợp này được xét sau khi đã thấy kết quả, khác luật chốt trước) |
| VGG-11, GoogLeNet, DenseNet-121, AlexNet, khối NiN | 2 | −24 đến −0,8 | bỏ |
| Train 300 epoch, sau đó ảnh 160×160 + 450 epoch | 2 | +0,7, rồi +2,1 | giữ |
| 128 kênh, SAM | 2 | +0,8 và +0,5 | đạt nhưng không ghép vào cấu hình cuối |
| MixUp/CutMix, khối SE, lấy mẫu cân bằng lớp, nhiễu hoặc làm mờ Gauss | 2 | −1,2 đến +0,6 | bỏ |

Khi chuyển sang Keras, vài chỗ chỉ làm gần đúng vì Keras không có lớp tương ứng hoặc mặc định khác PyTorch:
- phóng to ngẫu nhiên luôn vào giữa ảnh và giữ tỉ lệ khung, trong khi `RandomResizedCrop` của PyTorch cắt ở vị trí và tỉ lệ ngẫu nhiên;
- `RandAugment` một phép thay cho `TrivialAugmentWide`;
- learning rate tăng tuyến tính rồi giảm theo cosine, thay cho OneCycle (không đổi momentum theo chu kỳ);
- khởi tạo trọng số, epsilon của AdamW, kiểu số (float32 thay cho bfloat16) và cách resize ảnh val/test theo mặc định của Keras;
- epoch tốt nhất được chọn theo val accuracy không TTA.

Trên val (trung bình 10 epoch cuối, không TTA), bản Keras thấp hơn bản PyTorch khoảng 1,8 điểm ở cả Model 1 và Model 2, trong khi train accuracy lại cao hơn. Nguyên nhân nhiều khả năng là cách phóng to vào giữa ảnh ở trên: ảnh bị phóng to nên mờ hơn, và vị trí cắt không ngẫu nhiên. Phần này chưa được sửa.

## Lưu ý khi so sánh các model

- Model 3 dùng trọng số đã pretrain trên ImageNet (1,28 triệu ảnh), còn Model 1 và 2 chỉ học từ 3.436 ảnh train.
- Model 2 train ở ảnh lớn hơn và lâu hơn Model 1 (160×160 và 450 epoch, so với 128×128 và 200 epoch). Ở bản PyTorch, khi train cùng cách với Model 1, Model 2 hơn Model 1 khoảng 2,2 điểm test; phần chênh còn lại đến từ ảnh lớn hơn và train lâu hơn.
- Test của Model 2 bản PyTorch đã được xem 3 lần trong quá trình nâng cấp. Mỗi bản chỉ được chọn bằng val, nhưng việc nâng cấp tiếp được quyết định sau khi đã thấy test của bản trước, nên con số test của Model 2 có thể hơi lạc quan.
- Mỗi cấu hình Keras chỉ train một lần (seed 42). Kết quả thay đổi một chút giữa các lần chạy: bản PyTorch dao động 0,2–0,8 điểm test giữa 3 seed. Với 751 ảnh test, 1 điểm chỉ ứng với khoảng 7–8 ảnh, nên chênh lệch dưới khoảng 2–3 điểm chưa chắc có ý nghĩa.
- Công thức của Model 3 được chọn trên val qua một vòng thử (mục "Nâng cấp Model 3"), Model 1 qua một lượt thử và Model 2 qua 7 vòng ở bản PyTorch.
- Test của Model 3 đã được xem 2 lần: công thức ban đầu và công thức B. Công thức B được chọn chỉ bằng val, nhưng việc nâng cấp được quyết định sau khi đã thấy test của công thức ban đầu, giống trường hợp của Model 2.
- Không so trực tiếp với 78,9% của bài báo gốc. Bài báo dùng CaffeNet đã pretrain trên ImageNet và chia 5 fold ngẫu nhiên theo từng ảnh, cách chia này cũng bị rò rỉ ảnh cùng cảnh như mô tả ở trên.
- Pipeline trong `src/` chia ngẫu nhiên 70/15/15 theo từng ảnh (lưu ở `artifacts/split_manifest.json`), không theo nhóm như `youonlycodeonce/splits/`, nên không chung test set với ba model trong `youonlycodeonce/` và có thể bị rò rỉ ảnh cùng cảnh như mô tả ở mục chia dữ liệu.
