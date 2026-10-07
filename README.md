# Freiburg Groceries Classification (Deep Learning Comparison)

Dự án so sánh hiệu quả giữa 3 kiến trúc Deep Learning từ đơn giản đến phức tạp trên bộ dữ liệu Freiburg Groceries (25 nhóm sản phẩm):
1. **Model 1**: Simple CNN (Baseline), file `src/train_simple_cnn.py`
2. **Model 2**: Deep Convolutional Neural Network (Deep CNN)
3. **Model 3**: Transfer Learning (YOLOv5 Classification Backbone)

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
pip install -r requirements-tensorflow.txt   # chỉ khi cần chạy src/mnist.py
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

## Cách chia train/val/test dùng chung

Ba file `splits/train.txt`, `splits/val.txt`, `splits/test.txt` (3.436 / 760 / 751 ảnh) là cách chia **chung cho cả 3 model**, để mọi model được chấm trên cùng một test set. Mỗi dòng là một ảnh, dạng `LỚP/TÊN_FILE.png`.

Bộ dữ liệu có nhiều ảnh chụp lại cùng một kệ hàng từ góc hơi khác. Nếu chia ngẫu nhiên theo từng ảnh, khoảng 20% ảnh test có một ảnh "anh em" cùng cảnh nằm trong train, khiến accuracy cao hơn thực tế khoảng 2 điểm. Vì vậy các ảnh giống nhau được gom thành nhóm rồi chia theo nhóm, 70/15/15 trong từng lớp.

Ngoài ảnh cùng cảnh, cách gom này còn nối các ảnh có cùng mẫu bao bì (cùng logo, cùng hãng) dù chụp ở cảnh khác. Vì vậy cách chia này chặt hơn việc chỉ tách theo cảnh:
- một số mẫu bao bì ở val/test không có trong train;
- ở vài lớp (TEA, HONEY, CAKE, RICE, CORN), gần một nửa trở lên số ảnh val hoặc test thuộc cùng một nhóm, nên accuracy của riêng các lớp này dao động nhiều.

Cách gom nhóm được mô tả trong `src/make_splits.py`.

- Đọc split trong code Python (từ thư mục nào cũng được, miễn là `sys.path` trỏ tới `src/`):
  ```python
  import sys; sys.path.insert(0, "src")   # notebook trong thư mục notebooks/ thì dùng "../src"
  from train_simple_cnn import load_split

  class_names, train, val, test = load_split()  # list các cặp (đường dẫn ảnh, nhãn)
  ```
- Không sửa tay các file trong `splits/`. Chỉ tạo lại bằng `python src/make_splits.py` (khoảng 1,5 phút) khi dữ liệu thay đổi. Script tự dừng, không ghi đè, nếu `data/` chưa đủ 25 lớp và 4.947 ảnh.
- Chỉ dùng val để chọn cấu hình. Test chỉ đánh giá một lần, cho cấu hình cuối cùng.

## Model 1: Simple CNN (baseline)

`src/train_simple_cnn.py` train từ đầu một CNN đơn giản, không dùng trọng số pretrained:

- **Kiến trúc:** 6 khối Conv 3×3 → BatchNorm → ReLU → MaxBlurPool, rồi Global Average Pooling, Dropout và một lớp Linear (6,3 triệu tham số). MaxBlurPool là max pooling chống răng cưa: lấy max 2×2 nhưng trượt từng pixel, rồi lấy trung bình 2×2 khi thu nhỏ, nên dự đoán ít bị đổi khi vật thể dịch đi vài pixel.
- **Huấn luyện:**
  - ảnh 128×128, augmentation mạnh (RandomResizedCrop, lật ngang, TrivialAugmentWide, RandomErasing), label smoothing;
  - AdamW với lịch learning rate OneCycle trong 200 epoch;
  - lưu model của epoch có val accuracy cao nhất.
- **Đánh giá:** ảnh được phóng lên 160×160, lấy trung bình dự đoán của ảnh gốc và ảnh lật ngang (TTA).

Thử cấu hình (chỉ báo val accuracy):

```bash
python src/train_simple_cnn.py --epochs 50 --num_blocks 5
```

Chạy cấu hình cuối và đánh giá trên test set, mỗi seed một lần:

```bash
python src/train_simple_cnn.py --eval_test 1 --seed 42
python src/train_simple_cnn.py --eval_test 1 --seed 1
python src/train_simple_cnn.py --eval_test 1 --seed 2
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

Với SWA, val accuracy là của model trung bình. Ở ảnh 160×160, MaxPool thường bỏ mất hàng và cột cuối của feature map ở khối cuối, còn MaxBlurPool thì không. Vì vậy một phần mức tăng của MaxBlurPool có thể đến từ điểm này chứ không chỉ từ việc chống răng cưa.

## Lưu ý khi so sánh các model

- Báo cáo trung bình ± độ lệch chuẩn qua ít nhất 3 seed (`--seed`). Với 751 ảnh test, chênh lệch dưới khoảng 2–3 điểm chưa chắc có ý nghĩa.
- Không so trực tiếp với 78,9% của bài báo gốc. Bài báo dùng CaffeNet đã pretrain trên ImageNet và chia 5 fold ngẫu nhiên theo từng ảnh, cách chia này cũng bị rò rỉ ảnh cùng cảnh như mô tả ở trên.
- `src/train_baseline.py` dùng cách chia riêng nên không chung test set với các model trên.
- Cùng một seed chỉ tái lập đúng kết quả khi cùng thiết bị (MPS/CUDA/CPU), cùng phiên bản thư viện và cùng `--num_workers`.
