# Freiburg Groceries Classification (Deep Learning Comparison)

Dự án so sánh hiệu quả giữa 3 kiến trúc Deep Learning từ đơn giản đến phức tạp trên bộ dữ liệu Freiburg Groceries (25 nhóm sản phẩm):
1. **Model 1**: Simple CNN (Baseline), file `src/train_simple_cnn.py`
2. **Model 2**: Deep Convolutional Neural Network (Deep CNN), file `src/train_deep_cnn.py`
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

Thử cấu hình (chỉ báo val accuracy). Nhớ đặt `--checkpoint` riêng cho mỗi lần thử: mặc định mọi lần chạy cùng seed đều ghi vào `checkpoints/simple_cnn_seed<seed>.pt` và `checkpoints/simple_cnn_seed<seed>.history.json`, nên lần thử chạy sau sẽ xoá model và kết quả test của lần chạy cuối.

```bash
python src/train_simple_cnn.py --epochs 50 --num_blocks 5 --checkpoint checkpoints/simple_cnn_try.pt
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

## Model 2: Deep CNN (ResNet)

`src/train_deep_cnn.py` train từ đầu một mạng kiểu ResNet, không dùng trọng số pretrained:

- **Kiến trúc:**
  - stem gồm 3 Conv 3×3 (Conv đầu bước 2) và MaxBlurPool, thu nhỏ ảnh 4 lần;
  - 4 tầng, mỗi tầng 1 khối residual (bố cục ResNet-10), số kênh 96 → 192 → 384 → 768;
  - Global Average Pooling, Dropout và một lớp Linear (11,1 triệu tham số).
- **Khối residual:** hai Conv 3×3 và một đường tắt cộng đầu vào vào đầu ra. Ở khối thu nhỏ ảnh (đầu tầng 2, 3, 4), đường tắt dùng AvgPool 2×2 + Conv 1×1, còn đường chính chống răng cưa: Conv 3×3 bước 1 → BatchNorm → ReLU → BlurPool (làm mờ bằng bộ lọc [1, 2, 1] × [1, 2, 1] / 16 rồi lấy mẫu bước 2) thay cho Conv bước 2. γ của BatchNorm cuối mỗi khối khởi tạo bằng 0.
- **Huấn luyện và đánh giá:** giống hệt Model 1 (dùng chung code đọc dữ liệu, augmentation, công thức train, đánh giá ở 160×160 với TTA), thêm stochastic depth 0,1: lúc train, ngẫu nhiên bỏ đường chính của một khối với xác suất tăng dần theo độ sâu.

Thử cấu hình (chỉ báo val accuracy), nhớ đặt `--checkpoint` riêng để không ghi đè model và kết quả test của lần chạy cuối:

```bash
python src/train_deep_cnn.py --blocks 2,2,2,2 --checkpoint checkpoints/deep_cnn_r18.pt
```

Chạy cấu hình cuối và đánh giá trên test set, mỗi seed một lần:

```bash
python src/train_deep_cnn.py --eval_test 1 --seed 42
python src/train_deep_cnn.py --eval_test 1 --seed 1
python src/train_deep_cnn.py --eval_test 1 --seed 2
```

- Mỗi seed lưu `checkpoints/deep_cnn_seed<seed>.pt` và `checkpoints/deep_cnn_seed<seed>.history.json` (kèm kết quả test).
- Thời gian: khoảng 11 phút mỗi seed trên Apple M5 Max khi chạy một mình (khoảng 19 phút nếu chạy 2 seed cùng lúc).
- Bản đầu của Model 2 (64 kênh, không chống răng cưa) chạy lại được bằng `--width 64 --aa 0`, kèm `--checkpoint` riêng (ví dụ `--checkpoint checkpoints/deep_cnn_v1_seed42.pt`) để không ghi đè model và kết quả test của Model 2.

### Kết quả

Trung bình ± độ lệch chuẩn qua 3 seed (42, 1, 2). "Model 2 bản đầu" là cấu hình trước khi nâng cấp (64 kênh, không chống răng cưa):

| | Model 1 | Model 2 bản đầu | Model 2 |
|---|---|---|---|
| Val, có TTA (trung bình 10 epoch cuối) | 87,1 ± 0,1 | 87,1 ± 0,5 | 89,5 ± 0,4 |
| **Test, có TTA** | **85,0 ± 0,8** | **85,8 ± 0,7** | **87,2 ± 0,7** |
| Test, không TTA | 84,2 ± 0,8 | 85,5 ± 0,9 | 87,1 ± 0,5 |
| Test, trung bình accuracy của 25 lớp | 84,0 ± 0,9 | 84,5 ± 0,7 | 85,5 ± 1,2 |
| Tham số | 6,3 triệu | 4,9 triệu | 11,1 triệu |

Test accuracy theo từng seed (42 / 1 / 2): Model 2 87,7 / 86,4 / 87,5; bản đầu 86,4 / 85,1 / 85,8. Trên cùng tập test, Model 2 hơn bản đầu 1,3–1,7 điểm ở cả 3 seed, và hơn Model 1 khoảng 2,2 điểm. Với 751 ảnh test, 1,5 điểm chỉ ứng với khoảng 11 ảnh, nên riêng test chưa đủ để kết luận chắc. Val cho kết quả cùng chiều: trên 6 seed, seed nào Model 2 cũng hơn bản đầu (xem vòng 4 bên dưới). Tuy vậy val cũng chỉ có 760 ảnh và là tập đã dùng để chọn cấu hình, nên mức tăng trên val (+2,0) có thể cao hơn thực tế. TTA gần như không còn giúp (87,1% không TTA, 87,2% có TTA). Model 2 tự tin thấp: trên val, xác suất trung bình cho dự đoán là 60% so với 72% của Model 1, dù accuracy cao hơn (89,5% so với 87,6%, checkpoint seed 42).

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

Model 1 trong cùng điều kiện thí nghiệm đạt 87,3 ± 1,1. Các lần thử trong bảng, kể cả lần thử Model 1 này, chạy với `--num_workers 6`, còn lần chạy cuối dùng mặc định `--num_workers 8`, nên dù cùng seed, augmentation ngẫu nhiên vẫn khác: cấu hình được chọn đạt 87,9 ± 0,5 trong bảng nhưng 87,1 ± 0,5 khi chạy lại (cột "Model 2 bản đầu" ở mục Kết quả), ngang Model 1 (87,1 ± 0,1). Vì vậy chênh lệch dưới khoảng 1 điểm, giữa các dòng hay giữa hai model, có thể chỉ là nhiễu. Với 3.436 ảnh train và train từ đầu, mạng sâu hơn không tự động tốt hơn: độ rộng (số kênh) và cách chống overfit quan trọng hơn độ sâu.

#### Vòng 4: nâng cấp sau khi đã có kết quả test của bản đầu

Mọi lần chạy dùng `--num_workers 8` như lần chạy cuối của bản đầu, và chênh lệch so với bản đầu được tính theo từng cặp seed. Với thay đổi không làm đổi số tham số (chống răng cưa, stochastic depth), cùng seed cho khởi tạo, thứ tự ảnh và augmentation giống hệt bản đầu, nên mỗi cặp so sánh đúng cùng điều kiện. Đổi độ rộng hay độ sâu làm đổi số tham số, nên khác cả khởi tạo, thứ tự ảnh và augmentation (model được tạo trước khi DataLoader lấy số ngẫu nhiên); khi đó hai lần chạy chỉ chung số seed. Luật chốt trước khi chạy:
- thay đổi giữ nguyên số tham số phải tăng trung bình ít nhất 0,5 điểm và tăng ở ít nhất 2/3 seed; thay đổi độ rộng hay độ sâu phải tăng trung bình ít nhất 0,8 điểm và tăng ở cả 3 seed;
- nếu hai thay đổi khác loại cùng đạt thì thử tổ hợp của chúng; tổ hợp chỉ được chọn nếu hơn thay đổi đơn tốt nhất ít nhất 0,3 điểm;
- cấu hình được chọn phải được xác nhận thêm trên seed 3, 4, 5 (cùng với bản đầu trên các seed đó): trung bình 6 seed phải tăng ít nhất 0,5 điểm, tăng ở ít nhất 4/6 seed, và trung bình 3 seed mới phải tăng.

| Cấu hình (seed 42, 1, 2) | Tham số | Val accuracy (%) | So với bản đầu | Quyết định |
|---|---|---|---|---|
| Bản đầu: ResNet-10, 64 kênh, stochastic depth 0,1 | 4,9 triệu | 87,1 ± 0,5 | | mốc so sánh |
| + chống răng cưa ở khối thu nhỏ | 4,9 triệu | 88,3 ± 0,5 | +1,2 (3/3 seed) | đạt; trên 6 seed +1,2, cả 6 seed đều tăng |
| Ảnh train 160 × 160 (đánh giá 200 × 200) | 4,9 triệu | 87,6 ± 0,7 | +0,4 (2/3) | bỏ |
| Stochastic depth 0,2 | 4,9 triệu | 87,1 ± 0,8 | +0,0 (1/3) | bỏ |
| Stochastic depth 0,3 | 4,9 triệu | 86,5 ± 0,1 | −0,7 (0/3) | bỏ |
| ResNet-18 + stochastic depth 0,2 | 11,2 triệu | 86,9 (seed 42, 1) | −0,5 (0/2) | bỏ |
| 96 kênh, không chống răng cưa | 11,1 triệu | 88,1 ± 1,3 | +0,9 (2/3) | bỏ |
| **96 kênh + chống răng cưa** | 11,1 triệu | **89,5 ± 0,4** | **+2,4 (3/3)** | **giữ**: trên 6 seed +2,0, cả 6 seed đều tăng; hơn chỉ chống răng cưa +0,8 (5/6 seed) |

Do một lỗi (chống răng cưa được bật làm mặc định trong code khi các lần thử khác chưa chạy xong), 4 lần chạy bị kèm chống răng cưa ngoài ý muốn. Lần chạy seed 2 của ResNet-18 + stochastic depth 0,2 không được tính; hai seed còn lại đều không tăng, nên cấu hình này không đạt dù seed 2 ra sao. Ba lần chạy seed 42, 1, 2 của dòng "96 kênh + chống răng cưa" vốn định là 96 kênh không chống răng cưa (dòng "96 kênh, không chống răng cưa" được chạy bù sau đó). Luật chốt trước chỉ cho thử tổ hợp hai thay đổi khi cả hai cùng đạt, mà riêng 96 kênh không đạt; việc vẫn xét tổ hợp này (chỉ cần hơn chỉ chống răng cưa ít nhất 0,3 điểm) được quyết định sau khi đã thấy 3 seed đó. Vì vậy, bằng chứng không phụ thuộc vào lần chọn này chỉ gồm 3 seed mới 3, 4, 5 (trên val: hơn bản đầu +1,6; hơn chỉ chống răng cưa +0,5, tăng ở 2/3 seed) và tập test (hơn bản đầu 1,3–1,7 điểm ở cả 3 seed).

Chống răng cưa ở các bước thu nhỏ là thay đổi đáng giá nhất, giống như MaxBlurPool ở Model 1. Hai thay đổi được giữ cộng gần như dồn vào nhau: riêng chống răng cưa +1,2, riêng 96 kênh +0,9 (không ổn định giữa các seed), cả hai +2,4. Mạng sâu hơn (ResNet-18), stochastic depth mạnh hơn và ảnh train lớn hơn đều không giúp.

## Lưu ý khi so sánh các model

- Báo cáo trung bình ± độ lệch chuẩn qua ít nhất 3 seed (`--seed`). Với 751 ảnh test, chênh lệch dưới khoảng 2–3 điểm chưa chắc có ý nghĩa.
- Không so trực tiếp với 78,9% của bài báo gốc. Bài báo dùng CaffeNet đã pretrain trên ImageNet và chia 5 fold ngẫu nhiên theo từng ảnh, cách chia này cũng bị rò rỉ ảnh cùng cảnh như mô tả ở trên.
- `src/train_baseline.py` dùng cách chia riêng nên không chung test set với các model trên.
- Cùng một seed chỉ tái lập đúng kết quả khi cùng thiết bị (MPS/CUDA/CPU), cùng phiên bản thư viện và cùng `--num_workers`.
