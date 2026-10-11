"""Model 1: Simple CNN phân loại 25 loại sản phẩm (Freiburg Groceries), train từ đầu.

Chạy: python youonlycodeonce/split_data.py --split nhom   (một lần)
      KERAS_BACKEND=torch python youonlycodeonce/train_simple_cnn.py --split nhom
(KERAS_BACKEND=torch để train trên GPU qua PyTorch, Mac hay NVIDIA đều được; máy chỉ có CPU thì bỏ đi.)
"""
import argparse
import gc
import json
from pathlib import Path

import keras
import numpy as np
from keras import layers
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="nhom", choices=["nhom", "ngaunhien"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)

root = Path(__file__).resolve().parents[1]
data_dir = root / "artifacts" / "split_data" / args.split
save_path = root / "checkpoints" / f"simple_cnn_{args.split}_seed{args.seed}"

# 1. Đọc dữ liệu
train_data = keras.utils.image_dataset_from_directory(data_dir / "train", label_mode="categorical",
    batch_size=None, image_size=(128, 128), shuffle=True, seed=args.seed)
val_data = keras.utils.image_dataset_from_directory(data_dir / "val", label_mode="categorical",
    batch_size=64, image_size=(160, 160), shuffle=False)  # 160 > 128 vì ảnh train bị zoom vào
test_data = keras.utils.image_dataset_from_directory(data_dir / "test", label_mode="categorical",
    batch_size=64, image_size=(160, 160), shuffle=False)
class_names = train_data.class_names
zoom, flip = layers.RandomZoom((-0.4, 0)), layers.RandomFlip("horizontal")  # số âm = zoom vào
rand_augment = layers.RandAugment(num_ops=1)  # mỗi ảnh 1 phép, gần giống TrivialAugmentWide
train_data = train_data.map(lambda x, y: (rand_augment(flip(zoom(x))), y), num_parallel_calls=8)
train_data = train_data.batch(64)  # augment xong mới chia batch để mỗi ảnh có phép RandAugment riêng

# 2. Xây dựng model
model = keras.Sequential([keras.Input(shape=(None, None, 3))])  # None: nhận cả ảnh 128 lẫn 160
model.add(layers.Rescaling(1 / 255))  # mean, std ở dòng dưới tính trên ảnh train
model.add(layers.Normalization(mean=(0.521, 0.458, 0.391), variance=(0.259**2, 0.247**2, 0.245**2)))
model.add(layers.RandomErasing(factor=(0.25, 0.25), fill_value=0))  # xoá 1 vùng ở 25% ảnh train
for i in range(6):
    model.add(layers.Conv2D(32 * 2**i, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(2, strides=1, padding="same"))  # MaxBlurPool = max bước 1
    model.add(layers.AveragePooling2D(2))  # + trung bình 2x2, đỡ nhạy khi vật lệch vài pixel
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(len(class_names), activation="softmax"))

# 3. Train
steps = args.epochs * len(train_data)  # lr tăng dần trong 30% số bước đầu rồi giảm theo cosine về 0
lr = keras.optimizers.schedules.CosineDecay(
    2e-3 / 25, int(0.7 * steps), warmup_target=2e-3, warmup_steps=int(0.3 * steps))
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.05),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
checkpoint = keras.callbacks.ModelCheckpoint(
    f"{save_path}.weights.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True)
# với PyTorch, Keras để sót tensor trong vòng tham chiếu, phải gọi gc mỗi epoch để không tràn RAM
free_memory = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())
history = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                    callbacks=[checkpoint, free_memory], verbose=2).history
model.load_weights(f"{save_path}.weights.h5")  # lấy lại trọng số của epoch có val accuracy cao nhất

# 4. Đánh giá trên tập test
x_test = np.concatenate([x for x, y in test_data])
y_test = np.concatenate([y for x, y in test_data]).argmax(axis=1)
probs = model.predict(x_test, verbose=0)
y_pred = (probs + model.predict(np.flip(x_test, axis=2), verbose=0)).argmax(axis=1)  # TTA lật ngang
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
val_acc = history["val_accuracy"]
results = {"split": args.split, "seed": args.seed, "best_val_accuracy": max(val_acc),
           "best_epoch": int(np.argmax(val_acc)) + 1, "test_acc": np.mean(y_pred == y_test),
           "test_acc_no_tta": np.mean(probs.argmax(axis=1) == y_test),
           "macro_f1": f1_score(y_test, y_pred, average="macro"),
           "mean_per_class_acc": balanced_accuracy_score(y_test, y_pred), "history": history}
Path(f"{save_path}.json").write_text(json.dumps(results, indent=1))
print("Test accuracy {test_acc:.3f} (không TTA {test_acc_no_tta:.3f})".format(**results))
