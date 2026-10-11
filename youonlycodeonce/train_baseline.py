"""Model baseline: Simple CNN như Model 1 nhưng không augmentation, train ngắn với early stopping.

Dùng để so sánh với Model 1 (xem README, mục "Baseline và early stopping").
Chạy: python youonlycodeonce/split_data.py --split nhom   (một lần)
      KERAS_BACKEND=torch python youonlycodeonce/train_baseline.py --split nhom
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
parser.add_argument("--epochs", type=int, default=100)  # tối đa, early stopping thường dừng sớm hơn
parser.add_argument("--lr", type=float, default=3e-3)
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)

root = Path(__file__).resolve().parents[1]
data_dir = root / "artifacts" / "split_data" / args.split
save_path = root / "checkpoints" / f"baseline_{args.split}_seed{args.seed}"

# 1. Đọc dữ liệu: ảnh chỉ được thu về 128x128, không augmentation
train_data = keras.utils.image_dataset_from_directory(data_dir / "train", label_mode="categorical",
    batch_size=64, image_size=(128, 128), shuffle=True, seed=args.seed)
val_data = keras.utils.image_dataset_from_directory(data_dir / "val", label_mode="categorical",
    batch_size=64, image_size=(128, 128), shuffle=False)
test_data = keras.utils.image_dataset_from_directory(data_dir / "test", label_mode="categorical",
    batch_size=64, image_size=(128, 128), shuffle=False)
class_names = train_data.class_names

# 2. Xây dựng model: như Model 1, chỉ bỏ lớp RandomErasing
model = keras.Sequential([keras.Input(shape=(128, 128, 3))])
model.add(layers.Rescaling(1 / 255))  # mean, std ở dòng dưới tính trên ảnh train
model.add(layers.Normalization(mean=(0.521, 0.458, 0.391), variance=(0.259**2, 0.247**2, 0.245**2)))
for i in range(6):
    model.add(layers.Conv2D(32 * 2**i, 3, padding="same", use_bias=False))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D(2, strides=1, padding="same"))  # MaxBlurPool = max bước 1
    model.add(layers.AveragePooling2D(2))  # + trung bình 2x2, đỡ nhạy khi vật lệch vài pixel
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(len(class_names), activation="softmax"))

# 3. Train: learning rate cố định, val không tăng 5 epoch thì giảm một nửa, không tăng 15 epoch thì dừng
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.05),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
checkpoint = keras.callbacks.ModelCheckpoint(
    f"{save_path}.weights.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5)
early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)
# với PyTorch, Keras để sót tensor trong vòng tham chiếu, phải gọi gc mỗi epoch để không tràn RAM
free_memory = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())
history = model.fit(train_data, validation_data=val_data, epochs=args.epochs,
                    callbacks=[checkpoint, reduce_lr, early_stop, free_memory], verbose=2).history
model.load_weights(f"{save_path}.weights.h5")  # lấy lại trọng số của epoch có val accuracy cao nhất

# 4. Đánh giá trên tập test (không TTA)
x_test = np.concatenate([x for x, y in test_data])
y_test = np.concatenate([y for x, y in test_data]).argmax(axis=1)
y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
val_acc = history["val_accuracy"]
results = {"split": args.split, "seed": args.seed, "lr": args.lr, "epochs_run": len(val_acc),
           "best_val_accuracy": max(val_acc), "best_epoch": int(np.argmax(val_acc)) + 1,
           "test_acc": np.mean(y_pred == y_test), "macro_f1": f1_score(y_test, y_pred, average="macro"),
           "mean_per_class_acc": balanced_accuracy_score(y_test, y_pred), "history": history}
Path(f"{save_path}.json").write_text(json.dumps(results, indent=1))
print("Test accuracy {test_acc:.3f} sau {epochs_run} epoch".format(**results))
