"""Model 3: ResNet50 phân loại 25 loại sản phẩm (Freiburg Groceries), dùng trọng số ImageNet.

Chạy: python youonlycodeonce/split_data.py --split nhom   (một lần)
      KERAS_BACKEND=torch python youonlycodeonce/train_resnet50.py --split nhom
(KERAS_BACKEND=torch để train trên GPU qua PyTorch, Mac hay NVIDIA đều được; máy chỉ có CPU thì bỏ đi.)
"""
import argparse
import gc
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="nhom", choices=["nhom", "ngaunhien"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=20)  # số epoch của giai đoạn 2
args = parser.parse_args()
keras.utils.set_random_seed(args.seed)

root = Path(__file__).resolve().parents[1]
data_dir = root / "artifacts" / "split_data" / args.split
save_path = root / "checkpoints" / f"resnet50_{args.split}_seed{args.seed}"


# cắt ngẫu nhiên như RandomResizedCrop: 35-100% diện tích ảnh, tỉ lệ khung 3/4-4/3, ở vị trí bất kỳ
def random_crop(image, size):
    begin, crop, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), tf.zeros((1, 0, 4)),
        area_range=(0.35, 1), aspect_ratio_range=(3 / 4, 4 / 3), use_image_if_no_bounding_boxes=True)
    image = tf.image.crop_to_bounding_box(image, begin[0], begin[1], crop[0], crop[1])
    return tf.image.resize(image, size, antialias=True)


# 1. Đọc dữ liệu
train_data = keras.utils.image_dataset_from_directory(data_dir / "train", label_mode="categorical",
    batch_size=None, image_size=(256, 256), shuffle=True, seed=args.seed)
val_data = keras.utils.image_dataset_from_directory(data_dir / "val", label_mode="categorical",
    batch_size=32, image_size=(320, 320), shuffle=False)  # 320 > 256 vì ảnh train là phần cắt phóng to
test_data = keras.utils.image_dataset_from_directory(data_dir / "test", label_mode="categorical",
    batch_size=32, image_size=(320, 320), shuffle=False)
class_names = train_data.class_names
flip, rand_augment = layers.RandomFlip("horizontal"), layers.RandAugment(num_ops=1)  # mỗi ảnh 1 phép
train_data = train_data.map(lambda x, y: (rand_augment(flip(random_crop(x, (256, 256)))), y),
                            num_parallel_calls=8)
train_data = train_data.batch(32)  # augment xong mới chia batch để mỗi ảnh có phép RandAugment riêng

# 2. Xây dựng model
base_model = keras.applications.ResNet50(weights="imagenet", include_top=False)
model = keras.Sequential([
    keras.Input(shape=(None, None, 3)),  # None: nhận cả ảnh train 256 lẫn ảnh val/test 320
    layers.Lambda(keras.applications.resnet50.preprocess_input),  # chuẩn hoá như lúc train ImageNet
    layers.RandomErasing(factor=(0.25, 0.25), fill_value=0),  # xoá 1 vùng ở 25% ảnh, 0 = màu trung bình
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax"),
])

# 3. Train
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
checkpoint = keras.callbacks.ModelCheckpoint(
    f"{save_path}.weights.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True)
# với PyTorch, Keras để sót tensor trong vòng tham chiếu, phải gọi gc mỗi epoch để không tràn RAM
free_memory = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: gc.collect())
base_model.trainable = False  # giai đoạn 1: khoá ResNet50, train lớp Dense mới với lr lớn
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.05),
              loss=loss, metrics=["accuracy"])
h1 = model.fit(train_data, validation_data=val_data, epochs=2, callbacks=[checkpoint, free_memory],
               verbose=2)
base_model.trainable = True  # giai đoạn 2: fine-tune toàn bộ, đổi trainable phải compile lại
steps = args.epochs * len(train_data)  # lr tăng dần trong epoch đầu rồi giảm theo cosine về 0
lr = keras.optimizers.schedules.CosineDecay(
    1e-4 / 25, steps - len(train_data), warmup_target=1e-4, warmup_steps=len(train_data))
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.05),
              loss=loss, metrics=["accuracy"])
h2 = model.fit(train_data, validation_data=val_data, initial_epoch=2, epochs=2 + args.epochs,
               callbacks=[checkpoint, free_memory], verbose=2)
history = {key: h1.history[key] + h2.history[key] for key in h1.history}  # ghép 2 giai đoạn
model.load_weights(f"{save_path}.weights.h5")  # epoch có val accuracy cao nhất của cả 2 giai đoạn

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
