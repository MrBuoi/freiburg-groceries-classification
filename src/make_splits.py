"""Tạo cách chia train/val/test dùng chung cho cả nhóm, chia theo nhóm ảnh giống nhau để tránh rò rỉ.

Bộ Freiburg có nhiều ảnh chụp lại cùng một kệ hàng từ góc hơi khác. Nếu chia ngẫu nhiên theo
từng ảnh, một ảnh nằm ở test còn "anh em" của nó nằm ở train, model chỉ cần nhớ cảnh là đoán
đúng (với cách chia cũ, khoảng 20% ảnh test rơi vào trường hợp này). Script này gom các ảnh
giống nhau thành một nhóm rồi chia theo nhóm, 70/15/15 trong từng lớp:
  1. Ảnh trùng nhau hoàn toàn về pixel → cùng nhóm.
  2. Hai ảnh cùng lớp khớp hình học với nhau (ORB + RANSAC, ít nhất 10 điểm khớp) → cùng nhóm.
     Thử trên 40.000 cặp ảnh khác lớp, không cặp nào vượt quá 5 điểm khớp. Ngoài ảnh cùng cảnh,
     luật này cũng nối các ảnh có cùng mẫu bao bì (cùng logo, cùng hãng) dù chụp ở cảnh khác,
     và các cặp nối tiếp nhau thành chuỗi. Ví dụ nhóm 28 ảnh lớn nhất của lớp TEA gồm nhiều loại
     trà của cùng một hãng. Vì vậy cách chia chặt hơn việc chỉ tách theo cảnh: một số mẫu bao bì
     ở val/test không có trong train, và val/test của vài lớp có nhiều ảnh từ cùng một nhóm.
  3. Hai ảnh cùng lớp khớp ít nhất 6 điểm và có số thứ tự file cách nhau không quá 3 (chụp
     liên tiếp) → cùng nhóm. Một cặp ảnh cùng lớp bất kỳ chỉ có 2.7% khả năng có số file liền
     nhau như vậy, trong khi gần một nửa các cặp khớp rất mạnh (≥ 30 điểm) có số file liền nhau.
     Luật này bắt thêm các ảnh chụp lại cùng kệ hàng nhưng khớp yếu hơn.

Kết quả ghi vào splits/{train,val,test}.txt (mỗi dòng một ảnh, dạng LỚP/TÊN_FILE.png) và
splits/groups.csv. train_simple_cnn.py và các model khác đọc các file này, nên chỉ cần chạy lại
script khi dữ liệu thay đổi. Script dừng, không ghi đè splits/, nếu data/ không có đủ 25 lớp và
4.947 ảnh. Cần thêm thư viện: pip install scikit-image

Chạy:
    python src/make_splits.py
"""

import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES, EXPECTED_IMAGES = 25, 4947  # bộ Freiburg Groceries đầy đủ
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

N_KEYPOINTS = 256  # số điểm đặc trưng ORB tối đa mỗi ảnh
MAX_HAMMING = 64  # hai descriptor (256 bit) khác nhau quá 64 bit thì không coi là khớp
RATIO_TEST = 0.8  # điểm khớp tốt nhất phải rõ ràng tốt hơn điểm khớp thứ hai
MIN_MATCHES = 6  # cần ít nhất chừng này cặp điểm khớp mới chạy RANSAC
MIN_INLIERS = 10  # ngưỡng để coi hai ảnh là cùng cảnh hoặc cùng mẫu bao bì
MIN_INLIERS_CONSECUTIVE = 6  # ngưỡng thấp hơn cho hai ảnh chụp liên tiếp
MAX_SHOT_GAP = 3  # hai file có số thứ tự cách nhau không quá 3 được coi là chụp liên tiếp


# 1. Đọc ảnh và trích đặc trưng ORB
def list_images():
    class_names = sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    images = []
    for class_name in class_names:
        images += sorted(
            f"{class_name}/{p.name}"
            for p in (DATA_DIR / class_name).iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return class_names, images


def crop_padding(rgb):
    """Bỏ viền xám (128, 128, 128) mà tác giả bộ dữ liệu chèn vào, để hai ảnh không bị coi là
    giống nhau chỉ vì cùng có viền."""
    gray = np.all(np.abs(rgb.astype(np.int16) - 128) <= 3, axis=2)

    def span(is_padding):
        lo, hi = 0, len(is_padding)
        while lo < hi and is_padding[lo]:
            lo += 1
        while hi > lo and is_padding[hi - 1]:
            hi -= 1
        return lo, hi

    top, bottom = span(gray.mean(axis=1) >= 0.98)
    left, right = span(gray.mean(axis=0) >= 0.98)
    if bottom - top < 32 or right - left < 32:
        return slice(None), slice(None)
    return slice(top, bottom), slice(left, right)


def describe(rel_path):
    """Trả về mã băm của pixel, toạ độ các điểm đặc trưng (số phức x + iy) và descriptor ORB."""
    from skimage.feature import ORB

    with Image.open(DATA_DIR / rel_path) as image:
        rgb = np.asarray(image.convert("RGB"))
        luma = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    digest = hashlib.md5(rgb.tobytes()).hexdigest()
    rows, cols = crop_padding(rgb)
    orb = ORB(n_keypoints=N_KEYPOINTS, fast_threshold=0.05, n_scales=6, downscale=1.25)
    try:
        orb.detect_and_extract(luma[rows, cols])
    except RuntimeError:  # ảnh quá trơn, không tìm được điểm đặc trưng
        return digest, np.zeros(0, np.complex128), np.zeros((0, 4), np.uint64)
    points = orb.keypoints[:, 1] + 1j * orb.keypoints[:, 0]
    descriptors = np.packbits(orb.descriptors, axis=1).view(np.uint64)  # 256 bit → 4 số uint64
    return digest, points, descriptors


# 2. So khớp hình học giữa các cặp ảnh trong cùng một lớp
def similarity_inliers(src, dst, rng, n_hypotheses=400):
    """Số điểm khớp đúng nhiều nhất của một phép biến đổi dst ≈ a·src + b (phóng to/thu nhỏ
    0.5–2 lần, xoay dưới 30°), thử trên các cặp điểm chọn ngẫu nhiên (RANSAC)."""
    n = len(src)
    if n * (n - 1) // 2 <= n_hypotheses:
        p, q = np.triu_indices(n, 1)
    else:
        p, q = rng.integers(n, size=n_hypotheses), rng.integers(n, size=n_hypotheses)
        p, q = p[p != q], q[p != q]
    dz = src[p] - src[q]
    keep = np.abs(dz) > 8  # hai điểm quá gần nhau cho ra phép biến đổi không ổn định
    p, q, dz = p[keep], q[keep], dz[keep]
    if len(p) == 0:
        return 0
    a = (dst[p] - dst[q]) / dz
    b = dst[p] - a * src[p]
    keep = (np.abs(a) > 0.5) & (np.abs(a) < 2.0) & (np.abs(np.angle(a)) < np.pi / 6)
    if not keep.any():
        return 0
    residual = np.abs(a[keep, None] * src[None, :] + b[keep, None] - dst[None, :])
    return int((residual < 4.0).sum(axis=1).max())


def class_edges(task):
    """Các cặp ảnh trong một lớp có ít nhất MIN_INLIERS_CONSECUTIVE điểm khớp hình học."""
    ids, points, descriptors, valid = task  # mỗi ảnh có tối đa N_KEYPOINTS điểm, phần thiếu valid=False
    edges = []
    for a in range(len(ids) - 1):
        for start in range(a + 1, len(ids), 32):  # so một ảnh với 32 ảnh khác mỗi lần cho nhanh
            js = np.arange(start, min(start + 32, len(ids)))
            dist = np.bitwise_count(descriptors[a][None, :, None, :] ^ descriptors[js][:, None, :, :])
            dist = dist.sum(axis=-1, dtype=np.uint16)  # (ảnh j, điểm của a, điểm của j)
            dist[:, ~valid[a], :] = 999
            dist[np.broadcast_to(~valid[js][:, None, :], dist.shape)] = 999
            two_best = np.partition(dist, 1, axis=2)[:, :, :2]
            best_in_j = dist.argmin(axis=2)
            best_in_a = dist.argmin(axis=1)
            mutual = np.take_along_axis(best_in_a, best_in_j, axis=1) == np.arange(N_KEYPOINTS)
            good = (
                mutual
                & (two_best[:, :, 0] <= MAX_HAMMING)
                & (two_best[:, :, 0] < RATIO_TEST * two_best[:, :, 1])
            )
            for k in np.where(good.sum(axis=1) >= MIN_MATCHES)[0]:
                j = js[k]
                matched = np.where(good[k])[0]
                rng = np.random.default_rng([SPLIT_SEED, int(ids[a]), int(ids[j])])
                inliers = similarity_inliers(
                    points[a, matched], points[j, best_in_j[k, matched]], rng
                )
                if inliers >= MIN_INLIERS_CONSECUTIVE:
                    edges.append((int(ids[a]), int(ids[j]), inliers))
    return edges


def shot_number(rel_path):
    """Số thứ tự ở cuối tên file, ví dụ BEANS/BEANS0083.png → 83 (None nếu không có)."""
    match = re.search(r"(\d+)$", Path(rel_path).stem)
    return int(match.group(1)) if match else None


def same_scene(rel_a, rel_b, inliers):
    if inliers >= MIN_INLIERS:
        return True
    a, b = shot_number(rel_a), shot_number(rel_b)
    return a is not None and b is not None and abs(a - b) <= MAX_SHOT_GAP


# 3. Gom nhóm và chia theo nhóm
def find_groups(n, edges):
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in edges:
        parent[root(i)] = root(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[root(i)].append(i)
    return sorted(groups.values())


def assign_splits(groups, image_class, class_names):
    """Trong từng lớp, xáo các nhóm rồi lần lượt đưa mỗi nhóm vào tập đang thiếu nhiều nhất
    (tính theo tỉ lệ so với mục tiêu 70/15/15)."""
    rng = random.Random(SPLIT_SEED)
    split_of = {}
    for class_name in class_names:
        class_groups = [g for g in groups if image_class[g[0]] == class_name]
        rng.shuffle(class_groups)
        total = sum(len(g) for g in class_groups)
        counts = dict.fromkeys(SPLIT_RATIOS, 0)
        for group in class_groups:
            name = max(
                SPLIT_RATIOS,
                key=lambda s: (SPLIT_RATIOS[s] * total - counts[s]) / (SPLIT_RATIOS[s] * total),
            )
            counts[name] += len(group)
            for i in group:
                split_of[i] = name
    return split_of


def main():
    for stream in (sys.stdout, sys.stderr):  # Windows: tránh lỗi khi in tiếng Việt ra file/pipe
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    class_names, images = list_images()
    if (len(class_names), len(images)) != (EXPECTED_CLASSES, EXPECTED_IMAGES):
        sys.exit(
            f"{DATA_DIR} có {len(class_names)} lớp, {len(images)} ảnh; cần {EXPECTED_CLASSES} lớp, "
            f"{EXPECTED_IMAGES} ảnh (xem mục tải dữ liệu trong README). Không ghi đè {SPLIT_DIR}."
        )
    image_class = [rel.split("/")[0] for rel in images]
    workers = max(1, min(12, (os.cpu_count() or 2) - 1))
    print(f"{len(images)} ảnh, {len(class_names)} lớp | trích đặc trưng ORB bằng {workers} tiến trình...")
    with Pool(workers) as pool:
        described = pool.map(describe, images, chunksize=16)

    # ảnh trùng pixel hoàn toàn
    by_digest = defaultdict(list)
    for i, (digest, _, _) in enumerate(described):
        by_digest[digest].append(i)
    duplicate_edges = []
    for same in by_digest.values():
        if len({image_class[i] for i in same}) > 1:
            raise ValueError(f"Ảnh trùng pixel nhưng khác lớp: {[images[i] for i in same]}")
        duplicate_edges += [(same[0], i) for i in same[1:]]

    # so khớp ORB trong từng lớp; xếp lớp đông ảnh lên trước để chia việc đều hơn
    tasks = []
    for class_name in sorted(class_names, key=lambda c: -image_class.count(c)):
        ids = np.array([i for i, c in enumerate(image_class) if c == class_name])
        points = np.zeros((len(ids), N_KEYPOINTS), np.complex128)
        descriptors = np.zeros((len(ids), N_KEYPOINTS, 4), np.uint64)
        valid = np.zeros((len(ids), N_KEYPOINTS), bool)
        for row, i in enumerate(ids):
            _, pts, desc = described[i]
            points[row, : len(pts)], descriptors[row, : len(desc)], valid[row, : len(desc)] = pts, desc, True
        tasks.append((ids, points, descriptors, valid))
    print("So khớp các cặp ảnh trong từng lớp...")
    with Pool(workers) as pool:
        matches = [edge for edges in pool.map(class_edges, tasks, chunksize=1) for edge in edges]
    orb_edges = [(i, j) for i, j, n in matches if same_scene(images[i], images[j], n)]
    strong = sum(1 for _, _, n in matches if n >= MIN_INLIERS)

    groups = find_groups(len(images), duplicate_edges + orb_edges)
    split_of = assign_splits(groups, image_class, class_names)

    # kiểm tra: không cặp ảnh cùng cảnh nào bị tách sang hai tập khác nhau
    crossing = [(i, j) for i, j in duplicate_edges + orb_edges if split_of[i] != split_of[j]]
    assert not crossing, f"{len(crossing)} cặp ảnh cùng cảnh nằm ở hai tập khác nhau"

    SPLIT_DIR.mkdir(exist_ok=True)
    for name in SPLIT_RATIOS:
        lines = sorted(images[i] for i in range(len(images)) if split_of[i] == name)
        (SPLIT_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    group_id = {i: k for k, group in enumerate(groups) for i in group}
    rows = [f"{images[i]},{group_id[i]},{split_of[i]}" for i in range(len(images))]
    (SPLIT_DIR / "groups.csv").write_text("image,group,split\n" + "\n".join(rows) + "\n", encoding="utf-8")

    sizes = sorted((len(g) for g in groups), reverse=True)
    multi = [s for s in sizes if s > 1]
    print(
        f"Cặp trùng pixel: {len(duplicate_edges)} | cặp cùng cảnh: {len(orb_edges)} "
        f"({strong} khớp mạnh, {len(orb_edges) - strong} chụp liên tiếp) | "
        f"{len(groups)} nhóm, trong đó {len(multi)} nhóm nhiều ảnh gồm {sum(multi)} ảnh "
        f"(nhóm lớn nhất {sizes[0]} ảnh)"
    )
    for name in SPLIT_RATIOS:
        count = sum(1 for s in split_of.values() if s == name)
        print(f"  {name:5s}: {count:4d} ảnh ({count / len(images):.1%})")
    print(f"Đã ghi {SPLIT_DIR}/train.txt, val.txt, test.txt và groups.csv")


if __name__ == "__main__":
    main()
