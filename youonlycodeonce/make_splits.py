"""Chia train/val/test theo nhóm ảnh giống nhau để tránh rò rỉ giữa train và test.
"""
import hashlib
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import ORB

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def describe(name):
    with Image.open(ROOT / "data" / name) as image:
        rgb = np.asarray(image.convert("RGB"))
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255
    digest = hashlib.md5(rgb.tobytes()).hexdigest()
    # bỏ viền xám từ dữ liệu gốc
    border = np.all(np.abs(rgb.astype(np.int16) - 128) <= 3, axis=2)
    rows, cols = np.where(border.mean(axis=1) < 0.98)[0], np.where(border.mean(axis=0) < 0.98)[0]
    if len(rows) and len(cols):
        gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    orb = ORB(n_keypoints=256, fast_threshold=0.05, n_scales=6, downscale=1.25)
    orb.detect_and_extract(gray)  
    points = orb.keypoints[:, 1] + 1j * orb.keypoints[:, 0]
    return digest, points, np.packbits(orb.descriptors, axis=1).view(np.uint64)


def ransac(src, dst, rng, tries=400):
    n = len(src)
    if n * (n - 1) // 2 <= tries:
        p, q = np.triu_indices(n, 1)
    else:
        p, q = rng.integers(n, size=tries), rng.integers(n, size=tries)
        p, q = p[p != q], q[p != q]
    # đặt min khoảng cách hai điểm để đảm bảo phép biến đổi ổn định
    far = np.abs(src[p] - src[q]) > 8  
    p, q = p[far], q[far]
    if len(p) == 0:
        return 0
    a = (dst[p] - dst[q]) / (src[p] - src[q])
    b = dst[p] - a * src[p]
    ok = (np.abs(a) > 0.5) & (np.abs(a) < 2) & (np.abs(np.angle(a)) < np.pi / 6)
    if not ok.any():
        return 0
    return int((np.abs(a[ok, None] * src + b[ok, None] - dst) < 4).sum(axis=1).max())


def match(task):
    i, later = task
    pi, di = features[i][1:]
    edges = []
    for j in later:
        pj, dj = features[j][1:]
        if len(di) < 6 or len(dj) < 6:  
            continue
        # khoảng cách Hamming giữa các điểm
        dist = np.bitwise_count(di[:, None] ^ dj[None]).sum(axis=2) 
        # điểm gần nhất ở ảnh j của mỗi điểm ở anh i  
        best = dist.argmin(axis=1)  
        first = dist.min(axis=1)
        # khoảng cách gần thứ hai
        second = np.partition(dist, 1, axis=1)[:, 1]  
        # ghép hai điểm nếu gần nhau nhất
        good = (dist.argmin(axis=0)[best] == np.arange(len(di))) & (first <= 64) & (first < 0.8 * second)
        if good.sum() >= 6:
            n = ransac(pi[good], pj[best[good]], np.random.default_rng([SEED, i, j]))
            if n >= 6:
                edges.append((i, j, n))
    return edges

def share(data): 
    global features
    features = data

if __name__ == "__main__":
    names = sorted(f"{p.parent.name}/{p.name}" for p in (ROOT / "data").glob("*/*.png")
                   if not p.name.startswith("."))  
    classes = [name.split("/")[0] for name in names]
    assert len(names) == 4947 and len(set(classes)) == 25, (
        f"data/ có {len(set(classes))} lớp, {len(names)} ảnh; cần đúng 25 lớp, 4.947 ảnh (xem README)")
    with Pool() as pool:
        features = pool.map(describe, names, chunksize=16)

    edges, first_seen = [], {}
    for i, (digest, _, _) in enumerate(features):  
        if digest in first_seen:
            edges.append((first_seen[digest], i))
        first_seen.setdefault(digest, i)
    tasks = [(i, [j for j in range(i + 1, len(names)) if classes[j] == classes[i]]) for i in range(len(names))]
    with Pool(initializer=share, initargs=(features,)) as pool:
        for found in pool.map(match, tasks, chunksize=8):
            for i, j, n in found:
                if n >= 10 or abs(int(names[i][-8:-4]) - int(names[j][-8:-4])) <= 3:  
                    edges.append((i, j))
    # gom nhóm
    label = list(range(len(names)))
    for i, j in edges:
        old, new = label[j], label[i]
        label = [new if x == old else x for x in label]
    members = defaultdict(list)
    for i, x in enumerate(label):
        members[x].append(i)
    groups = sorted(members.values())

    # trong từng lớp, xáo các nhóm rồi đưa lần lượt mỗi nhóm vào train/val/test cho đúng tỉ lệ 70/15/15
    rng, split_of = random.Random(SEED), {}
    for c in sorted(set(classes)):
        class_groups = [g for g in groups if classes[g[0]] == c]
        rng.shuffle(class_groups)
        total, counts = sum(map(len, class_groups)), dict.fromkeys(RATIOS, 0)
        for g in class_groups:
            s = max(RATIOS, key=lambda s: (RATIOS[s] * total - counts[s]) / (RATIOS[s] * total))
            counts[s] += len(g)
            split_of.update(dict.fromkeys(g, s))

    out = Path(__file__).resolve().parent / "splits"
    out.mkdir(exist_ok=True)
    for s in RATIOS:
        (out / f"{s}.txt").write_text("\n".join(n for i, n in enumerate(names) if split_of[i] == s) + "\n",
                                      encoding="utf-8")
    group_of = {i: k for k, g in enumerate(groups) for i in g}
    rows = [f"{n},{group_of[i]},{split_of[i]}" for i, n in enumerate(names)]
    (out / "groups.csv").write_text("image,group,split\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(len(groups), "groups", {s: list(split_of.values()).count(s) for s in RATIOS})
