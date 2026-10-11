"""Chia ảnh trong data/ thành train/val/test theo danh sách có sẵn trong youonlycodeonce/.

--split nhom: cách chia của nhóm (splits/), ảnh gần giống nhau được gom nhóm và nằm chung một tập.
--split ngaunhien: chia ngẫu nhiên từng ảnh 70/15/15 (splits_ngaunhien/), dùng để so sánh.
Ảnh được chép vào artifacts/split_data/<cách chia>/{train,val,test}/<LỚP>/.
Chạy: python youonlycodeonce/split_data.py --split nhom
"""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="nhom", choices=["nhom", "ngaunhien"])
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
list_dir = root / "youonlycodeonce" / ("splits" if args.split == "nhom" else "splits_ngaunhien")
out_dir = root / "artifacts" / "split_data" / args.split
shutil.rmtree(out_dir, ignore_errors=True)  # chép lại từ đầu mỗi lần chạy

for subset in ["train", "val", "test"]:
    images = (list_dir / f"{subset}.txt").read_text().split()  # mỗi dòng: LỚP/TÊN_ẢNH.png
    for image in images:
        (out_dir / subset / image).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "data" / image, out_dir / subset / image)
    print(subset, len(images))
