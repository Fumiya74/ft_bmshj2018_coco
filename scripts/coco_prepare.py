# -*- coding: utf-8 -*-
"""
COCO 2017 train/val のダウンロード後に、224×224 への事前クロップを行うスクリプト。
- 学習: train2017 → out_dir/train
- 評価: val2017   → out_dir/val  （--include_val True のとき）
アノテーションは不要です（画像再構成タスクのため）。
"""
import argparse, glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_dir', type=str, default='/content/data/coco', help='COCOのルート')
    p.add_argument('--out_dir', type=str, default='/content/data/coco224', help='224クロップの保存先')
    p.add_argument('--limit_train', type=int, default=0, help='trainの最大枚数（0で無制限）')
    p.add_argument('--limit_val', type=int, default=0, help='valの最大枚数（0で無制限）')
    p.add_argument('--include_val', type=lambda x: str(x).lower()=='true', default=False,
                   help='val2017 も 224 クロップして保存する')
    return p.parse_args()

def center_crop_resize_224(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((224, 224), Image.BICUBIC)
    return img

def process_split(src_dir: Path, dst_root: Path, limit: int):
    files = sorted(glob.glob(str(src_dir / "*.jpg")))
    if limit > 0:
        files = files[:limit]
    # 大量ファイル対策として先頭文字でサブフォルダ分散
    for fp in tqdm(files, desc=f"Cropping {src_dir.name} -> 224"):
        name = Path(fp).name
        sub = name[0]
        dst_dir = dst_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(fp).convert("RGB")
            img224 = center_crop_resize_224(img)
            img224.save(dst_dir / name, quality=95)
        except Exception:
            continue

def main():
    args = get_args()
    coco = Path(args.coco_dir)
    out = Path(args.out_dir)
    out_train = out / "train"
    out_val = out / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    if args.include_val:
        out_val.mkdir(parents=True, exist_ok=True)

    train_src = coco / "train2017"
    assert train_src.exists(), f"{train_src} が見つかりません。"
    process_split(train_src, out_train, args.limit_train)

    if args.include_val:
        val_src = coco / "val2017"
        assert val_src.exists(), f"{val_src} が見つかりません。"
        process_split(val_src, out_val, args.limit_val)

    print("224クロップを保存しました:", out)

if __name__ == "__main__":
    main()
