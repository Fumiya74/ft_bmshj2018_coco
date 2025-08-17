# -*- coding: utf-8 -*-
"""
COCO 2017 train/val のダウンロード後に、指定サイズ（デフォルト: 224）への事前クロップを行うスクリプト。
- 学習: train2017 → out_dir/train
- 評価: val2017   → out_dir/val  （--include_val True のとき）
アノテーションは不要です（画像再構成タスクのため）。
"""
import argparse, glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DEFAULT_OUT_DIR = '/content/data/coco224'

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_dir', type=str, default='/content/data/coco', help='COCOのルート')
    p.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR, help='クロップ画像の保存先（デフォルトはサイズ224想定）')
    p.add_argument('--size', type=int, default=224, help='出力正方形の一辺のピクセル数（デフォルト: 224）')
    p.add_argument('--limit_train', type=int, default=0, help='trainの最大枚数（0で無制限）')
    p.add_argument('--limit_val', type=int, default=0, help='valの最大枚数（0で無制限）')
    p.add_argument('--include_val', type=lambda x: str(x).lower()=='true', default=False,
                   help='val2017 も 指定サイズ でクロップして保存する（True/False）')
    return p.parse_args()

def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """中央正方形クロップ後、size×size にリサイズ"""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
    return img

def process_split(src_dir: Path, dst_root: Path, limit: int, size: int):
    files = sorted(glob.glob(str(src_dir / "*.jpg")))
    if limit > 0:
        files = files[:limit]
    # 大量ファイル対策として先頭文字でサブフォルダ分散
    for fp in tqdm(files, desc=f"Cropping {src_dir.name} -> {size}"):
        name = Path(fp).name
        sub = name[0]
        dst_dir = dst_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(fp).convert("RGB")
            out_img = center_crop_resize(img, size)
            out_img.save(dst_dir / name, quality=95)
        except Exception:
            continue

def main():
    args = get_args()

    # out_dir がデフォルト（coco224）で、size が 224 以外のときは自動で coco{size} に調整
    out_dir_path = Path(args.out_dir)
    if args.out_dir == DEFAULT_OUT_DIR and args.size != 224:
        out_dir_path = Path(f"/content/data/coco{args.size}")

    coco = Path(args.coco_dir)
    out = out_dir_path
    out_train = out / "train"
    out_val = out / "val"
    out_train.mkdir(parents=True, exist_ok=True)
    if args.include_val:
        out_val.mkdir(parents=True, exist_ok=True)

    train_src = coco / "train2017"
    assert train_src.exists(), f"{train_src} が見つかりません。"
    process_split(train_src, out_train, args.limit_train, args.size)

    if args.include_val:
        val_src = coco / "val2017"
        assert val_src.exists(), f"{val_src} が見つかりません。"
        process_split(val_src, out_val, args.limit_val, args.size)

    print(f"{args.size} ピクセルのクロップを保存しました:", out)

if __name__ == "__main__":
    main()

