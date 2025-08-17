# -*- coding: utf-8 -*-
"""
COCO 2017 の画像のみを使用するデータローダ（学習: train / train2017、評価: val / val2017）。
- 224×224 への変換はオンザフライ（学習: RandomResizedCrop / 評価: CenterCrop）
- 事前に224をディスクへ保存した場合（scripts/coco_prepare.py）、--use_prepared=True で高速ロード
- ディレクトリ名ゆらぎ（train vs train2017）や拡張子ゆらぎ（jpg/jpeg/png 等）に強い実装
"""

from pathlib import Path
from typing import List, Sequence
import glob

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


def _collect_images_recursive(base: Path,
                              exts: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")) -> List[str]:
    """base 配下を再帰で探索し、指定拡張子の画像パス一覧を返す（昇順ソート）。"""
    paths = []
    for ext in exts:
        paths += glob.glob(str(base / "**" / ext), recursive=True)
    return sorted(paths)


def _first_existing_dir(candidates: Sequence[Path]) -> Path | None:
    """候補のうち最初に存在するディレクトリを返す。なければ None。"""
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


class COCO224Dataset(Dataset):
    def __init__(
        self,
        coco_dir: str,
        split: str = "train",
        use_prepared: bool = False,
        max_images: int = 0,
        random_crop: bool = True,
    ):
        """
        Args:
            coco_dir: COCO ルート。生COCOなら train2017/val2017 を直下に、事前クロップ済みなら train/val を直下に想定。
            split: 'train' or 'val'
            use_prepared: 事前クロップ（224）済みのディレクトリ構成を使う場合 True（train/val）
            max_images: 使用する最大枚数（0 = 制限なし）
            random_crop: Trueなら学習時に RandomResizedCrop(224)、Falseなら常に CenterCrop(224)
        """
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split}"
        self.split = split
        self.use_prepared = use_prepared
        self.random_crop = random_crop

        root = Path(coco_dir)

        if use_prepared:
            # 事前クロップ: train / val を優先しつつ、train2017 / val2017 に置かれていても拾う
            candidates = [
                root / ("train" if split == "train" else "val"),
                root / ("train2017" if split == "train" else "val2017"),
            ]
        else:
            # 生COCO: train2017 / val2017 を優先しつつ、train / val でも拾う
            candidates = [
                root / ("train2017" if split == "train" else "val2017"),
                root / ("train" if split == "train" else "val"),
            ]

        base = _first_existing_dir(candidates)
        self.paths: List[str] = _collect_images_recursive(base) if base else []

        if max_images > 0:
            self.paths = self.paths[:max_images]

        if len(self.paths) == 0:
            msg = [
                f"No images found. split='{split}', use_prepared={use_prepared}",
                f"coco_dir='{root}'",
                "Looked for these directories (first existing one is used):",
            ] + [f"  - {c}" for c in candidates]
            raise FileNotFoundError("\n".join(msg))

        # 変換
        if random_crop and split == "train":
            self.tf = T.Compose([
                T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.ToTensor(),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        # 壊れ画像を避けたい場合は try/except で読み替えガードを入れてもよい
        img = Image.open(p).convert("RGB")
        x = self.tf(img)  # [C,H,W], [0,1] float32
        return x
