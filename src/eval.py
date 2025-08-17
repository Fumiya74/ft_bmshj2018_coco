
# -*- coding: utf-8 -*-
"""
検証用スクリプト（PSNR / MS-SSIM）。
"""
import argparse, torch
from torch.utils.data import DataLoader
from src.custom_decoder import StudentBM
from src.dataset_coco import COCO224Dataset
from src.losses import psnr, msssim_loss

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_dir', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--quality', type=int, default=8)
    p.add_argument('--use_prepared', type=lambda x: str(x).lower()=='true', default=False)
    p.add_argument('--max_images', type=int, default=0)
    return p.parse_args()

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StudentBM(quality=args.quality).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.student.load_state_dict(ckpt['student_state'])
    model.eval()

    ds = COCO224Dataset(args.coco_dir, split='val', use_prepared=args.use_prepared, max_images=args.max_images, random_crop=False)
    ld = DataLoader(ds, batch_size=32, shuffle=False)

    p_sum, m_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x in ld:
            x = x.to(device)
            out = model(x)
            x_s, x_t = out["x_student"], out["x_teacher"]
            bs = x.size(0)
            p_sum += psnr(x_s, x_t).item() * bs
            m_sum += (1.0 - msssim_loss(x_s, x_t).item()) * bs
            n += bs

    if n > 0:
        print(f"Val PSNR: {p_sum / n:.2f} dB | Val MS-SSIM: {m_sum / n:.4f}")
    else:
        print("検証データが見つかりませんでした。")

if __name__ == "__main__":
    main()
