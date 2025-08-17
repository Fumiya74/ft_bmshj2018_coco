# -*- coding: utf-8 -*-
"""
蒸留学習のトレーニングスクリプト（COCO-train 画像のみ使用）。
Teacher/Student の再構成画像を一致させるように学習します。
"""
import os, argparse, time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils  # 追加：再構成画像の保存

from src.custom_decoder import StudentBM
from src.dataset_coco import COCO224Dataset
from src.losses import l1_loss, msssim_loss, psnr

def get_args():
    p = argparse.ArgumentParser()
    # データセット
    p.add_argument('--coco_dir', type=str, required=True, help='COCOのルート（train2017 が直下にあること）')
    p.add_argument('--use_prepared', type=lambda x: str(x).lower()=='true', default=False,
                   help='事前に224クロップをディスクへ保存している場合 True')
    p.add_argument('--max_images', type=int, default=0, help='使用枚数の上限（0で無制限）')
    p.add_argument('--num_workers', type=int, default=2)
    # モデル/学習
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--quality', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_msssim', type=float, default=0.2)
    p.add_argument('--train_igdn', action='store_true')
    p.add_argument('--unfreeze_all', action='store_true')
    # ログ/保存
    p.add_argument('--save_dir', type=str, default='./checkpoints')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--wandb_project', type=str, default='')

    # 再構成画像の保存（※Student 出力のみ保存）
    p.add_argument('--recon_every', type=int, default=5,
                   help='何エポックごとにval再構成を保存するか（0で無効）')
    p.add_argument('--recon_count', type=int, default=16,
                   help='再構成に用いるval画像の枚数（先頭から固定。実行中は変更しない）')
    p.add_argument('--recon_subdir', type=str, default='recon',
                   help='再構成画像の保存サブディレクトリ名（save_dir配下）')
    return p.parse_args()

def maybe_init_wandb(args):
    use = bool(args.wandb_project) and (os.environ.get("WANDB_DISABLED","false").lower() != "true")
    if not use:
        return None
    try:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        return wandb
    except Exception as e:
        print(f"W&B 無効化: {e}")
        return None

def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

@torch.no_grad()
def save_val_student_recons(model, loader, device, save_root, tag, wb=None, grid_max=16):
    """
    Student の再構成結果のみを PNG 保存。
    loader は『固定サブセット』の DataLoader を渡すこと。
    出力先: save_root/tag/00000.png, 00001.png, ...
    """
    model.eval()
    out_dir = Path(save_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    grid_samples = []
    for x in loader:
        x = x.to(device)
        out = model(x)
        x_s = out["x_student"].clamp(0, 1)  # Student のみ

        bs = x_s.size(0)
        for b in range(bs):
            vutils.save_image(x_s[b], out_dir / f"{saved:05d}.png")
            if len(grid_samples) < grid_max:
                grid_samples.append(x_s[b].cpu())
            saved += 1

    if wb and len(grid_samples) > 0:
        grid = vutils.make_grid(grid_samples, nrow=4, padding=2)
        wb.log({f"recon/{tag}": [wb.Image(grid, caption=tag)]})
    print(f"[recon] Saved {saved} student recon images to {out_dir}")

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wb = maybe_init_wandb(args)

    # 学生/教師モデルの準備
    model = StudentBM(quality=args.quality, train_igdn=args.train_igdn, unfreeze_all=args.unfreeze_all).to(device)
    print(f"置換された層数 (ConvTranspose2d → Upsample+Conv2d): {model.n_swapped}")

    # データローダ（train/val はファイル名ハッシュで分割）
    train_ds = COCO224Dataset(args.coco_dir, split='train', use_prepared=args.use_prepared,
                              max_images=args.max_images, random_crop=True)
    val_ds   = COCO224Dataset(args.coco_dir, split='val',   use_prepared=args.use_prepared,
                              max_images=max(0, args.max_images // 10) if args.max_images>0 else 0,
                              random_crop=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # 再構成用に、実行中固定の val サブセットを作る
    recon_n = max(1, min(args.recon_count, len(val_ds))) if len(val_ds) > 0 else 0
    recon_ld = None
    if recon_n > 0:
        recon_indices = list(range(recon_n))   # 先頭から固定。実行中は不変
        recon_subset  = Subset(val_ds, recon_indices)
        recon_ld      = DataLoader(recon_subset, batch_size=min(8, args.batch_size), shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    # 学習対象パラメータのみ最適化
    params = [p for p in model.student.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best_msssim = -1.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu')
        model.student.load_state_dict(ckpt['student_state'])
        opt.load_state_dict(ckpt['opt_state'])
        sched.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_msssim = ckpt.get('best_msssim', best_msssim)
        print(f"{args.resume} から再開（epoch={start_epoch}）")

    # 学習開始前に Student 再構成を保存
    recon_root = Path(args.save_dir) / args.recon_subdir
    if recon_ld is not None:
        tag = f"pre_e{start_epoch:03d}"
        save_val_student_recons(model, recon_ld, device, recon_root, tag, wb=wb)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for i, x in enumerate(train_ld):
            x = x.to(device)
            out = model(x)
            x_s, x_t = out["x_student"], out["x_teacher"]
            loss = args.lambda_l1 * l1_loss(x_s, x_t) + args.lambda_msssim * msssim_loss(x_s, x_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_loss += loss.item()

            if wb and (i % 50 == 0):
                with torch.no_grad():
                    wb.log({
                        'train/loss': loss.item(),
                        'train/psnr_to_teacher': psnr(x_s, x_t).item(),
                        'lr': opt.param_groups[0]['lr'],
                        'epoch': epoch
                    })

        sched.step()

        # 検証
        model.eval()
        val_psnr = 0.0
        val_msssim = 0.0
        n_pix = 0
        with torch.no_grad():
            for x in val_ld:
                x = x.to(device)
                out = model(x)
                x_s, x_t = out["x_student"], out["x_teacher"]
                bs = x.size(0)
                val_psnr += psnr(x_s, x_t).item() * bs
                val_msssim += (1.0 - msssim_loss(x_s, x_t).item()) * bs
                n_pix += bs
        if n_pix > 0:
            val_psnr /= n_pix
            val_msssim /= n_pix

        print(f"Epoch {epoch:03d} | loss {run_loss/max(1,len(train_ld)):.4f} | PSNR {val_psnr:.2f} | MS-SSIM {val_msssim:.4f} | {time.time()-t0:.1f}s")

        if wb:
            wb.log({
                'val/psnr_to_teacher': val_psnr,
                'val/msssim_to_teacher': val_msssim,
                'epoch': epoch
            })

        # 既定：5エポックごとに Student の再構成だけ保存
        if recon_ld is not None and args.recon_every > 0 and ((epoch + 1) % args.recon_every == 0):
            tag = f"e{epoch:03d}"
            save_val_student_recons(model, recon_ld, device, recon_root, tag, wb=wb)

        # CKPT 保存
        last_path = Path(args.save_dir) / "last.pt"
        save_ckpt({
            'epoch': epoch,
            'student_state': model.student.state_dict(),
            'opt_state': opt.state_dict(),
            'sched_state': sched.state_dict(),
            'best_msssim': best_msssim,
        }, last_path)

        if val_msssim > best_msssim:
            best_msssim = val_msssim
            best_path = Path(args.save_dir) / "best_msssim.pt"
            save_ckpt({
                'epoch': epoch,
                'student_state': model.student.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': sched.state_dict(),
                'best_msssim': best_msssim,
            }, best_path)

    print("学習完了")

if __name__ == "__main__":
    main()
