# -*- coding: utf-8 -*-
"""
蒸留後のファインチューニングスクリプト。
- 蒸留で学習した student 側の層のみを最適化（それ以外は凍結）
- 目的は GT(入力画像) に対する通常の再構成最適化
- 蒸留スクリプトで保存した checkpoint から student の重みを読み込んで開始
- 学習前 + N エポックごとに、Student の val 再構成のみ PNG 保存
"""
import os, argparse, time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils

from src.custom_decoder import StudentBM
from src.dataset_coco import COCO224Dataset
from src.losses import l1_loss, msssim_loss, psnr

def get_args():
    p = argparse.ArgumentParser()
    # データセット
    p.add_argument('--coco_dir', type=str, required=True, help='COCOのルート（train2017 が直下にあること）')
    p.add_argument('--use_prepared', type=lambda x: str(x).lower()=='true', default=False)
    p.add_argument('--max_images', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=2)

    # 学習
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--quality', type=int, default=8, help='蒸留時と同じ quality を指定')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_msssim', type=float, default=0.2)

    # 凍結ポリシー
    p.add_argument('--freeze_igdn', action='store_true', help='student 内の IGDN/GDN を凍結')

    # ログ/保存・読み込み
    p.add_argument('--save_dir', type=str, default='./checkpoints_ft')
    p.add_argument('--resume_student', type=str, required=True,
                   help='蒸留チェックポイント（best_msssim.pt など）へのパス。student_state を読み込みます')
    p.add_argument('--resume_ft', type=str, default='')
    p.add_argument('--wandb_project', type=str, default='')

    # 再構成画像の保存（Student のみ）
    p.add_argument('--recon_every', type=int, default=5)
    p.add_argument('--recon_count', type=int, default=16)
    p.add_argument('--recon_subdir', type=str, default='recon_ft')
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

def freeze_except_student(model, freeze_igdn=False):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.student.named_parameters():
        p.requires_grad = True
    if freeze_igdn:
        for name, p in model.student.named_parameters():
            lname = name.lower()
            if 'igdn' in lname or 'gdn' in lname:
                p.requires_grad = False

def forward_student_only(model, x):
    if hasattr(model, 'forward_student') and callable(getattr(model, 'forward_student')):
        return model.forward_student(x)
    out = model(x)
    if isinstance(out, dict) and 'x_student' in out:
        return out['x_student']
    return out  # フォールバック

@torch.no_grad()
def save_val_student_recons(model, loader, device, save_root, tag, wb=None, grid_max=16):
    model.eval()
    out_dir = Path(save_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    grid_samples = []
    for x in loader:
        x = x.to(device)
        x_s = forward_student_only(model, x).clamp(0, 1)

        bs = x_s.size(0)
        for b in range(bs):
            vutils.save_image(x_s[b], out_dir / f"{saved:05d}.png")
            if len(grid_samples) < grid_max:
                grid_samples.append(x_s[b].cpu())
            saved += 1

    if wb and len(grid_samples) > 0:
        grid = vutils.make_grid(grid_samples, nrow=4, padding=2)
        wb.log({f"recon_ft/{tag}": [wb.Image(grid, caption=tag)]})
    print(f"[recon-FT] Saved {saved} student recon images to {out_dir}")

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wb = maybe_init_wandb(args)

    # モデル作成・蒸留 ckpt から student 差し替え
    model = StudentBM(quality=args.quality, train_igdn=True, unfreeze_all=False).to(device)
    if not Path(args.resume_student).exists():
        raise FileNotFoundError(f"resume_student が見つかりません: {args.resume_student}")
    distill_ckpt = torch.load(args.resume_student, map_location='cpu')
    if 'student_state' not in distill_ckpt:
        raise KeyError("チェックポイントに 'student_state' がありません（蒸留出力を指定してください）")
    model.student.load_state_dict(distill_ckpt['student_state'], strict=False)

    # データ
    train_ds = COCO224Dataset(args.coco_dir, split='train', use_prepared=args.use_prepared,
                              max_images=args.max_images, random_crop=True)
    val_ds   = COCO224Dataset(args.coco_dir, split='val',   use_prepared=args.use_prepared,
                              max_images=max(0, args.max_images // 10) if args.max_images>0 else 0,
                              random_crop=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # 再構成用・固定サブセット
    recon_n = max(1, min(args.recon_count, len(val_ds))) if len(val_ds) > 0 else 0
    recon_ld = None
    if recon_n > 0:
        recon_indices = list(range(recon_n))
        recon_subset  = Subset(val_ds, recon_indices)
        recon_ld      = DataLoader(recon_subset, batch_size=min(8, args.batch_size), shuffle=False,
                                   num_workers=args.num_workers, pin_memory=True)

    # 凍結設定
    freeze_except_student(model, freeze_igdn=args.freeze_igdn)

    # Optimizer / Scheduler
    params = [p for p in model.student.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("学習対象パラメータが 0 です。--freeze_igdn の指定などをご確認ください。")
    opt = optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best_msssim_to_gt = -1.0
    if args.resume_ft and Path(args.resume_ft).exists():
        ckpt = torch.load(args.resume_ft, map_location='cpu')
        model.student.load_state_dict(ckpt['student_state'], strict=False)
        opt.load_state_dict(ckpt['opt_state'])
        sched.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_msssim_to_gt = ckpt.get('best_msssim_to_gt', best_msssim_to_gt)
        print(f"{args.resume_ft} から再開（epoch={start_epoch}）")

    # 学習前に Student 再構成を保存
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
            x_s = forward_student_only(model, x)
            loss = args.lambda_l1 * l1_loss(x_s, x) + args.lambda_msssim * msssim_loss(x_s, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_loss += loss.item()

            if wb and (i % 50 == 0):
                with torch.no_grad():
                    wb.log({
                        'train/loss_ft': loss.item(),
                        'train/psnr_to_gt': psnr(x_s, x).item(),
                        'lr': opt.param_groups[0]['lr'],
                        'epoch': epoch
                    })

        sched.step()

        # 検証（GT への PSNR / MS-SSIM）
        model.eval()
        val_psnr_to_gt = 0.0
        val_msssim_to_gt = 0.0
        n_pix = 0
        with torch.no_grad():
            for x in val_ld:
                x = x.to(device)
                x_s = forward_student_only(model, x)
                bs = x.size(0)
                val_psnr_to_gt += psnr(x_s, x).item() * bs
                val_msssim_to_gt += (1.0 - msssim_loss(x_s, x).item()) * bs
                n_pix += bs
        if n_pix > 0:
            val_psnr_to_gt /= n_pix
            val_msssim_to_gt /= n_pix

        print(f"[FT] Epoch {epoch:03d} | loss {run_loss/max(1,len(train_ld)):.4f} | "
              f"PSNR(gt) {val_psnr_to_gt:.2f} | MS-SSIM(gt) {val_msssim_to_gt:.4f} | "
              f"{time.time()-t0:.1f}s")

        if wb:
            wb.log({
                'val/psnr_to_gt': val_psnr_to_gt,
                'val/msssim_to_gt': val_msssim_to_gt,
                'epoch': epoch
            })

        # 既定：5エポックごとに Student の再構成だけ保存
        if recon_ld is not None and args.recon_every > 0 and ((epoch + 1) % args.recon_every == 0):
            tag = f"e{epoch:03d}"
            save_val_student_recons(model, recon_ld, device, recon_root, tag, wb=wb)

        # 保存
        last_path = Path(args.save_dir) / "ft_last.pt"
        save_ckpt({
            'epoch': epoch,
            'student_state': model.student.state_dict(),
            'opt_state': opt.state_dict(),
            'sched_state': sched.state_dict(),
            'best_msssim_to_gt': best_msssim_to_gt,
        }, last_path)

        if val_msssim_to_gt > best_msssim_to_gt:
            best_msssim_to_gt = val_msssim_to_gt
            best_path = Path(args.save_dir) / "ft_best.pt"
            save_ckpt({
                'epoch': epoch,
                'student_state': model.student.state_dict(),
                'opt_state': opt.state_dict(),
                'sched_state': sched.state_dict(),
                'best_msssim_to_gt': best_msssim_to_gt,
            }, best_path)

    print("FT 完了")

if __name__ == "__main__":
    main()
