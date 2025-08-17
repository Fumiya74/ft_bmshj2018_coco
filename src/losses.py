
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

def l1_loss(a, b):
    return (a - b).abs().mean()

def msssim_loss(a, b):
    # (1 - MS-SSIM) を返す（小さいほど類似）
    return 1.0 - ms_ssim(a, b, data_range=1.0, size_average=True)

def psnr(a, b, eps=1e-8):
    mse = F.mse_loss(a, b)
    return 10.0 * torch.log10(1.0 / (mse + eps))
