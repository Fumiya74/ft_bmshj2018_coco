
# -*- coding: utf-8 -*-
"""
デコーダ内の ConvTranspose2d を Upsample + Conv2d に置換するためのユーティリティ。
Teacher/Student 2つの bmshj2018_factorized を用意し、Teacherは凍結、Studentのみ学習します。
"""
import torch
import torch.nn as nn
from compressai.zoo import bmshj2018_factorized

def _swap_deconv(module: nn.Module):
    """
    モジュールを再帰的に探索し、nn.ConvTranspose2d を
    nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(...)) に置き換える。
    返り値: (置換後モジュール, 置換数)
    """
    count = 0

    def make_block(deconv: nn.ConvTranspose2d) -> nn.Sequential:
        # 通常、CompressAIのデコーダでは stride=2 の転置畳み込みが使われるため、
        # 近傍補間Upsample(×2) + Conv2d で近似します。
        k = deconv.kernel_size[0]
        p = deconv.padding[0]
        s = deconv.stride[0]
        up = nn.Upsample(scale_factor=s, mode="nearest")
        conv = nn.Conv2d(
            in_channels=deconv.in_channels,
            out_channels=deconv.out_channels,
            kernel_size=k,
            padding=p,
            bias=(deconv.bias is not None),
        )
        # 置換Convの初期化（ReLU前提のKaiming Normal）
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        return nn.Sequential(up, conv)

    def _recurse(m: nn.Module) -> nn.Module:
        nonlocal count
        for name, child in list(m.named_children()):
            if isinstance(child, nn.ConvTranspose2d):
                setattr(m, name, make_block(child))
                count += 1
            else:
                setattr(m, name, _recurse(child))
        return m

    return _recurse(module), count


class StudentBM(nn.Module):
    """
    Studentモデル:
      - Teacher/Student ともに bmshj2018_factorized をロード
      - Student側のデコーダ g_s の ConvTranspose2d を Upsample+Conv2d に置換
      - 既定では置換した Conv のみ requires_grad=True とし、他は凍結
    """
    def __init__(self, quality: int = 8, train_igdn: bool = False, unfreeze_all: bool = False):
        super().__init__()
        # Teacherは推論専用（勾配不要）
        self.teacher = bmshj2018_factorized(quality=quality, pretrained=True).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # StudentはTeacherの重みで初期化
        self.student = bmshj2018_factorized(quality=quality, pretrained=True)
        # デコーダの転置畳み込みを置換
        self.student.g_s, n_swapped = _swap_deconv(self.student.g_s)

        # まず全凍結
        for p in self.student.parameters():
            p.requires_grad = False

        # 置換したConv2dのみ学習対象にする
        for m in self.student.g_s.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad_(True)

        # IGDNも一緒に学習したい場合
        if train_igdn:
            for m in self.student.modules():
                # compressai.layers.GDN / InverseGDN の両方に対応
                if m.__class__.__name__ in ("GDN", "InverseGDN", "GDN1"):
                    for p in m.parameters():
                        p.requires_grad = True

        # 全層を解凍（計算は重いが収束しやすい場合あり）
        if unfreeze_all:
            for p in self.student.parameters():
                p.requires_grad = True

        self.n_swapped = n_swapped

    @torch.no_grad()
    def encode(self, x):
        return self.student.g_a(x)

    def forward(self, x):
        """
        返り値:
          x_teacher: Teacherの再構成画像
          x_student: Studentの再構成画像
          likelihoods_student: RD目的拡張用に保持（本レシピでは未使用）
        """
        with torch.no_grad():
            out_t = self.teacher(x)
            x_t = out_t["x_hat"]

        out_s = self.student(x)
        x_s = out_s["x_hat"]

        return {
            "x_teacher": x_t,
            "x_student": x_s,
            "likelihoods_student": out_s["likelihoods"],
        }
