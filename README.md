# CompressAI `bmshj2018_factorized` デコーダ置換 & 蒸留→通常FT（COCO-train）

本プロジェクトは、**CompressAI** の `bmshj2018_factorized` を教師モデルとして、
デコーダ内の **ConvTranspose2d** を **Upsample + Conv2d** に置換した **学生モデル（Student）** を作成し、
まず **知識蒸留**（Teacher出力にStudentが近づく）で学習、その後 **通常の再構成最適化（対GT）でFT** を行うレシピです。

- 画像サイズ: **224×224**
- データセット: **COCO 2017 train**（画像のみ使用。アノテーション不要）
- 実行環境: **Google Colab Pro** を想定

---

## できること（更新）
- 🔁 **層置換**: `ConvTranspose2d → Upsample(scale=2) + Conv2d(kernel=5, padding=2)` をデコーダ `g_s` に自動適用  
- 🧊 **選択的微調整**: 置換Convのみ学習（既定）／IGDNも学習（`--train_igdn`）／全層解凍（`--unfreeze_all`）
- 🎓 **知識蒸留**: Teacher再構成にStudentが近づくよう **L1 + (1 − MS-SSIM)** を最小化
- 🖼️ **再構成の保存（重要）**:  
  - **学習開始前**と **既定5エポックごと**に、**val画像の Student 再構成のみ**をPNGで保存  
  - 確認に用いるval画像は **先頭から固定のサブセット**（既定16枚）。**実行中は不変**  
  - 保存先: 蒸留は `save_dir/recon/<tag>/00000.png ...`、FTは `save_dir/recon_ft/<tag>/...`  
  - `tag` 例: `pre_e000`, `e004`, `e009`, …（エポック0始まり表記）
- 📊 **W&B可視化**（任意）: ロス・PSNR・MS-SSIM・LR・**Student再構成のグリッド画像** をログ
- 💾 **チェックポイント**: 毎エポック `last.pt`、検証MS-SSIMベスト `best_msssim.pt`（蒸留）/ `ft_best.pt`（FT）
- 🧰 **前処理**: COCO train2017 の **DL/展開**、任意で **224クロップを事前作成**
- 📓 **Colabノートブック**: セットアップ→データ準備→学習→評価まで

---

## ディレクトリ構成
```
src/
  custom_decoder.py        # 置換ロジック（JPコメント）
  dataset_coco.py          # COCO画像を224にトランスフォーム（オンザフライ/事前クロップ両対応）
  losses.py                # L1, MS-SSIM, PSNR
  train_distill.py         # 蒸留学習（Student↔Teacher） + Student再構成保存
  eval.py                  # 検証（PSNR/MS-SSIM）
scripts/
  coco_prepare.py          # COCO train2017 のDL/展開 & 224クロップ（任意）
checkpoints/               # 蒸留の保存先（例）
checkpoints_ft/            # FTの保存先（例）
finetune_after_distill.py  # 蒸留後の通常FT（Student↔GT） + Student再構成保存
fine_tune_bmshj2018_coco_colab.ipynb  # Colabノートブック
```

---

## Colabでの手順（概要）
1. ノートブック `fine_tune_bmshj2018_coco_colab.ipynb` を開く  
2. **セットアップ**セルを実行（依存関係のインストール）  
3. **データ準備**セルを実行（COCO train2017 をDL＆展開。必要に応じて224クロップを事前生成）  
4. **蒸留学習**セルを実行（W&Bは任意）  
5. **FT学習**セルを実行（蒸留のベストckptから継続）  
6. **評価**セルを実行（PSNR / MS-SSIM を表示）  

> ⚠️ COCO train2017 は容量が大きい（~18GB）ため、Colabの一時ディスク使用量に注意してください。  
> 省メモリ・省ストレージでテストする場合は、`--max_images` で使用枚数を制限できます。

---

## 学習レシピ（既定）
- Teacher: `compressai.zoo.pretrained.bmshj2018_factorized(quality=8)`
- Student: Teacher重みで初期化 → デコーダ`g_s`のConvTranspose2dを **Upsample+Conv2d** に置換
- 微調整範囲: 置換Convのみ（`--train_igdn`でIGDNも、`--unfreeze_all`で全層）
- 損失: `λ1 * L1 + λ2 * (1 - MS-SSIM)`（既定 λ1=1.0, λ2=0.2）
- Optimizer: Adam (lr=1e-4), CosineAnnealingLR
- バッチ: 16（Colab T4/A100想定、適宜調整）
- エポック: 10〜30目安（打ち切り可）

---

## 代表的なコマンド

### 1) 224クロップを事前生成してから学習（任意）
```bash
python -m scripts.coco_prepare \
  --coco_dir /content/data/coco \
  --out_dir /content/data/coco224 \
  --val_ratio 0.02
```

### 2) 蒸留（Student↔Teacher、Student再構成のみ保存）
```bash
python src/train_distill.py \
  --coco_dir /content/data/coco224 \
  --use_prepared true \
  --quality 8 --epochs 15 --batch_size 16 --lr 1e-4 \
  --recon_every 5 --recon_count 16 --recon_subdir recon \
  --wandb_project bmshj2018_ft_coco \
  --save_dir ./checkpoints
```
- **再構成保存**: 学習前は `./checkpoints/recon/pre_e000/`、以降 `e004/`, `e009/` … に **Student出力のみ** をPNG保存  
- **W&B**: `recon/<tag>` としてグリッドサムネイルをログ（任意）

### 3) 蒸留の続きから再開
```bash
python src/train_distill.py \
  --coco_dir /content/data/coco224 \
  --use_prepared true \
  --resume ./checkpoints/last.pt
```

### 4) 蒸留後の通常FT（Student↔GT、Student再構成のみ保存）
```bash
python finetune_after_distill.py \
  --coco_dir /content/data/coco224 \
  --use_prepared true \
  --quality 8 --epochs 10 --batch_size 16 --lr 1e-4 \
  --resume_student ./checkpoints/best_msssim.pt \
  --recon_every 5 --recon_count 16 --recon_subdir recon_ft \
  --wandb_project bmshj2018_ft_coco \
  --save_dir ./checkpoints_ft
```
- **凍結方針**: Student 以外は全凍結。`--freeze_igdn` で Student 内の IGDN/GDN を凍結可能  
- **チェックポイント**: `./checkpoints_ft/ft_last.pt`, `./checkpoints_ft/ft_best.pt`  
- **再構成保存**: `./checkpoints_ft/recon_ft/<tag>/00000.png ...` に **Student出力のみ** を保存

### 5) 評価（例：PSNR/MS-SSIM）
```bash
python -m src.eval \
  --coco_dir /content/data/coco224 \
  --checkpoint ./checkpoints/best_msssim.pt
```

---

## 再構成保存の仕様（重要）
- **保存対象**: **Student の再構成のみ**（入力やTeacherは保存しません。入力/Teacherは決定的で冗長なため）  
- **タイミング**: 学習 **開始前** と、`--recon_every`（既定5）エポックごと  
- **対象画像**: valセット先頭から `--recon_count`（既定16）枚を **固定サブセット** として使用（**実行中は変更しない**）  
- **出力形式**: 1サンプル=1PNG（`00000.png` から連番）  
- **出力先**: 蒸留は `save_dir/recon/<tag>/`、FTは `save_dir/recon_ft/<tag>/`

> メモ: `dataset_coco.py` の分割はファイル名ハッシュに基づく決定的分割です。`recon_count` による固定サブセットも、同一設定・同一データ環境では再現的に選ばれます。

---

## 収束の目安（COCO train2017, 224crop, batch=16, lr=1e-4）
- **蒸留のみ**: Teacherにほぼ一致（PSNR差 ≤0.1dB / MS-SSIM ≥0.995）まで **約3–7エポック**  
- **蒸留後FT**: 対GTで微差を詰めるのに **+1–3エポック**  
- **合計**: おおよそ **5–10エポック** でTeacher同等〜僅差上回りが目安（以降は改善逓減）

---

## よくあるTips
- VRAMに余裕があれば **大きめbatch** + **AMP** で安定かつ高速収束
- FT時は **蒸留と同LRか、少し下げる（1/2など）** と1–2エポックで詰まりやすい
- `--max_images` でサブサンプルする場合は、必要エポック数が若干増える傾向

---

## 環境メモ
- W&Bを使わない場合は `WANDB_DISABLED=true` を環境変数に設定してください。  
- 追加指標（LPIPSなど）は計算コストが増えます。必要に応じて拡張してください。

