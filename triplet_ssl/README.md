# Triplet SSL — DINOv3 三聯圖持續預訓練(die-to-die 檢測)

針對 optical short-wavelength die-to-die 檢測影像的 DINOv3 ViT-B/16 持續預訓練
(continued SSL pretraining)。每筆樣本是一組三聯圖(triplet):

- **target**:待檢圖,可能含缺陷(~4–6 px PSF 光斑)
- **ref1 / ref2**:同一 pattern 位置的無缺陷鄰 die 參考圖

核心想法:保留 same-image DINO 做 domain adaptation，跨 die 僅讓 target
匹配「最佳的一張 reference」，不做矛盾的全面 ref1↔ref2 pull；搭配 robust top-k
豁免，避免真缺陷 patch 被強迫拉成正常。Phase 3 再用保留三通道角色的
order-aware fusion head，直接學習只有 `(target,ref1,ref2)=(1,0,0)` 才是缺陷。

本模組**純增量**:不改動 `dinov3/` 既有程式,只唯讀重用其元件。

---

## 環境

只有這個 interpreter 有 torch + cv2 + matplotlib(CPU 版):

```powershell
$PY = "C:\Users\Peter Peng\AppData\Local\Programs\Python\Python312\python.exe"
```

所有指令都在 repo 根目錄(`claude\dinov3`)執行。
Scientific TIFF reader 使用 `tifffile`（已列入 `requirements.txt`）。

## 目錄結構

```
triplet_ssl/
  data_gen/generate_triplets.py   # 合成 triplet 產生器(真資料來之前的替身)
  make_mimic_checkpoint.py        # 假權重 .pth 產生器(真權重來之前的替身)
  data/triplet_dataset.py         # dataloader:共享幾何/獨立光度增強、次像素對位、缺陷過採樣
  data/synth_defect.py            # Phase 3 合成缺陷 + 增強存活率檢查
  losses/triplet_loss.py          # best-reference pull + robust top-k + bounded margin
  models/backbone.py              # canonical 官方 DINOv3 ViT-B/16 架構契約
  models/order_aware_fusion.py    # target-asymmetric / ref-symmetric 部署 head
  eval/separability.py            # patch 殘差 AUROC 評估(唯一會讀 GT mask 的地方)
  eval/plot_curves.py             # 訓練曲線 PNG
  train_triplet.py                # 主訓練/評估入口
  configs/                        # base.yaml + phase0~3.yaml
```

---

## 快速上手(合成資料 smoke)

### 第 1 步:一次性準備

```powershell
# 假權重(之後換官方 DINOv3 ViT-B/16,程式不用改)
& $PY triplet_ssl\make_mimic_checkpoint.py --out checkpoints\dinov3_vitb16_mimic.pth

# 合成 triplet 資料(--misalign_px N 可模擬 die-to-die 偏移來測對位)
& $PY triplet_ssl\data_gen\generate_triplets.py --out data\sem_triplet `
    --n_train 200 --n_val 60 --n_test 60 --defect_frac 0.5
```

### 第 2 步:依序跑各 Phase

**所有 Phase 必須用同一個 `--out-dir`**,Phase 1+ 的 gate 才讀得到 Phase 0 baseline:

```powershell
$OUT = "triplet_ssl\runs\exp1"
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase0.yaml --out-dir $OUT   # 凍結 baseline(不訓練)
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase1.yaml --out-dir $OUT   # Triplet DINO
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase2.yaml --out-dir $OUT `
    --checkpoint "$OUT\phase1_teacher.pth"   # + 缺陷過採樣 + top-k 豁免
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase3.yaml --out-dir $OUT `
    --checkpoint "$OUT\phase2_teacher.pth"   # + bounded margin + fusion head
```

每個 Phase 跑完會印出 AUROC 與相對 Phase 0 的 delta(PASS/FAIL),**通過 gate 再進下一個 Phase**。
Phase 只自動載入 config 指定的 checkpoint；因此 Phase 2/3 必須像上面明確傳入前一階段
teacher checkpoint，否則會重新從 base checkpoint 開始。

常用覆寫參數:

```powershell
--checkpoint <真權重.pth>   # 換官方 DINOv3 權重
--data-root  <資料根目錄>    # 換真實資料
--num-steps 500 --batch-size 16 --device cuda
```

---

## 換成真實資料

### 資料格式

```
<root>/
  train/
    <triplet_id>/ref1.png  ref2.png  target.png          # 訓練不需要 mask
    manifest.json
  val/
    <triplet_id>/ref1.png  ref2.png  target.png  mask.png # mask 只給 eval 用
    manifest.json
```

`manifest.json` 是一個 list,每筆至少要有:

```json
[{"id": "loc_00001", "has_defect": true}, ...]
```

- `has_defect`:訓練期用來做**缺陷過採樣**(不會讀 mask);eval 期用來挑熱圖樣本。
- `mask.png`:單通道,>0 = 缺陷像素。**只有 eval split 需要**;訓練路徑有
  `assert_no_masks()` 保證絕不讀取。
- 灰階圖請存成 3 通道 PNG(灰階複製三份),或存單通道由 cv2 讀成 3 通道亦可。

TIFF 模式是一個檔案包含 target/ref1/ref2 三個語意 samples。reader 依 TIFF
metadata 接受單頁 `YXS=(H,W,3)`、單頁 `SYX=(3,H,W)` 或三個 grayscale pages，
統一成 HWC 後才套 `channel_order`。目前 production 規格為 float32、
SampleFormat=3、0..255，會原值保留；NaN、Inf 或越界會直接失敗，不做 per-image min-max。

### 真資料 checklist(重要)

1. **重算 normalization**:`triplet_ssl/__init__.py` 的 `IMG_MEAN / IMG_STD`
   目前是合成 SEM 統計,務必用 `tools/compute_stats.py` 換成 optical training
   set 統計；float32 直接使用 0..255，uint16 才需傳入相同 black/white level。
2. **保留原生 PSF**:`phase3_tiff.yaml` 會從約 480px TIFF 裁原生 128px window，
   不 resize，因此 4–6px PSF 不會先被縮掉。不要改回 224px whole-image resize。
3. **對位**:die-to-die 有偏移就開 `data.register: true`。對位是
   phaseCorrelate 粗對位 + ECC 次像素精修(實測精度 ~0.03 px),偏移超過半個
   patch(8 px)會警告。`dataset.offset_log` 存有每筆量到的偏移。
4. **翻轉合法性**:垂直翻轉已在程式內禁用;水平翻轉若對你的 pattern 也不合法,
   把 `data.flip_prob` 設 0。

### 換真權重

把官方 DINOv3 ViT-B/16 checkpoint 放到任意路徑,跑的時候加
`--checkpoint <路徑>` 即可(load hook 會自動剝 `backbone.` / `model` /
`teacher` 等前綴)。訓練與推論都使用官方 canonical factory（4 storage tokens、
LayerScale、DINOv3 RoPE/norm/mask-key 設定），任何 missing 或 unexpected backbone
key 都會直接失敗，不會默默丟棄結構權重。舊版 generic `vit_base` mimic/bundle
不相容，請重跑 `make_mimic_checkpoint.py` 或從官方權重重新訓練。
mimic 權重是隨機的,**所有數字在換真權重前都沒有意義**。

---

## 多 GPU(torchrun,例:2×H200 大 batch)

```bash
torchrun --nproc_per_node=2 triplet_ssl/train_triplet.py \
    --config triplet_ssl/configs/phase3_tiff.yaml \
    --out-dir triplet_ssl/runs/exp_h200 --checkpoint <真權重.pth>
```

- `optim.batch_size` 是**每卡** batch;global = batch_size × 卡數。大 global batch 時
  依線性法則手動放大 LR(`lr_eff = lr × global_batch / 調參時的 batch`)。
- `optim.amp: true` 開 bf16 autocast(H100/H200 建議)。
- `data.num_workers: 4` 左右,讓撒點/對位不卡 GPU。
- 機制:student 包在單一 forward 的容器裡進 DDP;teacher 每卡各持一份(EMA 自
  同步的 student 而來,天然一致);centering 跨卡 all-reduce;每卡不同 seed 取
  不同 batch;eval / 存檔 / 曲線只在 rank 0。
- Windows 的 torch build 缺 libuv 與 gloo transport,多程序只能在 Linux 跑;
  `DDP_INIT_FILE` 環境變數是 FileStore 逃生口(除錯用)。

## Phase 說明

| Phase | 內容 | 開啟的 flag |
|-------|------|------------|
| 0 | 凍結 backbone,量 baseline AUROC(後面每階段要超越的門檻)| — |
| 1 | Triplet DINO:best-reference patch loss + 同圖 DINO + 共享幾何增強 | — |
| 2 | + 缺陷 triplet 過採樣(batch 的 5–20%)+ robust top-k 豁免 | `oversample.enable`、`loss.topk_exempt.enable` |
| 3 | + bounded margin + order-aware target-only fusion head | `synth_defect.enable`、`fusion_head.enable` |

**Loss 配對表**(teacher ↔ student,CLS 與 patch 兩層都算):

| 配對 | 權重 | top-k 豁免 |
|------|------|-----------|
| best[t(ref1),t(ref2)]↔s(target) | `w_target2ref=0.5` | **是**(逐 patch 選 loss 較低的 ref)|
| 同圖兩增強(輪替 t/r1/r2)| `w_traditional=0.5` | 否 |

不做全面 ref1↔ref2 pull：單一 reference 上的 PSF 是 nuisance，但它的 morphology
仍必須保留，才能和 target-only PSF 做 presence 比較。

**Phase 3 合成 PSF 事件(排列組合放置)**:每 triplet 撒 `n_events` 顆 nuisance
PSF 點,presence 組合 (target, ref1, ref2) 隨機取自六種非 defect 組合、位置跨
die 共享;另以 `defect_prob` 在無真缺陷的 triplet 上放**一顆 (1,0,0) 真 defect**
(target 有、兩 ref 都沒有,位置避開 nuisance)。`synth_mask` 只標 (1,0,0) positive；
`synth_event_mask` 則標出所有已知人工事件，讓其他組合（特別是 (0,1,1)）成為
fusion hard negative，但不會把未標註真影像背景硬標 normal。合成 defect token
另以有界 L2 margin 推離兩張 reference，margin 達標後 loss 為零。
極性依局部背景決定(亮區放暗點、暗區放亮點,防 clip 飽和)。

穩定化照 stock DINO:EMA teacher(momentum 0.996→1.0 cosine)、centering、
teacher temp 0.04→0.07 warmup、cosine LR + warmup。backbone LR 小(1e-5,
layer-wise decay 0.9),head LR 正常(1e-3)。

---

## Config 重點(`configs/base.yaml`,各 phaseN.yaml 只覆寫差異)

| Key | 預設 | 說明 |
|-----|------|------|
| `checkpoint` | mimic 路徑 | 初始化權重(**絕不從零訓練**)|
| `data.register` | false | die-to-die 次像素對位 |
| `data.uint16_black_level` | 0 | uint16 optical acquisition 的固定 black level |
| `data.uint16_white_level` | 65535 | 固定 white level；right-aligned 12-bit 請設 4095 |
| `data.crop_scale` | [0.4, 1.0] | 共享幾何裁切比例 |
| `data.flip_prob` | 0.5 | 水平翻轉(垂直翻轉永遠禁用)|
| `loss.topk_exempt.pct` | 0.02 | 豁免最高 loss 的 patch 比例(只作用在 target↔ref)|
| `oversample.defect_frac` | 0.15 | 每 batch 缺陷 triplet 佔比 |
| `synth_defect.lambda` | 0.1 | bounded backbone-token margin loss 強度 |
| `synth_defect.margin` | 0.5 | L2-normalized token 距離目標(範圍 0–2)|
| `fusion_head.loss_weight` | 0.5 | order-aware balanced BCE 權重 |
| `fusion_head.warmup_steps` | 10 | fusion loss 線性 warmup |
| `fusion_head.ema_momentum` | 0.9 | 從零初始化 fusion head 的快速 EMA |
| `fusion_head.background_negative_frac` | 0.25 | 每張圖最低 residual 的安全背景 negative 比例 |
| `optim.backbone_lr` | 1e-5 | 持續預訓練用小 LR |
| `optim.num_steps` | 60 | 訓練步數(正式訓練請加大)|
| `eval.overlap_thresh` | 0 | patch 標成缺陷的 GT 重疊門檻(0=碰到就算)|
| `separability_gate.min_auroc_delta` | 0.0 | 過關門檻(AUROC 需 ≥ Phase0 + 此值)|

---

## 輸出解讀(都在 `--out-dir` 下)

| 檔案 | 內容 |
|------|------|
| `summary.tsv` | 總表:phase / AUROC / 缺陷÷正常殘差比 / guardrail / 備註 |
| `phaseN_metrics.json` | 完整指標(AUROC、delta、gate 結果、殘差統計、guardrail)|
| `phaseN_heatmaps/` | ~10 張殘差熱圖疊 GT 輪廓(白線)— 看殘差有沒有咬中缺陷 |
| `phaseN_curves.png` | 訓練曲線:各 loss 分項 + LR / teacher temp / momentum |
| `phaseN_train_log.json` | 每步數值(可重繪曲線)|
| `phaseN_exempt_*.png` | top-k 豁免的 patch 紅框疊圖(Phase 2+)— **紅框應落在缺陷區**,這是 top-k 有沒有咬對的 sanity 訊號 |
| `phaseN_teacher.pth` | teacher backbone；Phase 3 同時含 EMA fusion head 與部署 metadata |
| `phase0_auroc.txt` | baseline 門檻(Phase 1+ 的 gate 讀這個)|

**Guardrail**:每個 Phase 同時回報「無缺陷 triplet」的殘差均值/p99 —
訓練後應維持低且均勻,若升高代表誤報傾向在惡化。

---

## 已知限制(真資料上線前要處理)

1. `phase3_tiff` 已使用原生 128px crop，不 resize；舊的 folder PNG pipeline
   仍會 resize，只適合原本的 224px 範例資料，不可當 optical deployment gate。
2. **top-k 是百分比**:224px 時 2% = 3 patches,剛好 ≈ 一顆 PSF 缺陷;
   換解析度時記得重估(512px 時 2% = 20 patches,豁免過頭)。
3. Sinkhorn 未用(dinov3 版綁 distributed),以 centering 取代 —
   單機/CPU 可跑;多卡要 Sinkhorn 時再換。
4. 無標註 optical TIFF 不宣稱 AUROC；請用相同 registration/tile/channel order 的
   `infer.py --calib` 做 known-normal guardrail。224px folder eval 僅可顯式開啟為 proxy。

## 常見問題

| 症狀 | 處理 |
|------|------|
| `no phase-0 baseline in this --out-dir` | 先在同一個 out-dir 跑 phase0 |
| loss 變 NaN | 降 `backbone_lr`,或檢查 `teacher_temp` 是否過低 |
| 對位警告 `> 8px` | 偏移超過半 patch,patch 配對失效 — 檢查資料或先做粗對位 |
| AUROC 都是 0.97+ 但沒訓練 | 你還在用 mimic 隨機權重,數字無意義 |
| 讀不到影像 | 檢查目錄結構是否為 `<split>/<id>/{ref1,ref2,target}.png` + `manifest.json` |
## Order-aware triplet fusion head 部署

Phase 3 現在會訓練並部署具方向性的 patch head，其分數只代表：

```text
(target, ref1, ref2) = (1, 0, 0)  -> target-unique defect
其他所有 presence pattern          -> nuisance / normal
```

target 是特殊角色；ref1/ref2 共用權重，互換後輸出完全不變。訓練使用
`synth_mask` 標記 `100` positive，並用 `synth_event_mask` 標出所有 truth table
已知的人工事件位置。另只從 best-reference residual 最低的背景 patch 抽取安全
`000` negatives；高 residual 的未標註背景仍忽略，避免把未知真缺陷拉成 normal。

`phase3_teacher.pth` 是 version-2 部署 bundle，包含 EMA teacher backbone、EMA
fusion head、架構契約、patch size、channel order、registration、source dtype/layout
以及必要的 uint16 black/white level。省略 `--tile` 時會自動使用 checkpoint 的 training tile。
Fusion 預設用 32px context halo：每次仍送 128px tile，但只拼中央 64px，避免
6px PSF 恰好跨 tile seam；`--chunk 1` 的 VRAM 不增加，計算量約為非重疊的 4 倍。
如要關閉可明確傳 `--context-halo 0`，但必須重新校準 threshold。
先用正常影像校準，再執行：

```powershell
& $PY triplet_ssl\infer.py --checkpoint triplet_ssl\runs\exp1\phase3_teacher.pth `
    --input data\known_normal_tiffs --method fusion --calib

& $PY triplet_ssl\infer.py --checkpoint triplet_ssl\runs\exp1\phase3_teacher.pth `
    --input data\production_tiffs --method fusion --threshold <CALIBRATED_THRESHOLD> `
    --out-dir triplet_ssl\runs\fusion_infer --chunk 1
```

Fusion map 是 `sigmoid(target_unique_logit)`；在 held-out production data 校準前，
不要把它解讀成真實機率。校準結果會寫入 `--out-dir/calibration.json`，正式
threshold 必須由 held-out normal 與少量人工複核共同決定。舊的 backbone-only checkpoint 仍可使用
`--method residual` 或 `knn`；沒有 fusion head 卻指定 `--method fusion` 會明確報錯。
`--method auto` 有 head 時用 fusion，否則警告後 fallback 到 residual-min。

CPU regression tests：

```powershell
& $PY -m unittest discover -s triplet_ssl\tests -v
```
