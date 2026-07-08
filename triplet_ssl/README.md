# Triplet SSL — DINOv3 三聯圖持續預訓練(die-to-die 檢測)

針對 optical short-wavelength die-to-die 檢測影像的 DINOv3 ViT-B/16 持續預訓練
(continued SSL pretraining)。每筆樣本是一組三聯圖(triplet):

- **target**:待檢圖,可能含缺陷(~4–6 px PSF 光斑)
- **ref1 / ref2**:同一 pattern 位置的無缺陷鄰 die 參考圖

核心想法:**用 triplet 結構本身提供正樣本對**(ref↔ref、target↔ref),取代傳統
DINO「同一張圖兩次增強」;搭配 robust top-k 豁免,讓缺陷 patch 不被強迫拉向正常
特徵(避免 normality collapse)。訓練後以 patch 殘差
`‖f(target) − mean(f(ref1), f(ref2))‖` 的缺陷/正常 AUROC 做分階段驗收。

本模組**純增量**:不改動 `dinov3/` 既有程式,只唯讀重用其元件。

---

## 環境

只有這個 interpreter 有 torch + cv2 + matplotlib(CPU 版):

```powershell
$PY = "C:\Users\Peter Peng\AppData\Local\Programs\Python\Python312\python.exe"
```

所有指令都在 repo 根目錄(`claude\dinov3`)執行。

## 目錄結構

```
triplet_ssl/
  data_gen/generate_triplets.py   # 合成 triplet 產生器(真資料來之前的替身)
  make_mimic_checkpoint.py        # 假權重 .pth 產生器(真權重來之前的替身)
  data/triplet_dataset.py         # dataloader:共享幾何/獨立光度增強、次像素對位、缺陷過採樣
  data/synth_defect.py            # Phase 3 合成缺陷 + 增強存活率檢查
  losses/triplet_loss.py          # 配對表 + robust top-k 豁免 + repulsion
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
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase2.yaml --out-dir $OUT   # + 缺陷過採樣 + top-k 豁免
& $PY triplet_ssl\train_triplet.py --config triplet_ssl\configs\phase3.yaml --out-dir $OUT   # + 合成缺陷 repulsion(可選)
```

每個 Phase 跑完會印出 AUROC 與相對 Phase 0 的 delta(PASS/FAIL),**通過 gate 再進下一個 Phase**。

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

### 真資料 checklist(重要)

1. **重算 normalization**:`triplet_ssl/__init__.py` 的 `IMG_MEAN / IMG_STD`
   目前是合成 SEM 統計,務必換成你的資料統計。
2. **解析度警告(尚未修)**:目前訓練增強與 eval 都會把影像 resize 到
   `model.img_size`(預設 224)。**若原圖遠大於 224,縮圖會把 4–6 px 的 PSF
   缺陷抹掉**。真資料進來前需改成原生解析度視窗裁切(見「已知限制」)。
3. **對位**:die-to-die 有偏移就開 `data.register: true`。對位是
   phaseCorrelate 粗對位 + ECC 次像素精修(實測精度 ~0.03 px),偏移超過半個
   patch(8 px)會警告。`dataset.offset_log` 存有每筆量到的偏移。
4. **翻轉合法性**:垂直翻轉已在程式內禁用;水平翻轉若對你的 pattern 也不合法,
   把 `data.flip_prob` 設 0。

### 換真權重

把官方 DINOv3 ViT-B/16 checkpoint 放到任意路徑,跑的時候加
`--checkpoint <路徑>` 即可(load hook 會自動剝 `backbone.` / `model` /
`teacher` 等前綴)。mimic 權重是隨機的,**所有數字在換真權重前都沒有意義**。

---

## Phase 說明

| Phase | 內容 | 開啟的 flag |
|-------|------|------------|
| 0 | 凍結 backbone,量 baseline AUROC(後面每階段要超越的門檻)| — |
| 1 | Triplet DINO:配對表 + patch-level loss + 共享幾何增強 | — |
| 2 | + 缺陷 triplet 過採樣(batch 的 5–20%)+ robust top-k 豁免 | `oversample.enable`、`loss.topk_exempt.enable` |
| 3 | + 合成缺陷 repulsion(把合成缺陷 patch 主動推離參考特徵)| `synth_defect.enable` |

**Loss 配對表**(teacher ↔ student,CLS 與 patch 兩層都算):

| 配對 | 權重 | top-k 豁免 |
|------|------|-----------|
| t(ref1)↔s(ref2) + 對稱 | `w_ref2ref=0.4` | 否(兩邊都正常,全額一致性)|
| t(ref1\|ref2)↔s(target) | `w_target2ref=0.3` | **是**(缺陷 patch 本來就不該匹配參考)|
| 同圖兩增強(輪替 t/r1/r2)| `w_traditional=0.3` | 否 |

穩定化照 stock DINO:EMA teacher(momentum 0.996→1.0 cosine)、centering、
teacher temp 0.04→0.07 warmup、cosine LR + warmup。backbone LR 小(1e-5,
layer-wise decay 0.9),head LR 正常(1e-3)。

---

## Config 重點(`configs/base.yaml`,各 phaseN.yaml 只覆寫差異)

| Key | 預設 | 說明 |
|-----|------|------|
| `checkpoint` | mimic 路徑 | 初始化權重(**絕不從零訓練**)|
| `data.register` | false | die-to-die 次像素對位 |
| `data.crop_scale` | [0.4, 1.0] | 共享幾何裁切比例 |
| `data.flip_prob` | 0.5 | 水平翻轉(垂直翻轉永遠禁用)|
| `loss.topk_exempt.pct` | 0.02 | 豁免最高 loss 的 patch 比例(只作用在 target↔ref)|
| `oversample.defect_frac` | 0.15 | 每 batch 缺陷 triplet 佔比 |
| `synth_defect.lambda` | 0.1 | repulsion 強度 |
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
| `phaseN_teacher.pth` | 訓練後的 teacher backbone |
| `phase0_auroc.txt` | baseline 門檻(Phase 1+ 的 gate 讀這個)|

**Guardrail**:每個 Phase 同時回報「無缺陷 triplet」的殘差均值/p99 —
訓練後應維持低且均勻,若升高代表誤報傾向在惡化。

---

## 已知限制(真資料上線前要處理)

1. **縮圖問題(P0,未修)**:增強與 eval 的 resize 會毀掉 4–6 px 缺陷,
   原圖 > `img_size` 時必須先改成原生解析度裁切。
2. **合成缺陷形貌(P0,未修)**:`synth_defect.py` 目前是硬邊方塊/線條,
   與 PSF 光斑不符;真的要用 Phase 3 前應改成低對比 Gaussian blob(σ≈0.8–1.5)。
3. **top-k 是百分比**:224px 時 2% = 3 patches,剛好 ≈ 一顆 PSF 缺陷;
   換解析度時記得重估(512px 時 2% = 20 patches,豁免過頭)。
4. Sinkhorn 未用(dinov3 版綁 distributed),以 centering 取代 —
   單機/CPU 可跑;多卡要 Sinkhorn 時再換。

## 常見問題

| 症狀 | 處理 |
|------|------|
| `no phase-0 baseline in this --out-dir` | 先在同一個 out-dir 跑 phase0 |
| loss 變 NaN | 降 `backbone_lr`,或檢查 `teacher_temp` 是否過低 |
| 對位警告 `> 8px` | 偏移超過半 patch,patch 配對失效 — 檢查資料或先做粗對位 |
| AUROC 都是 0.97+ 但沒訓練 | 你還在用 mimic 隨機權重,數字無意義 |
| 讀不到影像 | 檢查目錄結構是否為 `<split>/<id>/{ref1,ref2,target}.png` + `manifest.json` |
