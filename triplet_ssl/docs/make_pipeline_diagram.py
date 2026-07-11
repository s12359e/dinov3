# -*- coding: utf-8 -*-
"""Render the triplet-SSL pipeline schematic to PNG.

Regenerate:  python triplet_ssl/docs/make_pipeline_diagram.py
Output:      triplet_ssl/docs/pipeline.png

Style follows the dataviz method: light surface, ink text tokens, validated
categorical palette used only as section accents (text never wears series color).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
BOX = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
EDGE = "#c6c5c0"
ACC = {"data": "#2a78d6", "model": "#1baf7a", "loss": "#eda100",
       "update": "#008300", "eval": "#4a3aa7", "phase": "#e34948"}

W, H = 100, 163
fig, ax = plt.subplots(figsize=(12, H * 0.12), facecolor=SURFACE)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def box(x, y, w, h, title, lines, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                facecolor=BOX, edgecolor=EDGE, linewidth=1.0))
    ax.add_patch(Rectangle((x - 0.15, y), 0.9, h, facecolor=accent, edgecolor="none"))
    ax.text(x + 1.8, y + 1.6, title, fontsize=9.5, fontweight="bold", color=INK, va="center")
    for i, ln in enumerate(lines):
        ax.text(x + 1.8, y + 3.6 + i * 1.9, ln, fontsize=8.3, color=INK2, va="center")


def header(y, key, text):
    ax.add_patch(Rectangle((4, y - 0.8), 1.6, 1.6, facecolor=ACC[key], edgecolor="none"))
    ax.text(6.6, y, text, fontsize=11, fontweight="bold", color=INK, va="center")


def arrow(x, y0, y1, color=INK2):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))


ax.text(50, 2.2, "Triplet SSL Pipeline — DINOv3 持續預訓練(單一 training step)",
        fontsize=13.5, fontweight="bold", color=INK, ha="center", va="center")

# ── ① DATA ──────────────────────────────────────────────────────────────────
header(6.5, "data", "① 資料路徑(dataloader)")
box(4, 8.5, 54, 7.5, "輸入(disk)",
    ["<id>/ref1.png · ref2.png · target.png",
     "manifest.json(has_defect 旗標)"], ACC["data"])
box(62, 8.5, 34, 7.5, "Batch 取樣",
    ["DefectOversampleBatchSampler",
     "phase>=2:缺陷 triplet 佔 15%"], ACC["data"])
arrow(31, 16.3, 18.6)
box(4, 19, 92, 9.5, "對位 register(config flag,預設關)",
    ["phaseCorrelate(Hanning)粗對位 + ECC 次像素精修(實測 ~0.03px)",
     "warpAffine + replicate 邊界(不繞邊);偏移 > 8px(半 patch)→ 警告",
     "ref2 → ref1,target → ref1(ref1 為錨點)"], ACC["data"])
arrow(31, 28.8, 30.6)
box(4, 31, 92, 9.5, "合成 PSF 事件 TripletSyntheticPSF(phase 3 才開)— 排列組合放置",
    ["nuisance:每 triplet 3–8 顆,presence (t,r1,r2) in {010,001,011,110,101,111},位置跨 die 共享",
     "真 defect = (1,0,0):target 有、兩 ref 都沒有;只放無真缺陷 triplet(prob 0.5),避開 nuisance 窗",
     "PSF:σ 1.1–1.7 → 4–6 px;極性看局部背景(防飽和);只有 (1,0,0) 進 loss mask"], ACC["data"])
arrow(31, 40.8, 42.6)
box(4, 43, 92, 9.5, "共享幾何增強(每 triplet 只抽一次,三張同步套用)",
    ["crop:scale U(0.4, 1.0) 隨機位置 + h-flip p=0.5;v-flip 永遠禁用(方向性 pattern)",
     "→ resize 224 → 三張圖的 patch k = 同一實體位置(patch 級配對的前提)",
     "synth pixel mask 跟著同一個幾何變換走 → patch mask (B, 196)"], ACC["data"])
arrow(31, 52.8, 54.6)
box(4, 55, 92, 7.5, "獨立光度增強 ×2(每張圖各自抽兩份 → student / teacher view)",
    ["brightness ±15 · contrast 0.9–1.1 · gamma 0.9–1.1 · blur 關(保護低對比小缺陷)"],
    ACC["data"])
arrow(31, 62.8, 64.6)
box(4, 65, 92, 7.5, "batch 輸出",
    ["ref1 / ref2 / target:各 (B, 2, 3, 224, 224) — view0 → student,view1 → teacher",
     "synth_mask:(B, 196)"], ACC["data"])

# ── ② MODELS ────────────────────────────────────────────────────────────────
header(76, "model", "② 模型(student / teacher 雙塔)")
arrow(31, 72.8, 74.6)   # stop above the section header, don't cross its text
box(4, 78, 42, 15.5, "STUDENT(訓練)",
    ["ViT-B/16 backbone",
     "init:DINOv3 pretrained ckpt",
     "lr 1e-5 · layerwise decay 0.9",
     "CLS head 768→1024 protos(lr 1e-3)",
     "patch head 768→1024 protos(lr 1e-3)"], ACC["model"])
box(54, 78, 42, 15.5, "TEACHER(EMA,無梯度)",
    ["ViT-B/16 backbone(student 複本)",
     "EMA momentum 0.996→1.0(cosine)",
     "CLS / patch head(EMA 複本)",
     "輸出銳化:centering(EMA 0.9)",
     "+ temp 0.04→0.07 warmup"], ACC["model"])
ax.annotate("", xy=(53.6, 85.5), xytext=(46.4, 85.5),
            arrowprops=dict(arrowstyle="-|>", color=ACC["model"], lw=1.6))
ax.text(50, 84.3, "EMA", fontsize=8.3, color=ACC["model"], ha="center", fontweight="bold")
ax.text(50, 95.6, "每 step 6 次 forward:s(ref1·v0) s(ref2·v0) s(tgt·v0) / t(ref1·v1) t(ref2·v1) t(tgt·v1)",
        fontsize=8.3, color=INK2, ha="center")
ax.text(50, 97.6, "每次輸出:cls logits (B, 1024) + patch logits (B, 196, 1024)",
        fontsize=8.3, color=INK2, ha="center")

# ── ③ LOSS ──────────────────────────────────────────────────────────────────
header(101.5, "loss", "③ Loss 配對表(每格 = patch 級 CE + 1.0 × CLS 級 CE)")
arrow(31, 98.8, 100.1)
box(4, 103.5, 44, 15.5, "配對矩陣(teacher 銳化後 <-> student)",
    ["              t(ref1)   t(ref2)  t(tgt)",
     "s(ref1)         ·        RR 0.4     ·",
     "s(ref2)       RR 0.4       ·        ·",
     "s(target)     TR 0.3     TR 0.3     ·",
     "同圖自己      TD 0.3(輪替 tgt/r1/r2)"], ACC["loss"])
box(52, 103.5, 44, 15.5, "TR 內部(唯一有特殊處理)",
    ["tr_pp (B,196) = ½[CE(s·tgt,t·r1)+CE(s·tgt,t·r2)]",
     "PULL:synth_mask 確定排除(不佔名額)",
     "  → top-k:剩餘最高 2%(3 patch)豁免",
     "  → 其餘取均值拉向 ref(壓 nuisance)",
     "REPEL(ph3):-0.1×mean(tr_pp[mask]) 推開"], ACC["loss"])
box(4, 121, 92, 7.5, "total = 0.4·RR + 0.3·TR + 0.3·TD(+ repel)",
    ["RR:全額一致性,無豁免 → nuisance 不變性的老師 | TD:傳統 DINO 對 → 特徵品質保底"],
    ACC["loss"])

# ── ④ UPDATE ────────────────────────────────────────────────────────────────
header(132, "update", "④ 更新順序(每 step)")
arrow(31, 128.8, 130.6)
box(4, 134, 92, 5.5, "backward → grad clip 3.0 → AdamW(cosine LR + warmup)→ EMA(backbone + 兩 head)→ centering 更新",
    [], ACC["update"])

# ── ⑤ EVAL ──────────────────────────────────────────────────────────────────
header(143, "eval", "⑤ Eval — phase 之間執行(唯一讀 GT mask 的地方,訓練路徑絕不碰)")
arrow(31, 139.8, 141.6)
box(4, 145, 92, 9.5, "residual_k = ‖ f(target)_k - ½( f(ref1)_k + f(ref2)_k ) ‖(teacher patch tokens,L2 norm)",
    ["AUROC(defect vs normal patch)· defect/normal 殘差比 · 熱圖疊 GT 輪廓 · 訓練曲線",
     "gate:AUROC >= phase0 baseline + δ → PASS 才進下一 phase;guardrail:正常 triplet 殘差 mean/p99"],
    ACC["eval"])

# ── Phase 開關 ───────────────────────────────────────────────────────────────
box(4, 156.5, 92, 5.5, "Phase 開關",
    ["0:凍結只 eval  |  1:RR/TR/TD  |  2:+ oversample 15% + top-k 2%  |  3:+ PSF 合成缺陷 + repel"],
    ACC["phase"])

out = "triplet_ssl/docs/pipeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {out}")
