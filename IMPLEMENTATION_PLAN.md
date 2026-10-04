# Implementation Plan — Detecting Near-Duplicate Face Images

**Paper:** Banerjee & Ross, TBIOM 2025 / arXiv:2408.07689  
**Slug:** `detecting-near-duplicate-face-images`  
**Package:** `neardup`  
**Upstream:** https://github.com/sudban3089/DetectingNear-Duplicates (TF GCN + MATLAB PRNU)

## Method

1. Build IPT graphs of near-duplicates via photometric/geometric transforms (Table 1).
2. Node features: 96×96 grayscale pixels (+ PRNU residual proxy).
3. **ChebNet** (`gcn_cheby`, K=3) predicts depth labels (Nodeembedding.py defaults).
4. **Link prediction:** PRNU pairwise distances + depth constraints → directed IPT.
5. Metrics: root ID accuracy; IPT edge reconstruction accuracy.

## Data (local)

| Set | Role | Path |
|-----|------|------|
| LFW | synthesize IPT train/test (paper training source) | `projects/datasets/lfw/lfw` |
| CelebA | optional extra identities | `projects/datasets/celeba/...` |
| NDFI / CelebAHQ-FM / MORPH | paper test sets | unavailable → D_n |

## Training (upstream Nodeembedding.py)

- Adam lr **0.01**, epochs **100**, hidden **16**, dropout **0.5**, weight_decay **5e-4**, early_stopping **10**, Chebyshev **K=3**

## Deviations expected

- MATLAB Enhanced PRNU → Python high-pass / wavelet-lite residual (D1)
- Official NDFI test set missing → hold-out LFW-synthesized IPTs (D2)
- TF → PyTorch ChebNet refactor (D3)
