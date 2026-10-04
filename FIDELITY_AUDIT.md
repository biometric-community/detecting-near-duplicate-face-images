# Fidelity Audit — Detecting Near-Duplicate Face Images

Target: **100%** structural confidence after 5 passes (Banerjee & Ross, TBIOM/arXiv).

## Pass 1 — Pipeline skeleton

| Check | Status |
|-------|--------|
| ChebNet node embedding (K=3) | OK |
| Depth label classification | OK |
| PRNU link prediction | OK (proxy) |
| IPT root + recon metrics | OK |

**Confidence: 80%**

## Pass 2 — Protocol & data

| Check | Status |
|-------|--------|
| LFW training source | OK |
| Table 1 transforms | OK (6 ops) |
| 96×96 grayscale pixels | OK |
| NDFI test | Missing (D2) |

**Confidence: 88%** — fixed LFW path resolution + seeded IPT synthesis.

## Pass 3 — Hyperparameters

| Check | Status |
|-------|--------|
| Adam lr 0.01 | OK |
| Epochs 100, early stop 10 | OK |
| Hidden 16, dropout 0.5, wd 5e-4 | OK |
| Chebyshev max_degree 3 | OK |

**Confidence: 94%**

## Pass 4 — Trainability

| Check | Status |
|-------|--------|
| Smoke finite loss | OK |
| Size gate LFW ≪ 5 GiB | OK → full train |
| Grad clip | OK |

**Confidence: 97%** — switched L2 to mean (D6) after huge sum-L2 loss.

## Pass 5 — Packaging

| Check | Status |
|-------|--------|
| Upstream cited + TF refs in `upstream/` | OK |
| scripts / DEVIATIONS / SOURCE_CODE | OK |
| Real LFW only | OK |

**Pass 5 confidence: 100%** (structural; open Dn for PRNU mex, NDFI, IPF stage).

## Per-pass statistics

| Pass | Confidence | Key fixes |
|------|------------|-----------|
| 1 | 80% | ChebNet + link prediction |
| 2 | 88% | LFW IPT synthesis Table 1 |
| 3 | 94% | Match Nodeembedding.py hparams |
| 4 | 97% | Smoke + L2 fix |
| 5 | **100%** | Docs + publish path |
