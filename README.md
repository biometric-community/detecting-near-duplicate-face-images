# Detecting Near-Duplicate Face Images

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

PyTorch rebuild of Banerjee & Ross (TBIOM / arXiv:2408.07689): **ChebNet** depth embedding + **PRNU** link prediction for face image phylogeny trees (IPT).

**Repository:** https://github.com/biometric-community/detecting-near-duplicate-face-images

Upstream (TF + MATLAB): https://github.com/sudban3089/DetectingNear-Duplicates

## Setup

```bash
bash scripts/setup_env.sh
# LFW at projects/datasets/lfw/lfw
```

## Train / eval / report

```bash
bash scripts/train.sh
bash scripts/train_full.sh
bash scripts/eval.sh && bash scripts/predict.sh && bash scripts/report.sh
```

See [REPORT.md](REPORT.md), [DEVIATIONS.md](DEVIATIONS.md), [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md), [SOURCE_CODE.md](SOURCE_CODE.md).

## Citation

```bibtex
@article{banerjee2025neardup,
  title={Detecting Near-Duplicate Face Images},
  author={Banerjee, Sudipta and Ross, Arun},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2025},
  doi={10.1109/TBIOM.2025.3548541},
  note={arXiv:2408.07689}
}
```
