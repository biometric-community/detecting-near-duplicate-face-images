"""REPORT.md + SVG figures from real train/eval logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neardup.plot_style import (
    FIGSIZE,
    LABEL_SIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    hist_path = log_dir / "train_history.json"
    if not hist_path.is_file():
        print("No train_history.json")
        return
    hist = json.loads(hist_path.read_text())
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hist["epochs"], hist["train_loss"], color=MULTI_SERIES_COLORS[0], lw=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
    ax.set_title("ChebNet training loss", fontsize=TITLE_SIZE)
    apply_style(ax)
    fig.savefig(fig_dir / "train_loss.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hist["epochs"], hist["val_depth_acc"], color=MULTI_SERIES_COLORS[0], lw=LINEWIDTH_MAIN, label="Depth acc")
    ax.plot(hist["epochs"], hist["val_root_acc"], color=MULTI_SERIES_COLORS[1], lw=LINEWIDTH_MAIN, label="Root ID")
    ax.plot(hist["epochs"], hist["val_ipt_recon"], color=MULTI_SERIES_COLORS[2], lw=LINEWIDTH_MAIN, label="IPT recon")
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
    ax.set_title("Validation metrics", fontsize=TITLE_SIZE)
    ax.legend()
    apply_style(ax)
    fig.savefig(fig_dir / "val_metrics.svg", bbox_inches="tight")
    plt.close(fig)

    summary = json.loads((log_dir / "train_summary.json").read_text()) if (log_dir / "train_summary.json").is_file() else {}
    eval_js = json.loads((log_dir / "eval_results.json").read_text()) if (log_dir / "eval_results.json").is_file() else {}

    def fmt(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return "n/a"

    report = f"""# Detecting Near-Duplicate Face Images — Results Report

Paper: Banerjee & Ross (TBIOM / arXiv:2408.07689). ChebNet depth embedding + PRNU link prediction on LFW-synthesized IPTs.

## Training

| Metric | Value |
|--------|------:|
| Protocol | {summary.get('protocol', 'n/a')} |
| Epoch | {summary.get('epoch', 'n/a')} |
| Train loss | {fmt(summary.get('train_loss'))} |
| Val depth acc | {fmt(summary.get('val_depth_acc'))} |
| Val root ID | {fmt(summary.get('val_root_acc'))} |
| Val IPT recon | {fmt(summary.get('val_ipt_recon'))} |

![loss](outputs/figures/train_loss.svg)

![val](outputs/figures/val_metrics.svg)

## Test

| Metric | Value |
|--------|------:|
| Depth acc | {fmt(eval_js.get('depth_acc'))} |
| Root ID | {fmt(eval_js.get('root_acc'))} |
| IPT recon | {fmt(eval_js.get('ipt_recon'))} |

Real LFW images only; NDFI official test set not available locally (see DEVIATIONS.md).
"""
    Path("REPORT.md").write_text(report)
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
