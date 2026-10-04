"""Evaluate ChebNet + PRNU link prediction on held-out LFW IPTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neardup.data import build_dataloaders
from neardup.metrics.ipt import batch_link_metrics
from neardup.models import build_model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("cuda_device") is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(cfg['cuda_device'])}")

    _, _, test_loader = build_dataloaders(cfg)
    sample = next(iter(test_loader))
    in_dim = sample["x"].shape[-1]
    model = build_model(cfg, in_dim).to(device)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    blob = torch.load(ckpt, map_location=device)
    model.load_state_dict(blob["model"])
    model.eval()

    total_correct, total_n = 0, 0
    root_sum, recon_sum, nb = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            depth = batch["depth"].to(device)
            pred = model(x, adj).argmax(-1)
            total_correct += (pred == depth).sum().item()
            total_n += depth.numel()
            m = batch_link_metrics(pred, batch["prnu"].to(device), batch["parents"].to(device))
            root_sum += m["root_acc"]
            recon_sum += m["ipt_recon"]
            nb += 1
    results = {
        "depth_acc": total_correct / max(total_n, 1),
        "root_acc": root_sum / max(nb, 1),
        "ipt_recon": recon_sum / max(nb, 1),
        "n_batches": nb,
        "checkpoint": ckpt,
    }
    out = Path(cfg["paths"]["log_dir"]) / "eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
