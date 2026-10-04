"""Predict depth labels and IPT edges for a few graphs; save JSON."""

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
from neardup.metrics.ipt import predict_parents_from_depth_prnu
from neardup.models import build_model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(cfg)
    sample = next(iter(test_loader))
    model = build_model(cfg, sample["x"].shape[-1]).to(device)
    ckpt = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            pred = model(x, adj).argmax(-1).cpu()
            for i in range(pred.size(0)):
                d = pred[i].numpy()
                prnu = batch["prnu"][i].numpy()
                edges = predict_parents_from_depth_prnu(d, prnu)
                rec = {
                    "depth_pred": d.tolist(),
                    "depth_gt": batch["depth"][i].tolist(),
                    "edges_pred": edges,
                    "parents_gt": batch["parents"][i].tolist(),
                }
                path = out_dir / f"ipt_{len(saved):04d}.json"
                path.write_text(json.dumps(rec, indent=2))
                saved.append(str(path))
                if len(saved) >= args.n:
                    print(json.dumps({"saved": saved}, indent=2))
                    return
    print(json.dumps({"saved": saved}, indent=2))


if __name__ == "__main__":
    main()
