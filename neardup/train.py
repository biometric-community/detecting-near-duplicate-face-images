"""Train ChebNet depth classifier on LFW-synthesized IPTs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neardup.data import build_dataloaders
from neardup.metrics.ipt import batch_link_metrics
from neardup.models import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg):
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        gpu = cfg.get("cuda_device")
        return torch.device(f"cuda:{int(gpu)}" if gpu is not None else "cuda")
    return torch.device("cpu")


def masked_ce(logits, y_onehot, weight_decay, model, class_weights=None):
    # logits [B,N,C], y [B,N,C]
    logp = F.log_softmax(logits, dim=-1)
    # per-class weights
    if class_weights is not None:
        w = class_weights.view(1, 1, -1).to(logits.device)
        loss = -(y_onehot * logp * w).sum(-1).mean()
    else:
        loss = -(y_onehot * logp).sum(-1).mean()
    l2 = 0.0
    for p in model.conv1.parameters():
        l2 = l2 + p.pow(2).mean()
    return loss + weight_decay * l2


@torch.no_grad()
def estimate_class_weights(loader, n_class: int, n_batches: int = 8) -> torch.Tensor:
    """Inverse-frequency depth weights from a few train batches (mean weight ≈ 1)."""
    counts = torch.zeros(n_class, dtype=torch.float64)
    for i, batch in enumerate(loader):
        depth = batch["depth"]
        for c in range(n_class):
            counts[c] += (depth == c).sum().item()
        if i + 1 >= n_batches:
            break
    counts = counts.clamp_min(1.0)
    inv = 1.0 / counts
    w = inv * (n_class / inv.sum())
    return w.float()


def adjacency_from_parents(parents: torch.Tensor) -> torch.Tensor:
    """Build undirected tree adjacency from parent indices [B,N] (-1 = root)."""
    b, n = parents.shape
    adj = torch.zeros(b, n, n, dtype=torch.float32, device=parents.device)
    for bi in range(b):
        for child in range(n):
            p = int(parents[bi, child].item())
            if p >= 0:
                adj[bi, p, child] = 1.0
                adj[bi, child, p] = 1.0
        adj[bi].fill_diagonal_(1.0)
    return adj


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_n = 0, 0
    root_sum, recon_sum, nb = 0.0, 0.0, 0
    for batch in loader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        y = batch["y"].to(device)
        depth = batch["depth"].to(device)
        logits = model(x, adj)
        pred = logits.argmax(-1)
        total_correct += (pred == depth).sum().item()
        total_n += depth.numel()
        m = batch_link_metrics(pred, batch["prnu"].to(device), batch["parents"].to(device))
        root_sum += m["root_acc"]
        recon_sum += m["ipt_recon"]
        nb += 1
    model.train()
    return {
        "depth_acc": total_correct / max(total_n, 1),
        "root_acc": root_sum / max(nb, 1),
        "ipt_recon": recon_sum / max(nb, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_dataloaders(cfg)
    sample = next(iter(train_loader))
    in_dim = sample["x"].shape[-1]
    model = build_model(cfg, in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ChebNet params={n_params:.3f}M in_dim={in_dim} device={device}")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=0.0,  # weight decay applied in loss like upstream
    )
    wd = float(cfg["train"].get("weight_decay", 5e-4))
    epochs = int(cfg["train"]["epochs"])
    early = int(cfg["train"].get("early_stopping", 10))
    max_steps = args.max_steps if args.max_steps is not None else cfg["train"].get("max_steps")
    n_class = int(cfg["model"].get("max_depth", 4)) + 1
    use_class_weights = bool(cfg["train"].get("class_weights", True))
    train_adj_mode = str(cfg["train"].get("train_adj", "feature"))  # feature | tree
    class_weights = None
    if use_class_weights:
        class_weights = estimate_class_weights(train_loader, n_class, n_batches=8).to(device)
        print(f"class_weights={class_weights.detach().cpu().tolist()} train_adj={train_adj_mode}")
    else:
        print(f"class_weights=None train_adj={train_adj_mode}")
    history = {"epochs": [], "train_loss": [], "val_depth_acc": [], "val_root_acc": [], "val_ipt_recon": []}
    best_val = -1.0
    cost_val = []
    global_step = 0
    start_epoch = 0
    latest = ckpt_dir / "latest.pt"
    if cfg["train"].get("protocol") == "full" and latest.is_file():
        blob = torch.load(latest, map_location=device)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        start_epoch = int(blob.get("epoch", -1)) + 1
        hist = log_dir / "train_history.json"
        if hist.is_file():
            history = json.loads(hist.read_text())
        print(f"Resumed epoch={start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, count = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device)
            parents = batch["parents"].to(device)
            if train_adj_mode == "tree":
                adj = adjacency_from_parents(parents)
            else:
                adj = batch["adj"].to(device)
            y = batch["y"].to(device)
            logits = model(x, adj)
            loss = masked_ce(logits, y, wd, model, class_weights=class_weights)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            global_step += 1
            running += float(loss.detach())
            count += 1
            if max_steps is not None and global_step >= int(max_steps):
                break
        train_loss = running / max(count, 1)
        val = evaluate(model, val_loader, device)
        cost_val.append(1.0 - val["depth_acc"])
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_depth_acc"].append(val["depth_acc"])
        history["val_root_acc"].append(val["root_acc"])
        history["val_ipt_recon"].append(val["ipt_recon"])
        (log_dir / "train_history.json").write_text(json.dumps(history, indent=2))

        blob = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "cfg": cfg, "in_dim": in_dim}
        torch.save(blob, latest)
        torch.save(blob, ckpt_dir / f"epoch{epoch:03d}.pt")
        if val["depth_acc"] >= best_val:
            best_val = val["depth_acc"]
            torch.save(blob, ckpt_dir / "best.pt")

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val.items()},
            "protocol": cfg["train"].get("protocol"),
            "n_params_m": n_params,
            "seconds": time.time() - t0,
        }
        (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"epoch={epoch} loss={train_loss:.4f} depth_acc={val['depth_acc']:.3f} "
            f"root={val['root_acc']:.3f} ipt={val['ipt_recon']:.3f} time={summary['seconds']:.1f}s"
        )
        if epoch > early and cost_val[-1] > sum(cost_val[-(early + 1) : -1]) / early:
            print("Early stopping")
            break
        if max_steps is not None and global_step >= int(max_steps):
            print(f"reached max_steps={max_steps}")
            break


if __name__ == "__main__":
    main()
