"""LFW-based IPT graph synthesis (paper: train on LFW near-duplicates)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .prnu import flatten_norm, prnu_residual
from .transforms import TRANSFORM_BANK, apply_random_transform


def resolve_lfw_root(root: str | Path) -> Path:
    p = Path(root)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    proj = Path(__file__).resolve().parents[2]
    candidates.append((proj / p).resolve())
    repo = proj
    while repo != repo.parent:
        if (repo / "docs" / "PAPERS.md").exists():
            break
        repo = repo.parent
    stripped = Path(*[x for x in Path(root).parts if x != ".."])
    candidates.append((repo / "projects" / stripped).resolve())
    candidates.append(repo / "projects" / "datasets" / "lfw" / "lfw")
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            # prefer nested lfw/lfw
            if (c / "lfw").is_dir():
                return c / "lfw"
            return c
    raise FileNotFoundError(f"LFW root not found from {root}")


def list_lfw_images(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"No images under {root}")
    return files


def load_gray96(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    return cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)


def synthesize_ipt(
    root_img: np.ndarray,
    n_nodes: int,
    rng: np.random.RandomState,
    max_depth: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Build a random IPT: node 0 is root (depth 0).
    Returns images [N,H,W], depths [N], parents [N] (-1 for root), edges list (parent, child).
    """
    images = [root_img.copy()]
    depths = [0]
    parents = [-1]
    edges: List[Tuple[int, int]] = []
    # Prefer chain + branches similar to paper IPT configs
    while len(images) < n_nodes:
        # pick a parent among current nodes with depth < max_depth
        candidates = [i for i, d in enumerate(depths) if d < max_depth]
        if not candidates:
            candidates = list(range(len(images)))
        parent = int(candidates[int(rng.randint(0, len(candidates)))])
        child_img, _ = apply_random_transform(images[parent], rng)
        # optionally stack a second mild transform
        if rng.rand() < 0.3:
            child_img, _ = apply_random_transform(child_img, rng)
        images.append(child_img)
        depths.append(depths[parent] + 1)
        parents.append(parent)
        edges.append((parent, len(images) - 1))
    imgs = np.stack(images, axis=0)
    return imgs, np.asarray(depths, dtype=np.int64), np.asarray(parents, dtype=np.int64), edges


def node_features(imgs: np.ndarray, pool: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel + PRNU feature matrices [N, D]. Pixel pooled to ``pool×pool`` for GNN tractability."""
    pix, prnu = [], []
    for i in range(imgs.shape[0]):
        g = imgs[i]
        g_pix = cv2.resize(g, (pool, pool), interpolation=cv2.INTER_AREA) if pool != 96 else g
        pix.append(flatten_norm(g_pix))
        # PRNU at full 96 for link prediction; pooled copy for optional concat features
        r = prnu_residual(g)
        r_pix = cv2.resize(r, (pool, pool), interpolation=cv2.INTER_AREA) if pool != 96 else r
        prnu.append(flatten_norm(r_pix))
    return np.stack(pix, 0), np.stack(prnu, 0)


def edit_aware_scalars(imgs: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Compact per-node edit features that accumulate along IPT paths (not wiped by
    per-image L2 normalize of pixels/PRNU). Returns [N, 8]:
      dist_to_root, mean|diff_to_root|, HF energy, image std,
      mean gray, PRNU energy, max|diff|, and dist*HF interaction.
    ``scale`` multiplies the vector (use ~10–20 when concatenating with unit-norm
    appearance features so ChebNet does not ignore the 8-D block).
    """
    n = imgs.shape[0]
    root = imgs[0].astype(np.float64)
    root_flat = root.reshape(-1)
    denom = float(np.sqrt(root_flat.size))
    rows = []
    for i in range(n):
        g = imgs[i].astype(np.float64)
        flat = g.reshape(-1)
        diff = flat - root_flat
        dist = float(np.linalg.norm(diff) / max(denom, 1.0))
        dist_n = min(dist / 80.0, 1.0)
        mad = float(np.mean(np.abs(diff)) / 255.0)
        mx = float(np.max(np.abs(diff)) / 255.0)
        hf = float(np.mean(np.abs(prnu_residual(imgs[i]))) / 20.0)
        hf = min(hf, 1.0)
        pr = prnu_residual(imgs[i])
        pr_e = float(np.sqrt(np.mean(pr.astype(np.float64) ** 2)) / 20.0)
        pr_e = min(pr_e, 1.0)
        std = float(g.std() / 80.0)
        mean = float(g.mean() / 255.0)
        inter = dist_n * hf
        row = np.asarray([dist_n, mad, hf, std, mean, pr_e, mx, inter], dtype=np.float32)
        rows.append(row * float(scale))
    return np.stack(rows, 0)


def adjacency_from_features(feat: np.ndarray, self_loop: bool = True) -> np.ndarray:
    """Dense similarity adjacency (Gaussian kernel on L2)."""
    # feat [N,D]
    x = feat.astype(np.float64)
    # pairwise squared distances
    xx = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(xx + xx.T - 2 * (x @ x.T), 0.0)
    # median heuristic bandwidth
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    med = max(med, 1e-6)
    a = np.exp(-d2 / med)
    np.fill_diagonal(a, 1.0 if self_loop else 0.0)
    return a.astype(np.float32)


class IPTGraphDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        n_nodes: int = 10,
        max_depth: int = 4,
        n_graphs: int = 500,
        seed: int = 42,
        feature: str = "pixel",  # pixel | prnu | concat
    ):
        self.paths = image_paths
        self.n_nodes = n_nodes
        self.max_depth = max_depth
        self.n_graphs = n_graphs
        self.feature = feature
        self.pool = 32
        self.rng = np.random.RandomState(seed)
        # pre-sample root image indices for reproducibility
        self.root_idx = self.rng.randint(0, len(image_paths), size=n_graphs)

    def __len__(self) -> int:
        return self.n_graphs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(int(self.root_idx[idx]) + idx * 9973)
        root = load_gray96(self.paths[int(self.root_idx[idx])])
        imgs, depths, parents, edges = synthesize_ipt(root, self.n_nodes, rng, self.max_depth)
        pool = int(getattr(self, "pool", 32))
        pix, prnu = node_features(imgs, pool=pool)
        # Compact edit features (scale so they dominate unit-norm appearance dims)
        edit = edit_aware_scalars(imgs, scale=20.0)
        edit_exp = np.concatenate([edit, edit * edit / 20.0], axis=1)  # [N, 16]
        # Full-res PRNU for link prediction
        prnu_full = []
        for i in range(imgs.shape[0]):
            prnu_full.append(flatten_norm(prnu_residual(imgs[i])))
        prnu_full_a = np.stack(prnu_full, 0)
        # Down-weight L2-normalized appearance so 16-D edit block drives depth logits
        pix_s = (pix * 0.05).astype(np.float32)
        prnu_s = (prnu * 0.05).astype(np.float32)
        if self.feature == "prnu":
            feat = np.concatenate([prnu_s, edit_exp], axis=1)
        elif self.feature == "edit":
            feat = edit_exp.astype(np.float32)
        elif self.feature == "concat":
            feat = np.concatenate([pix_s, prnu_s, edit_exp], axis=1)
        else:
            feat = np.concatenate([pix_s, edit_exp], axis=1)
        # Adjacency from compact edit features (depth-correlated similarity)
        adj = adjacency_from_features(edit)
        # one-hot depth labels (classes = max_depth+1)
        n_class = self.max_depth + 1
        y = np.zeros((self.n_nodes, n_class), dtype=np.float32)
        for i, d in enumerate(depths):
            y[i, min(int(d), self.max_depth)] = 1.0
        edge_t = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 2), dtype=torch.long)
        return {
            "x": torch.from_numpy(feat),
            "adj": torch.from_numpy(adj),
            "y": torch.from_numpy(y),
            "depth": torch.from_numpy(depths),
            "parents": torch.from_numpy(parents),
            "edges": edge_t,
            "prnu": torch.from_numpy(prnu_full_a),
        }


def build_dataloaders(cfg: dict):
    root = resolve_lfw_root(cfg["data"]["root"])
    paths = list_lfw_images(root)
    seed = int(cfg["train"].get("seed", 42))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    n_train = int(cfg["data"].get("n_train_graphs", 400))
    n_val = int(cfg["data"].get("n_val_graphs", 50))
    n_test = int(cfg["data"].get("n_test_graphs", 50))
    # partition identity pools to avoid root image reuse across splits
    n_img = len(paths)
    i1 = int(0.8 * n_img)
    i2 = int(0.9 * n_img)
    train_paths, val_paths, test_paths = paths[:i1], paths[i1:i2], paths[i2:]
    n_nodes = int(cfg["model"].get("n_nodes", 10))
    max_depth = int(cfg["model"].get("max_depth", 4))
    feat = cfg["model"].get("feature", "pixel")
    train_ds = IPTGraphDataset(train_paths, n_nodes, max_depth, n_train, seed, feat)
    val_ds = IPTGraphDataset(val_paths, n_nodes, max_depth, n_val, seed + 1, feat)
    test_ds = IPTGraphDataset(test_paths, n_nodes, max_depth, n_test, seed + 2, feat)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 2))

    def _collate(batch):
        # variable edge lists — keep as list; stack fixed tensors
        out = {
            "x": torch.stack([b["x"] for b in batch], 0),
            "adj": torch.stack([b["adj"] for b in batch], 0),
            "y": torch.stack([b["y"] for b in batch], 0),
            "depth": torch.stack([b["depth"] for b in batch], 0),
            "parents": torch.stack([b["parents"] for b in batch], 0),
            "prnu": torch.stack([b["prnu"] for b in batch], 0),
            "edges": [b["edges"] for b in batch],
        }
        return out

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=_collate),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=_collate),
        DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=_collate),
    )
