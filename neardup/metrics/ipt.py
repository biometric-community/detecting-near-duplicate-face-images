"""PRNU-based IPT link prediction (Linkprediction.m logic, simplified)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def predict_parents_from_depth_prnu(
    depth: np.ndarray,
    prnu: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Assign each non-root node a parent among shallower nodes with nearest PRNU.
    Root = argmin depth (break ties by min mean PRNU distance to others).
    """
    n = len(depth)
    # resolve multi-root: keep one with depth==min
    dmin = int(depth.min())
    roots = np.where(depth == dmin)[0]
    if len(roots) > 1:
        # choose root with largest mean distance (more "original" noise) — match script sort descend
        dist = _seuclidean(prnu)
        scores = []
        for r in roots:
            others = [x for x in roots if x != r]
            scores.append(float(np.mean(dist[r, others])) if others else 0.0)
        root = int(roots[int(np.argmax(scores))])
        # bump other roots to depth+1 conceptually
        depth = depth.copy()
        for r in roots:
            if r != root:
                depth[r] = dmin + 1
    else:
        root = int(roots[0])

    dist = _seuclidean(prnu)
    edges: List[Tuple[int, int]] = []
    for child in range(n):
        if child == root:
            continue
        # parents must be strictly shallower
        cands = [i for i in range(n) if depth[i] < depth[child] and i != child]
        if not cands:
            cands = [root]
        parent = min(cands, key=lambda i: dist[child, i])
        edges.append((parent, child))
    return edges


def _seuclidean(x: np.ndarray) -> np.ndarray:
    # x [N,D] row-normalized preferred
    v = x.astype(np.float64)
    # standardize columns
    std = v.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    v = (v - v.mean(axis=0, keepdims=True)) / std
    xx = np.sum(v * v, axis=1, keepdims=True)
    d2 = np.maximum(xx + xx.T - 2 * (v @ v.T), 0.0)
    return np.sqrt(d2)


def ipt_metrics(
    pred_edges: List[Tuple[int, int]],
    gt_edges: List[Tuple[int, int]],
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
) -> dict:
    # root id
    pred_root = int(np.argmin(pred_depth))
    gt_root = int(np.argmin(gt_depth))
    root_ok = float(pred_root == gt_root)
    gt_set = {(int(a), int(b)) for a, b in gt_edges}
    pred_set = {(int(a), int(b)) for a, b in pred_edges}
    if not gt_set:
        recon = 1.0 if not pred_set else 0.0
    else:
        recon = len(gt_set & pred_set) / len(gt_set)
    return {"root_acc": root_ok, "ipt_recon": recon}


@torch.no_grad()
def batch_link_metrics(depth_pred: torch.Tensor, prnu: torch.Tensor, parents_gt: torch.Tensor) -> dict:
    """
    depth_pred: [B,N] int, prnu [B,N,D], parents_gt [B,N] with -1 root.
    """
    b, n = depth_pred.shape
    root_sum, recon_sum = 0.0, 0.0
    for i in range(b):
        d = depth_pred[i].cpu().numpy()
        p = prnu[i].cpu().numpy()
        parents = parents_gt[i].cpu().numpy()
        gt_edges = [(int(parents[c]), c) for c in range(n) if parents[c] >= 0]
        pred_edges = predict_parents_from_depth_prnu(d, p)
        m = ipt_metrics(pred_edges, gt_edges, d, parents * 0 + (parents >= 0))  # depth from gt via parents
        # fix gt depth from parents
        gt_depth = np.zeros(n, dtype=np.int64)
        # BFS depths from root
        root = int(np.where(parents < 0)[0][0])
        gt_depth[:] = -1
        gt_depth[root] = 0
        changed = True
        while changed:
            changed = False
            for c in range(n):
                if parents[c] >= 0 and gt_depth[parents[c]] >= 0 and gt_depth[c] < 0:
                    gt_depth[c] = gt_depth[parents[c]] + 1
                    changed = True
        m = ipt_metrics(pred_edges, gt_edges, d, gt_depth)
        root_sum += m["root_acc"]
        recon_sum += m["ipt_recon"]
    return {"root_acc": root_sum / b, "ipt_recon": recon_sum / b}
