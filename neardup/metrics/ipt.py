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
    # Root = argmin predicted depth. If all depths identical (collapsed classifier),
    # do not credit root_acc=1 from argmin always picking index 0 — unless the
    # single class is depth 0 and the chosen root index equals the GT root.
    uniq = np.unique(pred_depth)
    gt_root = int(np.argmin(gt_depth))
    if len(uniq) == 1:
        if int(uniq[0]) == 0:
            pred_root = int(np.argmin(pred_depth))  # all equal → index 0
            root_ok = float(pred_root == gt_root)
        else:
            root_ok = 0.0
    else:
        dmin = int(pred_depth.min())
        cand = np.where(pred_depth == dmin)[0]
        pred_root = int(cand[0]) if len(cand) == 1 else int(np.argmin(pred_depth))
        # multi-candidate at min depth: prefer node already chosen by link pred (first)
        root_ok = float(pred_root == gt_root)
    gt_set = {(int(a), int(b)) for a, b in gt_edges}
    pred_set = {(int(a), int(b)) for a, b in pred_edges}
    if not gt_set:
        recon = 1.0 if not pred_set else 0.0
    else:
        recon = len(gt_set & pred_set) / len(gt_set)
    return {"root_acc": root_ok, "ipt_recon": recon, "collapsed": float(len(uniq) == 1)}


@torch.no_grad()
def batch_link_metrics(depth_pred: torch.Tensor, prnu: torch.Tensor, parents_gt: torch.Tensor) -> dict:
    """
    depth_pred: [B,N] int, prnu [B,N,D], parents_gt [B,N] with -1 root.
    Root selection uses PRNU tie-break when multiple nodes share min depth
    (via ``predict_parents_from_depth_prnu``); collapsed uniform depths → root_acc=0.
    """
    b, n = depth_pred.shape
    root_sum, recon_sum, collapsed_sum = 0.0, 0.0, 0.0
    for i in range(b):
        d = depth_pred[i].cpu().numpy().astype(np.int64)
        p = prnu[i].cpu().numpy()
        parents = parents_gt[i].cpu().numpy()
        gt_edges = [(int(parents[c]), c) for c in range(n) if parents[c] >= 0]
        pred_edges = predict_parents_from_depth_prnu(d.copy(), p)
        # BFS depths from GT root
        gt_depth = np.full(n, -1, dtype=np.int64)
        root = int(np.where(parents < 0)[0][0])
        gt_depth[root] = 0
        changed = True
        while changed:
            changed = False
            for c in range(n):
                if parents[c] >= 0 and gt_depth[parents[c]] >= 0 and gt_depth[c] < 0:
                    gt_depth[c] = gt_depth[parents[c]] + 1
                    changed = True
        # Predicted root from link predictor (PRNU-aware among min-depth candidates).
        # Collapsed uniform depths → root_acc=0 unless single class is 0 and
        # the PRNU-chosen root equals GT root.
        if len(np.unique(d)) == 1:
            collapsed_sum += 1.0
            if int(np.unique(d)[0]) == 0:
                dist = _seuclidean(p)
                scores = []
                for r in range(n):
                    others = [x for x in range(n) if x != r]
                    scores.append(float(np.mean(dist[r, others])) if others else 0.0)
                pred_root = int(np.argmax(scores))
                root_ok = float(pred_root == root)
            else:
                root_ok = 0.0
            recon = 0.0
            gt_set = {(int(a), int(b)) for a, b in gt_edges}
            pred_set = {(int(a), int(b)) for a, b in pred_edges}
            if gt_set:
                recon = len(gt_set & pred_set) / len(gt_set)
            root_sum += root_ok
            recon_sum += recon
            continue
        # Non-collapsed: root = unique min-depth node after PRNU multi-root resolve
        d_for_root = d.copy()
        dmin = int(d_for_root.min())
        roots = np.where(d_for_root == dmin)[0]
        if len(roots) > 1:
            dist = _seuclidean(p)
            scores = []
            for r in roots:
                others = [x for x in roots if x != r]
                scores.append(float(np.mean(dist[r, others])) if others else 0.0)
            pred_root = int(roots[int(np.argmax(scores))])
        else:
            pred_root = int(roots[0])
        root_sum += float(pred_root == root)
        gt_set = {(int(a), int(b)) for a, b in gt_edges}
        pred_set = {(int(a), int(b)) for a, b in pred_edges}
        recon_sum += (len(gt_set & pred_set) / len(gt_set)) if gt_set else (1.0 if not pred_set else 0.0)
    return {
        "root_acc": root_sum / b,
        "ipt_recon": recon_sum / b,
        "collapsed_frac": collapsed_sum / b,
    }
