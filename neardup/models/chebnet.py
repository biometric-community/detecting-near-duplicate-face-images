"""PyTorch ChebNet (Kipf/Defferrard-style) for IPT depth prediction."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops. adj: [B,N,N] or [N,N]."""
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).expand(b, -1, -1)
    a = adj + eye
    deg = a.sum(-1).clamp_min(1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * a * d_inv_sqrt.unsqueeze(-2)


def chebyshev_supports(adj: torch.Tensor, k: int) -> List[torch.Tensor]:
    """
    Compute Chebyshev polynomials T_0..T_k of scaled Laplacian.
    Returns list of [B,N,N] tensors length k+1.
    """
    a_norm = normalize_adj(adj)  # ~ I - L_sym for GCN; for Cheby use Laplacian
    # Laplacian L = I - A_norm
    b, n, _ = a_norm.shape
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).expand(b, -1, -1)
    L = eye - a_norm
    # Scale to [-1,1]: L_hat = 2L/lambda_max - I ; approx lambda_max=2
    L_hat = L - eye
    supports = [eye, L_hat]
    for _ in range(2, k + 1):
        # T_k = 2 L_hat T_{k-1} - T_{k-2}
        Tk = 2 * L_hat.bmm(supports[-1]) - supports[-2]
        supports.append(Tk)
    return supports[: k + 1]


class ChebConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 3):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(k + 1, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        # x: [B,N,F]
        outs = []
        for i in range(self.k + 1):
            # supports[i] @ x  -> [B,N,F]
            sx = supports[i].bmm(x)
            outs.append(sx.matmul(self.weight[i]))
        return sum(outs) + self.bias


class ChebNet(nn.Module):
    """Two-layer ChebNet matching upstream gcn_cheby defaults (hidden=16)."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 16, k: int = 3, dropout: float = 0.5):
        super().__init__()
        self.k = k
        self.dropout = dropout
        self.conv1 = ChebConv(in_dim, hidden, k)
        self.conv2 = ChebConv(hidden, n_classes, k)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        supports = chebyshev_supports(adj, self.k)
        h = F.relu(self.conv1(x, supports))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, supports)


def build_model(cfg: dict, in_dim: int) -> ChebNet:
    m = cfg["model"]
    return ChebNet(
        in_dim=in_dim,
        n_classes=int(m.get("max_depth", 4)) + 1,
        hidden=int(m.get("hidden", 16)),
        k=int(m.get("max_degree", 3)),
        dropout=float(m.get("dropout", 0.5)),
    )
