"""PRNU / sensor-noise residual proxy (replaces MATLAB Enhanced PRNU)."""

from __future__ import annotations

import cv2
import numpy as np


def prnu_residual(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    High-pass residual approximating SPN/PRNU for link prediction.

    Paper uses Li Enhanced PRNU (MATLAB mex). We use image − Gaussian
    denoised estimate as a portable proxy (documented as D1).
    """
    g = gray.astype(np.float32)
    k = max(3, int(2 * round(3 * sigma) + 1) | 1)
    den = cv2.GaussianBlur(g, (k, k), sigmaX=sigma)
    return (g - den).astype(np.float32)


def flatten_norm(feat: np.ndarray) -> np.ndarray:
    v = feat.reshape(-1).astype(np.float32)
    mn, mx = float(v.min()), float(v.max())
    if mx - mn < 1e-8:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)
