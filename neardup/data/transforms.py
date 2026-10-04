"""Photometric / geometric transforms (paper Table 1) for IPT synthesis."""

from __future__ import annotations

from typing import Callable, List, Tuple

import cv2
import numpy as np


def brightness(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    a = float(rng.uniform(0.9, 1.5))
    b = float(rng.uniform(-30, 30))
    out = img.astype(np.float32) * a + b
    return np.clip(out, 0, 255).astype(np.float32)


def median_filter(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    k = int(rng.randint(2, 7))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img.astype(np.uint8), k).astype(np.float32)


def gaussian_smooth(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    sigma = float(rng.uniform(1.0, 3.0))
    k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma)


def gamma_transform(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    g = float(rng.uniform(0.5, 1.5))
    x = np.clip(img.astype(np.float32) / 255.0, 0, 1)
    return np.clip((x**g) * 255.0, 0, 255).astype(np.float32)


def translate(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    tx = float(rng.uniform(5, 20)) * (1 if rng.rand() < 0.5 else -1)
    ty = float(rng.uniform(5, 20)) * (1 if rng.rand() < 0.5 else -1)
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img.astype(np.float32), m, (w, h), borderMode=cv2.BORDER_REFLECT)


def scale_transform(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    s = float(rng.uniform(0.9, 1.1))
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    resized = cv2.resize(img.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img, dtype=np.float32)
    y0 = max(0, (h - nh) // 2)
    x0 = max(0, (w - nw) // 2)
    y1 = min(h, y0 + nh)
    x1 = min(w, x0 + nw)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return canvas


TRANSFORM_BANK: List[Tuple[str, Callable]] = [
    ("brightness", brightness),
    ("median", median_filter),
    ("gaussian", gaussian_smooth),
    ("gamma", gamma_transform),
    ("translate", translate),
    ("scale", scale_transform),
]


def apply_random_transform(img: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, str]:
    name, fn = TRANSFORM_BANK[int(rng.randint(0, len(TRANSFORM_BANK)))]
    return fn(img, rng), name
