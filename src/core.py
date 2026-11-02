"""Noise simulation, classical filters, local features, and full-reference metrics."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def noisy_image(clean: np.ndarray, sigma: float, image_id: str, seed: int = 302) -> np.ndarray:
    tag = hashlib.sha256(f"{seed}:{image_id}:{sigma}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(tag[:8], "little"))
    return np.clip(clean.astype(np.float32) + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)


def filters(noisy: np.ndarray, nlm_h: float, sigma: float) -> dict[str, np.ndarray]:
    return {
        "noisy": noisy,
        "gaussian": cv2.GaussianBlur(noisy, (5, 5), 1.1),
        "bilateral": cv2.bilateralFilter(noisy, 7, 2.5 * sigma, 3),
        "nlm": cv2.fastNlMeansDenoising(noisy, None, float(nlm_h), 7, 21),
    }


def features(outputs: dict[str, np.ndarray], sigma: float, include_texture: bool = True) -> np.ndarray:
    n = outputs["noisy"].astype(np.float32)
    g = outputs["gaussian"].astype(np.float32)
    b = outputs["bilateral"].astype(np.float32)
    m = outputs["nlm"].astype(np.float32)
    base = [n / 255, g / 255, b / 255, m / 255, (n - m) / 50, (b - m) / 50,
            np.full_like(n, sigma / 50)]
    if include_texture:
        local_mean = cv2.GaussianBlur(n, (5, 5), 0)
        local_sq = cv2.GaussianBlur(n * n, (5, 5), 0)
        std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0))
        gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
        base.extend([local_mean / 255, std / 50, edge / 400])
    return np.stack(base, axis=-1).reshape(-1, len(base)).astype(np.float32)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2)
    return float("inf") if error == 0 else float(10 * np.log10(255**2 / error))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """11x11 Gaussian SSIM, sigma=1.5, C1=(.01L)^2, C2=(.03L)^2."""
    a = reference.astype(np.float64)
    b = estimate.astype(np.float64)
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    aa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    bb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    quality = ((2 * mu_a * mu_b + c1) * (2 * ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (aa + bb + c2))
    return float(quality[5:-5, 5:-5].mean())
