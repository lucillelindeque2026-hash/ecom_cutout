"""Shadow handling for product photography.

Two strategies:
  * preserve_contact_shadow  - recovers the real contact shadow from the source
                               photo and re-applies it on the new background, so
                               the product stays "grounded" instead of floating.
  * synthesize_drop_shadow   - generates a clean soft shadow when the source has
                               none (studio-lit or already-white backgrounds).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

try:
    from .config import ShadowMode
except ImportError:
    from config import ShadowMode

log = logging.getLogger(__name__)


def _bbox_bottom_band(alpha: np.ndarray, band_ratio: float = 0.25) -> np.ndarray:
    """Mask that is 1 only within the lower band of the subject bbox — where a
    contact shadow physically lives. Prevents darkening the whole frame."""
    ys = np.where(alpha.max(axis=1) > 8)[0]
    xs = np.where(alpha.max(axis=0) > 8)[0]
    if len(ys) == 0 or len(xs) == 0:
        return np.ones_like(alpha, dtype=np.float32)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    band_top = int(y1 - (y1 - y0) * band_ratio)
    m = np.zeros_like(alpha, dtype=np.float32)
    m[band_top:y1 + 1, max(0, x0 - 20):min(alpha.shape[1], x1 + 20)] = 1.0
    return m


def preserve_contact_shadow(rgb: np.ndarray, alpha: np.ndarray,
                            strength: float = 0.55) -> np.ndarray:
    """Return a single-channel float32 multiplier (1.0 = keep background,
    <1.0 = darken as shadow) for compositing onto the new background."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    darkness = (255.0 - gray) / 255.0            # 1 where very dark
    soft = (alpha < 250).astype(np.float32)      # not fully opaque subject
    band = _bbox_bottom_band(alpha)

    shadow = darkness * soft * band
    # gate faint noise, scale by strength
    shadow[shadow < 0.08] = 0.0
    shadow *= strength
    k = max(3, int(min(rgb.shape[:2]) * 0.02) | 1)  # odd kernel ~2% of image
    shadow = cv2.GaussianBlur(shadow, (k, k), 0)
    shadow = np.clip(shadow, 0.0, 1.0)
    return (1.0 - shadow).astype(np.float32) * 255.0  # 255=no shadow


def synthesize_drop_shadow(alpha: np.ndarray, strength: float = 0.45,
                           offset_ratio: float = 0.015,
                           blur_ratio: float = 0.03) -> np.ndarray:
    """Generate a soft elliptical shadow from the subject silhouette."""
    h, w = alpha.shape
    base = (alpha > 128).astype(np.uint8)
    if base.sum() == 0:
        return np.full((h, w), 255.0, dtype=np.float32)

    ys = np.where(base.max(axis=1) > 0)[0]
    xs = np.where(base.max(axis=0) > 0)[0]
    cx = int((xs.min() + xs.max()) / 2)
    bottom = int(ys.max())
    rx = int((xs.max() - xs.min()) * 0.42)
    ry = max(6, int((ys.max() - ys.min()) * 0.06))
    off = int(min(h, w) * offset_ratio)

    ellipse = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse, (cx, min(h - 1, bottom + off)), (rx, ry),
                0, 0, 360, 255, -1)
    blur_k = max(5, int(min(h, w) * blur_ratio) | 1)
    shadow = cv2.GaussianBlur(ellipse, (blur_k, blur_k), 0).astype(np.float32)
    shadow = shadow / (shadow.max() + 1e-6) * strength
    return ((1.0 - shadow) * 255.0).astype(np.float32)


def build_shadow_layer(mode: ShadowMode, rgb: np.ndarray, alpha: np.ndarray,
                       strength: float) -> Optional[np.ndarray]:
    if mode == ShadowMode.NONE:
        return None
    if mode == ShadowMode.PRESERVE:
        return preserve_contact_shadow(rgb, alpha, strength)
    if mode == ShadowMode.SYNTHESIZE:
        return synthesize_drop_shadow(alpha, strength)
    return None