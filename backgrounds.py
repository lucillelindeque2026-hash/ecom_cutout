"""Background compositing: transparent, solid colour, or a supplied image."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from .config import BackgroundType, CutoutConfig
except ImportError:
    from config import BackgroundType, CutoutConfig

log = logging.getLogger(__name__)
RGB = Tuple[int, int, int]


def _solid_bg(h: int, w: int, color: RGB) -> np.ndarray:
    bg = np.empty((h, w, 3), dtype=np.uint8)
    bg[:, :] = color
    return bg


def _image_bg(h: int, w: int, path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Background image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cover-fit then centre-crop to (h, w)
    scale = max(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                     interpolation=cv2.INTER_AREA)
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def composite(rgb: np.ndarray, alpha: np.ndarray, cfg: CutoutConfig,
              shadow_layer: Optional[np.ndarray] = None) -> np.ndarray:
    """Alpha-composite `rgb` (H,W,3) onto the configured background.

    Returns RGB for TRANSPARENT (caller stores alpha separately) or a finished
    RGB canvas otherwise. `shadow_layer` (H,W uint8, 255=no shadow) darkens the
    background beneath the subject when provided.
    """
    h, w = alpha.shape
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]

    if cfg.background == BackgroundType.TRANSPARENT:
        return rgb  # caller keeps RGBA

    if cfg.background == BackgroundType.IMAGE:
        if not cfg.bg_image_path:
            raise ValueError("BackgroundType.IMAGE requires cfg.bg_image_path")
        bg = _image_bg(h, w, cfg.bg_image_path).astype(np.float32)
    else:
        bg = _solid_bg(h, w, cfg.bg_color).astype(np.float32)

    if shadow_layer is not None:
        s = (shadow_layer.astype(np.float32) / 255.0)[:, :, None]
        bg = bg * s  # shadow_layer holds 1.0 where unshadowed, <1 in shadow

    out = bg * (1.0 - a) + rgb.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)