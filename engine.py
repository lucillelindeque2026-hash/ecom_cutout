"""Segmentation engines. Primary backend is rembg (ONNX Runtime, GPU-capable),
which exposes U2-Net, IS-Net and the BiRefNet family. The engine returns a
single-channel uint8 alpha matte (0=background, 255=foreground)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SegmentationEngine(ABC):
    @abstractmethod
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """Return an alpha matte (H, W) uint8 for an RGB image (H, W, 3)."""


class RembgEngine(SegmentationEngine):
    """Wraps rembg with a cached session so batch runs reuse the model."""

    def __init__(self, model: str = "birefnet-general", alpha_matting: bool = True,
                 fg_threshold: int = 240, bg_threshold: int = 15,
                 erode_size: int = 11, post_process_mask: bool = True,
                 gpu: bool = True) -> None:
        from rembg import new_session  # deferred: heavy import
        try:
            self._session = new_session(model)
        except ValueError as exc:
            raise ValueError(
                f"Unknown model '{model}'. Try 'birefnet-general', "
                f"'isnet-general-use', 'u2net'."
            ) from exc
        self._alpha_matting = alpha_matting
        self._fg = fg_threshold
        self._bg = bg_threshold
        self._erode = erode_size
        self._post = post_process_mask
        self._gpu = gpu
        log.info("RembgEngine ready: model=%s alpha_matting=%s gpu=%s",
                 model, alpha_matting, gpu)

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        from rembg import remove
        # rembg operates on BGR uint8 arrays / PIL; we pass an ndarray.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cut = remove(
            bgr,
            session=self._session,
            alpha_matting=self._alpha_matting,
            alpha_matting_foreground_threshold=self._fg,
            alpha_matting_background_threshold=self._bg,
            alpha_matting_erode_size=self._erode,
            post_process_mask=self._post,
        )
        # `remove` returns BGR(A); we only need the alpha channel.
        if cut.ndim == 3 and cut.shape[2] == 4:
            alpha = cut[:, :, 3]
        else:  # model returned a mask only
            alpha = cut if cut.ndim == 2 else cut[:, :, 0]
        return np.ascontiguousarray(alpha, dtype=np.uint8)


def refine_mask(alpha: np.ndarray, feather_px: float = 1.0,
                denoise_px: int = 0) -> np.ndarray:
    """Optional post-refinement: light feathering and speckle removal.
    Keeps soft edges (hair, glass, contact shadow) intact by default."""
    out = alpha
    if denoise_px > 0:
        k = denoise_px * 2 + 1
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if feather_px > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=feather_px, sigmaY=feather_px)
    return out


def build_engine(cfg) -> SegmentationEngine:
    return RembgEngine(
        model=cfg.model, alpha_matting=cfg.alpha_matting,
        fg_threshold=cfg.fg_threshold, bg_threshold=cfg.bg_threshold,
        erode_size=cfg.erode_size, post_process_mask=cfg.post_process_mask,
        gpu=cfg.gpu,
    )