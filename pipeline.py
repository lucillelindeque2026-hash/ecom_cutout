"""End-to-end cutout pipeline: load → segment → composite → frame → save."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from .backgrounds import composite
    from .config import BackgroundType, CutoutConfig, OutputFormat
    from .engine import build_engine, refine_mask, SegmentationEngine
    from .shadows import build_shadow_layer
except ImportError:
    from backgrounds import composite
    from config import BackgroundType, CutoutConfig, OutputFormat
    from engine import build_engine, refine_mask, SegmentationEngine
    from shadows import build_shadow_layer

log = logging.getLogger(__name__)


@dataclass
class Result:
    path: Path
    width: int
    height: int
    ok: bool
    error: Optional[str] = None


def _load_rgb(path: Path) -> Tuple[np.ndarray, Optional[Image.Image]]:
    pil = Image.open(path)
    exif = pil.info.get("exif")
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    rgb = np.array(pil)
    return rgb, (pil if exif else None)


def _auto_crop(rgba_layers: Tuple[np.ndarray, np.ndarray],
               pad_ratio: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    rgb, alpha = rgba_layers
    ys = np.where(alpha.max(axis=1) > 4)[0]
    xs = np.where(alpha.max(axis=0) > 4)[0]
    if len(ys) == 0 or len(xs) == 0:
        return rgb, alpha
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph = int((y1 - y0) * pad_ratio)
    pw = int((x1 - x0) * pad_ratio)
    h, w = alpha.shape
    y0, y1 = max(0, y0 - ph), min(h, y1 + ph + 1)
    x0, x1 = max(0, x0 - pw), min(w, x1 + pw + 1)
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def _frame(rgb: np.ndarray, alpha: np.ndarray, cfg: CutoutConfig
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply fill_ratio scaling, optional square canvas, and target size."""
    if cfg.auto_crop:
        rgb, alpha = _auto_crop((rgb, alpha))

    h, w = alpha.shape
    if cfg.target_size:
        scale = cfg.target_size / max(h, w)
        if scale < 1.0 or cfg.square_canvas:  # downscale large, keep squares
            nh, nw = int(round(h * scale)), int(round(w * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
            rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
            alpha = cv2.resize(alpha, (nw, nh), interpolation=interp)
            h, w = nh, nw

    if cfg.square_canvas or cfg.fill_ratio:
        # canvas so the subject occupies ~fill_ratio of the longest edge
        longest = max(h, w)
        canvas_side = int(round(longest / max(cfg.fill_ratio, 0.05)))
        if cfg.square_canvas:
            cw = ch = canvas_side
        else:
            cw = max(w, int(round(w / cfg.fill_ratio)))
            ch = max(h, int(round(h / cfg.fill_ratio)))
        new_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
        new_alpha = np.zeros((ch, cw), dtype=np.uint8)
        # place colour/transparent background default (composite fills later)
        new_rgb[:, :] = cfg.bg_color if cfg.background != BackgroundType.TRANSPARENT else 0
        y = (ch - h) // 2
        x = (cw - w) // 2
        new_rgb[y:y + h, x:x + w] = rgb
        new_alpha[y:y + h, x:x + w] = alpha
        rgb, alpha = new_rgb, new_alpha

    return rgb, alpha


class CutoutPipeline:
    """Reusable object; construct once and call `process` per image (the model
    session is loaded a single time — critical for batch throughput)."""

    def __init__(self, cfg: CutoutConfig,
                 engine: Optional[SegmentationEngine] = None) -> None:
        self.cfg = cfg
        self.engine = engine or build_engine(cfg)

    def process(self, src: Path | str, dst: Path | str) -> Result:
        return self._process(src, dst, None)

    def process_pair(self, src: Path | str, dst: Path | str,
                     silhouette_dst: Path | str) -> Result:
        """Save the normal cutout and a transparent black silhouette."""
        return self._process(src, dst, Path(silhouette_dst))

    def _process(self, src: Path | str, dst: Path | str,
                 silhouette_dst: Optional[Path]) -> Result:
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if silhouette_dst is not None:
            silhouette_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            rgb, exif_pil = _load_rgb(src)
            alpha = self.engine.segment(rgb)
            alpha = refine_mask(alpha, feather_px=1.0)

            rgb, alpha = _frame(rgb, alpha, self.cfg)
            shadow = build_shadow_layer(self.cfg.shadow, rgb, alpha,
                                        self.cfg.shadow_strength)

            transparent = self.cfg.background == BackgroundType.TRANSPARENT
            out_rgb = composite(rgb, alpha, self.cfg, shadow_layer=shadow)

            self._save(out_rgb, alpha if transparent else None, dst, exif_pil)
            if silhouette_dst is not None:
                silhouette_rgb = np.zeros_like(rgb)
                self._save(silhouette_rgb, alpha, silhouette_dst, None)
            h, w = out_rgb.shape[:2]
            return Result(dst, w, h, ok=True)
        except Exception as exc:  # keep batch resilient
            log.exception("Failed on %s", src)
            return Result(dst, 0, 0, ok=False, error=str(exc))

    def _save(self, rgb: np.ndarray, alpha: Optional[np.ndarray],
              dst: Path, exif_pil: Optional[Image.Image]) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        fmt = self.cfg.fmt

        if alpha is not None:  # PNG/WEBP with transparency
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = alpha
            pil = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))
        else:
            if fmt == OutputFormat.PNG and self.cfg.background == BackgroundType.TRANSPARENT:
                raise ValueError("Transparent background cannot be saved as opaque.")
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        save_kwargs = {}
        if self.cfg.keep_exif and exif_pil is not None:
            exif = exif_pil.info.get("exif")
            if exif:
                save_kwargs["exif"] = exif

        suffix = dst.suffix.lower().lstrip(".")
        if fmt == OutputFormat.JPEG or suffix in ("jpg", "jpeg"):
            if pil.mode in ("RGBA", "LA"):
                pil = pil.convert("RGB")
            pil.save(dst.with_suffix(".jpg"), format="JPEG",
                     quality=self.cfg.jpeg_quality, optimize=True, **save_kwargs)
        elif fmt == OutputFormat.WEBP or suffix == "webp":
            pil.save(dst.with_suffix(".webp"), format="WEBP",
                     quality=95, method=6, **save_kwargs)
        else:
            pil.save(dst.with_suffix(".png"), format="PNG", optimize=True,
                     **save_kwargs)