"""Configuration dataclasses and e-commerce marketplace presets."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

RGB = Tuple[int, int, int]


class OutputFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class BackgroundType(str, Enum):
    TRANSPARENT = "transparent"
    COLOR = "color"
    IMAGE = "image"


class ShadowMode(str, Enum):
    NONE = "none"                    # hard cutout, product may "float"
    PRESERVE = "preserve"            # keep contact shadow from original
    SYNTHESIZE = "synthesize"        # generate a soft drop shadow


@dataclass
class CutoutConfig:
    # --- segmentation ---
    model: str = "birefnet-general"  # SOTA; use "isnet-general-use" for speed
    alpha_matting: bool = True
    fg_threshold: int = 240
    bg_threshold: int = 15
    erode_size: int = 11
    post_process_mask: bool = True
    gpu: bool = True

    # --- output ---
    background: BackgroundType = BackgroundType.COLOR
    bg_color: RGB = (255, 255, 255)  # pure white = Amazon compliant
    bg_image_path: Optional[str] = None
    fmt: OutputFormat = OutputFormat.PNG
    jpeg_quality: int = 95

    # --- framing ---
    auto_crop: bool = True
    fill_ratio: float = 0.85         # subject occupies 85% of canvas (Amazon)
    square_canvas: bool = True       # pad to 1:1
    target_size: Optional[int] = 2000  # longest edge px; None = keep source

    # --- shadows ---
    shadow: ShadowMode = ShadowMode.PRESERVE
    shadow_strength: float = 0.55

    # --- misc ---
    keep_exif: bool = True


@dataclass
class Preset:
    """A named bundle of overrides applied on top of CutoutConfig."""
    name: str
    overrides: dict = field(default_factory=dict)


# Marketplace requirements distilled into presets.
PRESETS: dict[str, Preset] = {
    "amazon-main": Preset(
        "amazon-main",
        dict(background=BackgroundType.COLOR, bg_color=(255, 255, 255),
             fmt=OutputFormat.JPEG, jpeg_quality=95, square_canvas=True,
             fill_ratio=0.85, target_size=2000, shadow=ShadowMode.PRESERVE),
    ),
    "amazon-alpha": Preset(
        "amazon-alpha",
        dict(background=BackgroundType.TRANSPARENT, fmt=OutputFormat.PNG,
             square_canvas=False, target_size=2000, shadow=ShadowMode.NONE),
    ),
    "ebay": Preset(
        "ebay",
        dict(background=BackgroundType.COLOR, bg_color=(255, 255, 255),
             fmt=OutputFormat.JPEG, square_canvas=True, fill_ratio=0.80,
             target_size=1600, shadow=ShadowMode.PRESERVE),
    ),
    "shopify": Preset(
        "shopify",
        dict(background=BackgroundType.TRANSPARENT, fmt=OutputFormat.PNG,
             square_canvas=False, target_size=2048, shadow=ShadowMode.SYNTHESIZE),
    ),
}


def config_from_preset(preset: str, **overrides) -> CutoutConfig:
    """Build a CutoutConfig from a named preset plus ad-hoc overrides."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Options: {list(PRESETS)}")
    base = dict(PRESETS[preset].overrides)
    base.update(overrides)
    return CutoutConfig(**base)