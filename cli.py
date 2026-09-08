"""Command-line interface: single image or whole-folder batch processing."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from .config import (BackgroundType, CutoutConfig, OutputFormat, PRESETS,
                     ShadowMode, config_from_preset)
from .pipeline import CutoutPipeline, Result

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
log = logging.getLogger("ecom_cutout")


def _gather_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input not found: {input_path}")


def _out_path(src: Path, in_root: Path, out_root: Path, ext: str) -> Path:
    rel = src.relative_to(in_root) if src.is_relative_to(in_root) else src.name
    return (out_root / rel).with_suffix(ext)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecom-cutout",
        description="Professional e-commerce background removal (GPU, batch).")
    p.add_argument("input", type=Path, help="image file or folder to process")
    p.add_argument("output", type=Path, help="output file or folder")
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="marketplace preset (amazon-main, ebay, shopify, ...)")
    p.add_argument("--model", default=None,
                   help="birefnet-general (quality) | isnet-general-use (fast)")
    p.add_argument("--bg", choices=[b.value for b in BackgroundType], default=None)
    p.add_argument("--bg-color", default=None, help="hex, e.g. FFFFFF")
    p.add_argument("--bg-image", default=None, help="path to background photo")
    p.add_argument("--shadow", choices=[s.value for s in ShadowMode], default=None)
    p.add_argument("--format", choices=[f.value for f in OutputFormat], default=None)
    p.add_argument("--size", type=int, default=None, help="longest edge px")
    p.add_argument("--fill", type=float, default=None, help="subject fill ratio")
    p.add_argument("--no-square", action="store_true", help="disable 1:1 canvas")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _config_from_args(a: argparse.Namespace) -> CutoutConfig:
    overrides: dict = {}
    if a.model: overrides["model"] = a.model
    if a.bg: overrides["background"] = BackgroundType(a.bg)
    if a.bg_color:
        h = a.bg_color.lstrip("#")
        overrides["bg_color"] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    if a.bg_image:
        overrides["bg_image_path"] = a.bg_image
        overrides["background"] = BackgroundType.IMAGE
    if a.shadow: overrides["shadow"] = ShadowMode(a.shadow)
    if a.format: overrides["fmt"] = OutputFormat(a.format)
    if a.size: overrides["target_size"] = a.size
    if a.fill: overrides["fill_ratio"] = a.fill
    if a.no_square: overrides["square_canvas"] = False
    if a.cpu: overrides["gpu"] = False

    cfg = config_from_preset(a.preset, **overrides) if a.preset \
        else CutoutConfig(**overrides)
    return cfg


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if a.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        cfg = _config_from_args(a)
        inputs = _gather_inputs(a.input)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print("No images found.", file=sys.stderr)
        return 1

    pipeline = CutoutPipeline(cfg)          # model loads once
    ext = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[cfg.fmt.value]
    in_root = a.input if a.input.is_dir() else a.input.parent
    out_root = a.output if a.output.is_dir() or len(inputs) > 1 else a.output.parent

    results: list[Result] = []
    for src in tqdm(inputs, desc="Cutout", unit="img"):
        dst = out_root / src.name if len(inputs) == 1 and a.output.suffix \
            else _out_path(src, in_root, out_root, ext)
        results.append(pipeline.process(src, dst))

    ok = sum(r.ok for r in results)
    failed = [r for r in results if not r.ok]
    print(f"\nDone: {ok}/{len(results)} succeeded.")
    for r in failed:
        print(f"  ✗ {r.path}: {r.error}", file=sys.stderr)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())