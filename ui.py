"""Optional Gradio UI for interactive review before committing to a batch."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import (BackgroundType, CutoutConfig, OutputFormat, PRESETS,
                     ShadowMode, config_from_preset)
from .pipeline import CutoutPipeline

_pipe_cache: dict[str, CutoutPipeline] = {}


def _pipeline(cfg: CutoutConfig) -> CutoutPipeline:
    key = f"{cfg.model}|{cfg.gpu}"
    if key not in _pipe_cache:
        _pipe_cache[key] = CutoutPipeline(cfg)
    return _pipe_cache[key]


def run(pil_img: Image.Image, preset: str, model: str, bg_mode: str,
    bg_hex: str, shadow_mode: str, fill: float, size: int
    ) -> tuple[Image.Image, Image.Image]:
    if pil_img is None:
        raise ValueError("Upload an image first.")
    cfg = config_from_preset(preset, model=model) if preset in PRESETS \
        else CutoutConfig(model=model)
    cfg.background = BackgroundType.TRANSPARENT if bg_mode == "transparent" \
        else BackgroundType.COLOR
    if bg_mode == "color":
        h = bg_hex.lstrip("#")
        cfg.bg_color = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    cfg.shadow = ShadowMode(shadow_mode)
    cfg.fill_ratio = fill
    cfg.target_size = int(size) if size else None

    import tempfile, os
    from pathlib import Path
    tmp_in = Path(tempfile.gettempdir()) / "_ui_in.png"
    tmp_out = Path(tempfile.gettempdir()) / ("_ui_out.png" if cfg.background ==
                BackgroundType.TRANSPARENT else "_ui_out.jpg")
    tmp_silhouette = Path(tempfile.gettempdir()) / "_ui_silhouette.png"
    cfg.fmt = OutputFormat.PNG if cfg.background == BackgroundType.TRANSPARENT \
        else OutputFormat.JPEG
    pil_img.save(tmp_in)
    res = _pipeline(cfg).process_pair(tmp_in, tmp_out, tmp_silhouette)
    if not res.ok:
        raise RuntimeError(res.error)
    return Image.open(res.path), Image.open(tmp_silhouette)


def launch(share: bool = False, port: int = 7860) -> None:
    import gradio as gr
    with gr.Blocks(title="Ecom Cutout Studio") as demo:
        gr.Markdown("# E-commerce Background Remover\nGPU · BiRefNet/IS-Net · "
                    "white-bg & shadow aware.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Source", height=420)
            out = gr.Image(type="pil", label="Result", height=420)
            silhouette = gr.Image(type="pil", label="Black silhouette", height=420)
        with gr.Row():
            preset = gr.Dropdown(list(PRESETS), value="amazon-main", label="Preset")
            model = gr.Dropdown(["birefnet-general", "isnet-general-use", "u2net"],
                                value="birefnet-general", label="Model")
            bg_mode = gr.Radio(["color", "transparent"], value="color", label="Background")
            bg_hex = gr.Textbox(value="#FFFFFF", label="Background colour")
            shadow_mode = gr.Radio([s.value for s in ShadowMode],
                                   value="preserve", label="Shadow")
            fill = gr.Slider(0.5, 0.95, value=0.85, step=0.01, label="Fill ratio")
            size = gr.Slider(800, 3000, value=2000, step=100, label="Longest edge")
        btn = gr.Button("Remove background", variant="primary")
        btn.click(run, [inp, preset, model, bg_mode, bg_hex, shadow_mode, fill, size],
              [out, silhouette])
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    launch()