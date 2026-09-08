# E-Commerce Background Removal Toolkit

Professional e-commerce product image background removal toolkit with GPU acceleration and batch processing capabilities.

## Features

- **AI-Powered Segmentation**: Uses state-of-the-art BiRefNet model for precise product cutouts
- **GPU Acceleration**: Fast inference with ONNX Runtime GPU support (CPU fallback available)
- **Batch Processing**: Process single images or entire folders
- **Marketplace Presets**: Pre-configured settings for Amazon, eBay, Shopify, and more
- **Shadow Handling**: Preserve original shadows or synthesize realistic drop shadows
- **Flexible Output**: PNG (transparent), JPEG, or WebP formats with customizable backgrounds
- **CLI & UI**: Command-line interface for automation + optional Gradio web UI

## Installation

```bash
pip install -r requirements.txt
```

### GPU Requirements

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA support
- Matching CUDA/cuDNN drivers installed
- `onnxruntime-gpu` package (included in requirements.txt)

For CPU-only systems, replace `rembg[gpu]` with `rembg[cpu]` and `onnxruntime-gpu` with `onnxruntime` in requirements.txt.

**Note**: First run will download model weights (~170 MB for birefnet-general) to `~/.u2net`.

## Usage

### Command-Line Interface

#### Basic Usage

```bash
# Single image
python -m ecom_cutout.cli input.jpg output.png

# Process entire folder
python -m ecom_cutout.cli ./input_folder ./output_folder
```

#### Using Presets

```bash
# Amazon main product image (white background, 2000px, JPEG)
python -m ecom_cutout.cli product.jpg amazon_output.jpg --preset amazon-main

# Amazon alpha cutout (transparent background, PNG)
python -m ecom_cutout.cli product.png amazon_alpha.png --preset amazon-alpha

# eBay listing
python -m ecom_cutout.cli item.jpg ebay_output.jpg --preset ebay

# Shopify store (transparent with shadow)
python -m ecom_cutout.cli product.png shopify_output.png --preset shopify
```

#### Custom Configuration

```bash
# Custom white background
python -m ecom_cutout.cli input.jpg output.png --bg color --bg-color FFFFFF

# Transparent background
python -m ecom_cutout.cli input.jpg output.png --bg transparent

# Custom background image
python -m ecom_cutout.cli input.jpg output.png --bg image --bg-image background.jpg

# Add synthesized shadow
python -m ecom_cutout.cli input.jpg output.png --shadow synthesize

# Force CPU inference
python -m ecom_cutout.cli input.jpg output.png --cpu

# Verbose output
python -m ecom_cutout.cli input.jpg output.png -v
```

### Available Options

| Option | Description |
|--------|-------------|
| `--preset` | Marketplace preset: `amazon-main`, `amazon-alpha`, `ebay`, `shopify` |
| `--model` | Segmentation model: `birefnet-general` (quality) or `isnet-general-use` (fast) |
| `--bg` | Background type: `transparent`, `color`, `image` |
| `--bg-color` | Hex color code (e.g., `FFFFFF` for white) |
| `--bg-image` | Path to background image file |
| `--shadow` | Shadow mode: `none`, `preserve`, `synthesize` |
| `--format` | Output format: `png`, `jpeg`, `webp` |
| `--size` | Longest edge in pixels (e.g., `2000`) |
| `--fill` | Subject fill ratio (0.0-1.0, default 0.85) |
| `--no-square` | Disable 1:1 canvas padding |
| `--cpu` | Force CPU inference |
| `-v`, `--verbose` | Enable verbose logging |

### Python API

```python
from ecom_cutout import CutoutConfig, CutoutPipeline, BackgroundType, OutputFormat

# Create custom configuration
config = CutoutConfig(
    background=BackgroundType.COLOR,
    bg_color=(255, 255, 255),  # White background
    fmt=OutputFormat.JPEG,
    target_size=2000,
    shadow="preserve"
)

# Initialize pipeline (model loads once)
pipeline = CutoutPipeline(config)

# Process single image
result = pipeline.process("input.jpg", "output.jpg")
print(f"Success: {result.ok}, Error: {result.error}")

# Batch process
for input_path, output_path in images:
    result = pipeline.process(input_path, output_path)
```

## Marketplace Presets

| Preset | Background | Format | Size | Canvas | Shadow |
|--------|------------|--------|------|--------|--------|
| `amazon-main` | White (#FFFFFF) | JPEG | 2000px | 1:1 square | Preserve |
| `amazon-alpha` | Transparent | PNG | 2000px | Original | None |
| `ebay` | White (#FFFFFF) | JPEG | 1600px | 1:1 square | Preserve |
| `shopify` | Transparent | PNG | 2048px | Original | Synthesize |

## Project Structure

```
ecom-cutout/
├── __init__.py      # Package exports and version
├── cli.py           # Command-line interface
├── config.py        # Configuration and marketplace presets
├── engine.py        # Rembg segmentation engine
├── pipeline.py      # Image processing pipeline
├── backgrounds.py   # Background replacement utilities
├── shadows.py       # Shadow detection and synthesis
├── ui.py            # Optional Gradio web UI
└── requirements.txt # Dependencies
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, for acceleration)
- ~200 MB disk space for model weights

### Dependencies

- `rembg[gpu]` - Background removal library
- `onnxruntime-gpu` - GPU inference engine
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `Pillow` - Image I/O
- `gradio` - Web UI (optional)
- `tqdm` - Progress bars

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.
