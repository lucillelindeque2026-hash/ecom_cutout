"""E-commerce background removal toolkit."""
from .config import (BackgroundType, CutoutConfig, OutputFormat, PRESETS,
                     ShadowMode, config_from_preset)
from .engine import RembgEngine, build_engine
from .pipeline import CutoutPipeline, Result

__all__ = ["CutoutConfig", "CutoutPipeline", "RembgEngine", "build_engine",
           "BackgroundType", "OutputFormat", "ShadowMode", "PRESETS",
           "config_from_preset", "Result"]
__version__ = "1.0.0"