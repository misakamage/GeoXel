"""Public GeoXel inference API.

The neural backbone remains importable as :mod:`streamvggt`; this package adds
path-independent map loading and a small command-line entry point.
"""

__version__ = "0.1.0"

from .inference import GeoXelMap, load_checkpoint, load_image_sequence

__all__ = ["GeoXelMap", "load_checkpoint", "load_image_sequence", "__version__"]
