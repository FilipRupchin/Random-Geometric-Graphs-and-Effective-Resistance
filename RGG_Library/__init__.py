# Package public surface for RGG_Library

"""
RGG_Library — lightweight graph visualization utilities.
"""

# explicit version (update on release)
__version__ = "0.1.0"

from .graph_build import RGGBuilder
from .graph_viz import RGGVisualizer

__all__ = ["RGGBuilder", "RGGVisualizer", "__version__"]