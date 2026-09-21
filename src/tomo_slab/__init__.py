"""Command-line tool for slab-boundary segmentation of tomograms."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tomo-slab")
except PackageNotFoundError:
    __version__ = "uninstalled"
