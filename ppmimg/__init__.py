"""ppmimg - a tiny pure-stdlib PPM image processing library."""

from .image import Image
from .io import read_ppm, write_ppm, load, save, loads_ppm, dumps_ppm
from .filters import (
    to_grayscale,
    invert,
    adjust_brightness,
    adjust_contrast,
    gaussian_blur,
    sharpen,
    sobel_edge,
    resize,
    rotate,
)

__version__ = "0.1.0"

__all__ = [
    "Image",
    "read_ppm",
    "write_ppm",
    "load",
    "save",
    "loads_ppm",
    "dumps_ppm",
    "to_grayscale",
    "invert",
    "adjust_brightness",
    "adjust_contrast",
    "gaussian_blur",
    "sharpen",
    "sobel_edge",
    "resize",
    "rotate",
]
