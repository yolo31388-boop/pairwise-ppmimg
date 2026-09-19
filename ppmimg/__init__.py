"""ppmimg - a tiny pure-stdlib PPM image processing library.

Example::

    from ppmimg import Image, ppm, filters

    img = ppm.load("input.ppm")
    gray = filters.to_grayscale(img)
    ppm.save(gray, "output.ppm", fmt="P3")

Filters are also available as methods on :class:`Image`, e.g.
``img.to_grayscale()``.
"""

from . import filters, ppm
from .image import Image
from .ppm import PPMError, load, loads, save, dumps
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


def _bind_methods():
    """Attach filter functions to Image as non-mutating convenience methods."""

    def _factory(func, **fixed):
        def method(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        method.__name__ = func.__name__
        method.__doc__ = func.__doc__
        return method

    for name in (
        "to_grayscale",
        "invert",
        "adjust_brightness",
        "adjust_contrast",
        "gaussian_blur",
        "sharpen",
        "sobel_edge",
        "resize",
        "rotate",
    ):
        func = getattr(filters, name)
        setattr(Image, name, _factory(func))

    def save_method(self, path, fmt=None):
        return ppm.save(self, path, fmt)

    def dumps_method(self, fmt="P6"):
        return ppm.dumps(self, fmt)

    save_method.__name__ = "save"
    dumps_method.__name__ = "dumps"
    Image.save = save_method
    Image.dumps = dumps_method

    @classmethod
    def load(cls, path):
        return ppm.load(path)

    @classmethod
    def loads(cls, data):
        return ppm.loads(data)

    Image.load = load
    Image.loads = loads


_bind_methods()

__all__ = [
    "Image",
    "ppm",
    "filters",
    "PPMError",
    "load",
    "loads",
    "save",
    "dumps",
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