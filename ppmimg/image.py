"""Pure-stdlib RGB image type used by the ppmimg package."""


class Image:
    """A simple in-memory RGB image.

    Pixels are stored row-major as ``(r, g, b)`` integer tuples, each
    component in the range 0-255.
    """

    def __init__(self, width, height, pixels=None):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self._width = int(width)
        self._height = int(height)
        total = self._width * self._height
        if pixels is None:
            self._pixels = [(0, 0, 0)] * total
        else:
            if len(pixels) != total:
                raise ValueError(
                    "pixel count %d does not match width*height (%d)"
                    % (len(pixels), total)
                )
            self._pixels = []
            for value in pixels:
                r, g, b = value
                for component in (r, g, b):
                    if not 0 <= int(component) <= 255:
                        raise ValueError("pixel components must be within 0-255")
                self._pixels.append((int(r), int(g), int(b)))

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def pixels(self):
        """All pixels as a flat row-major list of ``(r, g, b)`` tuples."""
        return list(self._pixels)

    def get_pixel(self, x, y):
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError("pixel coordinate (%d, %d) out of range" % (x, y))
        return self._pixels[y * self._width + x]

    def set_pixel(self, x, y, value):
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError("pixel coordinate (%d, %d) out of range" % (x, y))
        r, g, b = int(value[0]), int(value[1]), int(value[2])
        for component in (r, g, b):
            if not 0 <= component <= 255:
                raise ValueError("pixel components must be within 0-255")
        self._pixels[y * self._width + x] = (r, g, b)

    def copy(self):
        return Image(self._width, self._height, self._pixels)

    def __eq__(self, other):
        if not isinstance(other, Image):
            return NotImplemented
        return (
            self._width == other._width
            and self._height == other._height
            and self._pixels == other._pixels
        )

    def __repr__(self):
        return "Image(width=%d, height=%d)" % (self._width, self._height)