"""Core Image class: width, height and RGB pixel data."""

__all__ = ["Image", "clamp"]


def clamp(value):
    """Clamp a numeric value to the 0-255 byte range, rounding to int."""
    value = int(round(value))
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


class Image:
    """An RGB image stored as a flat list of (r, g, b) tuples, row-major."""

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width, height, pixels=None):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        if pixels is None:
            self.pixels = [(0, 0, 0)] * (self.width * self.height)
        else:
            if len(pixels) != self.width * self.height:
                raise ValueError(
                    "pixel data length %d does not match %dx%d"
                    % (len(pixels), self.width, self.height)
                )
            self.pixels = [
                (clamp(r), clamp(g), clamp(b)) for r, g, b in pixels
            ]

    def get_pixel(self, x, y):
        """Return the (r, g, b) tuple at column x, row y."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("pixel coordinate (%d, %d) out of bounds" % (x, y))
        return self.pixels[y * self.width + x]

    def set_pixel(self, x, y, color):
        """Set the pixel at column x, row y to the (r, g, b) tuple color."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("pixel coordinate (%d, %d) out of bounds" % (x, y))
        r, g, b = color
        self.pixels[y * self.width + x] = (clamp(r), clamp(g), clamp(b))

    def copy(self):
        """Return an independent copy of this image."""
        return Image(self.width, self.height, list(self.pixels))

    def __eq__(self, other):
        return (
            isinstance(other, Image)
            and self.width == other.width
            and self.height == other.height
            and self.pixels == other.pixels
        )

    def __repr__(self):
        return "Image(width=%d, height=%d)" % (self.width, self.height)
