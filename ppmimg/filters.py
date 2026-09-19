"""Image filters for ppmimg.

Every filter returns a brand new :class:`~ppmimg.image.Image`; the input
image is never modified. Spatial filters (blur, sharpen, sobel) handle
borders by *edge replication* (clamping out-of-range coordinates to the
nearest valid pixel).
"""

import math

from .image import Image


def _split_channels(image):
    """Return three flat ``bytearray`` channel buffers (R, G, B)."""
    red = bytearray(image.width * image.height)
    green = bytearray(len(red))
    blue = bytearray(len(red))
    i = 0
    for r, g, b in image.pixels:
        red[i] = r
        green[i] = g
        blue[i] = b
        i += 1
    return red, green, blue


def _merge_channels(width, height, channels):
    total = width * height
    red, green, blue = channels
    return Image(
        width,
        height,
        [(red[i], green[i], blue[i]) for i in range(total)],
    )


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clamped_index_map(radius, limit):
    """For each kernel offset in ``-radius..radius``, the clamped source
    coordinate for every index along one axis (``0..limit-1``)."""
    return [
        [_clamp(i + off, 0, limit - 1) for i in range(limit)]
        for off in range(-radius, radius + 1)
    ]


def to_grayscale(image):
    """Convert an RGB image to grayscale (R = G = B for every pixel).

    Uses the Rec. 601 luma weights: ``0.299 R + 0.587 G + 0.114 B``.
    """
    out = bytearray(image.width * image.height)
    i = 0
    for r, g, b in image.pixels:
        out[i] = round(0.299 * r + 0.587 * g + 0.114 * b)
        i += 1
    return _merge_channels(image.width, image.height, (out, bytearray(out), bytearray(out)))


def invert(image):
    """Photographic negative: ``255 - value`` per channel."""
    red, green, blue = _split_channels(image)
    for i in range(len(red)):
        red[i] = 255 - red[i]
        green[i] = 255 - green[i]
        blue[i] = 255 - blue[i]
    return _merge_channels(image.width, image.height, (red, green, blue))


def adjust_brightness(image, factor):
    """Multiply every channel by *factor* (1.0 unchanged, <1 darker, >1 brighter)."""
    red, green, blue = _split_channels(image)
    for i in range(len(red)):
        red[i] = _clamp(round(red[i] * factor), 0, 255)
        green[i] = _clamp(round(green[i] * factor), 0, 255)
        blue[i] = _clamp(round(blue[i] * factor), 0, 255)
    return _merge_channels(image.width, image.height, (red, green, blue))


def adjust_contrast(image, factor):
    """Scale pixel differences from mid-gray (128) by *factor*.

    ``0.0`` flattens everything to gray, ``1.0`` is unchanged.
    """
    red, green, blue = _split_channels(image)
    for i in range(len(red)):
        red[i] = _clamp(round(128 + (red[i] - 128) * factor), 0, 255)
        green[i] = _clamp(round(128 + (green[i] - 128) * factor), 0, 255)
        blue[i] = _clamp(round(128 + (blue[i] - 128) * factor), 0, 255)
    return _merge_channels(image.width, image.height, (red, green, blue))


def _gaussian_weights(radius, sigma=None):
    """Normalized integer Gaussian weights in fixed point (scale 1024)."""
    if sigma is None:
        sigma = max(radius / 2.0, 0.5)
    scale = 1024
    raw = [
        math.exp(-(off * off) / (2.0 * sigma * sigma))
        for off in range(-radius, radius + 1)
    ]
    total = sum(raw)
    weights = [round(w * scale / total) for w in raw]
    weights[radius] += scale - sum(weights)  # absorb rounding error at center
    return weights


def gaussian_blur(image, radius=3):
    """Gaussian blur with kernel radius 1-5.

    Implemented as two separable 1-D convolutions (horizontal then vertical)
    in fixed-point integer arithmetic. Borders use edge-replicated pixels.
    """
    radius = int(radius)
    if not 1 <= radius <= 5:
        raise ValueError("radius must be between 1 and 5")
    width, height = image.width, image.height
    channels = _split_channels(image)
    weights = _gaussian_weights(radius)
    scale = 1024
    offsets = range(-radius, radius + 1)
    xmap = _clamped_index_map(radius, width)
    ymap = _clamped_index_map(radius, height)
    out_channels = []

    for src in channels:
        # Weighted sums exceed 255, so both intermediate buffers are int lists.
        temp = [0] * (width * height)
        dst = bytearray(width * height)

        # Horizontal pass, divide by scale once to stay in byte value range.
        for y in range(height):
            row_start = y * width
            for x in range(width):
                acc = 0
                for k, dx in enumerate(offsets):
                    acc += weights[k] * src[row_start + xmap[k][x]]
                temp[row_start + x] = acc // scale

        # Vertical pass.
        for y in range(height):
            dst_start = y * width
            for x in range(width):
                acc = 0
                for k, dy in enumerate(offsets):
                    acc += weights[k] * temp[ymap[k][y] * width + x]
                dst[dst_start + x] = _clamp(acc // scale, 0, 255)

        out_channels.append(dst)
    return _merge_channels(width, height, out_channels)


# Kernel for sharpen(): center boosted by 1, each 4-neighbor by -1.
_SHARPEN = {
    (-1, -1): 0, (0, -1): -1, (1, -1): 0,
    (-1, 0): -1, (0, 0): 5, (1, 0): -1,
    (-1, 1): 0, (0, 1): -1, (1, 1): 0,
}


def _convolve_3x3(channels, width, height, kernel):
    """Apply a 3x3 kernel with edge replication.

    Returns Python-int lists so signed convolution values (e.g. Sobel)
    are preserved without clamping.
    """
    out_channels = []
    for src in channels:
        dst = [0] * (width * height)
        for y in range(height):
            y0 = y - 1 if y > 0 else 0
            y2 = y + 1 if y + 1 < height else y
            r0, r1, r2 = y0 * width, y * width, y2 * width
            for x in range(width):
                x0 = x - 1 if x > 0 else 0
                x2 = x + 1 if x + 1 < width else x
                dst[r1 + x] = (
                    kernel[(-1, -1)] * src[r0 + x0]
                    + kernel[(0, -1)] * src[r0 + x]
                    + kernel[(1, -1)] * src[r0 + x2]
                    + kernel[(-1, 0)] * src[r1 + x0]
                    + kernel[(0, 0)] * src[r1 + x]
                    + kernel[(1, 0)] * src[r1 + x2]
                    + kernel[(-1, 1)] * src[r2 + x0]
                    + kernel[(0, 1)] * src[r2 + x]
                    + kernel[(1, 1)] * src[r2 + x2]
                )
        out_channels.append(dst)
    return out_channels


def sharpen(image):
    """Sharpen with a 5-center / -1-neighbor kernel (edge-replicated borders)."""
    width, height = image.width, image.height
    channels = _split_channels(image)
    out = [
        bytearray(_clamp(v, 0, 255) for v in ch)
        for ch in _convolve_3x3(channels, width, height, _SHARPEN)
    ]
    return _merge_channels(width, height, out)


_SOBEL_X = {
    (-1, -1): -1, (0, -1): 0, (1, -1): 1,
    (-1, 0): -2, (0, 0): 0, (1, 0): 2,
    (-1, 1): -1, (0, 1): 0, (1, 1): 1,
}
_SOBEL_Y = {
    (-1, -1): -1, (0, -1): -2, (1, -1): -1,
    (-1, 0): 0, (0, 0): 0, (1, 0): 0,
    (-1, 1): 1, (0, 1): 2, (1, 1): 1,
}


def sobel_edge(image):
    """Sobel edge detection; returns a grayscale image with bright edges.

    The image is first converted to Rec. 601 luma; each output value is
    ``clamp(sqrt(gx**2 + gy**2), 0, 255)``. Flat regions are 0.
    """
    gray, _, _ = _split_channels(to_grayscale(image))
    width, height = image.width, image.height
    (gx,) = _convolve_3x3((gray,), width, height, _SOBEL_X)
    (gy,) = _convolve_3x3((gray,), width, height, _SOBEL_Y)
    out = bytearray(width * height)
    for i in range(len(out)):
        out[i] = int(_clamp(round(math.hypot(gx[i], gy[i])), 0, 255))
    return _merge_channels(width, height, (out, bytearray(out), bytearray(out)))


def resize(image, width, height):
    """Resize to ``width x height`` using bilinear interpolation.

    Source samples map via ``sx = dx * (src_w - 1) / (dst_w - 1)`` (and
    likewise for y), so corner pixels map exactly to corners.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    src_w, src_h = image.width, image.height
    red, green, blue = _split_channels(image)

    if width == 1:
        x_params = [(0, 0, 0.0, 1.0)]
    else:
        x_params = []
        for dx in range(width):
            sx = dx * (src_w - 1) / (width - 1)
            x0 = int(math.floor(sx))
            x1 = min(x0 + 1, src_w - 1)
            fx = sx - x0
            x_params.append((x0, x1, 1.0 - fx, fx))

    if height == 1:
        y_params = [(0, 0, 0.0, 1.0)]
    else:
        y_params = []
        for dy in range(height):
            sy = dy * (src_h - 1) / (height - 1)
            y0 = int(math.floor(sy))
            y1 = min(y0 + 1, src_h - 1)
            fy = sy - y0
            y_params.append((y0, y1, 1.0 - fy, fy))

    def resample(src):
        dst = bytearray(width * height)
        for dy, (y0, y1, w0y, w1y) in enumerate(y_params):
            r0, r1 = y0 * src_w, y1 * src_w
            dst_start = dy * width
            for dx, (x0, x1, w0x, w1x) in enumerate(x_params):
                top = src[r0 + x0] * w0x + src[r0 + x1] * w1x
                bottom = src[r1 + x0] * w0x + src[r1 + x1] * w1x
                dst[dst_start + dx] = _clamp(round(top * w0y + bottom * w1y), 0, 255)
        return dst

    return _merge_channels(
        width,
        height,
        (resample(red), resample(green), resample(blue)),
    )


def rotate(image, degrees):
    """Rotate in 90-degree increments: 90, 180 or 270 (clockwise)."""
    degrees = int(degrees) % 360
    if degrees not in (90, 180, 270):
        raise ValueError("rotate supports 90, 180 or 270 degrees")
    src_w, src_h = image.width, image.height
    src = image.pixels

    if degrees == 90:
        dst_w, dst_h = src_h, src_w
        pixels = [
            src[(src_h - 1 - dx) * src_w + dy]
            for dy in range(dst_h)
            for dx in range(dst_w)
        ]
    elif degrees == 180:
        dst_w, dst_h = src_w, src_h
        pixels = [
            src[(src_h - 1 - dy) * src_w + (src_w - 1 - dx)]
            for dy in range(dst_h)
            for dx in range(dst_w)
        ]
    else:  # 270
        dst_w, dst_h = src_h, src_w
        pixels = [
            src[dx * src_w + (src_w - 1 - dy)]
            for dy in range(dst_h)
            for dx in range(dst_w)
        ]
    return Image(dst_w, dst_h, pixels)
