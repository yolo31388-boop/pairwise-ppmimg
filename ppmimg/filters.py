"""Pure-Python image filters. Every function returns a new Image."""

import math

from .image import Image, clamp

__all__ = [
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

# ITU-R BT.601 luma coefficients.
_R_COEF = 0.299
_G_COEF = 0.587
_B_COEF = 0.114


def to_grayscale(image):
    """Convert to grayscale using the BT.601 luma formula."""
    out = Image(image.width, image.height)
    out.pixels = [
        (lambda v: (v, v, v))(clamp(_R_COEF * r + _G_COEF * g + _B_COEF * b))
        for r, g, b in image.pixels
    ]
    return out


def invert(image):
    """Invert every channel (photo negative)."""
    out = Image(image.width, image.height)
    out.pixels = [(255 - r, 255 - g, 255 - b) for r, g, b in image.pixels]
    return out


def adjust_brightness(image, factor):
    """Scale every channel by factor (1.0 = unchanged)."""
    out = Image(image.width, image.height)
    out.pixels = [
        (clamp(r * factor), clamp(g * factor), clamp(b * factor))
        for r, g, b in image.pixels
    ]
    return out


def adjust_contrast(image, factor):
    """Scale the distance from mid-gray 128 by factor (1.0 = unchanged)."""
    out = Image(image.width, image.height)
    out.pixels = [
        (
            clamp((r - 128) * factor + 128),
            clamp((g - 128) * factor + 128),
            clamp((b - 128) * factor + 128),
        )
        for r, g, b in image.pixels
    ]
    return out


def _gaussian_kernel(radius):
    sigma = max(radius / 2.0, 1e-6)
    kernel = []
    for i in range(-radius, radius + 1):
        kernel.append(math.exp(-(i * i) / (2.0 * sigma * sigma)))
    total = sum(kernel)
    return [k / total for k in kernel]


def gaussian_blur(image, radius=1):
    """Gaussian blur with the given radius (1-5). Edge pixels are replicated.

    Implemented as two separable 1-D passes for speed.
    """
    radius = int(radius)
    if not 1 <= radius <= 5:
        raise ValueError("radius must be between 1 and 5")
    kernel = _gaussian_kernel(radius)
    w, h = image.width, image.height
    src = image.pixels

    # Horizontal pass (float intermediate to keep precision).
    tmp = [(0.0, 0.0, 0.0)] * (w * h)
    for y in range(h):
        row = y * w
        for x in range(w):
            ar = ag = ab = 0.0
            for k in range(-radius, radius + 1):
                xx = x + k
                if xx < 0:
                    xx = 0
                elif xx >= w:
                    xx = w - 1
                weight = kernel[k + radius]
                r, g, b = src[row + xx]
                ar += r * weight
                ag += g * weight
                ab += b * weight
            tmp[row + x] = (ar, ag, ab)

    # Vertical pass.
    out = Image(w, h)
    dst = out.pixels
    for y in range(h):
        for x in range(w):
            ar = ag = ab = 0.0
            for k in range(-radius, radius + 1):
                yy = y + k
                if yy < 0:
                    yy = 0
                elif yy >= h:
                    yy = h - 1
                weight = kernel[k + radius]
                r, g, b = tmp[yy * w + x]
                ar += r * weight
                ag += g * weight
                ab += b * weight
            dst[y * w + x] = (clamp(ar), clamp(ag), clamp(ab))
    return out


def _convolve3x3(image, kernel):
    """Generic 3x3 convolution with edge-replication boundary handling."""
    w, h = image.width, image.height
    src = image.pixels
    out = Image(w, h)
    dst = out.pixels
    for y in range(h):
        ym = y - 1 if y > 0 else 0
        yp = y + 1 if y < h - 1 else h - 1
        for x in range(w):
            xm = x - 1 if x > 0 else 0
            xp = x + 1 if x < w - 1 else w - 1
            ar = ag = ab = 0.0
            idx = 0
            for yy in (ym, y, yp):
                base = yy * w
                for xx in (xm, x, xp):
                    weight = kernel[idx]
                    idx += 1
                    if weight:
                        r, g, b = src[base + xx]
                        ar += r * weight
                        ag += g * weight
                        ab += b * weight
            dst[y * w + x] = (clamp(ar), clamp(ag), clamp(ab))
    return out


_SHARPEN_KERNEL = (0, -1, 0, -1, 5, -1, 0, -1, 0)


def sharpen(image):
    """Sharpen with a classic 3x3 kernel."""
    return _convolve3x3(image, _SHARPEN_KERNEL)


def sobel_edge(image):
    """Sobel edge detection. Returns a grayscale Image (edges are bright)."""
    gray = to_grayscale(image)
    w, h = gray.width, gray.height
    src = gray.pixels
    out = Image(w, h)
    dst = out.pixels
    for y in range(h):
        ym = y - 1 if y > 0 else 0
        yp = y + 1 if y < h - 1 else h - 1
        for x in range(w):
            xm = x - 1 if x > 0 else 0
            xp = x + 1 if x < w - 1 else w - 1
            tl = src[ym * w + xm][0]
            tc = src[ym * w + x][0]
            tr = src[ym * w + xp][0]
            ml = src[y * w + xm][0]
            mr = src[y * w + xp][0]
            bl = src[yp * w + xm][0]
            bc = src[yp * w + x][0]
            br = src[yp * w + xp][0]
            gx = -tl - 2 * ml - bl + tr + 2 * mr + br
            gy = -tl - 2 * tc - tr + bl + 2 * bc + br
            mag = clamp(math.sqrt(gx * gx + gy * gy))
            dst[y * w + x] = (mag, mag, mag)
    return out


def resize(image, new_width, new_height):
    """Resize using bilinear interpolation."""
    new_width = int(new_width)
    new_height = int(new_height)
    if new_width <= 0 or new_height <= 0:
        raise ValueError("new dimensions must be positive")
    w, h = image.width, image.height
    src = image.pixels
    out = Image(new_width, new_height)
    dst = out.pixels
    x_ratio = w / new_width
    y_ratio = h / new_height
    for y in range(new_height):
        src_y = (y + 0.5) * y_ratio - 0.5
        y0 = math.floor(src_y)
        fy = src_y - y0
        y0c = min(max(y0, 0), h - 1)
        y1c = min(max(y0 + 1, 0), h - 1)
        row0 = y0c * w
        row1 = y1c * w
        for x in range(new_width):
            src_x = (x + 0.5) * x_ratio - 0.5
            x0 = math.floor(src_x)
            fx = src_x - x0
            x0c = min(max(x0, 0), w - 1)
            x1c = min(max(x0 + 1, 0), w - 1)
            w00 = (1 - fx) * (1 - fy)
            w10 = fx * (1 - fy)
            w01 = (1 - fx) * fy
            w11 = fx * fy
            p00 = src[row0 + x0c]
            p10 = src[row0 + x1c]
            p01 = src[row1 + x0c]
            p11 = src[row1 + x1c]
            dst[y * new_width + x] = (
                clamp(p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11),
                clamp(p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11),
                clamp(p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11),
            )
    return out


def rotate(image, degrees):
    """Rotate clockwise by 90, 180 or 270 degrees."""
    w, h = image.width, image.height
    src = image.pixels
    if degrees == 90:
        out = Image(h, w)
        for y in range(w):
            for x in range(h):
                out.pixels[y * h + x] = src[(h - 1 - x) * w + y]
        return out
    if degrees == 180:
        out = Image(w, h)
        out.pixels = [src[w * h - 1 - i] for i in range(w * h)]
        return out
    if degrees == 270:
        out = Image(h, w)
        for y in range(w):
            for x in range(h):
                out.pixels[y * h + x] = src[x * w + (w - 1 - y)]
        return out
    raise ValueError("degrees must be 90, 180 or 270")
