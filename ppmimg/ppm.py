"""Read and write PPM images (P3 ASCII and P6 binary), using only the
Python standard library.

Only 8-bit-per-sample images (maxval == 255) are written, but files with
other maxvals are read correctly by rescaling to 0-255.
"""

from .image import Image


class PPMError(ValueError):
    """Raised when a PPM file is malformed."""


def _skip_ws_comments(data, pos):
    """Advance past PPM whitespace and ``# ...`` comments."""
    n = len(data)
    while pos < n:
        ch = data[pos]
        if ch in b" \t\n\r\x0b\x0c":
            pos += 1
        elif ch == 0x23:  # '#'
            while pos < n and data[pos] not in b"\n\r":
                pos += 1
        else:
            break
    return pos


def _read_token(data, pos):
    """Read one whitespace-delimited ASCII token from *data*."""
    pos = _skip_ws_comments(data, pos)
    start = pos
    n = len(data)
    while pos < n and data[pos] not in b" \t\n\r\x0b\x0c":
        pos += 1
    if pos == start:
        raise PPMError("unexpected end of header data")
    return data[start:pos], pos


def _read_header(data):
    """Parse magic/width/height/maxval, returning ``(magic, w, h, maxval, pos)``.

    After a P6 header exactly one single whitespace character separates
    maxval from the raw raster, as required by the PPM specification.
    """
    if len(data) < 2 or data[0] != 0x50:  # 'P'
        raise PPMError("not a PPM file (bad magic number)")
    magic = data[1]
    if magic not in (0x33, 0x36):
        raise PPMError("unsupported PPM magic 'P%c' (only P3 and P6)" % chr(magic))
    pos = 2
    width_tok, pos = _read_token(data, pos)
    height_tok, pos = _read_token(data, pos)
    maxval_tok, pos = _read_token(data, pos)
    width = int(width_tok)
    height = int(height_tok)
    maxval = int(maxval_tok)
    if width <= 0 or height <= 0:
        raise PPMError("image width and height must be positive")
    if not 1 <= maxval <= 65535:
        raise PPMError("maxval must be between 1 and 65535")
    if magic == 0x36:
        if pos >= len(data):
            raise PPMError("P6 file is truncated (missing raster separator)")
        if data[pos] not in b" \t\n\r\x0b\x0c":
            raise PPMError("malformed P6 header")
        pos += 1
    return magic, width, height, maxval, pos


def _rescale(value, maxval):
    scaled = value * 255 // maxval
    return max(0, min(255, scaled))


def _pixels_from_p6(data, pos, width, height, maxval):
    channels = width * height * 3
    if maxval < 256:
        expected = channels
        raw = data[pos:pos + expected]
        if len(raw) != expected:
            raise PPMError(
                "P6 raster is %d bytes but %d were expected" % (len(raw), expected)
            )
        samples = raw if maxval == 255 else bytes(_rescale(b, maxval) for b in raw)
    else:
        expected = channels * 2
        raw = data[pos:pos + expected]
        if len(raw) != expected:
            raise PPMError(
                "P6 raster is %d bytes but %d were expected" % (len(raw), expected)
            )
        samples = bytes(
            _rescale((raw[i] << 8) | raw[i + 1], maxval)
            for i in range(0, expected, 2)
        )
    return [
        (samples[i], samples[i + 1], samples[i + 2])
        for i in range(0, len(samples), 3)
    ]


def loads(data):
    """Parse PPM bytes into an :class:`~ppmimg.image.Image`."""
    if isinstance(data, str):
        raise TypeError("PPM data must be bytes, not str")
    magic, width, height, maxval, pos = _read_header(data)
    if magic == 0x33:
        tokens = data[pos:].split()
        expected = width * height * 3
        if len(tokens) < expected:
            raise PPMError(
                "P3 raster contains %d samples, expected %d" % (len(tokens), expected)
            )
        try:
            samples = [_rescale(int(tok), maxval) for tok in tokens[:expected]]
        except ValueError:
            raise PPMError("P3 raster contains a non-numeric sample")
        pixels = [
            (samples[i], samples[i + 1], samples[i + 2])
            for i in range(0, expected, 3)
        ]
    else:
        pixels = _pixels_from_p6(data, pos, width, height, maxval)
    return Image(width, height, pixels)


def load(path):
    """Read a PPM file (P3 or P6) from *path*."""
    with open(path, "rb") as fh:
        return loads(fh.read())


def _save_p3(image):
    lines = ["P3", "%d %d" % (image.width, image.height), "255"]
    row = []
    width = image.width
    for idx, (r, g, b) in enumerate(image.pixels):
        row.append("%d %d %d" % (r, g, b))
        if len(row) == width:
            lines.append(" ".join(row))
            row = []
    return ("\n".join(lines) + "\n").encode("ascii")


def _save_p6(image):
    header = ("P6\n%d %d\n255\n" % (image.width, image.height)).encode("ascii")
    pix = image.pixels
    raster = bytearray(len(pix) * 3)
    flat = memoryview(raster)
    i = 0
    for r, g, b in pix:
        flat[i] = r
        flat[i + 1] = g
        flat[i + 2] = b
        i += 3
    return header + bytes(raster)


def dumps(image, fmt="P6"):
    """Serialize *image* to PPM bytes in the requested format.

    *fmt* must be ``"P3"`` (ASCII) or ``"P6"`` (binary).
    """
    if fmt == "P3":
        return _save_p3(image)
    if fmt == "P6":
        return _save_p6(image)
    raise ValueError("unsupported PPM format %r (use 'P3' or 'P6')" % fmt)


def save(image, path, fmt=None):
    """Write *image* to *path*.

    The format is inferred from *fmt* when given, otherwise from the file
    extension (``.ppm``/``.pnm`` default to P6).
    """
    if fmt is None:
        lowered = path.lower()
        if lowered.endswith(".p3") or lowered.endswith(".ascii"):
            fmt = "P3"
        elif lowered.endswith(".p6") or lowered.endswith(".ppm") or lowered.endswith(".pnm"):
            fmt = "P6"
        else:
            raise ValueError("cannot infer PPM format from path %r" % path)
    with open(path, "wb") as fh:
        fh.write(dumps(image, fmt))