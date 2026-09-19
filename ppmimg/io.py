"""Reading and writing PPM images (P3 plain text and P6 binary)."""

from .image import Image, clamp

__all__ = ["read_ppm", "write_ppm", "load", "save"]


def _next_token(data, pos):
    """Read one whitespace-delimited token from bytes, skipping comments.

    Returns (token_bytes, position_after_token). '#' starts a comment that
    runs to the end of the line.
    """
    n = len(data)
    while pos < n:
        c = data[pos]
        if c == 35:  # '#'
            while pos < n and data[pos] not in (10, 13):
                pos += 1
        elif chr(c).isspace():
            pos += 1
        else:
            break
    start = pos
    while pos < n and not chr(data[pos]).isspace() and data[pos] != 35:
        pos += 1
    if start == pos:
        raise ValueError("unexpected end of PPM header")
    return data[start:pos], pos


def read_ppm(path):
    """Read a P3 or P6 PPM file and return an Image."""
    with open(path, "rb") as fh:
        data = fh.read()
    return loads_ppm(data)


def loads_ppm(data):
    """Parse PPM bytes (P3 or P6) and return an Image."""
    magic, pos = _next_token(data, 0)
    if magic not in (b"P3", b"P6"):
        raise ValueError("not a PPM file (magic: %r)" % magic)
    width_tok, pos = _next_token(data, pos)
    height_tok, pos = _next_token(data, pos)
    maxval_tok, pos = _next_token(data, pos)
    width = int(width_tok)
    height = int(height_tok)
    maxval = int(maxval_tok)
    if width <= 0 or height <= 0:
        raise ValueError("invalid PPM dimensions %dx%d" % (width, height))
    if not (0 < maxval < 65536):
        raise ValueError("invalid PPM maxval %d" % maxval)

    count = width * height * 3
    if magic == b"P3":
        values = []
        for _ in range(count):
            tok, pos = _next_token(data, pos)
            values.append(int(tok))
    else:
        # Exactly one whitespace character separates the header from the data.
        if pos < len(data) and chr(data[pos]).isspace():
            pos += 1
        raw = data[pos:]
        if maxval < 256:
            if len(raw) < count:
                raise ValueError("truncated P6 pixel data")
            values = list(raw[:count])
        else:
            if len(raw) < count * 2:
                raise ValueError("truncated P6 pixel data")
            values = [
                (raw[i] << 8) | raw[i + 1] for i in range(0, count * 2, 2)
            ]

    if maxval != 255:
        values = [clamp(v * 255.0 / maxval) for v in values]

    pixels = [
        (values[i], values[i + 1], values[i + 2])
        for i in range(0, count, 3)
    ]
    return Image(width, height, pixels)


def write_ppm(image, path, binary=True):
    """Write an Image to path. binary=True -> P6, binary=False -> P3."""
    with open(path, "wb") as fh:
        fh.write(dumps_ppm(image, binary=binary))


def dumps_ppm(image, binary=True):
    """Serialize an Image to PPM bytes (P6 if binary else P3)."""
    header = b"P6" if binary else b"P3"
    lines = [header,
             ("%d %d" % (image.width, image.height)).encode("ascii"),
             b"255"]
    if binary:
        body = bytearray()
        for r, g, b in image.pixels:
            body += bytes((r, g, b))
        return b"\n".join(lines) + b"\n" + bytes(body)
    rows = []
    w = image.width
    for y in range(image.height):
        parts = []
        for r, g, b in image.pixels[y * w:(y + 1) * w]:
            parts.append("%d %d %d" % (r, g, b))
        rows.append("  ".join(parts).encode("ascii"))
    return b"\n".join(lines + rows) + b"\n"


# Friendly aliases
load = read_ppm
save = write_ppm
