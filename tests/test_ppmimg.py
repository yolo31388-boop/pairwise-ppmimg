"""Tests for the ppmimg library, mirroring the acceptance scenarios."""

import time

import pytest

from ppmimg import (
    Image,
    read_ppm,
    write_ppm,
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


def make_4x4():
    """A 4x4 image with known, distinct pixel values."""
    pixels = []
    for y in range(4):
        for x in range(4):
            pixels.append(((x * 60 + y) % 256, (y * 60 + x) % 256,
                           (x * 37 + y * 91) % 256))
    return Image(4, 4, pixels)


# 1. PPM round-trip -------------------------------------------------------

def test_p3_roundtrip(tmp_path):
    img = make_4x4()
    path = tmp_path / "test_p3.ppm"
    write_ppm(img, str(path), binary=False)
    back = read_ppm(str(path))
    assert (back.width, back.height) == (4, 4)
    assert back.pixels == img.pixels


def test_p6_roundtrip(tmp_path):
    img = make_4x4()
    path = tmp_path / "test_p6.ppm"
    write_ppm(img, str(path), binary=True)
    back = read_ppm(str(path))
    assert (back.width, back.height) == (4, 4)
    assert back.pixels == img.pixels


def test_p3_p6_cross_conversion_lossless(tmp_path):
    img = make_4x4()
    p3 = tmp_path / "a.ppm"
    p6 = tmp_path / "b.ppm"
    write_ppm(img, str(p3), binary=False)
    write_ppm(img, str(p6), binary=True)
    assert read_ppm(str(p3)).pixels == read_ppm(str(p6)).pixels == img.pixels


def test_ppm_header_with_comments(tmp_path):
    content = b"P6\n# a comment\n4 4\n# another\n255\n" + bytes(range(48))
    path = tmp_path / "commented.ppm"
    path.write_bytes(content)
    img = read_ppm(str(path))
    assert (img.width, img.height) == (4, 4)
    assert img.get_pixel(0, 0) == (0, 1, 2)
    assert img.get_pixel(3, 3) == (45, 46, 47)


def test_get_set_pixel():
    img = Image(3, 2)
    img.set_pixel(2, 1, (10, 20, 30))
    assert img.get_pixel(2, 1) == (10, 20, 30)
    with pytest.raises(IndexError):
        img.get_pixel(3, 0)
    with pytest.raises(IndexError):
        img.set_pixel(0, 2, (0, 0, 0))


# 2. Grayscale ------------------------------------------------------------

def test_grayscale_pure_red():
    img = Image(1, 1, [(255, 0, 0)])
    gray = to_grayscale(img)
    expected = round(0.299 * 255)  # 76
    assert gray.get_pixel(0, 0) == (expected, expected, expected)


def test_grayscale_formula_and_channel_equality():
    img = Image(2, 1, [(10, 200, 90), (255, 255, 255)])
    gray = to_grayscale(img)
    for (r, g, b), (gr, gg, gb) in zip(img.pixels, gray.pixels):
        assert gr == gg == gb
        assert gr == round(0.299 * r + 0.587 * g + 0.114 * b)
    assert img.get_pixel(0, 0) == (10, 200, 90)  # original untouched


# 3. Invert ---------------------------------------------------------------

def test_invert():
    img = Image(2, 1, [(0, 255, 128), (255, 0, 1)])
    inv = invert(img)
    assert inv.get_pixel(0, 0) == (255, 0, 127)
    assert inv.get_pixel(1, 0) == (0, 255, 254)
    assert invert(inv) == img


# Brightness / contrast ----------------------------------------------------

def test_adjust_brightness():
    img = Image(2, 1, [(100, 200, 255), (50, 50, 50)])
    bright = adjust_brightness(img, 1.5)
    assert bright.get_pixel(0, 0) == (150, 255, 255)  # clamped at 255
    dark = adjust_brightness(img, 0.5)
    assert dark.get_pixel(1, 0) == (25, 25, 25)
    assert adjust_brightness(img, 1.0) == img


def test_adjust_contrast():
    img = Image(2, 1, [(128, 128, 128), (200, 60, 128)])
    up = adjust_contrast(img, 2.0)
    assert up.get_pixel(0, 0) == (128, 128, 128)  # mid-gray fixed point
    assert up.get_pixel(1, 0) == (255, 0, 128)    # clamped
    assert adjust_contrast(img, 1.0) == img


# 4. Gaussian blur ---------------------------------------------------------

def test_blur_solid_color_unchanged():
    img = Image(6, 6, [(30, 120, 200)] * 36)
    for radius in (1, 3, 5):
        blurred = gaussian_blur(img, radius)
        assert blurred.pixels == img.pixels


def test_blur_boundary_produces_middle_values():
    # Top half black, bottom half white (horizontal edge between rows 2/3).
    img = Image(6, 6, [(0, 0, 0)] * 18 + [(255, 255, 255)] * 18)
    blurred = gaussian_blur(img, 1)
    above = blurred.get_pixel(2, 2)[0]   # last black row, near the edge
    below = blurred.get_pixel(2, 3)[0]   # first white row, near the edge
    assert 0 < above < 255
    assert 0 < below < 255
    # Symmetric halves: the two edge rows should sum to ~255.
    assert abs(above + below - 255) <= 2
    # Far from the edge the color is preserved.
    assert blurred.get_pixel(0, 0)[0] == 0
    assert blurred.get_pixel(5, 5)[0] == 255


def test_blur_tiny_image_no_crash():
    img = Image(1, 1, [(7, 8, 9)])
    assert gaussian_blur(img, 5).get_pixel(0, 0) == (7, 8, 9)


def test_blur_invalid_radius():
    with pytest.raises(ValueError):
        gaussian_blur(Image(2, 2), 0)
    with pytest.raises(ValueError):
        gaussian_blur(Image(2, 2), 6)


# Sharpen ------------------------------------------------------------------

def test_sharpen_solid_unchanged():
    img = Image(5, 5, [(80, 160, 240)] * 25)
    assert sharpen(img).pixels == img.pixels


def test_sharpen_enhances_contrast():
    img = Image(3, 3, [(100, 100, 100)] * 9)
    img.set_pixel(1, 1, (200, 200, 200))
    out = sharpen(img)
    assert out.get_pixel(1, 1)[0] == 255  # 5*200 - 4*100 = 600, clamped
    assert out.get_pixel(1, 0)[0] < 100   # neighbor pulled down


# 5. Sobel -----------------------------------------------------------------

def test_sobel_solid_near_zero():
    img = Image(8, 8, [(123, 45, 67)] * 64)
    edges = sobel_edge(img)
    assert max(p[0] for p in edges.pixels) == 0
    for r, g, b in edges.pixels:
        assert r == g == b


def test_sobel_vertical_edge_high():
    # Left half black, right half white.
    pixels = []
    for y in range(8):
        for x in range(8):
            pixels.append((0, 0, 0) if x < 4 else (255, 255, 255))
    img = Image(8, 8, pixels)
    edges = sobel_edge(img)
    # Interior of the flat regions is zero.
    assert edges.get_pixel(1, 4)[0] == 0
    assert edges.get_pixel(6, 4)[0] == 0
    # The columns straddling the edge light up strongly.
    assert edges.get_pixel(3, 4)[0] > 200
    assert edges.get_pixel(4, 4)[0] > 200


# 6. Resize ----------------------------------------------------------------

def test_resize_2x2_to_4x4_bilinear():
    # R channel values 0, 100 / 200, 255; G=B=0 for easy math.
    img = Image(2, 2, [(0, 0, 0), (100, 0, 0), (200, 0, 0), (255, 0, 0)])
    out = resize(img, 4, 4)
    assert (out.width, out.height) == (4, 4)
    # Hand-computed bilinear: src = (i + 0.5) * 0.5 - 0.5, edges clamped.
    assert [out.get_pixel(x, 0)[0] for x in range(4)] == [0, 25, 75, 100]
    assert [out.get_pixel(x, 1)[0] for x in range(4)] == [50, 72, 117, 139]
    assert [out.get_pixel(x, 2)[0] for x in range(4)] == [150, 167, 200, 216]
    assert [out.get_pixel(x, 3)[0] for x in range(4)] == [200, 214, 241, 255]


def test_resize_downscale_and_identity():
    img = make_4x4()
    same = resize(img, 4, 4)
    assert same.pixels == img.pixels
    small = resize(img, 2, 2)
    assert (small.width, small.height) == (2, 2)


# Rotate -------------------------------------------------------------------

def test_rotate_90():
    img = Image(2, 3, [(i, i, i) for i in range(6)])
    out = rotate(img, 90)
    assert (out.width, out.height) == (3, 2)
    # Clockwise: dst(x, y) = src(y, h-1-x)
    assert out.get_pixel(0, 0) == (4, 4, 4)  # src(0, 2)
    assert out.get_pixel(2, 1) == (1, 1, 1)  # src(1, 0)


def test_rotate_180_and_270():
    img = Image(2, 3, [(i, i, i) for i in range(6)])
    out180 = rotate(img, 180)
    assert (out180.width, out180.height) == (2, 3)
    assert out180.pixels == list(reversed(img.pixels))
    assert rotate(rotate(img, 90), 270) == img
    assert rotate(rotate(img, 90), 90) == out180
    with pytest.raises(ValueError):
        rotate(img, 45)


# 7. Performance -----------------------------------------------------------

def test_performance_800x600():
    pixels = [((i * 7) % 256, (i * 13) % 256, (i * 29) % 256)
              for i in range(800 * 600)]
    img = Image(800, 600, pixels)

    start = time.perf_counter()
    blurred = gaussian_blur(img, 3)
    blur_time = time.perf_counter() - start
    assert (blurred.width, blurred.height) == (800, 600)
    assert blur_time < 10, "gaussian_blur took %.2fs" % blur_time

    start = time.perf_counter()
    resized = resize(img, 400, 300)
    resize_time = time.perf_counter() - start
    assert (resized.width, resized.height) == (400, 300)
    assert resize_time < 10, "resize took %.2fs" % resize_time
