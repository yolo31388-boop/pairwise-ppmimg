import os

import pytest

from ppmimg import Image, filters, load, loads, save, PPMError
from ppmimg.ppm import dumps


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_4x4():
    pixels = [
        (0, 0, 0),       (10, 20, 30),    (200, 100, 50),  (255, 255, 255),
        (1, 2, 3),       (40, 80, 120),   (160, 90, 20),   (10, 10, 10),
        (255, 0, 0),     (0, 255, 0),     (0, 0, 255),     (255, 255, 0),
        (128, 128, 128), (30, 60, 90),    (180, 70, 210),  (0, 0, 0),
    ]
    return Image(4, 4, pixels)


@pytest.fixture
def img():
    return make_4x4()


# ---------------------------------------------------------------------------
# Image basics
# ---------------------------------------------------------------------------


def test_image_dimensions_and_pixels():
    img = make_4x4()
    assert img.width == 4
    assert img.height == 4
    assert img.get_pixel(0, 0) == (0, 0, 0)
    assert img.get_pixel(2, 0) == (200, 100, 50)
    assert img.get_pixel(0, 2) == (255, 0, 0)
    assert img.get_pixel(3, 3) == (0, 0, 0)


def test_set_pixel():
    img = make_4x4()
    img.set_pixel(1, 1, (7, 8, 9))
    assert img.get_pixel(1, 1) == (7, 8, 9)


def test_out_of_range_coordinates():
    img = make_4x4()
    with pytest.raises(IndexError):
        img.get_pixel(4, 0)
    with pytest.raises(IndexError):
        img.get_pixel(0, 4)
    with pytest.raises(IndexError):
        img.set_pixel(-1, 0, (1, 2, 3))


def test_constructor_validates_pixel_count():
    with pytest.raises(ValueError):
        Image(2, 2, [(0, 0, 0)])


# ---------------------------------------------------------------------------
# PPM P3 / P6 round trips
# ---------------------------------------------------------------------------


P3_SAMPLE = b"""\
P3
# a comment
4 4
255
0 0 0    10 20 30   200 100 50  255 255 255
1 2 3    40 80 120  160 90 20   10 10 10
255 0 0  0 255 0    0 0 255     255 255 0
128 128 128  30 60 90  180 70 210  0 0 0
"""


def test_load_p3_text_format():
    img = loads(P3_SAMPLE)
    assert img.width == 4
    assert img.height == 4
    assert img.get_pixel(0, 0) == (0, 0, 0)
    assert img.get_pixel(2, 0) == (200, 100, 50)
    assert img.get_pixel(0, 2) == (255, 0, 0)
    assert img.get_pixel(3, 3) == (0, 0, 0)
    assert img == make_4x4()


def test_p6_roundtrip_preserves_pixels(img):
    data = dumps(img, "P6")
    assert data[:2] == b"P6"
    assert loads(data) == img


def test_p3_roundtrip_preserves_pixels(img):
    data = dumps(img, "P3")
    assert data[:2] == b"P3"
    assert loads(data) == img


def test_p3_to_p6_and_back(img):
    p3 = dumps(img, "P3")
    p6 = dumps(loads(p3), "P6")
    assert loads(p6) == img
    assert dumps(loads(p6), "P3").startswith(b"P3")


def test_p6_header_no_whitespace_inside_raster():
    img = Image(2, 1, [(1, 2, 3), (4, 5, 6)])
    data = dumps(img, "P6")
    header, _, raster = data.partition(b"255\n")
    assert header.startswith(b"P6\n2 1\n")
    assert raster == bytes([1, 2, 3, 4, 5, 6])


def test_comments_between_header_tokens():
    data = b"P6\n# comment one\n2 # width\n2\n255\n" + bytes(range(12))
    img = loads(data)
    assert img.width == 2
    assert img.get_pixel(0, 0) == (0, 1, 2)


def test_save_and_load_file(tmp_path, img):
    path = os.path.join(tmp_path, "x.ppm")
    save(img, path, "P6")
    assert load(path) == img
    path3 = os.path.join(tmp_path, "y.p3")
    save(img, path3)
    assert load(path3) == img


def test_truncated_files_raise():
    with pytest.raises(PPMError):
        loads(b"P6\n2 2\n255\n\x00\x00")
    with pytest.raises(PPMError):
        loads(b"P3\n2 2\n255\n0 0 0 0 0 0")

# ---------------------------------------------------------------------------
# Filters: immutability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func,args",
    [
        (filters.to_grayscale, ()),
        (filters.invert, ()),
        (filters.adjust_brightness, (1.2,)),
        (filters.adjust_contrast, (1.5,)),
        (filters.gaussian_blur, (3,)),
        (filters.sharpen, ()),
        (filters.sobel_edge, ()),
        (filters.resize, (8, 8)),
        (filters.rotate, (90,)),
    ],
)
def test_filters_return_new_image(img, func, args):
    before = img.pixels
    out = func(img, *args)
    assert out is not img
    assert img.pixels == before  # original untouched


def test_methods_match_functions(img):
    assert img.to_grayscale() == filters.to_grayscale(img)
    assert img.gaussian_blur(2) == filters.gaussian_blur(img, 2)
    assert img.rotate(180) == filters.rotate(img, 180)


# ---------------------------------------------------------------------------
# Grayscale / invert / brightness / contrast
# ---------------------------------------------------------------------------


def test_grayscale_formula():
    img = Image(1, 1, [(255, 0, 0)])
    expected = round(0.299 * 255)
    gray = filters.to_grayscale(img)
    r, g, b = gray.get_pixel(0, 0)
    assert r == g == b == expected


def test_grayscale_green_and_blue():
    img = Image(3, 1, [(0, 255, 0), (0, 0, 255), (100, 100, 100)])
    gray = filters.to_grayscale(img)
    assert gray.get_pixel(0, 0)[0] == round(0.587 * 255)
    assert gray.get_pixel(1, 0)[0] == round(0.114 * 255)
    assert gray.get_pixel(2, 0) == (100, 100, 100)


def test_invert():
    img = Image(2, 1, [(255, 0, 128), (0, 255, 10)])
    out = filters.invert(img)
    assert out.get_pixel(0, 0) == (0, 255, 127)
    assert out.get_pixel(1, 0) == (255, 0, 245)


def test_brightness_factor():
    img = Image(2, 1, [(100, 50, 200), (255, 255, 255)])
    out = filters.adjust_brightness(img, 0.5)
    assert out.get_pixel(0, 0) == (50, 25, 100)
    out = filters.adjust_brightness(img, 2.0)
    assert out.get_pixel(0, 0) == (200, 100, 255)
    assert out.get_pixel(1, 0) == (255, 255, 255)


def test_contrast_flatten_and_boost():
    img = Image(2, 1, [(100, 100, 100), (200, 200, 200)])
    flat = filters.adjust_contrast(img, 0.0)
    assert flat.get_pixel(0, 0) == flat.get_pixel(1, 0) == (128, 128, 128)
    same = filters.adjust_contrast(img, 1.0)
    assert same == img


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5])
def test_blur_uniform_color_unchanged(radius):
    color = (47, 99, 13)
    img = Image(8, 6, [color] * 48)
    out = filters.gaussian_blur(img, radius)
    assert out == img


def test_blur_black_white_boundary_has_mid_values():
    # Left half black, right half white, one row.
    pixels = [(0, 0, 0)] * 4 + [(255, 255, 255)] * 4
    img = Image(8, 1, pixels)
    out = filters.gaussian_blur(img, 2)
    middle = [out.get_pixel(x, 0)[0] for x in range(8)]
    assert 0 < middle[3] < 255
    assert 0 < middle[4] < 255
    # monotonic ramp across the boundary
    assert middle[3] <= middle[4]
    # far edges stay near their original values
    assert middle[0] < 5
    assert middle[7] > 250


def test_blur_does_not_crash_on_borders(img):
    out = filters.gaussian_blur(img, 5)
    assert (out.width, out.height) == (4, 4)
    # Every output pixel is a valid 0-255 RGB tuple
    for y in range(out.height):
        for x in range(out.width):
            r, g, b = out.get_pixel(x, y)
            assert 0 <= min(r, g, b) and max(r, g, b) <= 255


def test_blur_bad_radius(img):
    with pytest.raises(ValueError):
        filters.gaussian_blur(img, 0)
    with pytest.raises(ValueError):
        filters.gaussian_blur(img, 6)


def test_blur_center_of_small_black_white_image():
    # radius 1 => 3 weights summing to 1024 (fixed point).
    from ppmimg.filters import _gaussian_weights
    weights = _gaussian_weights(1)
    assert len(weights) == 3
    assert sum(weights) == 1024
    assert weights[0] == weights[2]  # symmetric
    # A single white pixel on a black row: 1-D center weight applied once.
    img = Image(3, 3, [(255, 255, 255) if (x, y) == (1, 1) else (0, 0, 0)
                       for y in range(3) for x in range(3)])
    out = filters.gaussian_blur(img, 1)
    # Separable fixed point: each pass divides by 1024 with floor.
    after_h = (255 * weights[1]) // 1024
    expected = (after_h * weights[1]) // 1024
    assert out.get_pixel(1, 1)[0] == expected


# ---------------------------------------------------------------------------
# Sharpen / Sobel
# ---------------------------------------------------------------------------


def test_sharpen_uniform_color_unchanged():
    img = Image(6, 6, [(120, 60, 200)] * 36)
    assert filters.sharpen(img) == img


def test_sobel_flat_region_near_zero():
    img = Image(8, 8, [(90, 90, 90)] * 64)
    out = filters.sobel_edge(img)
    assert all(
        out.get_pixel(x, y) == (0, 0, 0)
        for y in range(8)
        for x in range(8)
    )


def test_sobel_vertical_edge_is_bright():
    # Black left half, white right half: vertical edge -> strong gradient gx.
    black = (0, 0, 0)
    white = (255, 255, 255)
    pixels = [black if x < 4 else white for y in range(6) for x in range(8)]
    img = Image(8, 6, pixels)
    out = filters.sobel_edge(img)
    # The black/white seam lies between x=3 and x=4; both respond strongly.
    edge_values = [out.get_pixel(x, y)[0] for x in (3, 4) for y in range(1, 5)]
    flat_values = [out.get_pixel(0, y)[0] for y in range(1, 5)]
    assert max(edge_values) == 255  # Sobel magnitude saturates at the edge
    assert min(edge_values) > 200
    assert max(flat_values) < 5     # far from the edge
    r, g, b = out.get_pixel(3, 2)
    assert r == g == b  # grayscale output


def test_sobel_values_clamped_to_255():
    pixels = [(0, 0, 0) if x < 4 else (255, 255, 255)
              for y in range(6) for x in range(8)]
    img = Image(8, 6, pixels)
    out = filters.sobel_edge(img)
    for y in range(out.height):
        for x in range(out.width):
            assert max(out.get_pixel(x, y)) <= 255
    # rotate mapping tests below use the corrected CW convention

# ---------------------------------------------------------------------------
# Resize (bilinear) and rotate
# ---------------------------------------------------------------------------


def test_resize_dimensions(img):
    out = filters.resize(img, 10, 6)
    assert (out.width, out.height) == (10, 6)


def test_resize_upscale_2x2_to_4x4_manual():
    # Mapping sx = dx*(2-1)/(4-1) = dx/3, so x samples at 0, 1/3, 2/3, 1.
    a = (0, 0, 0)
    b = (30, 60, 90)
    c = (120, 30, 210)
    d = (255, 255, 255)
    img = Image(2, 2, [a, b, c, d])
    out = filters.resize(img, 4, 4)
    assert (out.width, out.height) == (4, 4)

    def bil(v00, v10, v01, v11, fx, fy):
        top = v00 * (1 - fx) + v10 * fx
        bottom = v01 * (1 - fx) + v11 * fx
        return round(top * (1 - fy) + bottom * fy)

    for dy in range(4):
        fy = dy / 3
        for dx in range(4):
            fx = dx / 3
            r = bil(a[0], b[0], c[0], d[0], fx, fy)
            g = bil(a[1], b[1], c[1], d[1], fx, fy)
            bl = bil(a[2], b[2], c[2], d[2], fx, fy)
            assert out.get_pixel(dx, dy) == (r, g, bl)


def test_resize_keeps_corners_exact():
    a, b, c, d = (10, 20, 30), (40, 50, 60), (70, 80, 90), (100, 110, 120)
    img = Image(2, 2, [a, b, c, d])
    out = filters.resize(img, 5, 7)
    assert out.get_pixel(0, 0) == a
    assert out.get_pixel(4, 0) == b
    assert out.get_pixel(0, 6) == c
    assert out.get_pixel(4, 6) == d


def test_resize_same_size_identity(img):
    assert filters.resize(img, 4, 4) == img


def test_resize_single_column():
    img = Image(3, 1, [(10, 10, 10), (20, 20, 20), (30, 30, 30)])
    out = filters.resize(img, 1, 1)
    assert (out.width, out.height) == (1, 1)
    # Single sample lands on the center edge mapping -> first pixel
    assert out.get_pixel(0, 0) == (10, 10, 10)


def test_rotate_90_dimensions_and_pixels(img):
    out = filters.rotate(img, 90)
    assert (out.width, out.height) == (4, 4)
    # 90 CW: dst(dx, dy) = src(dy, W-1-dx)
    for dy in range(4):
        for dx in range(4):
            assert out.get_pixel(dx, dy) == img.get_pixel(dy, 3 - dx)


def test_rotate_180_and_270(img):
    r180 = filters.rotate(img, 180)
    assert r180.get_pixel(0, 0) == img.get_pixel(3, 3)
    assert r180.get_pixel(3, 3) == img.get_pixel(0, 0)

    r90 = filters.rotate(img, 90)
    assert filters.rotate(r90, 90) == r180
    assert filters.rotate(r90, 270) == img

    r270 = filters.rotate(img, 270)
    for dy in range(4):
        for dx in range(4):
            assert r270.get_pixel(dx, dy) == img.get_pixel(3 - dy, dx)


def test_rotate_rectangular_dimensions():
    img = Image(4, 2, [(i, i, i) for i in range(8)])
    assert (filters.rotate(img, 90).width, filters.rotate(img, 90).height) == (2, 4)
    assert (filters.rotate(img, 270).width, filters.rotate(img, 270).height) == (2, 4)
    assert (filters.rotate(img, 180).width, filters.rotate(img, 180).height) == (4, 2)


def test_rotate_invalid_angle(img):
    with pytest.raises(ValueError):
        filters.rotate(img, 45)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def _make_800x600():
    total = 800 * 600
    pixels = [
        ((x * 13 + y * 7) % 256, (x * 5 + y * 11) % 256, (x + y * 3) % 256)
        for y in range(600)
        for x in range(800)
    ]
    assert len(pixels) == total
    return Image(800, 600, pixels)


def test_performance_blur_800x600():
    import time
    img = _make_800x600()
    start = time.perf_counter()
    out = filters.gaussian_blur(img, 3)
    elapsed = time.perf_counter() - start
    assert out.width == 800
    assert elapsed < 10.0, "gaussian_blur(3) took %.2fs" % elapsed


def test_performance_resize_800x600():
    import time
    img = _make_800x600()
    start = time.perf_counter()
    out = filters.resize(img, 1024, 768)
    elapsed = time.perf_counter() - start
    assert out.width == 1024
    assert elapsed < 10.0, "resize took %.2fs" % elapsed