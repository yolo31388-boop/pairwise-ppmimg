"""Command-line batch filter tool.

Usage::

    python -m ppmimg -f blur:3 -f grayscale -o out/ *.ppm

Filter syntax (applied in order, repeat ``-f`` as needed):

    grayscale | invert | sharpen | sobel
    brightness:<factor>   contrast:<factor>
    blur:[radius]         resize:<width>x<height>
    rotate:<90|180|270>
"""

import argparse
import glob as globlib
import os
import sys

from . import filters
from .ppm import PPMError, load, save


def _parse_filter(spec):
    if ":" in spec:
        name, arg = spec.split(":", 1)
    else:
        name, arg = spec, None
    table = {
        "grayscale": (filters.to_grayscale, lambda a: ()),
        "invert": (filters.invert, lambda a: ()),
        "sharpen": (filters.sharpen, lambda a: ()),
        "sobel": (filters.sobel_edge, lambda a: ()),
        "brightness": (filters.adjust_brightness, lambda a: (float(a),)),
        "contrast": (filters.adjust_contrast, lambda a: (float(a),)),
        "blur": (filters.gaussian_blur, lambda a: (int(a) if a else 3,)),
        "rotate": (filters.rotate, lambda a: (int(a),)),
    }
    if name == "resize":
        if not arg or "x" not in arg:
            raise argparse.ArgumentTypeError("resize needs WIDTHxHEIGHT, e.g. resize:320x240")
        w, h = arg.lower().split("x", 1)
        return (filters.resize, (int(w), int(h)))
    if name not in table:
        raise argparse.ArgumentTypeError("unknown filter: %s" % name)
    func, parse = table[name]
    try:
        return (func, parse(arg))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("bad argument for %s: %s" % (name, exc))


def _iter_input_paths(patterns):
    for pattern in patterns:
        matches = globlib.glob(pattern)
        if matches:
            for match in matches:
                yield match
        else:
            yield pattern  # let open() raise a clear error


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m ppmimg",
        description="Batch-apply filters to PPM images (pure standard library).",
    )
    parser.add_argument("inputs", nargs="+", help="PPM files or glob patterns")
    parser.add_argument("-o", "--outdir", default=None, help="output directory")
    parser.add_argument(
        "--format",
        choices=("P3", "P6"),
        default=None,
        help="output format (defaults to P6, or matches .p3/.p6 suffix)",
    )
    parser.add_argument(
        "-f",
        "--filter",
        dest="filter_specs",
        action="append",
        default=[],
        help="filter to apply, e.g. blur:3 or resize:320x240 (repeatable)",
    )
    args = parser.parse_args(argv)

    chain = [_parse_filter(spec) for spec in args.filter_specs]
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    ok = 0
    for path in _iter_input_paths(args.inputs):
        try:
            image = load(path)
            for func, fargs in chain:
                image = func(image, *fargs)
            base = os.path.basename(path)
            out_path = os.path.join(args.outdir, base) if args.outdir else path
            save(image, out_path, args.format)
            print("%s -> %s" % (path, out_path))
            ok += 1
        except (OSError, PPMError, ValueError) as exc:
            print("failed: %s (%s)" % (path, exc), file=sys.stderr)
    if ok == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())