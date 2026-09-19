"""Batch-apply a filter to every PPM in a directory.

Usage:
    python batch_filter.py INPUT_DIR OUTPUT_DIR FILTER [OPTIONS]

Examples:
    python batch_filter.py in/ out/ grayscale
    python batch_filter.py in/ out/ blur --radius 3
    python batch_filter.py in/ out/ brightness --factor 1.2
    python batch_filter.py in/ out/ resize --width 400 --height 300
    python batch_filter.py in/ out/ rotate --degrees 90
    python batch_filter.py in/ out/ sobel --p3
"""

import argparse
import os
import sys

from ppmimg import (
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-apply a filter to PPM images.")
    parser.add_argument("input_dir", help="directory containing .ppm files")
    parser.add_argument("output_dir", help="directory for processed files")
    parser.add_argument(
        "filter",
        choices=["grayscale", "invert", "brightness", "contrast", "blur",
                 "sharpen", "sobel", "resize", "rotate"],
        help="filter to apply")
    parser.add_argument("--factor", type=float, default=1.0,
                        help="factor for brightness/contrast (default 1.0)")
    parser.add_argument("--radius", type=int, default=1,
                        help="blur radius 1-5 (default 1)")
    parser.add_argument("--width", type=int, help="target width for resize")
    parser.add_argument("--height", type=int, help="target height for resize")
    parser.add_argument("--degrees", type=int, default=90,
                        choices=[90, 180, 270],
                        help="rotation angle (default 90)")
    parser.add_argument("--p3", action="store_true",
                        help="write plain-text P3 output (default is P6)")
    return parser


def apply_filter(img, args):
    if args.filter == "grayscale":
        return to_grayscale(img)
    if args.filter == "invert":
        return invert(img)
    if args.filter == "brightness":
        return adjust_brightness(img, args.factor)
    if args.filter == "contrast":
        return adjust_contrast(img, args.factor)
    if args.filter == "blur":
        return gaussian_blur(img, args.radius)
    if args.filter == "sharpen":
        return sharpen(img)
    if args.filter == "sobel":
        return sobel_edge(img)
    if args.filter == "resize":
        if not args.width or not args.height:
            raise SystemExit("resize requires --width and --height")
        return resize(img, args.width, args.height)
    if args.filter == "rotate":
        return rotate(img, args.degrees)
    raise SystemExit("unknown filter: %s" % args.filter)


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(args.input_dir)
                   if n.lower().endswith(".ppm"))
    if not names:
        print("no .ppm files found in %s" % args.input_dir)
        return 1
    for name in names:
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        img = read_ppm(src)
        out = apply_filter(img, args)
        write_ppm(out, dst, binary=not args.p3)
        print("%s -> %s (%dx%d)" % (src, dst, out.width, out.height))
    return 0


if __name__ == "__main__":
    sys.exit(main())
