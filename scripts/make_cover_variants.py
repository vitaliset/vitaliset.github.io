# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=9"]
# ///
"""Generate post-cover image variants from a single source image.

Revives just the image-resize piece of the old Gulp `img` task: from one cover
picture it writes `<slug>.jpg` plus the resized variants the templates expect
(`_thumb`, `_placehold`, `_thumb@2x`, and the `_xs/_sm/_md/_lg` responsive set)
into `assets/img/posts/`. The homepage card uses `_placehold`/`_thumb`/`_thumb@2x`;
the post hero and social preview use the base `<slug>.jpg`.

Usage (zero-setup via uv, reads the PEP 723 deps above):

    uv run scripts/make_cover_variants.py --source path/to/cover.jpg --slug my-slug

Or with any Python that has Pillow installed:

    python scripts/make_cover_variants.py --source path/to/cover.jpg --slug my-slug

Widths and JPEG quality match the original Gulp pipeline. Images are only ever
scaled DOWN (a target wider than the source falls back to the source width), so a
small source never gets upscaled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

# suffix -> width in px (same as the old gulp-responsive config). "" is the base file.
VARIANTS: dict[str, int] = {
    "": 1920,            # base: post hero + og:image
    "_lg": 1999,
    "_md": 991,
    "_sm": 767,
    "_xs": 575,
    "_thumb@2x": 1070,   # homepage card, retina
    "_thumb": 535,       # homepage card
    "_placehold": 230,   # homepage card, blur-up placeholder
}

QUALITY = 70
DEST = Path("assets/img/posts")


def flatten(img: Image.Image) -> Image.Image:
    """Return an RGB image, compositing any transparency onto white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate post-cover variants from one image.")
    parser.add_argument("--source", required=True, type=Path, help="path to the source cover image")
    parser.add_argument("--slug", required=True, help="cover slug = the post's featured-img value")
    parser.add_argument("--dest", default=DEST, type=Path, help=f"output dir (default: {DEST})")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")
    args.dest.mkdir(parents=True, exist_ok=True)

    src = flatten(Image.open(args.source))
    for suffix, width in VARIANTS.items():
        w = min(width, src.width)
        h = round(src.height * w / src.width)
        out = args.dest / f"{args.slug}{suffix}.jpg"
        src.resize((w, h), Image.LANCZOS).save(
            out, "JPEG", quality=QUALITY, progressive=True, optimize=True
        )
        print(f"  {out}  ({w}x{h})")

    print(f"done: {len(VARIANTS)} files for '{args.slug}' in {args.dest}/")


if __name__ == "__main__":
    main()
