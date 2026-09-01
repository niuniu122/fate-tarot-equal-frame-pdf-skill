#!/usr/bin/env python3
"""Embed a finalized opaque RGB print raster in a conservative PDF 1.4 page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zlib

from PIL import Image


POINTS_PER_MM = 72.0 / 25.4


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {value}")
    return float(value)


def write_print_pdf(
    *,
    image_path: str | Path,
    output_pdf: str | Path,
    trim_width_mm: float,
    trim_height_mm: float,
    bleed_left_mm: float,
    bleed_right_mm: float,
    bleed_top_mm: float,
    bleed_bottom_mm: float,
    aspect_tolerance: float = 0.002,
) -> dict[str, object]:
    """Write one direct RGB Image XObject without cropping or resampling.

    ``image_path`` must already be the complete media raster: trim plus bleed.
    The function deliberately fails when its aspect ratio does not match the
    requested physical page. It never silently crops, stretches, or resamples.
    """

    trim_width_mm = _positive("trim_width_mm", trim_width_mm)
    trim_height_mm = _positive("trim_height_mm", trim_height_mm)
    bleed_left_mm = _positive("bleed_left_mm", bleed_left_mm, allow_zero=True)
    bleed_right_mm = _positive("bleed_right_mm", bleed_right_mm, allow_zero=True)
    bleed_top_mm = _positive("bleed_top_mm", bleed_top_mm, allow_zero=True)
    bleed_bottom_mm = _positive("bleed_bottom_mm", bleed_bottom_mm, allow_zero=True)
    if aspect_tolerance <= 0:
        raise ValueError("aspect_tolerance must be positive")

    media_width_mm = trim_width_mm + bleed_left_mm + bleed_right_mm
    media_height_mm = trim_height_mm + bleed_top_mm + bleed_bottom_mm
    image_path = Path(image_path)
    output_pdf = Path(output_pdf)
    with Image.open(image_path) as source:
        if source.mode != "RGB":
            raise ValueError(
                f"final media image must be opaque RGB; got mode {source.mode!r}. "
                "Composite transparency explicitly before export."
            )
        image = source.copy()

    image_ratio = image.width / image.height
    page_ratio = media_width_mm / media_height_mm
    relative_error = abs(image_ratio - page_ratio) / page_ratio
    if relative_error > aspect_tolerance:
        raise ValueError(
            "image/page aspect ratio mismatch would require cropping, stretching, "
            f"or resampling: image={image.width}x{image.height}, "
            f"media={media_width_mm:g}x{media_height_mm:g} mm, "
            f"relative_error={relative_error:.6f}"
        )

    media_width_pt = media_width_mm * POINTS_PER_MM
    media_height_pt = media_height_mm * POINTS_PER_MM
    trim_left_pt = bleed_left_mm * POINTS_PER_MM
    trim_bottom_pt = bleed_bottom_mm * POINTS_PER_MM
    trim_right_pt = (bleed_left_mm + trim_width_mm) * POINTS_PER_MM
    trim_top_pt = (bleed_bottom_mm + trim_height_mm) * POINTS_PER_MM
    media_box = f"[0 0 {_number(media_width_pt)} {_number(media_height_pt)}]"
    trim_box = (
        f"[{_number(trim_left_pt)} {_number(trim_bottom_pt)} "
        f"{_number(trim_right_pt)} {_number(trim_top_pt)}]"
    )

    image_bytes = zlib.compress(image.tobytes(), level=9)
    content_bytes = (
        "q\n"
        f"{_number(media_width_pt)} 0 0 {_number(media_height_pt)} 0 0 cm\n"
        "/Im0 Do\n"
        "Q\n"
    ).encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R "
            f"/MediaBox {media_box} /CropBox {media_box} "
            f"/BleedBox {media_box} /TrimBox {trim_box} /ArtBox {trim_box} "
            "/Resources << /ProcSet [/PDF /ImageC] "
            "/XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        (
            "<< /Type /XObject /Subtype /Image "
            f"/Width {image.width} /Height {image.height} "
            "/ColorSpace /DeviceRGB /BitsPerComponent 8 "
            "/Interpolate false /Filter /FlateDecode "
            f"/Length {len(image_bytes)} >>\nstream\n"
        ).encode("ascii")
        + image_bytes
        + b"\nendstream",
        f"<< /Length {len(content_bytes)} >>\nstream\n".encode("ascii")
        + content_bytes
        + b"endstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.write_bytes(pdf)

    return {
        "pdf": str(output_pdf),
        "source_image": str(image_path),
        "image_px": [image.width, image.height],
        "media_mm": [media_width_mm, media_height_mm],
        "trim_mm": [trim_width_mm, trim_height_mm],
        "bleed_mm": {
            "left": bleed_left_mm,
            "right": bleed_right_mm,
            "top": bleed_top_mm,
            "bottom": bleed_bottom_mm,
        },
        "pdf_version": "1.4",
        "page_image": "direct /Im0 Image XObject",
        "resampled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trim-width-mm", required=True, type=float)
    parser.add_argument("--trim-height-mm", required=True, type=float)
    parser.add_argument("--bleed-left-mm", required=True, type=float)
    parser.add_argument("--bleed-right-mm", required=True, type=float)
    parser.add_argument("--bleed-top-mm", required=True, type=float)
    parser.add_argument("--bleed-bottom-mm", required=True, type=float)
    parser.add_argument("--aspect-tolerance", type=float, default=0.002)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = write_print_pdf(
        image_path=args.image,
        output_pdf=args.output,
        trim_width_mm=args.trim_width_mm,
        trim_height_mm=args.trim_height_mm,
        bleed_left_mm=args.bleed_left_mm,
        bleed_right_mm=args.bleed_right_mm,
        bleed_top_mm=args.bleed_top_mm,
        bleed_bottom_mm=args.bleed_bottom_mm,
        aspect_tolerance=args.aspect_tolerance,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
