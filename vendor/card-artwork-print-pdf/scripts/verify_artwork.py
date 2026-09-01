#!/usr/bin/env python3
"""Verify scoped pixel changes and conservative single-image print PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
from pypdf import PdfReader


def _read_exact_pixels(path: str | Path) -> tuple[np.ndarray, str]:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image).copy(), image.mode


def compare_pixels(
    before_path: str | Path,
    after_path: str | Path,
    rois: list[tuple[int, int, int, int]],
) -> dict[str, object]:
    before, before_mode = _read_exact_pixels(before_path)
    after, after_mode = _read_exact_pixels(after_path)
    if before.shape != after.shape or before_mode != after_mode:
        raise ValueError(
            "before/after pixel grids differ: "
            f"before={before.shape}/{before_mode}, after={after.shape}/{after_mode}"
        )
    height, width = before.shape[:2]
    roi_mask = np.zeros((height, width), dtype=bool)
    normalized: list[list[int]] = []
    for roi in rois:
        x0, y0, x1, y1 = (int(value) for value in roi)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"ROI outside image bounds {width}x{height}: {roi}")
        roi_mask[y0:y1, x0:x1] = True
        normalized.append([x0, y0, x1, y1])
    if not normalized:
        raise ValueError("at least one ROI is required for scoped-change verification")
    if before.ndim == 2:
        changed = before != after
    else:
        changed = np.any(before != after, axis=tuple(range(2, before.ndim)))
    ys, xs = np.where(changed)
    bbox = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return {
        "before": str(before_path),
        "after": str(after_path),
        "mode": before_mode,
        "size_px": [width, height],
        "rois_px": normalized,
        "changed_pixels_total": int(changed.sum()),
        "changed_pixels_inside_roi": int(np.sum(changed & roi_mask)),
        "changed_pixels_outside_roi": int(np.sum(changed & ~roi_mask)),
        "changed_bbox_px": bbox,
        "scope_gate_passed": bool(np.sum(changed & ~roi_mask) == 0 and np.sum(changed & roi_mask) > 0),
    }


def _box_mm(box: object) -> list[float]:
    return [float(value) * 25.4 / 72.0 for value in box]


def _assert_box(actual: list[float], expected: list[float], name: str) -> None:
    if any(abs(a - e) > 0.001 for a, e in zip(actual, expected)):
        raise ValueError(f"{name} mismatch: actual={actual}, expected={expected}")


def verify_pdf(
    pdf_path: str | Path,
    expected_image: str | Path,
    *,
    trim_width_mm: float,
    trim_height_mm: float,
    bleed_left_mm: float,
    bleed_right_mm: float,
    bleed_top_mm: float,
    bleed_bottom_mm: float,
    pdftoppm: str | Path | None = None,
) -> dict[str, object]:
    pdf_path = Path(pdf_path)
    if pdf_path.read_bytes()[:8] != b"%PDF-1.4":
        raise ValueError("expected conservative PDF-1.4 header")
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) != 1:
        raise ValueError(f"expected one page, got {len(reader.pages)}")
    page = reader.pages[0]
    media_expected = [
        0.0,
        0.0,
        bleed_left_mm + trim_width_mm + bleed_right_mm,
        bleed_bottom_mm + trim_height_mm + bleed_top_mm,
    ]
    trim_expected = [
        bleed_left_mm,
        bleed_bottom_mm,
        bleed_left_mm + trim_width_mm,
        bleed_bottom_mm + trim_height_mm,
    ]
    media = _box_mm(page.mediabox)
    trim = _box_mm(page.trimbox)
    bleed = _box_mm(page.bleedbox)
    _assert_box(media, media_expected, "MediaBox")
    _assert_box(bleed, media_expected, "BleedBox")
    _assert_box(trim, trim_expected, "TrimBox")

    resources = page["/Resources"]
    if "/Font" in resources or "/Group" in page:
        raise ValueError("unexpected font or transparency group in raster print PDF")
    xobjects = resources["/XObject"].get_object()
    if list(xobjects) != ["/Im0"]:
        raise ValueError(f"expected exactly direct /Im0, got {list(xobjects)}")
    image_object = xobjects["/Im0"].get_object()
    if image_object["/Subtype"] != "/Image" or image_object["/Filter"] != "/FlateDecode":
        raise ValueError("/Im0 is not a direct Flate image")
    if "/SMask" in image_object or "/Mask" in image_object:
        raise ValueError("transparency mask is not allowed")
    content = page.get_contents().get_data()
    if b"/Im0 Do" not in content:
        raise ValueError("page content does not paint /Im0")

    expected = np.asarray(Image.open(expected_image).convert("RGB"))
    embedded_images = list(page.images)
    if len(embedded_images) != 1:
        raise ValueError(f"expected one extractable image, got {len(embedded_images)}")
    embedded = np.asarray(embedded_images[0].image.convert("RGB"))
    if embedded.shape != expected.shape or not np.array_equal(embedded, expected):
        different = -1 if embedded.shape != expected.shape else int(np.any(embedded != expected, axis=2).sum())
        raise ValueError(f"embedded PDF pixels differ from approved media image: {different}")

    renderers: list[str] = []
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        pdf_page = document[0]
        bitmap = pdf_page.render(scale=1.5)
        rendered = bitmap.to_pil().convert("RGB")
        bitmap.close()
        pdf_page.close()
        document.close()
        if np.all(np.asarray(rendered) >= 254):
            raise ValueError("PDFium rendered a blank white page")
        renderers.append("PDFium")
    except ImportError:
        pass

    executable = Path(pdftoppm) if pdftoppm else None
    if executable is None:
        located = shutil.which("pdftoppm")
        executable = Path(located) if located else None
    if executable is not None and executable.exists():
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "render"
            subprocess.run(
                [str(executable), "-r", "108", "-png", "-singlefile", str(pdf_path), str(prefix)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            rendered = np.asarray(Image.open(prefix.with_suffix(".png")).convert("RGB"))
            if np.all(rendered >= 254):
                raise ValueError("Poppler rendered a blank white page")
        renderers.append("Poppler")

    return {
        "pdf": str(pdf_path),
        "pdf_version": "1.4",
        "media_box_mm": media,
        "trim_box_mm": trim,
        "bleed_box_mm": bleed,
        "direct_image_xobject": True,
        "embedded_pixels_equal_approved_image": True,
        "renderers": renderers,
        "render_nonblank": bool(renderers),
        "illustrator_structure_compatible": True,
        "illustrator_real_open_tested": False,
    }


def _parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rectangle must be x0,y0,x1,y1")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--roi", action="append", type=_parse_rect, default=[])
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--approved-image", type=Path)
    parser.add_argument("--trim-width-mm", type=float)
    parser.add_argument("--trim-height-mm", type=float)
    parser.add_argument("--bleed-left-mm", type=float, default=0.0)
    parser.add_argument("--bleed-right-mm", type=float, default=0.0)
    parser.add_argument("--bleed-top-mm", type=float, default=0.0)
    parser.add_argument("--bleed-bottom-mm", type=float, default=0.0)
    parser.add_argument("--pdftoppm", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {}
    failed = False
    if args.before or args.after or args.roi:
        if not args.before or not args.after or not args.roi:
            parser.error("--before, --after, and at least one --roi are required together")
        pixel_report = compare_pixels(args.before, args.after, args.roi)
        report["pixel_scope"] = pixel_report
        failed |= not bool(pixel_report["scope_gate_passed"])
    if args.pdf or args.approved_image:
        required = (args.pdf, args.approved_image, args.trim_width_mm, args.trim_height_mm)
        if any(value is None for value in required):
            parser.error("PDF verification requires --pdf, --approved-image, and trim dimensions")
        report["pdf"] = verify_pdf(
            args.pdf,
            args.approved_image,
            trim_width_mm=args.trim_width_mm,
            trim_height_mm=args.trim_height_mm,
            bleed_left_mm=args.bleed_left_mm,
            bleed_right_mm=args.bleed_right_mm,
            bleed_top_mm=args.bleed_top_mm,
            bleed_bottom_mm=args.bleed_bottom_mm,
            pdftoppm=args.pdftoppm,
        )
    if not report:
        parser.error("provide pixel-scope inputs, PDF inputs, or both")
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
