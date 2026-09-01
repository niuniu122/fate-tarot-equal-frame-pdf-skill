#!/usr/bin/env python3
"""Compatibility entry point for the subject-first fixed-bleed pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from subject_first_pipeline import (
    BackgroundModel,
    DetectionError,
    _perimeter_model,
    _render_background,
    prepare_subject_first,
)


def _fit_background(rgb: np.ndarray) -> BackgroundModel:
    """Compatibility alias for callers that render a provisional perimeter model."""
    return _perimeter_model(rgb)


def prepare_equal_frame(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    trim_width_mm: float = 80.0,
    trim_height_mm: float = 120.0,
    bleed_left_mm: float = 5.0,
    bleed_right_mm: float = 5.0,
    bleed_top_mm: float = 5.0,
    bleed_bottom_mm: float = 5.0,
    minimum_effective_ppi: float = 300.0,
    background_tolerance: int | None = None,
    validated_frame_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Forward the legacy public API to the authoritative subject-first pipeline."""
    return prepare_subject_first(
        source_path,
        output_dir,
        trim_width_mm=trim_width_mm,
        trim_height_mm=trim_height_mm,
        bleed_left_mm=bleed_left_mm,
        bleed_right_mm=bleed_right_mm,
        bleed_top_mm=bleed_top_mm,
        bleed_bottom_mm=bleed_bottom_mm,
        minimum_effective_ppi=minimum_effective_ppi,
        background_tolerance=background_tolerance,
        validated_frame_override=validated_frame_override,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trim-width-mm", type=float, default=80.0)
    parser.add_argument("--trim-height-mm", type=float, default=120.0)
    parser.add_argument("--bleed-left-mm", type=float, default=5.0)
    parser.add_argument("--bleed-right-mm", type=float, default=5.0)
    parser.add_argument("--bleed-top-mm", type=float, default=5.0)
    parser.add_argument("--bleed-bottom-mm", type=float, default=5.0)
    parser.add_argument("--minimum-effective-ppi", type=float, default=300.0)
    parser.add_argument("--background-tolerance", type=int)
    parser.add_argument(
        "--validated-frame-override",
        help=(
            "JSON object or JSON file containing left/right/top/bottom pixel "
            "anchors and radii (one number or four named corner radii)"
        ),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        validated_frame_override = None
        if args.validated_frame_override:
            raw_override = args.validated_frame_override
            try:
                if raw_override.lstrip().startswith("{"):
                    validated_frame_override = json.loads(raw_override)
                else:
                    validated_frame_override = json.loads(
                        Path(raw_override).read_text(encoding="utf-8")
                    )
            except (OSError, json.JSONDecodeError) as error:
                raise DetectionError(
                    "validated frame override must be valid JSON text or a readable JSON file"
                ) from error
        report = prepare_equal_frame(
            args.source,
            args.output_dir,
            trim_width_mm=args.trim_width_mm,
            trim_height_mm=args.trim_height_mm,
            bleed_left_mm=args.bleed_left_mm,
            bleed_right_mm=args.bleed_right_mm,
            bleed_top_mm=args.bleed_top_mm,
            bleed_bottom_mm=args.bleed_bottom_mm,
            minimum_effective_ppi=args.minimum_effective_ppi,
            background_tolerance=args.background_tolerance,
            validated_frame_override=validated_frame_override,
        )
    except DetectionError as error:
        parser.exit(2, f"unsafe input: {error}\n")
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
