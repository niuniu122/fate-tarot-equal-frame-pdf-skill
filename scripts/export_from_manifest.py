#!/usr/bin/env python3
"""Build and verify a print PDF from an approved subject-first manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from subject_first_pipeline import (
    CURRENT_PIPELINE_VERSION,
    DetectionError,
    SAMPLING_PROVENANCE_FIELDS,
    SAMPLING_PROVENANCE_SCHEMA_VERSION,
    _boundary_connected,
    _detect_dark_closed_frame,
    _evaluate,
    _frame_geometry_evidence,
    _polyfit_robust,
    _sampling_provenance_artifact_payload,
)


_PIPELINE_STAGES = (
    "detect-subject-frame",
    "extract-subject",
    "sample-exterior",
    "render-bleed-background",
    "place-subject",
)
_REQUIRED_ARTIFACTS = (
    "subject_rgba",
    "subject_mask",
    "exterior_sampling_mask",
    "bleed_background_rgb",
    "media_rgb",
)
_LEGACY_PIPELINE_VERSION = 2
_CURRENT_PIPELINE_VERSION = CURRENT_PIPELINE_VERSION
_CURRENT_SAMPLING_PROVENANCE_SCHEMA_VERSION = SAMPLING_PROVENANCE_SCHEMA_VERSION
_SAMPLING_PROVENANCE_FIELDS = SAMPLING_PROVENANCE_FIELDS
_V2_SAMPLING_FIELDS = frozenset(
    {
        "background_tolerance",
        "corner_coverage_fraction",
        "corner_median_rgb",
        "corner_method",
        "corner_sample_pixels",
        "exterior_sample_pixels",
        "non_exterior_sample_pixels",
        "region_policy",
        "sampling_mask_bbox_px",
        "sampling_mask_sha256",
        "sampling_quality_gate",
        "side_coverage_fraction",
        "side_median_rgb",
        "side_model_residual_p90",
        "side_sample_pixels",
        "side_texture_dispersion",
        "source_sha256",
        "subject_overlap_pixels",
    }
)
_GEOMETRY_TOLERANCE_MM = 1e-9
_SCALE_TOLERANCE = 1e-12
_EXACT_RATIO_TOLERANCE = 1e-12
# Existing approved manifests were produced before the export-time independent
# reconstruction was added.  Their dark-frame double-layer evidence differs by
# at most 0.01346 because the preparation evidence is measured before the final
# source-connected ornament mask.  Keep a small, explicit raster margin while
# still rejecting threshold-shaped fabricated values.
_GEOMETRY_DOUBLE_LAYER_TOLERANCE = 0.015


def _manifest_contract_evidence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select one explicit, versioned validation contract without field inference."""

    pipeline_version = manifest.get("pipeline_version")
    if isinstance(pipeline_version, bool) or not isinstance(pipeline_version, int):
        raise ValueError("manifest pipeline_version must be an explicit integer")
    schema_version = manifest.get("sampling_provenance_schema_version")
    if pipeline_version == _CURRENT_PIPELINE_VERSION:
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _CURRENT_SAMPLING_PROVENANCE_SCHEMA_VERSION
        ):
            raise ValueError(
                "manifest pipeline version 3 requires sampling provenance schema version 1"
            )
        return {
            "pipeline_version": _CURRENT_PIPELINE_VERSION,
            "sampling_provenance_schema_version": (
                _CURRENT_SAMPLING_PROVENANCE_SCHEMA_VERSION
            ),
            "validation_mode": "v3-exact-decoded-provenance",
        }
    if pipeline_version == _LEGACY_PIPELINE_VERSION:
        if "sampling_provenance_schema_version" in manifest:
            raise ValueError(
                "manifest pipeline version 2 cannot claim a current sampling provenance schema"
            )
        return {
            "pipeline_version": _LEGACY_PIPELINE_VERSION,
            "sampling_provenance_schema_version": None,
            "validation_mode": "v2-exact-deterministic-replay",
        }
    raise ValueError(f"unsupported manifest pipeline_version: {pipeline_version!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest does not contain a valid {label} object")
    return value


def _sequence(value: object, length: int, label: str) -> Sequence[Any]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"manifest does not contain a valid {label}")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"manifest {label} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest {label} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"manifest {label} must be a positive finite number")
    return number


def _declared_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"manifest does not contain a valid SHA-256 for {label}")
    return value.lower()


def _manifest_file(value: object, manifest_directory: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest does not contain a valid path for {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"manifest {label} is not an existing file: {path}")
    return path


def _file_sha256(path: Path, label: str) -> str:
    try:
        return _sha256(path)
    except OSError as exc:
        raise ValueError(f"cannot hash manifest {label}: {path}") from exc


def _decoded_image(path: Path, mode: str, label: str) -> np.ndarray:
    """Decode an artifact while preserving its declared pixel contract."""

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != mode:
                raise ValueError(
                    f"manifest {label} must be stored in {mode} mode, not {image.mode}"
                )
            return np.asarray(image).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"manifest {label} cannot be decoded as an image: {path}") from exc


def _decoded_source_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"manifest source cannot be decoded as an image: {path}") from exc


def _decoded_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest {label} cannot be decoded as JSON: {path}") from exc
    return _mapping(value, label)


def _integer_sequence(
    value: object,
    length: int,
    label: str,
) -> tuple[int, ...]:
    sequence = _sequence(value, length, label)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in sequence):
        raise ValueError(f"manifest does not contain a valid integer {label}")
    return tuple(int(item) for item in sequence)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"manifest {label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest {label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"manifest {label} must be a finite number")
    return number


def _binary_bbox(mask: np.ndarray) -> list[int] | None:
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) == 0:
        return None
    return [
        int(x_coordinates.min()),
        int(y_coordinates.min()),
        int(x_coordinates.max()) + 1,
        int(y_coordinates.max()) + 1,
    ]


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.concatenate(([False], values.astype(bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts)) if len(starts) else 0


def _named_floats(
    value: object,
    names: Sequence[str],
    label: str,
) -> dict[str, float]:
    mapping = _mapping(value, label)
    return {
        name: _finite_float(mapping.get(name), f"{label} {name}")
        for name in names
    }


def _require_recomputed_float(
    declared: float,
    recomputed: float,
    *,
    tolerance: float,
    label: str,
) -> None:
    if not math.isclose(
        declared,
        recomputed,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"manifest {label} counter does not match recomputed decoded-pixel evidence"
        )


def _named_nonnegative_integers(
    value: object,
    names: Sequence[str],
    label: str,
) -> dict[str, int]:
    mapping = _mapping(value, label)
    result: dict[str, int] = {}
    for name in names:
        item = mapping.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"manifest {label} {name} must be a non-negative integer")
        result[name] = int(item)
    return result


def _require_exact_typed_value(
    declared: object,
    recomputed: object,
    *,
    label: str,
) -> None:
    """Reject both value drift and JSON type substitution in audited evidence."""

    if type(declared) is not type(recomputed):
        raise ValueError(f"manifest {label} has the wrong JSON type")
    if isinstance(recomputed, dict):
        if declared.keys() != recomputed.keys():
            raise ValueError(f"manifest {label} has missing or unexpected fields")
        for key in recomputed:
            _require_exact_typed_value(
                declared[key],
                recomputed[key],
                label=f"{label} {key}",
            )
        return
    if isinstance(recomputed, list):
        if len(declared) != len(recomputed):
            raise ValueError(f"manifest {label} has the wrong list length")
        for index, (declared_item, recomputed_item) in enumerate(
            zip(declared, recomputed, strict=True)
        ):
            _require_exact_typed_value(
                declared_item,
                recomputed_item,
                label=f"{label}[{index}]",
            )
        return
    if declared != recomputed:
        raise ValueError(
            f"manifest {label} does not match recomputed decoded-pixel evidence"
        )


def _validate_v3_sampling_provenance(
    sampling: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    if sampling.get("side_interpolation_method") != (
        "same-side-valid-coordinate-linear"
    ):
        raise ValueError(
            "manifest v3 sampling provenance has an invalid interpolation method"
        )
    missing = [
        field for field in _SAMPLING_PROVENANCE_FIELDS if field not in sampling
    ]
    if missing:
        raise ValueError(
            "manifest pipeline version 3 sampling provenance is missing: "
            + ", ".join(missing)
        )
    expected = _mapping(
        recomputed.get("interpolation_provenance"),
        "recomputed sampling interpolation provenance",
    )
    for field in _SAMPLING_PROVENANCE_FIELDS:
        _require_exact_typed_value(
            sampling.get(field),
            expected.get(field),
            label=f"sampling provenance {field}",
        )


def _validate_v2_sampling_metrics(
    sampling: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    """Validate every reproducible v2 counter against the exact legacy replay."""

    replayed_fields = (
        "region_policy",
        "subject_overlap_pixels",
        "non_exterior_sample_pixels",
        "exterior_sample_pixels",
        "side_sample_pixels",
        "corner_sample_pixels",
        "side_coverage_fraction",
        "corner_coverage_fraction",
        "side_texture_dispersion",
        "side_model_residual_p90",
        "side_median_rgb",
        "corner_median_rgb",
        "corner_method",
        "sampling_quality_gate",
    )
    for field in replayed_fields:
        _require_exact_typed_value(
            sampling.get(field),
            recomputed.get(field),
            label=f"v2 sampling replay {field}",
        )
    tolerance = sampling.get("background_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError(
            "manifest v2 sampling background_tolerance must be a non-negative integer"
        )


def _validate_declared_geometry_metrics(
    manifest: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    double_layer_tolerance: float,
) -> None:
    frame_detection = _mapping(manifest.get("frame_detection"), "frame_detection")
    evidence = _mapping(
        frame_detection.get("geometry_evidence"),
        "frame geometry evidence",
    )
    side_names = ("top", "right", "bottom", "left")
    continuous = _named_floats(
        evidence.get("continuous_side_coverage"),
        side_names,
        "frame continuous side coverage",
    )
    if min(continuous.values()) < 0.50:
        raise ValueError(
            "manifest frame geometry coverage does not satisfy the four-side gate"
        )
    double_layer = _named_floats(
        evidence.get("double_layer_coverage"),
        side_names,
        "frame double-layer coverage",
    )
    rounded = evidence.get("rounded_corner_evidence") is True
    if not rounded and min(double_layer.values()) < 0.35:
        raise ValueError(
            "manifest frame geometry does not prove rounded corners or parallel layers"
        )
    recomputed_continuous = _named_floats(
        recomputed.get("continuous_side_coverage"),
        side_names,
        "recomputed frame continuous side coverage",
    )
    recomputed_double_layer = _named_floats(
        recomputed.get("double_layer_coverage"),
        side_names,
        "recomputed frame double-layer coverage",
    )
    for side in side_names:
        _require_recomputed_float(
            continuous[side],
            recomputed_continuous[side],
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"geometry continuous-side {side}",
        )
        _require_recomputed_float(
            double_layer[side],
            recomputed_double_layer[side],
            tolerance=double_layer_tolerance,
            label=f"geometry double-layer {side}",
        )
    recomputed_rounded = recomputed.get("rounded_corner_evidence") is True
    if rounded and not recomputed_rounded:
        raise ValueError(
            "manifest geometry rounded-corner counter claims evidence absent from "
            "the recomputed decoded pixels"
        )
    declared_structure_gate = rounded or min(double_layer.values()) >= 0.35
    recomputed_structure_gate = (
        recomputed_rounded or min(recomputed_double_layer.values()) >= 0.35
    )
    if declared_structure_gate != recomputed_structure_gate:
        raise ValueError(
            "manifest geometry rounded-or-double-layer gate does not match "
            "recomputed decoded-pixel evidence"
        )


def _validate_reviewed_flat_sampling_metrics(
    sampling: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    if sampling.get("sampling_method") != "reviewed-flat-exterior-median-v1":
        raise ValueError("manifest reviewed flat sampling method is invalid")
    if sampling.get("side_interpolation_method") != (
        "not-applicable-reviewed-flat-median"
    ):
        raise ValueError("manifest reviewed flat sampling provenance is invalid")
    expected_provenance = _mapping(
        recomputed.get("interpolation_provenance"),
        "recomputed reviewed flat sampling provenance",
    )
    for field in _SAMPLING_PROVENANCE_FIELDS:
        _require_exact_typed_value(
            sampling.get(field),
            expected_provenance.get(field),
            label=f"reviewed flat sampling provenance {field}",
        )
    replayed_fields = (
        "sampling_method",
        "region_policy",
        "flat_background_rgb",
        "flat_color_statistic",
        "source_edge_extension",
        "subject_overlap_pixels",
        "non_exterior_sample_pixels",
        "exterior_sample_pixels",
        "side_sample_pixels",
        "corner_sample_pixels",
        "side_coverage_fraction",
        "corner_coverage_fraction",
        "side_texture_dispersion",
        "side_model_residual_p90",
        "side_median_rgb",
        "corner_median_rgb",
        "corner_method",
        "sampling_quality_gate",
    )
    for field in replayed_fields:
        _require_exact_typed_value(
            sampling.get(field),
            recomputed.get(field),
            label=f"reviewed flat sampling metric {field}",
        )


def _validate_declared_sampling_metrics(
    sampling: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    source_width: int,
    source_height: int,
    pipeline_version: int,
) -> None:
    if sampling.get("sampling_method") == "reviewed-flat-exterior-median-v1":
        if pipeline_version != _CURRENT_PIPELINE_VERSION:
            raise ValueError(
                "reviewed flat exterior sampling requires the current exact manifest contract"
            )
        _validate_reviewed_flat_sampling_metrics(sampling, recomputed)
        return
    side_names = ("top", "right", "bottom", "left")
    corner_names = ("top_left", "top_right", "bottom_right", "bottom_left")
    present_provenance_fields = [
        field for field in _SAMPLING_PROVENANCE_FIELDS if field in sampling
    ]
    if pipeline_version == _CURRENT_PIPELINE_VERSION:
        _validate_v3_sampling_provenance(sampling, recomputed)
    elif pipeline_version == _LEGACY_PIPELINE_VERSION:
        if present_provenance_fields or "side_interpolation_method" in sampling:
            raise ValueError(
                "manifest pipeline version 2 cannot contain v3 sampling provenance"
            )
        actual_fields = frozenset(str(field) for field in sampling.keys())
        if actual_fields != _V2_SAMPLING_FIELDS:
            missing = sorted(_V2_SAMPLING_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _V2_SAMPLING_FIELDS)
            raise ValueError(
                "manifest pipeline version 2 sampling schema mismatch; missing="
                f"{missing}, unexpected={unexpected}"
            )
        _validate_v2_sampling_metrics(sampling, recomputed)
        return
    else:
        raise ValueError(f"unsupported manifest pipeline_version: {pipeline_version!r}")
    if sampling.get("non_exterior_sample_pixels") != 0:
        raise ValueError("manifest sampling metrics report non-exterior sample pixels")
    side_coverage = _named_floats(
        sampling.get("side_coverage_fraction"),
        side_names,
        "sampling side coverage",
    )
    if min(side_coverage.values()) < 0.08:
        raise ValueError("manifest sampling side coverage is below the hard quality gate")
    corner_coverage = _named_floats(
        sampling.get("corner_coverage_fraction"),
        corner_names,
        "sampling corner coverage",
    )
    if min(corner_coverage.values()) < 0.12:
        raise ValueError("manifest sampling corner coverage is below the hard quality gate")
    band = int(recomputed.get("band_px", 0))
    if band <= 0:
        raise ValueError("recomputed sampling evidence does not contain a valid side band")
    side_counts = _named_nonnegative_integers(
        sampling.get("side_sample_pixels"),
        side_names,
        "sampling side pixel count",
    )
    recomputed_side_evidence = _mapping(
        recomputed.get("side_evidence"),
        "recomputed sampling side evidence",
    )
    for side in side_names:
        axis_length = source_width if side in {"top", "bottom"} else source_height
        denominator = axis_length * band
        declared_from_count = side_counts[side] / denominator
        _require_recomputed_float(
            side_coverage[side],
            declared_from_count,
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling side-coverage/count {side}",
        )
        side_recomputed = _mapping(
            recomputed_side_evidence.get(side),
            f"recomputed sampling side evidence {side}",
        )
        recomputed_coverage = _finite_float(
            side_recomputed.get("pixel_coverage_fraction"),
            f"recomputed sampling side coverage {side}",
        )
        _require_recomputed_float(
            side_coverage[side],
            recomputed_coverage,
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling side coverage {side}",
        )
        recomputed_count = int(round(recomputed_coverage * denominator))
        if side_counts[side] != recomputed_count:
            raise ValueError(
                f"manifest sampling side-pixel {side} counter does not match "
                "recomputed decoded-pixel evidence"
            )

    corner_counts = _named_nonnegative_integers(
        sampling.get("corner_sample_pixels"),
        corner_names,
        "sampling corner pixel count",
    )
    recomputed_corner_coverage = _named_floats(
        recomputed.get("corner_coverage_fraction"),
        corner_names,
        "recomputed sampling corner coverage",
    )
    corner_extent = max(band, round(min(source_width, source_height) * 0.08))
    corner_denominator = corner_extent * corner_extent
    for corner in corner_names:
        _require_recomputed_float(
            corner_coverage[corner],
            corner_counts[corner] / corner_denominator,
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling corner-coverage/count {corner}",
        )
        _require_recomputed_float(
            corner_coverage[corner],
            recomputed_corner_coverage[corner],
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling corner coverage {corner}",
        )
        recomputed_count = int(
            round(recomputed_corner_coverage[corner] * corner_denominator)
        )
        if corner_counts[corner] != recomputed_count:
            raise ValueError(
                f"manifest sampling corner-pixel {corner} counter does not match "
                "recomputed decoded-pixel evidence"
            )
    dispersion = _mapping(
        sampling.get("side_texture_dispersion"),
        "sampling side texture dispersion",
    )
    for side in side_names:
        side_evidence = _mapping(
            dispersion.get(side),
            f"sampling texture dispersion {side}",
        )
        bad_fraction = _finite_float(
            side_evidence.get("bad_coordinate_fraction"),
            f"sampling bad-coordinate fraction {side}",
        )
        bad_run = _finite_float(
            side_evidence.get("longest_bad_run_fraction"),
            f"sampling bad-run fraction {side}",
        )
        if bad_fraction > 0.15 or bad_run > 0.10:
            raise ValueError(
                f"manifest sampling texture metrics fail the hard {side} quality gate"
            )
        p90 = _finite_float(
            side_evidence.get("coordinate_p90_second_difference_linf_p90"),
            f"sampling texture p90 {side}",
        )
        recomputed_side = _mapping(
            recomputed_side_evidence.get(side),
            f"recomputed sampling texture evidence {side}",
        )
        axis_length = source_width if side in {"top", "bottom"} else source_height
        declared_bad_coordinate_count = bad_fraction * axis_length
        recomputed_bad_run = _finite_float(
            recomputed_side.get("longest_bad_run_fraction"),
            f"recomputed sampling bad-run fraction {side}",
        )
        declared_bad_run_count = bad_run * axis_length
        recomputed_bad_run_count = recomputed_bad_run * axis_length
        if (
            not math.isclose(
                declared_bad_coordinate_count,
                round(declared_bad_coordinate_count),
                rel_tol=0.0,
                abs_tol=_EXACT_RATIO_TOLERANCE,
            )
            or
            not math.isclose(
                declared_bad_run_count,
                round(declared_bad_run_count),
                rel_tol=0.0,
                abs_tol=_EXACT_RATIO_TOLERANCE,
            )
        ):
            raise ValueError(
                f"manifest sampling texture {side} fractions do not encode integer runs"
            )
        if (
            not math.isclose(
                recomputed_bad_run_count,
                round(recomputed_bad_run_count),
                rel_tol=0.0,
                abs_tol=_EXACT_RATIO_TOLERANCE,
            )
            or round(declared_bad_run_count) != round(recomputed_bad_run_count)
        ):
            raise ValueError(
                f"manifest sampling bad-run {side} integer counter does not match "
                "recomputed decoded-pixel evidence"
            )
        _require_recomputed_float(
            bad_fraction,
            _finite_float(
                recomputed_side.get("bad_coordinate_fraction"),
                f"recomputed sampling bad-coordinate fraction {side}",
            ),
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling bad-coordinate fraction {side}",
        )
        _require_recomputed_float(
            p90,
            _finite_float(
                recomputed_side.get(
                    "coordinate_p90_second_difference_linf_p90"
                ),
                f"recomputed sampling texture p90 {side}",
            ),
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling texture p90 {side}",
        )

    gate = _mapping(
        sampling.get("sampling_quality_gate"),
        "sampling quality gate",
    )
    expected_gate = {
        "coordinate_second_difference_linf_p90_limit": 24.0,
        "maximum_bad_coordinate_fraction": 0.15,
        "maximum_bad_run_fraction": 0.10,
        "minimum_side_coverage_fraction": 0.08,
        "minimum_corner_coverage_fraction": 0.12,
    }
    for field, expected_value in expected_gate.items():
        _require_recomputed_float(
            _finite_float(gate.get(field), f"sampling quality gate {field}"),
            expected_value,
            tolerance=_EXACT_RATIO_TOLERANCE,
            label=f"sampling quality-gate {field}",
        )
    for field in (
        "side_sample_pixels",
        "corner_sample_pixels",
        "side_coverage_fraction",
        "corner_coverage_fraction",
        "side_texture_dispersion",
        "side_model_residual_p90",
        "side_median_rgb",
        "corner_median_rgb",
        "corner_method",
        "sampling_quality_gate",
    ):
        _require_exact_typed_value(
            sampling.get(field),
            recomputed.get(field),
            label=f"v3 sampling metric {field}",
        )


def _recompute_v2_sampling_semantics(
    source_rgb: np.ndarray,
    exterior: np.ndarray,
    subject: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exactly replay the pipeline-v2 side-fallback sampler from decoded pixels."""

    height, width = exterior.shape
    band = max(4, round(min(width, height) * 0.06))
    sample_mask = np.zeros_like(exterior, dtype=bool)
    side_masks: dict[str, np.ndarray] = {}
    side_values: dict[str, np.ndarray] = {}
    side_counts: dict[str, int] = {}

    def horizontal(side: str) -> None:
        top = side == "top"
        values = np.empty((width, 3), dtype=np.float64)
        selected_mask = np.zeros_like(exterior, dtype=bool)
        count = 0
        for x in range(width):
            positions = (
                np.arange(0, band)
                if top
                else np.arange(height - band, height)
            )
            allowed = exterior[positions, x]
            if np.any(allowed):
                positions = positions[allowed]
            else:
                positions = np.flatnonzero(exterior[:, x])
                if not len(positions):
                    raise ValueError(
                        f"recomputed v2 {side} sampler has no exterior in column {x}"
                    )
                count_to_take = min(band, len(positions))
                positions = (
                    positions[:count_to_take]
                    if top
                    else positions[-count_to_take:]
                )
            values[x] = np.median(source_rgb[positions, x], axis=0)
            selected_mask[positions, x] = True
            sample_mask[positions, x] = True
            count += len(positions)
        side_values[side] = values
        side_masks[side] = selected_mask
        side_counts[side] = int(count)

    def vertical(side: str) -> None:
        left = side == "left"
        values = np.empty((height, 3), dtype=np.float64)
        selected_mask = np.zeros_like(exterior, dtype=bool)
        count = 0
        for y in range(height):
            positions = (
                np.arange(0, band)
                if left
                else np.arange(width - band, width)
            )
            allowed = exterior[y, positions]
            if np.any(allowed):
                positions = positions[allowed]
            else:
                positions = np.flatnonzero(exterior[y])
                if not len(positions):
                    raise ValueError(
                        f"recomputed v2 {side} sampler has no exterior in row {y}"
                    )
                count_to_take = min(band, len(positions))
                positions = (
                    positions[:count_to_take]
                    if left
                    else positions[-count_to_take:]
                )
            values[y] = np.median(source_rgb[y, positions], axis=0)
            selected_mask[y, positions] = True
            sample_mask[y, positions] = True
            count += len(positions)
        side_values[side] = values
        side_masks[side] = selected_mask
        side_counts[side] = int(count)

    horizontal("top")
    horizontal("bottom")
    vertical("left")
    vertical("right")

    corner_extent = max(band, round(min(width, height) * 0.08))
    corner_regions = {
        "top_left": (slice(0, corner_extent), slice(0, corner_extent)),
        "top_right": (
            slice(0, corner_extent),
            slice(width - corner_extent, width),
        ),
        "bottom_right": (
            slice(height - corner_extent, height),
            slice(width - corner_extent, width),
        ),
        "bottom_left": (
            slice(height - corner_extent, height),
            slice(0, corner_extent),
        ),
    }
    corner_counts: dict[str, int] = {}
    corner_coverage: dict[str, float] = {}
    corner_medians: dict[str, list[int]] = {}
    for corner, (ys, xs) in corner_regions.items():
        selected = exterior[ys, xs]
        pixels = source_rgb[ys, xs][selected]
        if len(pixels) < 3:
            raise ValueError(f"recomputed v2 {corner} lacks exterior samples")
        region = sample_mask[ys, xs]
        region[selected] = True
        sample_mask[ys, xs] = region
        corner_counts[corner] = int(selected.sum())
        corner_coverage[corner] = float(
            selected.sum() / (corner_extent * corner_extent)
        )
        corner_medians[corner] = [
            int(round(value)) for value in np.median(pixels.reshape(-1, 3), axis=0)
        ]

    coefficients = {
        side: _polyfit_robust(values) for side, values in side_values.items()
    }
    coverage: dict[str, float] = {}
    dispersion: dict[str, dict[str, float]] = {}
    residual: dict[str, float] = {}
    for side in ("top", "right", "bottom", "left"):
        horizontal_side = side in {"top", "bottom"}
        axis_length = width if horizontal_side else height
        coverage[side] = float(side_counts[side] / (axis_length * band))
        roughness = np.zeros(axis_length, dtype=np.float64)
        mask = side_masks[side]
        for coordinate in range(axis_length):
            if horizontal_side:
                positions = np.flatnonzero(mask[:, coordinate])
                pixels = source_rgb[positions, coordinate]
            else:
                positions = np.flatnonzero(mask[coordinate])
                pixels = source_rgb[coordinate, positions]
            if len(pixels) < 3:
                continue
            second_difference = np.diff(pixels.astype(np.float64), n=2, axis=0)
            high_frequency = np.max(np.abs(second_difference), axis=1)
            roughness[coordinate] = float(np.percentile(high_frequency, 90))
        bad = roughness > 24.0
        dispersion[side] = {
            "coordinate_p90_second_difference_linf_p90": float(
                np.percentile(roughness, 90)
            ),
            "bad_coordinate_fraction": float(np.mean(bad)),
            "longest_bad_run_fraction": float(
                _longest_true_run(bad) / axis_length
            ),
        }
        fitted = _evaluate(coefficients[side], axis_length)
        residual[side] = float(
            np.percentile(
                np.max(np.abs(side_values[side] - fitted), axis=1),
                90,
            )
        )
        if (
            coverage[side] < 0.08
            or dispersion[side]["bad_coordinate_fraction"] > 0.15
            or dispersion[side]["longest_bad_run_fraction"] > 0.10
        ):
            raise ValueError(
                f"recomputed v2 exterior sampling quality failed on the {side} side"
            )
    weak_corners = [
        corner for corner, value in corner_coverage.items() if value < 0.12
    ]
    if weak_corners:
        raise ValueError(
            "recomputed v2 exterior sampling corner coverage failed on "
            + ", ".join(weak_corners)
        )

    overlap = int(np.count_nonzero(sample_mask & subject))
    non_exterior = int(np.count_nonzero(sample_mask & ~exterior))
    if overlap or non_exterior:
        raise ValueError(
            "recomputed v2 sampling mask overlaps subject or non-exterior pixels"
        )
    return sample_mask, {
        "band_px": int(band),
        "region_policy": "exterior-only",
        "subject_overlap_pixels": overlap,
        "non_exterior_sample_pixels": non_exterior,
        "exterior_sample_pixels": int(sample_mask.sum()),
        "side_sample_pixels": side_counts,
        "corner_sample_pixels": corner_counts,
        "side_coverage_fraction": coverage,
        "corner_coverage_fraction": corner_coverage,
        "side_texture_dispersion": dispersion,
        "side_model_residual_p90": residual,
        "side_median_rgb": {
            side: [
                int(round(value))
                for value in np.median(side_values[side], axis=0)
            ]
            for side in ("top", "right", "bottom", "left")
        },
        "corner_median_rgb": corner_medians,
        "corner_method": "direct-exterior-only-source-patch-median",
        "sampling_quality_gate": {
            "status": "passed",
            "coordinate_second_difference_linf_p90_limit": 24.0,
            "maximum_bad_coordinate_fraction": 0.15,
            "maximum_bad_run_fraction": 0.10,
            "minimum_side_coverage_fraction": 0.08,
            "minimum_corner_coverage_fraction": 0.12,
        },
    }


def _recompute_same_side_interpolation_provenance(
    valid: np.ndarray,
    source_edge_contact: np.ndarray,
    *,
    side: str,
) -> tuple[int, int, int, dict[str, Any]]:
    """Replay the v3 same-side missing-coordinate gate from decoded masks."""

    axis_length = len(valid)
    missing = ~valid.astype(bool)
    missing_count = int(missing.sum())
    maximum_allowed_gap = max(12, round(axis_length * 0.10))
    exception: dict[str, Any] = {
        "status": "not-needed",
        "reason": None,
        "base_maximum_gap_px": maximum_allowed_gap,
        "missing_run_px": None,
        "source_edge_contact_overlap_px": 0,
        "allowed_center_zone_px": None,
        "bracketing_direct_sample_coordinates_px": None,
    }
    if not missing_count:
        return 0, 0, maximum_allowed_gap, exception
    valid_coordinates = np.flatnonzero(valid)
    missing_coordinates = np.flatnonzero(missing)
    if len(valid_coordinates) < 2:
        raise ValueError(
            f"recomputed {side} sampling lacks two same-side direct coordinates"
        )
    if (
        missing_coordinates[0] < valid_coordinates[0]
        or missing_coordinates[-1] > valid_coordinates[-1]
    ):
        raise ValueError(
            f"recomputed {side} sampling does not bracket every missing coordinate"
        )
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    maximum_gap = int(np.max(stops - starts)) if len(starts) else 0
    oversized = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start > maximum_allowed_gap
    ]
    if oversized:
        center_width = max(12, round(axis_length * 0.24))
        center_start = math.floor((axis_length - center_width) / 2)
        center_stop = center_start + center_width
        allowed: tuple[int, int] | None = None
        if len(oversized) == 1 and side in {"top", "bottom"}:
            start, stop = oversized[0]
            contact_overlap = int(source_edge_contact[start:stop].sum())
            directly_bracketed = (
                start > 0
                and stop < axis_length
                and bool(valid[start - 1])
                and bool(valid[stop])
            )
            centered = start >= center_start and stop <= center_stop
            if contact_overlap > 0 and directly_bracketed and centered:
                allowed = (start, stop)
                exception = {
                    "status": "allowed",
                    "reason": "centered-source-edge-ornament-contact",
                    "base_maximum_gap_px": maximum_allowed_gap,
                    "missing_run_px": [start, stop],
                    "source_edge_contact_overlap_px": contact_overlap,
                    "allowed_center_zone_px": [center_start, center_stop],
                    "bracketing_direct_sample_coordinates_px": [start - 1, stop],
                }
        if allowed is None:
            raise ValueError(
                f"recomputed {side} sampling interpolation gap exceeds its local evidence limit"
            )
    return missing_count, maximum_gap, maximum_allowed_gap, exception


def _recompute_sampling_semantics(
    source_rgb: np.ndarray,
    exterior: np.ndarray,
    subject: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Independently rebuild the sampler mask and its hard quality evidence."""

    height, width = exterior.shape
    band = max(4, round(min(width, height) * 0.06))
    side_regions = {
        "top": (slice(0, band), slice(0, width)),
        "right": (slice(0, height), slice(width - band, width)),
        "bottom": (slice(height - band, height), slice(0, width)),
        "left": (slice(0, height), slice(0, band)),
    }
    expected_mask = np.zeros_like(exterior, dtype=bool)
    side_evidence: dict[str, dict[str, float]] = {}
    side_values: dict[str, np.ndarray] = {}
    side_counts: dict[str, int] = {}
    side_residuals: dict[str, float] = {}
    interpolated_counts: dict[str, int] = {}
    maximum_gaps: dict[str, int] = {}
    maximum_allowed_gaps: dict[str, int] = {}
    interpolation_exceptions: dict[str, dict[str, Any]] = {}
    for side, (ys, xs) in side_regions.items():
        selected = exterior[ys, xs]
        region = expected_mask[ys, xs]
        region[selected] = True
        expected_mask[ys, xs] = region
        horizontal = side in {"top", "bottom"}
        axis_length = width if horizontal else height
        valid_coordinates = np.any(selected, axis=0 if horizontal else 1)
        medians = np.full((axis_length, 3), np.nan, dtype=np.float64)
        source_edge_contact = {
            "top": subject[0],
            "right": subject[:, -1],
            "bottom": subject[-1],
            "left": subject[:, 0],
        }[side]
        (
            interpolated_counts[side],
            maximum_gaps[side],
            maximum_allowed_gaps[side],
            interpolation_exceptions[side],
        ) = _recompute_same_side_interpolation_provenance(
            valid_coordinates,
            source_edge_contact,
            side=side,
        )
        samples_per_coordinate = np.sum(selected, axis=0 if horizontal else 1)
        coordinate_roughness = np.zeros(axis_length, dtype=np.float64)
        for coordinate in range(axis_length):
            if horizontal:
                positions = np.flatnonzero(selected[:, coordinate])
                pixels = source_rgb[ys, coordinate][positions]
            else:
                positions = np.flatnonzero(selected[coordinate])
                pixels = source_rgb[coordinate, xs][positions]
            if len(pixels):
                medians[coordinate] = np.median(pixels, axis=0)
            if len(pixels) < 3:
                continue
            second_difference = np.diff(pixels.astype(np.float64), n=2, axis=0)
            high_frequency = np.max(np.abs(second_difference), axis=1)
            coordinate_roughness[coordinate] = float(
                np.percentile(high_frequency, 90)
            )
        bad = coordinate_roughness > 24.0
        pixel_coverage = float(selected.sum() / (axis_length * band))
        coordinate_coverage = float(np.mean(valid_coordinates))
        mean_samples = float(
            np.mean(samples_per_coordinate[valid_coordinates])
            if np.any(valid_coordinates)
            else 0.0
        )
        bad_fraction = float(np.mean(bad))
        bad_run_fraction = float(_longest_true_run(bad) / axis_length)
        side_evidence[side] = {
            "pixel_coverage_fraction": pixel_coverage,
            "coordinate_coverage_fraction": coordinate_coverage,
            "mean_samples_per_valid_coordinate": mean_samples,
            "bad_coordinate_fraction": bad_fraction,
            "longest_bad_run_fraction": bad_run_fraction,
            "coordinate_p90_second_difference_linf_p90": float(
                np.percentile(coordinate_roughness, 90)
            ),
        }
        coordinate_axis = np.arange(axis_length, dtype=np.float64)
        for channel in range(3):
            medians[~valid_coordinates, channel] = np.interp(
                coordinate_axis[~valid_coordinates],
                coordinate_axis[valid_coordinates],
                medians[valid_coordinates, channel],
            )
        side_values[side] = medians
        side_counts[side] = int(selected.sum())
        coefficients = _polyfit_robust(medians)
        fitted = _evaluate(coefficients, axis_length)
        side_residuals[side] = float(
            np.percentile(np.max(np.abs(medians - fitted), axis=1), 90)
        )
        has_dense_pixel_support = pixel_coverage >= 0.08
        has_compatible_coordinate_support = bool(
            coordinate_coverage >= 0.80
            and mean_samples >= max(3.0, band * 0.07)
        )
        if (
            not (has_dense_pixel_support or has_compatible_coordinate_support)
            or bad_fraction > 0.15
            or bad_run_fraction > 0.10
        ):
            raise ValueError(
                f"recomputed exterior sampling quality failed on the {side} side"
            )

    corner_extent = max(band, round(min(width, height) * 0.08))
    corner_regions = {
        "top_left": (slice(0, corner_extent), slice(0, corner_extent)),
        "top_right": (
            slice(0, corner_extent),
            slice(width - corner_extent, width),
        ),
        "bottom_right": (
            slice(height - corner_extent, height),
            slice(width - corner_extent, width),
        ),
        "bottom_left": (
            slice(height - corner_extent, height),
            slice(0, corner_extent),
        ),
    }
    corner_coverage: dict[str, float] = {}
    corner_counts: dict[str, int] = {}
    corner_medians: dict[str, list[int]] = {}
    for corner, (ys, xs) in corner_regions.items():
        selected = exterior[ys, xs]
        region = expected_mask[ys, xs]
        region[selected] = True
        expected_mask[ys, xs] = region
        coverage = float(selected.sum() / (corner_extent * corner_extent))
        corner_coverage[corner] = coverage
        corner_counts[corner] = int(selected.sum())
        pixels = source_rgb[ys, xs][selected]
        if len(pixels) < 3:
            raise ValueError(f"recomputed {corner} lacks exterior samples")
        corner_medians[corner] = [
            int(round(value)) for value in np.median(pixels.reshape(-1, 3), axis=0)
        ]
        if coverage < 0.12:
            raise ValueError(
                f"recomputed exterior sampling corner coverage failed on {corner}"
            )

    overlap = int(np.count_nonzero(expected_mask & subject))
    non_exterior = int(np.count_nonzero(expected_mask & ~exterior))
    if overlap or non_exterior:
        raise ValueError(
            "recomputed exterior sampling mask overlaps subject or non-exterior pixels"
        )
    return expected_mask, {
        "band_px": int(band),
        "side_evidence": side_evidence,
        "side_sample_pixels": side_counts,
        "side_coverage_fraction": {
            side: side_evidence[side]["pixel_coverage_fraction"]
            for side in ("top", "right", "bottom", "left")
        },
        "side_texture_dispersion": {
            side: {
                field: side_evidence[side][field]
                for field in (
                    "coordinate_p90_second_difference_linf_p90",
                    "bad_coordinate_fraction",
                    "longest_bad_run_fraction",
                )
            }
            for side in ("top", "right", "bottom", "left")
        },
        "corner_sample_pixels": corner_counts,
        "corner_coverage_fraction": corner_coverage,
        "side_model_residual_p90": side_residuals,
        "side_median_rgb": {
            side: [
                int(round(value))
                for value in np.median(side_values[side], axis=0)
            ]
            for side in ("top", "right", "bottom", "left")
        },
        "corner_median_rgb": corner_medians,
        "corner_method": "direct-exterior-only-source-patch-median",
        "sampling_quality_gate": {
            "status": "passed",
            "coordinate_second_difference_linf_p90_limit": 24.0,
            "maximum_bad_coordinate_fraction": 0.15,
            "maximum_bad_run_fraction": 0.10,
            "minimum_side_coverage_fraction": 0.08,
            "minimum_corner_coverage_fraction": 0.12,
        },
        "subject_overlap_pixels": overlap,
        "non_exterior_sample_pixels": non_exterior,
        "interpolation_provenance": {
            "side_interpolated_coordinate_count": interpolated_counts,
            "side_interpolation_maximum_gap_px": maximum_gaps,
            "side_interpolation_maximum_allowed_gap_px": maximum_allowed_gaps,
            "side_interpolation_exceptions": interpolation_exceptions,
        },
    }


def _recompute_reviewed_flat_sampling_semantics(
    source_rgb: np.ndarray,
    exterior: np.ndarray,
    subject: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Independently replay the reviewed continuous exterior-color sampler."""

    if exterior.shape != source_rgb.shape[:2] or subject.shape != source_rgb.shape[:2]:
        raise ValueError("reviewed flat exterior masks do not match the decoded source")
    if np.any(exterior & subject):
        raise ValueError("reviewed flat exterior sampling overlaps the protected subject")
    expected_mask = exterior.astype(bool, copy=True)
    sample_count = int(expected_mask.sum())
    minimum_samples = max(16, round(expected_mask.size * 0.005))
    if sample_count < minimum_samples:
        raise ValueError("recomputed reviewed flat exterior sampling is insufficient")

    flat_color = np.rint(np.median(source_rgb[expected_mask], axis=0)).astype(np.uint8)
    height, width = exterior.shape
    band = max(4, round(min(width, height) * 0.06))
    corner_extent = max(band, round(min(width, height) * 0.08))
    side_regions = {
        "top": (slice(0, band), slice(0, width)),
        "right": (slice(0, height), slice(width - band, width)),
        "bottom": (slice(height - band, height), slice(0, width)),
        "left": (slice(0, height), slice(0, band)),
    }
    corner_regions = {
        "top_left": (slice(0, corner_extent), slice(0, corner_extent)),
        "top_right": (
            slice(0, corner_extent),
            slice(width - corner_extent, width),
        ),
        "bottom_right": (
            slice(height - corner_extent, height),
            slice(width - corner_extent, width),
        ),
        "bottom_left": (
            slice(height - corner_extent, height),
            slice(0, corner_extent),
        ),
    }
    side_counts: dict[str, int] = {}
    side_coverage: dict[str, float] = {}
    side_medians: dict[str, list[int]] = {}
    for side, (ys, xs) in side_regions.items():
        selected = expected_mask[ys, xs]
        count = int(selected.sum())
        if count < 3:
            raise ValueError(
                f"recomputed reviewed flat exterior lacks {side} source samples"
            )
        side_counts[side] = count
        side_coverage[side] = float(count / selected.size)
        side_medians[side] = [
            int(round(value))
            for value in np.median(source_rgb[ys, xs][selected], axis=0)
        ]

    corner_counts: dict[str, int] = {}
    corner_coverage: dict[str, float] = {}
    corner_medians: dict[str, list[int]] = {}
    for corner, (ys, xs) in corner_regions.items():
        selected = expected_mask[ys, xs]
        count = int(selected.sum())
        if count < 3:
            raise ValueError(
                f"recomputed reviewed flat exterior lacks {corner} source samples"
            )
        corner_counts[corner] = count
        corner_coverage[corner] = float(count / selected.size)
        corner_medians[corner] = [
            int(round(value))
            for value in np.median(source_rgb[ys, xs][selected], axis=0)
        ]

    zeros_by_side = {side: 0 for side in side_regions}
    not_needed_by_side = {
        side: {
            "status": "not-needed",
            "reason": "reviewed-flat-exterior-median",
            "base_maximum_gap_px": 0,
            "missing_run_px": None,
            "source_edge_contact_overlap_px": 0,
            "allowed_center_zone_px": None,
            "bracketing_direct_sample_coordinates_px": None,
        }
        for side in side_regions
    }
    return expected_mask, {
        "sampling_method": "reviewed-flat-exterior-median-v1",
        "region_policy": "exterior-only",
        "flat_background_rgb": flat_color.tolist(),
        "flat_color_statistic": "per-channel-median-of-boundary-connected-exterior",
        "source_edge_extension": "forbidden",
        "subject_overlap_pixels": int(np.count_nonzero(expected_mask & subject)),
        "non_exterior_sample_pixels": int(np.count_nonzero(expected_mask & ~exterior)),
        "exterior_sample_pixels": sample_count,
        "band_px": int(band),
        "side_sample_pixels": side_counts,
        "corner_sample_pixels": corner_counts,
        "side_coverage_fraction": side_coverage,
        "corner_coverage_fraction": corner_coverage,
        "side_texture_dispersion": {
            side: {"used_for_flat_model": False} for side in side_regions
        },
        "side_model_residual_p90": {side: None for side in side_regions},
        "side_median_rgb": side_medians,
        "corner_median_rgb": corner_medians,
        "corner_method": "direct-exterior-only-source-patch-median",
        "sampling_quality_gate": {
            "status": "passed",
            "mode": "reviewed-flat-exterior-median",
            "minimum_total_exterior_sample_pixels": int(minimum_samples),
            "all_four_sides_sampled": True,
            "all_four_corners_sampled": True,
        },
        "interpolation_provenance": {
            "side_interpolated_coordinate_count": dict(zeros_by_side),
            "side_interpolation_maximum_gap_px": dict(zeros_by_side),
            "side_interpolation_maximum_allowed_gap_px": dict(zeros_by_side),
            "side_interpolation_exceptions": not_needed_by_side,
        },
    }


def _validate_artifact_aliases(
    manifest: Mapping[str, Any],
    manifest_directory: Path,
    artifact_paths: Mapping[str, Path],
) -> None:
    aliases = {
        "subject_image": "subject_rgba",
        "subject_mask": "subject_mask",
        "exterior_sampling_mask": "exterior_sampling_mask",
        "sampling_provenance_artifact": "sampling_provenance",
        "bleed_background_image": "bleed_background_rgb",
        "media_image": "media_rgb",
    }
    for field, artifact_name in aliases.items():
        if field not in manifest:
            continue
        alias_path = _manifest_file(manifest.get(field), manifest_directory, field)
        if alias_path != artifact_paths[artifact_name]:
            raise ValueError(
                f"manifest {field} path does not match the approved {artifact_name} artifact"
            )


def _validate_artifact_hash_links(
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    subject_extraction = _mapping(manifest.get("subject_extraction"), "subject_extraction")
    linked_hashes = [
        (
            subject_extraction,
            "subject_rgba_sha256",
            "subject_rgba",
            "subject extraction RGBA",
        ),
        (
            subject_extraction,
            "subject_mask_sha256",
            "subject_mask",
            "subject extraction mask",
        ),
        (
            _mapping(manifest.get("background_sampling"), "background_sampling"),
            "sampling_mask_sha256",
            "exterior_sampling_mask",
            "background sampling mask",
        ),
        (manifest, "bleed_background_sha256", "bleed_background_rgb", "bleed background"),
        (manifest, "media_sha256", "media_rgb", "media image"),
    ]
    if "sampling_provenance" in artifacts:
        linked_hashes.append(
            (
                manifest,
                "sampling_provenance_sha256",
                "sampling_provenance",
                "sampling provenance",
            )
        )
    for owner, field, artifact_name, label in linked_hashes:
        declared = _declared_sha256(owner.get(field), label)
        artifact = _mapping(artifacts.get(artifact_name), f"artifact {artifact_name}")
        artifact_hash = _declared_sha256(
            artifact.get("sha256"),
            f"artifact {artifact_name}",
        )
        if declared != artifact_hash:
            raise ValueError(f"manifest {label} hash does not match its approved artifact")


def _validate_decoded_artifacts(
    manifest: Mapping[str, Any],
    source_path: Path,
    artifact_paths: Mapping[str, Path],
    *,
    trim_width_mm: float,
    trim_height_mm: float,
    bleed: Mapping[str, float],
    contract_evidence: Mapping[str, Any],
) -> None:
    """Recompute the subject-first proof from decoded pixels, not self-reports."""

    source_rgb = _decoded_source_rgb(source_path)
    source_height, source_width = source_rgb.shape[:2]
    declared_source_size = _integer_sequence(
        manifest.get("source_size_px"),
        2,
        "source_size_px pair",
    )
    if declared_source_size != (source_width, source_height):
        raise ValueError("manifest source dimensions do not match the decoded source image")

    x0, y0, x1, y1 = _integer_sequence(
        manifest.get("subject_bbox_px"),
        4,
        "subject_bbox_px",
    )
    if not (0 <= x0 < x1 <= source_width and 0 <= y0 < y1 <= source_height):
        raise ValueError("manifest subject bbox is outside the decoded source dimensions")
    bbox = [x0, y0, x1, y1]
    for field in ("source_crop_bbox_px",):
        if field in manifest and list(_integer_sequence(manifest.get(field), 4, field)) != bbox:
            raise ValueError(f"manifest {field} does not match the subject bbox")
    extraction = _mapping(manifest.get("subject_extraction"), "subject_extraction")
    if list(_integer_sequence(extraction.get("subject_bbox_px"), 4, "extraction bbox")) != bbox:
        raise ValueError("manifest subject extraction bbox does not match the subject bbox")

    subject_rgba = _decoded_image(
        artifact_paths["subject_rgba"],
        "RGBA",
        "subject RGBA artifact",
    )
    subject_mask = _decoded_image(
        artifact_paths["subject_mask"],
        "L",
        "subject mask artifact",
    )
    subject_height = y1 - y0
    subject_width = x1 - x0
    if subject_rgba.shape != (subject_height, subject_width, 4):
        raise ValueError("subject RGBA dimensions do not match the approved subject bbox")
    if subject_mask.shape != (subject_height, subject_width):
        raise ValueError("subject mask dimensions do not match the approved subject bbox")
    declared_subject_size = _integer_sequence(
        manifest.get("subject_size_px"),
        2,
        "subject_size_px pair",
    )
    if declared_subject_size != (subject_width, subject_height):
        raise ValueError("manifest subject dimensions do not match the decoded artifacts")

    mask_values = np.unique(subject_mask)
    if not np.all(np.isin(mask_values, np.asarray((0, 255), dtype=np.uint8))):
        raise ValueError("decoded artifact is not a binary subject mask")
    subject_pixels = subject_mask == 255
    if not np.any(subject_pixels):
        raise ValueError("decoded binary subject mask is empty")
    if _binary_bbox(subject_pixels) != [0, 0, subject_width, subject_height]:
        raise ValueError("decoded subject mask does not agree with the approved subject bbox")

    alpha = subject_rgba[..., 3]
    if not np.array_equal(alpha > 0, subject_pixels):
        raise ValueError("subject RGBA alpha coverage does not match the binary subject mask")
    if np.any(subject_rgba[..., :3][~subject_pixels] != 0):
        raise ValueError("subject RGB contains pixels outside the binary subject mask")

    source_crop = source_rgb[y0:y1, x0:x1]
    if not np.array_equal(
        subject_rgba[..., :3][subject_pixels],
        source_crop[subject_pixels],
    ):
        raise ValueError("subject RGB pixels do not match the decoded source image")

    opaque = alpha == 255
    actual_opaque_pixels = int(opaque.sum())
    if manifest.get("opaque_subject_pixels") != actual_opaque_pixels:
        raise ValueError("manifest opaque subject pixel count does not match subject RGBA")
    actual_semitransparent_pixels = int(((alpha > 0) & (alpha < 255)).sum())
    if manifest.get("semi_transparent_edge_pixels") != actual_semitransparent_pixels:
        raise ValueError("manifest semi-transparent subject pixel count is not reproducible")

    sampling_mask = _decoded_image(
        artifact_paths["exterior_sampling_mask"],
        "L",
        "exterior sampling mask artifact",
    )
    if sampling_mask.shape != (source_height, source_width):
        raise ValueError("exterior sampling mask dimensions do not match the source image")
    sampling_values = np.unique(sampling_mask)
    if not np.all(np.isin(sampling_values, np.asarray((0, 255), dtype=np.uint8))):
        raise ValueError("decoded exterior sampling mask is not binary")
    sampling_pixels = sampling_mask == 255
    sampling = _mapping(manifest.get("background_sampling"), "background_sampling")
    actual_sampling_pixels = int(sampling_pixels.sum())
    if actual_sampling_pixels <= 0:
        raise ValueError("decoded exterior sampling mask is empty")
    if sampling.get("exterior_sample_pixels") != actual_sampling_pixels:
        raise ValueError("exterior sampling mask pixel count does not match the manifest")
    if "sampling_mask_bbox_px" in sampling:
        sampling_bbox = _binary_bbox(sampling_pixels)
        declared_sampling_bbox = list(
            _integer_sequence(
                sampling.get("sampling_mask_bbox_px"),
                4,
                "sampling_mask_bbox_px",
            )
        )
        if sampling_bbox != declared_sampling_bbox:
            raise ValueError("exterior sampling mask bbox does not match the manifest")

    source_subject = np.zeros((source_height, source_width), dtype=bool)
    source_subject[y0:y1, x0:x1] = subject_pixels
    try:
        recomputed_geometry = _frame_geometry_evidence(source_rgb, source_subject)
    except DetectionError as exc:
        raise ValueError(
            "recomputed frame geometry evidence did not pass on decoded pixels"
        ) from exc
    if recomputed_geometry.get("passed") is not True:
        raise ValueError(
            "recomputed frame geometry evidence did not pass on decoded pixels"
        )
    frame_detection = _mapping(manifest.get("frame_detection"), "frame_detection")
    geometry_for_counter_comparison: Mapping[str, Any] = recomputed_geometry
    double_layer_tolerance = _EXACT_RATIO_TOLERANCE
    if frame_detection.get("method") == (
        "continuous-dark-frame-plus-rounded-geometric-fill"
    ):
        try:
            _, independently_detected = _detect_dark_closed_frame(source_rgb)
        except DetectionError as exc:
            raise ValueError(
                "independent dark-frame detection failed while recomputing geometry counters"
            ) from exc
        geometry_for_counter_comparison = _mapping(
            independently_detected.get("geometry_evidence"),
            "independently recomputed dark-frame geometry evidence",
        )
        double_layer_tolerance = _GEOMETRY_DOUBLE_LAYER_TOLERANCE
    _validate_declared_geometry_metrics(
        manifest,
        geometry_for_counter_comparison,
        double_layer_tolerance=double_layer_tolerance,
    )

    actual_overlap = int(np.count_nonzero(sampling_pixels & source_subject))
    if actual_overlap != 0:
        raise ValueError("decoded exterior sampling mask overlaps the protected subject")
    if sampling.get("subject_overlap_pixels") != actual_overlap:
        raise ValueError("exterior sampling mask overlap count does not match the manifest")
    recomputed_exterior = _boundary_connected(~source_subject)
    actual_non_exterior = int(
        np.count_nonzero(sampling_pixels & ~recomputed_exterior)
    )
    if actual_non_exterior != 0:
        raise ValueError(
            "decoded sampling mask contains pixels outside boundary-connected exterior"
        )
    if sampling.get("non_exterior_sample_pixels") != actual_non_exterior:
        raise ValueError(
            "decoded sampling non-exterior count does not match the manifest"
        )
    pipeline_version = int(contract_evidence["pipeline_version"])
    reviewed_flat = sampling.get("sampling_method") == (
        "reviewed-flat-exterior-median-v1"
    )
    if pipeline_version == _LEGACY_PIPELINE_VERSION:
        expected_sampling_mask, recomputed_sampling = (
            _recompute_v2_sampling_semantics(
                source_rgb,
                recomputed_exterior,
                source_subject,
            )
        )
    elif reviewed_flat:
        expected_sampling_mask, recomputed_sampling = (
            _recompute_reviewed_flat_sampling_semantics(
                source_rgb,
                recomputed_exterior,
                source_subject,
            )
        )
    else:
        expected_sampling_mask, recomputed_sampling = _recompute_sampling_semantics(
            source_rgb,
            recomputed_exterior,
            source_subject,
        )
    if not np.array_equal(sampling_pixels, expected_sampling_mask):
        raise ValueError(
            "decoded sampling mask does not match recomputed exterior-only sampling"
        )
    if pipeline_version == _CURRENT_PIPELINE_VERSION:
        decoded_provenance = _decoded_json_mapping(
            artifact_paths["sampling_provenance"],
            "sampling provenance artifact",
        )
        expected_provenance = _sampling_provenance_artifact_payload(
            source_sha256=_file_sha256(source_path, "source"),
            sampling_mask_sha256=_file_sha256(
                artifact_paths["exterior_sampling_mask"],
                "exterior sampling mask",
            ),
            sampling=dict(sampling),
        )
        _require_exact_typed_value(
            dict(decoded_provenance),
            expected_provenance,
            label="sampling provenance artifact",
        )
    _validate_declared_sampling_metrics(
        sampling,
        recomputed_sampling,
        source_width=source_width,
        source_height=source_height,
        pipeline_version=pipeline_version,
    )

    media_width, media_height = _integer_sequence(
        manifest.get("media_size_px"),
        2,
        "media_size_px pair",
    )
    if media_width <= 0 or media_height <= 0:
        raise ValueError("manifest media dimensions must be positive integers")
    background_rgb = _decoded_image(
        artifact_paths["bleed_background_rgb"],
        "RGB",
        "bleed background artifact",
    )
    media_rgb = _decoded_image(
        artifact_paths["media_rgb"],
        "RGB",
        "media RGB artifact",
    )
    expected_media_shape = (media_height, media_width, 3)
    if background_rgb.shape != expected_media_shape:
        raise ValueError("bleed background dimensions do not match the manifest media dimensions")
    if media_rgb.shape != expected_media_shape:
        raise ValueError("media dimensions do not match the manifest media dimensions")
    if reviewed_flat:
        flat_color = np.asarray(
            recomputed_sampling["flat_background_rgb"],
            dtype=np.uint8,
        )
        if not np.all(background_rgb == flat_color[None, None, :]):
            raise ValueError(
                "reviewed flat bleed background is not uniform or does not match "
                "the recomputed current-card exterior color"
            )

    media_values = _sequence(manifest.get("media_mm"), 2, "media_mm pair")
    media_width_mm = _positive_float(media_values[0], "media width")
    media_height_mm = _positive_float(media_values[1], "media height")
    pixels_per_mm_x = media_width / media_width_mm
    pixels_per_mm_y = media_height / media_height_mm
    if not math.isclose(
        pixels_per_mm_x,
        pixels_per_mm_y,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("media pixel dimensions do not preserve the declared physical aspect ratio")
    pixels_per_mm = (pixels_per_mm_x + pixels_per_mm_y) / 2.0
    recomputed_effective_ppi = pixels_per_mm * 25.4
    declared_effective_ppi = _positive_float(
        manifest.get("effective_ppi"),
        "effective PPI",
    )
    minimum_effective_ppi = _positive_float(
        manifest.get("minimum_effective_ppi"),
        "minimum effective PPI",
    )
    if minimum_effective_ppi < 300.0:
        raise ValueError(
            "manifest minimum effective PPI is below the Skill hard floor of 300"
        )
    if not math.isclose(
        declared_effective_ppi,
        recomputed_effective_ppi,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "manifest effective PPI does not match the decoded media raster and media_mm"
        )
    if recomputed_effective_ppi + 1e-6 < minimum_effective_ppi:
        raise ValueError(
            "decoded media effective PPI is below the manifest minimum effective PPI"
        )

    placement_x, placement_y = _integer_sequence(
        manifest.get("placement_px"),
        2,
        "placement_px pair",
    )
    if not (
        0 <= placement_x <= media_width - subject_width
        and 0 <= placement_y <= media_height - subject_height
    ):
        raise ValueError("subject placement is outside the decoded media dimensions")

    placement = _mapping(manifest.get("subject_placement_mm"), "subject_placement_mm")
    expected_placement_mm = {
        "x_mm": placement_x / pixels_per_mm,
        "y_mm": placement_y / pixels_per_mm,
        "width_mm": subject_width / pixels_per_mm,
        "height_mm": subject_height / pixels_per_mm,
    }
    half_pixel_mm = 0.5 / pixels_per_mm + _GEOMETRY_TOLERANCE_MM
    for field, expected_value in expected_placement_mm.items():
        declared_value = _finite_float(placement.get(field), f"subject {field}")
        if not math.isclose(
            declared_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=half_pixel_mm,
        ):
            raise ValueError(f"subject placement {field} does not match placement_px")
    trim_box = _sequence(placement.get("trim_box_mm"), 4, "subject trim_box_mm")
    expected_trim_box = (
        bleed["left"],
        bleed["top"],
        bleed["left"] + trim_width_mm,
        bleed["top"] + trim_height_mm,
    )
    if any(
        not math.isclose(
            _finite_float(actual, "subject trim box coordinate"),
            expected,
            rel_tol=0.0,
            abs_tol=_GEOMETRY_TOLERANCE_MM,
        )
        for actual, expected in zip(trim_box, expected_trim_box)
    ):
        raise ValueError("subject placement trim box does not match trim and bleed geometry")
    if not math.isclose(
        _positive_float(placement.get("scale_x"), "subject scale_x"),
        1.0,
        rel_tol=0.0,
        abs_tol=_SCALE_TOLERANCE,
    ) or not math.isclose(
        _positive_float(placement.get("scale_y"), "subject scale_y"),
        1.0,
        rel_tol=0.0,
        abs_tol=_SCALE_TOLERANCE,
    ):
        raise ValueError("manifest no-resample placement does not use unit pixel scale")

    destination = media_rgb[
        placement_y : placement_y + subject_height,
        placement_x : placement_x + subject_width,
    ]
    subject_rgb = subject_rgba[..., :3]
    exact_opaque_pixels = int(np.all(destination[opaque] == subject_rgb[opaque], axis=1).sum())
    if exact_opaque_pixels != actual_opaque_pixels:
        raise ValueError("media does not preserve all exact opaque subject pixels")
    if manifest.get("opaque_subject_exact_pixels") != exact_opaque_pixels:
        raise ValueError("manifest exact opaque subject pixels are not reproducible")

    expected_media = background_rgb.copy()
    expected_destination = expected_media[
        placement_y : placement_y + subject_height,
        placement_x : placement_x + subject_width,
    ]
    alpha_float = alpha.astype(np.float32)[..., None] / 255.0
    composite = np.rint(
        subject_rgb.astype(np.float32) * alpha_float
        + expected_destination.astype(np.float32) * (1.0 - alpha_float)
    )
    expected_destination[:] = np.clip(composite, 0, 255).astype(np.uint8)
    expected_destination[opaque] = subject_rgb[opaque]
    if not np.array_equal(media_rgb, expected_media):
        raise ValueError(
            "decoded media RGB is not the approved bleed background plus placed subject"
        )


def _validate_pipeline_stages(manifest: Mapping[str, Any]) -> None:
    stages = manifest.get("pipeline_stages")
    if not isinstance(stages, list):
        raise ValueError("manifest does not prove the ordered subject-first pipeline stages")
    names: list[object] = []
    statuses: list[object] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("manifest contains an invalid subject-first pipeline stage")
        names.append(stage.get("name"))
        statuses.append(stage.get("status"))
    if names != list(_PIPELINE_STAGES):
        raise ValueError("manifest subject-first pipeline stages are missing or out of order")
    if any(status != "completed" for status in statuses):
        raise ValueError("manifest subject-first pipeline stages are not all completed")


def _validate_geometry(manifest: Mapping[str, Any]) -> tuple[float, float, dict[str, float]]:
    trim_values = _sequence(manifest.get("trim_mm"), 2, "trim_mm pair")
    media_values = _sequence(manifest.get("media_mm"), 2, "media_mm pair")
    bleed_values = _mapping(manifest.get("bleed_mm"), "bleed_mm")

    trim_width_mm = _positive_float(trim_values[0], "trim width")
    trim_height_mm = _positive_float(trim_values[1], "trim height")
    media_width_mm = _positive_float(media_values[0], "media width")
    media_height_mm = _positive_float(media_values[1], "media height")
    bleed = {
        side: _positive_float(bleed_values.get(side), f"{side} bleed")
        for side in ("left", "right", "top", "bottom")
    }

    expected_media_width = trim_width_mm + bleed["left"] + bleed["right"]
    expected_media_height = trim_height_mm + bleed["top"] + bleed["bottom"]
    if not math.isclose(
        media_width_mm,
        expected_media_width,
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE_MM,
    ) or not math.isclose(
        media_height_mm,
        expected_media_height,
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE_MM,
    ):
        raise ValueError("manifest media geometry does not equal trim plus all four fixed bleeds")

    placement = _mapping(manifest.get("subject_placement_mm"), "subject_placement_mm")
    if placement.get("cropped") is not False:
        raise ValueError("manifest does not prove uncropped subject placement")
    scale_x = _positive_float(placement.get("scale_x"), "subject scale_x")
    scale_y = _positive_float(placement.get("scale_y"), "subject scale_y")
    if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=_SCALE_TOLERANCE):
        raise ValueError("manifest subject placement uses unequal horizontal and vertical scale")

    return trim_width_mm, trim_height_mm, bleed


def _validate_subject_first_manifest(
    manifest: Mapping[str, Any],
    manifest_directory: Path,
) -> tuple[Path, float, float, dict[str, float]]:
    contract_evidence = _manifest_contract_evidence(manifest)
    fixed_provenance_sibling = (
        manifest_directory.resolve() / "sampling-provenance.json"
    )
    if (
        contract_evidence["pipeline_version"] == _LEGACY_PIPELINE_VERSION
        and fixed_provenance_sibling.is_file()
    ):
        raise ValueError(
            "manifest pipeline version 2 has a v3 sampling provenance sibling"
        )
    if manifest.get("resampled") is not False:
        raise ValueError("manifest does not prove a no-resample preparation")

    opaque_pixels = manifest.get("opaque_subject_pixels")
    exact_pixels = manifest.get("opaque_subject_exact_pixels")
    if (
        isinstance(opaque_pixels, bool)
        or not isinstance(opaque_pixels, int)
        or opaque_pixels <= 0
        or exact_pixels != opaque_pixels
    ):
        raise ValueError("manifest does not prove exact fully opaque subject pixels")

    _validate_pipeline_stages(manifest)
    trim_width_mm, trim_height_mm, bleed = _validate_geometry(manifest)

    frame_detection = _mapping(manifest.get("frame_detection"), "frame_detection")
    geometry_evidence = _mapping(
        frame_detection.get("geometry_evidence"),
        "frame geometry evidence",
    )
    if geometry_evidence.get("passed") is not True:
        raise ValueError("manifest frame geometry evidence did not pass")

    source_sha256 = _declared_sha256(manifest.get("source_sha256"), "source")
    source_path = _manifest_file(manifest.get("source"), manifest_directory, "source")
    if _file_sha256(source_path, "source") != source_sha256:
        raise ValueError("source image hash no longer matches the approved manifest")

    subject_extraction = _mapping(manifest.get("subject_extraction"), "subject_extraction")
    extraction_source = _declared_sha256(
        subject_extraction.get("source_sha256"),
        "subject extraction source",
    )
    if extraction_source != source_sha256:
        raise ValueError("subject extraction is tied to a different source image")

    sampling = _mapping(manifest.get("background_sampling"), "background_sampling")
    sampling_source = _declared_sha256(
        sampling.get("source_sha256"),
        "background sampling source",
    )
    if sampling_source != source_sha256:
        raise ValueError("background sampling is tied to a different source image")
    if sampling.get("region_policy") != "exterior-only":
        raise ValueError("background sampling policy is not exterior-only")
    if sampling.get("subject_overlap_pixels") != 0:
        raise ValueError("background sampling overlaps the protected subject")
    sampling_quality_gate = _mapping(
        sampling.get("sampling_quality_gate"),
        "sampling quality gate",
    )
    if sampling_quality_gate.get("status") != "passed":
        raise ValueError("manifest sampling quality gate did not pass")
    sample_pixels = sampling.get("exterior_sample_pixels")
    if isinstance(sample_pixels, bool) or not isinstance(sample_pixels, int) or sample_pixels <= 0:
        raise ValueError("manifest does not prove any exterior-only sampling pixels")

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    pipeline_version = int(contract_evidence["pipeline_version"])
    required_artifacts = _REQUIRED_ARTIFACTS + (
        ("sampling_provenance",)
        if pipeline_version == _CURRENT_PIPELINE_VERSION
        else ()
    )
    if (
        pipeline_version == _LEGACY_PIPELINE_VERSION
        and (
            "sampling_provenance" in artifacts
            or "sampling_provenance_artifact" in manifest
            or "sampling_provenance_sha256" in manifest
        )
    ):
        raise ValueError(
            "manifest pipeline version 2 cannot contain a v3 sampling provenance artifact"
        )
    missing_artifacts = [name for name in required_artifacts if name not in artifacts]
    if missing_artifacts:
        raise ValueError(
            "manifest is missing required independent artifacts: " + ", ".join(missing_artifacts)
        )

    artifact_paths: dict[str, Path] = {}
    for name, value in artifacts.items():
        artifact = _mapping(value, f"artifact {name}")
        path = _manifest_file(artifact.get("path"), manifest_directory, f"artifact {name}")
        declared_hash = _declared_sha256(artifact.get("sha256"), f"artifact {name}")
        if _file_sha256(path, f"artifact {name}") != declared_hash:
            raise ValueError(f"artifact hash no longer matches the approved manifest: {name}")
        artifact_source = _declared_sha256(
            artifact.get("source_sha256"),
            f"artifact {name} source",
        )
        if artifact_source != source_sha256:
            raise ValueError(f"artifact is tied to a different source image: {name}")
        artifact_paths[str(name)] = path

    if len(set(artifact_paths.values())) != len(artifact_paths):
        raise ValueError("manifest does not keep its approved artifacts as separate files")
    if (
        pipeline_version == _CURRENT_PIPELINE_VERSION
        and artifact_paths["sampling_provenance"] != fixed_provenance_sibling
    ):
        raise ValueError(
            "manifest v3 sampling provenance artifact is not its fixed sibling"
        )

    _validate_artifact_aliases(manifest, manifest_directory, artifact_paths)
    _validate_artifact_hash_links(manifest, artifacts)
    _validate_decoded_artifacts(
        manifest,
        source_path,
        artifact_paths,
        trim_width_mm=trim_width_mm,
        trim_height_mm=trim_height_mm,
        bleed=bleed,
        contract_evidence=contract_evidence,
    )

    media_path = artifact_paths["media_rgb"]
    if "media_image" in manifest:
        declared_media_path = _manifest_file(
            manifest.get("media_image"),
            manifest_directory,
            "media image",
        )
        if declared_media_path != media_path:
            raise ValueError("media image path does not match the approved media_rgb artifact")
    if "media_sha256" in manifest:
        media_sha256 = _declared_sha256(manifest.get("media_sha256"), "media image")
        if media_sha256 != _file_sha256(media_path, "media image"):
            raise ValueError("media image hash no longer matches the approved manifest")

    return media_path, trim_width_mm, trim_height_mm, bleed


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load required print module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _find_print_skill_root(explicit: str | Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    configured = os.environ.get("CARD_ARTWORK_PRINT_PDF_SKILL")
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "card-artwork-print-pdf"
    )
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home) / "skills" / "card-artwork-print-pdf")
    else:
        candidates.append(Path.home() / ".codex" / "skills" / "card-artwork-print-pdf")
    for candidate in candidates:
        if (candidate / "scripts" / "build_print_pdf.py").is_file() and (
            candidate / "scripts" / "verify_artwork.py"
        ).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "card-artwork-print-pdf Skill was not found; pass --print-skill-root or set "
        "CARD_ARTWORK_PRINT_PDF_SKILL, restore the bundled verifier, or install it "
        "under CODEX_HOME/skills"
    )


def _path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _approved_input_paths(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Path]:
    manifest_directory = manifest_path.parent
    inputs = {
        "manifest": manifest_path,
        "source": _manifest_file(manifest.get("source"), manifest_directory, "source"),
    }
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    for name, value in artifacts.items():
        artifact = _mapping(value, f"artifact {name}")
        inputs[f"artifact {name}"] = _manifest_file(
            artifact.get("path"),
            manifest_directory,
            f"artifact {name}",
        )
    return inputs


def _validate_output_paths(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    output_pdf: Path,
    report_path: Path | None,
) -> tuple[Path, Path, Path, Path | None]:
    output_pdf = output_pdf.expanduser().resolve()
    build_report = output_pdf.with_name(output_pdf.stem + "-build-report.json")
    verify_report = output_pdf.with_name(output_pdf.stem + "-verify-report.json")
    report_path = report_path.expanduser().resolve() if report_path is not None else None

    input_by_key = {
        _path_key(path): label
        for label, path in _approved_input_paths(manifest, manifest_path).items()
    }
    outputs = {
        "output PDF": output_pdf,
        "build report": build_report,
        "verify report": verify_report,
    }
    if report_path is not None:
        outputs["external report"] = report_path

    seen: dict[str, str] = {}
    for label, path in outputs.items():
        key = _path_key(path)
        if key in input_by_key:
            raise ValueError(
                f"{label} path collides with approved input {input_by_key[key]}: {path}"
            )
        if key in seen:
            raise ValueError(f"{label} path collides with {seen[key]}: {path}")
        seen[key] = label
    return output_pdf, build_report, verify_report, report_path


def _approved_media_sha256(manifest: Mapping[str, Any]) -> str:
    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    media = _mapping(artifacts.get("media_rgb"), "artifact media_rgb")
    artifact_hash = _declared_sha256(media.get("sha256"), "artifact media_rgb")
    if "media_sha256" in manifest:
        alias_hash = _declared_sha256(manifest.get("media_sha256"), "media image")
        if alias_hash != artifact_hash:
            raise ValueError("media image hash does not match the media_rgb artifact")
    return artifact_hash


def _read_verified_media_bytes(media_path: Path, approved_sha256: str) -> bytes:
    try:
        payload = media_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read approved media image: {media_path}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != approved_sha256:
        raise ValueError("media image changed after manifest validation")
    return payload


def _require_pdf_verification(verified: Mapping[str, Any]) -> None:
    if verified.get("direct_image_xobject") is not True:
        raise RuntimeError("PDF verification did not prove a direct image XObject")
    if verified.get("embedded_pixels_equal_approved_image") is not True:
        raise RuntimeError("PDF verification did not prove approved embedded pixels")
    renderers = verified.get("renderers")
    if (
        verified.get("render_nonblank") is not True
        or not isinstance(renderers, Sequence)
        or isinstance(renderers, (str, bytes))
        or not renderers
    ):
        raise RuntimeError("PDF verification requires a nonblank independent renderer")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_from_manifest(
    manifest_path: str | Path,
    output_pdf: str | Path,
    *,
    print_skill_root: str | Path | None = None,
    pdftoppm: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_pdf = Path(output_pdf)
    report_path = Path(report_path) if report_path is not None else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest root must be a JSON object")
    contract_evidence = _manifest_contract_evidence(manifest)
    media_path, trim_width_mm, trim_height_mm, bleed = _validate_subject_first_manifest(
        manifest,
        manifest_path.parent,
    )
    output_pdf, build_report, verify_report, report_path = _validate_output_paths(
        manifest,
        manifest_path,
        output_pdf,
        report_path,
    )
    approved_sha256 = _approved_media_sha256(manifest)
    approved_media = _read_verified_media_bytes(media_path, approved_sha256)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_pdf.stem}-export-",
        dir=output_pdf.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        snapshot_path = temporary_root / f"approved-media{media_path.suffix or '.png'}"
        snapshot_path.write_bytes(approved_media)
        if _file_sha256(snapshot_path, "approved media snapshot") != approved_sha256:
            raise RuntimeError("approved media snapshot hash mismatch")

        temporary_pdf = temporary_root / "candidate.pdf"
        skill_root = _find_print_skill_root(print_skill_root)
        builder = _load_module(
            "card_print_builder",
            skill_root / "scripts" / "build_print_pdf.py",
        )
        verifier = _load_module(
            "card_print_verifier",
            skill_root / "scripts" / "verify_artwork.py",
        )
        build_result = builder.write_print_pdf(
            image_path=snapshot_path,
            output_pdf=temporary_pdf,
            trim_width_mm=trim_width_mm,
            trim_height_mm=trim_height_mm,
            bleed_left_mm=bleed["left"],
            bleed_right_mm=bleed["right"],
            bleed_top_mm=bleed["top"],
            bleed_bottom_mm=bleed["bottom"],
        )
        if not isinstance(build_result, Mapping) or not temporary_pdf.is_file():
            raise RuntimeError("PDF builder did not produce a candidate PDF")
        verified_result = verifier.verify_pdf(
            temporary_pdf,
            snapshot_path,
            trim_width_mm=trim_width_mm,
            trim_height_mm=trim_height_mm,
            bleed_left_mm=bleed["left"],
            bleed_right_mm=bleed["right"],
            bleed_top_mm=bleed["top"],
            bleed_bottom_mm=bleed["bottom"],
            pdftoppm=pdftoppm,
        )
        if not isinstance(verified_result, Mapping):
            raise RuntimeError("PDF verifier did not return a verification report")
        _require_pdf_verification(verified_result)
        if _file_sha256(snapshot_path, "approved media snapshot") != approved_sha256:
            raise RuntimeError("approved media snapshot changed during PDF export")
        os.replace(temporary_pdf, output_pdf)

    build = dict(build_result)
    build["pdf"] = str(output_pdf)
    build["source_image"] = str(media_path.resolve())
    verified = dict(verified_result)
    verified["pdf"] = str(output_pdf)
    build_report_payload = dict(build)
    build_report_payload["manifest_validation"] = contract_evidence
    _atomic_write_json(build_report, build_report_payload)
    _atomic_write_json(verify_report, {"pdf": verified})
    result = {
        "output_pdf": str(output_pdf),
        "manifest": str(manifest_path),
        "build_report": str(build_report),
        "verify_report": str(verify_report),
        "manifest_validation": contract_evidence,
        "build": build,
        "pdf": verified,
    }
    if report_path is not None:
        result["report"] = str(report_path)
        _atomic_write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--print-skill-root", type=Path)
    parser.add_argument("--pdftoppm", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = export_from_manifest(
        args.manifest,
        args.output,
        print_skill_root=args.print_skill_root,
        pdftoppm=args.pdftoppm,
        report_path=args.report,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
