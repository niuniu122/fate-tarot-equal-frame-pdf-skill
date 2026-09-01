#!/usr/bin/env python3
"""Prepare one bordered card with a subject-first, exterior-only bleed pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree


class DetectionError(ValueError):
    """Raised when a safe, unique subject-first preparation cannot be proven."""


@dataclass(frozen=True)
class BackgroundModel:
    top: np.ndarray
    right: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    corners: np.ndarray


PIPELINE_STAGE_NAMES = (
    "detect-subject-frame",
    "extract-subject",
    "sample-exterior",
    "render-bleed-background",
    "place-subject",
)
CURRENT_PIPELINE_VERSION = 3
SAMPLING_PROVENANCE_SCHEMA_VERSION = 1
SAMPLING_PROVENANCE_FIELDS = (
    "side_interpolated_coordinate_count",
    "side_interpolation_maximum_gap_px",
    "side_interpolation_maximum_allowed_gap_px",
    "side_interpolation_exceptions",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampling_provenance_artifact_payload(
    *,
    source_sha256: str,
    sampling_mask_sha256: str,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    """Create the version-bound v3 sampling evidence sidecar payload."""

    provenance_fields = ("side_interpolation_method", *SAMPLING_PROVENANCE_FIELDS)
    metric_fields = (
        "background_tolerance",
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
        "sampling_mask_bbox_px",
    )
    return {
        "schema_version": SAMPLING_PROVENANCE_SCHEMA_VERSION,
        "pipeline_version": CURRENT_PIPELINE_VERSION,
        "source_sha256": source_sha256,
        "sampling_mask_sha256": sampling_mask_sha256,
        "provenance": {field: sampling.get(field) for field in provenance_fields},
        "hard_gate_metrics": {field: sampling.get(field) for field in metric_fields},
    }


def _polyfit_robust(samples: np.ndarray, degree: int = 3) -> np.ndarray:
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise DetectionError("background side does not contain enough exterior samples")
    x = np.linspace(0.0, 1.0, len(samples), dtype=np.float64)
    coefficients = np.empty((3, degree + 1), dtype=np.float64)
    for channel in range(3):
        values = samples[:, channel].astype(np.float64)
        current_degree = min(degree, len(values) - 1)
        keep = np.ones(len(values), dtype=bool)
        for _ in range(4):
            fit = np.polyfit(x[keep], values[keep], current_degree)
            residual = np.abs(values - np.polyval(fit, x))
            median = float(np.median(residual[keep]))
            cutoff = max(4.0, median * 3.5)
            next_keep = residual <= cutoff
            if int(next_keep.sum()) < current_degree + 2 or np.array_equal(next_keep, keep):
                break
            keep = next_keep
        coefficients[channel] = np.polyfit(x[keep], values[keep], current_degree)
    return coefficients


def _evaluate(coefficients: np.ndarray, length: int) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.stack(
        [np.polyval(coefficients[channel], coordinate) for channel in range(3)],
        axis=1,
    )


def _match_endpoints(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, len(values), dtype=np.float64)[:, None]
    correction = (1 - coordinate) * (start - values[0]) + coordinate * (end - values[-1])
    return values + correction


def _render_background(model: BackgroundModel, width: int, height: int) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = model.corners
    top = _match_endpoints(_evaluate(model.top, width), top_left, top_right)[None, :, :]
    bottom = _match_endpoints(_evaluate(model.bottom, width), bottom_left, bottom_right)[None, :, :]
    left = _match_endpoints(_evaluate(model.left, height), top_left, bottom_left)[:, None, :]
    right = _match_endpoints(_evaluate(model.right, height), top_right, bottom_right)[:, None, :]
    u = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :, None]
    v = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None, None]
    bilinear = (
        (1 - u) * (1 - v) * top_left
        + u * (1 - v) * top_right
        + u * v * bottom_right
        + (1 - u) * v * bottom_left
    )
    coons_patch = (1 - v) * top + v * bottom + (1 - u) * left + u * right - bilinear
    return np.clip(np.rint(coons_patch), 0, 255).astype(np.uint8)


def _perimeter_model(rgb: np.ndarray) -> BackgroundModel:
    """Fit a provisional model only to discover the exterior-connected region."""
    height, width = rgb.shape[:2]
    band = max(3, round(min(width, height) * 0.018))
    top = np.median(rgb[:band], axis=0)
    right = np.median(rgb[:, width - band :], axis=1)
    bottom = np.median(rgb[height - band :], axis=0)
    left = np.median(rgb[:, :band], axis=1)
    extent = max(3, round(min(width, height) * 0.035))
    corner_patches = (
        rgb[:extent, :extent],
        rgb[:extent, width - extent :],
        rgb[height - extent :, width - extent :],
        rgb[height - extent :, :extent],
    )
    corners = np.stack(
        [np.median(patch.reshape(-1, 3), axis=0) for patch in corner_patches]
    )
    return BackgroundModel(
        top=_polyfit_robust(top),
        right=_polyfit_robust(right),
        bottom=_polyfit_robust(bottom),
        left=_polyfit_robust(left),
        corners=corners,
    )


def _automatic_tolerance(rgb: np.ndarray, predicted: np.ndarray) -> int:
    height, width = rgb.shape[:2]
    band = max(2, round(min(width, height) * 0.012))
    residual = np.max(np.abs(rgb.astype(np.int16) - predicted.astype(np.int16)), axis=2)
    perimeter = np.concatenate(
        (
            residual[:band].ravel(),
            residual[height - band :].ravel(),
            residual[band : height - band, :band].ravel(),
            residual[band : height - band, width - band :].ravel(),
        )
    )
    return int(np.clip(math.ceil(float(np.percentile(perimeter, 72))) + 8, 12, 48))


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise DetectionError("subject mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _fit_circle(points: list[tuple[int, int]]) -> tuple[float, float, float]:
    coordinates = np.asarray(points, dtype=np.float64)
    if len(coordinates) < 12:
        raise DetectionError("not enough outer-frame points to fit a rounded corner")
    matrix = np.column_stack(
        (2 * coordinates[:, 0], 2 * coordinates[:, 1], np.ones(len(coordinates)))
    )
    squared = coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2
    center_x, center_y, constant = np.linalg.lstsq(matrix, squared, rcond=None)[0]
    radius = math.sqrt(max(0.0, constant + center_x**2 + center_y**2))
    return float(center_x), float(center_y), float(radius)


def _arc(
    center_x: float,
    center_y: float,
    radius: float,
    start_degrees: float,
    end_degrees: float,
    samples: int = 64,
) -> list[tuple[float, float]]:
    return [
        (
            center_x + radius * math.cos(math.radians(angle)),
            center_y + radius * math.sin(math.radians(angle)),
        )
        for angle in np.linspace(start_degrees, end_degrees, samples)
    ]


def _enumerate_dark_frame_candidates(
    gray: np.ndarray,
) -> tuple[list[dict[str, Any]], float, int]:
    """Enumerate every plausible closed dark frame using adaptive contrast.

    Contrast only proposes connected components.  Enclosure, centrality, size,
    aspect, and a real hollow-frame area gain decide whether a component is a
    geometrically plausible outer frame.
    """
    height, width = gray.shape
    border_samples = np.concatenate(
        (
            gray[round(height * 0.15) : round(height * 0.85), 8 : round(width * 0.034)].ravel(),
            gray[
                round(height * 0.15) : round(height * 0.85),
                width - round(width * 0.034) : width - 8,
            ].ravel(),
            gray[3 : round(height * 0.018), round(width * 0.15) : round(width * 0.85)].ravel(),
            gray[
                height - round(height * 0.015) : height,
                round(width * 0.15) : round(width * 0.85),
            ].ravel(),
        )
    )
    if not len(border_samples):
        raise DetectionError("source is too small for proportional outer-frame detection")
    threshold = float(np.percentile(border_samples, 45))
    structure = ndimage.generate_binary_structure(2, 2)
    dark = ndimage.binary_closing(gray < threshold, structure=structure, iterations=1)
    labels, component_count = ndimage.label(dark, structure=structure)
    if component_count == 0:
        raise DetectionError("no connected dark outer-frame component was detected")
    sizes = np.bincount(labels.ravel())
    canvas_area = height * width
    candidates: list[dict[str, Any]] = []
    for label_id in np.argsort(sizes[1:])[::-1] + 1:
        component_area = int(sizes[label_id])
        if component_area < max(12, round(canvas_area * 0.001)):
            break
        component = labels == label_id
        filled = ndimage.binary_fill_holes(component)
        if not filled[height // 2, width // 2]:
            continue
        x0, y0, x1, y1 = _tight_bbox(filled)
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        filled_area = int(filled.sum())
        if bbox_width < width * 0.45 or bbox_height < height * 0.45:
            continue
        if not 0.42 <= bbox_width / bbox_height <= 0.90:
            continue
        if not canvas_area * 0.20 <= filled_area <= canvas_area * 0.995:
            continue
        if filled_area - component_area < canvas_area * 0.05:
            continue
        try:
            geometry = _dark_frame_geometry(filled)
        except DetectionError:
            continue
        candidates.append(
            {
                "component": component,
                "filled": filled,
                "component_pixels": component_area,
                "filled_pixels": filled_area,
                "bbox_px": [x0, y0, x1, y1],
                "geometry": geometry,
            }
        )
    return candidates, threshold, component_count


def _dark_frame_geometry(filled: np.ndarray) -> dict[str, Any]:
    height, width = filled.shape
    left_samples: list[int] = []
    right_samples: list[int] = []
    for y in range(round(height * 0.10), round(height * 0.90)):
        xs = np.flatnonzero(filled[y])
        if len(xs):
            left_samples.append(int(xs[0]))
            right_samples.append(int(xs[-1]))
    if not left_samples:
        raise DetectionError("continuous vertical frame sides were not detected")
    left = int(round(float(np.median(left_samples))))
    right = int(round(float(np.median(right_samples))))

    columns = list(range(round(width * 0.15), round(width * 0.40))) + list(
        range(round(width * 0.60), round(width * 0.85))
    )
    top_samples: list[int] = []
    bottom_samples: list[int] = []
    for x in columns:
        ys = np.flatnonzero(filled[:, x])
        if len(ys):
            top_samples.append(int(ys[0]))
            bottom_samples.append(int(ys[-1]))
    if not top_samples:
        raise DetectionError("continuous horizontal frame sides were not detected")
    top = int(round(float(np.median(top_samples))))
    bottom = int(round(float(np.median(bottom_samples))))

    corner_depth = max(48, round(min(height, width) * 0.058))
    corner_point_sets: list[list[tuple[int, int]]] = [[], [], [], []]
    for y in range(max(0, top), min(height, top + corner_depth)):
        xs = np.flatnonzero(filled[y])
        if len(xs):
            corner_point_sets[0].append((int(xs[0]), y))
            corner_point_sets[1].append((int(xs[-1]), y))
    for y in range(max(0, bottom - corner_depth), min(height, bottom - 2)):
        xs = np.flatnonzero(filled[y])
        if len(xs):
            corner_point_sets[2].append((int(xs[0]), y))
            corner_point_sets[3].append((int(xs[-1]), y))
    radii = [_fit_circle(points)[2] for points in corner_point_sets]
    minimum_radius = min(height, width) * 0.035
    maximum_radius = min(height, width) * 0.065
    radii = [float(np.clip(radius, minimum_radius, maximum_radius)) for radius in radii]
    return {
        "left": left - 0.5,
        "right": right + 0.5,
        "top": top - 0.5,
        "bottom": bottom + 0.5,
        "radii": {
            "top_left": radii[0],
            "top_right": radii[1],
            "bottom_left": radii[2],
            "bottom_right": radii[3],
        },
        "dark_sides_px": {"left": left, "right": right, "top": top, "bottom": bottom},
    }


def _normalize_corner_radii_from_source_gradients(
    rgb: np.ndarray,
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize each radius to a source gradient arc with both tangents."""
    gray = rgb.astype(np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    gradient_y = ndimage.sobel(gray, axis=0) / 4.0
    gradient_x = ndimage.sobel(gray, axis=1) / 4.0
    gradient = np.hypot(gradient_x, gradient_y)
    height, width = gray.shape
    sides = geometry["dark_sides_px"]
    frame_width = int(sides["right"] - sides["left"])
    frame_height = int(sides["bottom"] - sides["top"])
    short_side = min(frame_width, frame_height)
    minimum_radius = max(4, round(short_side * 0.015))
    maximum_radius = max(
        minimum_radius,
        min(round(short_side * 0.18), short_side // 2 - 1),
    )
    probe_radius = max(2, round(min(width, height) * 0.004))
    outer_search_depth = probe_radius * 2
    gradient_threshold = 18.0
    minimum_normal_alignment = 0.80
    maximum_outward_competing_fraction = 0.32
    tangent_normal_alignment = 0.90
    tangent_normal_offset_tolerance = max(1, probe_radius // 2)
    tangent_confirmation_run_length = 4
    tangent_bridge_length = max(
        tangent_confirmation_run_length * 4,
        round(short_side * 0.04),
    )
    maximum_tangent_bridge_gap = max(1, probe_radius // 2)
    minimum_tangent_bridge_coverage = 0.70
    maximum_tangent_radius_delta = max(
        3,
        probe_radius * 2,
        round(short_side * 0.02),
    )
    maximum_connected_arc_gap_points = 2
    horizontal_span = np.concatenate(
        (
            np.arange(
                sides["left"] + round(frame_width * 0.14),
                sides["left"] + round(frame_width * 0.42),
            ),
            np.arange(
                sides["left"] + round(frame_width * 0.58),
                sides["left"] + round(frame_width * 0.86),
            ),
        )
    )
    vertical_span = np.arange(
        sides["top"] + round(frame_height * 0.15),
        sides["top"] + round(frame_height * 0.85),
    )
    horizontal_span = horizontal_span[
        (horizontal_span >= 0) & (horizontal_span < width)
    ]
    vertical_span = vertical_span[(vertical_span >= 0) & (vertical_span < height)]

    def locked_rail_outward_polarity(side: str) -> tuple[int, float]:
        anchor = int(sides[side])
        if side == "top":
            values = -gradient_y[anchor, horizontal_span]
        elif side == "bottom":
            values = gradient_y[anchor, horizontal_span]
        elif side == "left":
            values = -gradient_x[vertical_span, anchor]
        else:
            values = gradient_x[vertical_span, anchor]
        strong_values = values[np.abs(values) >= gradient_threshold]
        if not len(strong_values):
            raise DetectionError(
                f"locked outer {side} rail has no direct signed-gradient evidence"
            )
        positive_fraction = float(np.mean(strong_values > 0))
        negative_fraction = float(np.mean(strong_values < 0))
        if positive_fraction >= negative_fraction:
            return 1, positive_fraction
        return -1, negative_fraction

    rail_polarity: dict[str, int] = {}
    rail_polarity_confidence: dict[str, float] = {}
    for side in ("top", "right", "bottom", "left"):
        polarity, confidence = locked_rail_outward_polarity(side)
        rail_polarity[side] = polarity
        rail_polarity_confidence[side] = confidence

    def side_gradient_fields(side: str) -> tuple[np.ndarray, np.ndarray]:
        if side == "top":
            return -gradient_y, gradient_x
        if side == "bottom":
            return gradient_y, gradient_x
        if side == "left":
            return -gradient_x, gradient_y
        return gradient_x, gradient_y

    def locked_rail_reference(side: str) -> dict[str, Any]:
        anchor = int(sides[side])
        axis_limit = height if side in {"top", "bottom"} else width
        locked_band = geometry.get("gradient_bands_px", {}).get(
            side,
            [anchor, anchor],
        )
        band_start = max(0, int(locked_band[0]))
        band_stop = min(axis_limit - 1, int(locked_band[1]))
        positions = range(
            band_start,
            band_stop + 1,
        )
        along_span = horizontal_span if side in {"top", "bottom"} else vertical_span
        normal_gradient, tangent_gradient = side_gradient_fields(side)
        candidates: list[dict[str, Any]] = []
        for position in positions:
            if side in {"top", "bottom"}:
                normal_values = normal_gradient[position, along_span]
                tangent_values = tangent_gradient[position, along_span]
            else:
                normal_values = normal_gradient[along_span, position]
                tangent_values = tangent_gradient[along_span, position]
            magnitude = np.hypot(normal_values, tangent_values)
            alignment = np.divide(
                np.abs(normal_values),
                magnitude,
                out=np.zeros_like(magnitude, dtype=np.float32),
                where=magnitude > 0,
            )
            confirmed = (
                normal_values * rail_polarity[side] >= gradient_threshold
            ) & (alignment >= tangent_normal_alignment)
            candidates.append(
                {
                    "position_px": int(position),
                    "coverage": float(np.mean(confirmed)),
                }
            )
        maximum_coverage = max(candidate["coverage"] for candidate in candidates)
        best = [
            candidate
            for candidate in candidates
            if candidate["coverage"] == maximum_coverage
        ]
        selected = (
            min(best, key=lambda candidate: candidate["position_px"])
            if side in {"top", "left"}
            else max(best, key=lambda candidate: candidate["position_px"])
        )
        if selected["coverage"] < 0.50:
            raise DetectionError(
                f"locked outer {side} rail has insufficient straight-run evidence"
            )
        return {
            "anchor_px": anchor,
            "locked_gradient_band_px": [band_start, band_stop],
            "reference_normal_position_px": int(selected["position_px"]),
            "reference_normal_offset_px": int(selected["position_px"] - anchor),
            "reference_coverage": float(selected["coverage"]),
        }

    rail_reference = {
        side: locked_rail_reference(side)
        for side in ("top", "right", "bottom", "left")
    }

    def rail_alignment_at(
        side: str,
        along_position: int,
    ) -> dict[str, Any] | None:
        normal_gradient, tangent_gradient = side_gradient_fields(side)
        reference = int(rail_reference[side]["reference_normal_position_px"])
        band_start, band_stop = rail_reference[side]["locked_gradient_band_px"]
        axis_limit = height if side in {"top", "bottom"} else width
        matches: list[dict[str, Any]] = []
        for normal_position in range(
            max(0, band_start - 1, reference - tangent_normal_offset_tolerance),
            min(
                axis_limit,
                band_stop + 2,
                reference + tangent_normal_offset_tolerance + 1,
            ),
        ):
            if side in {"top", "bottom"}:
                normal_value = float(normal_gradient[normal_position, along_position])
                tangent_value = float(tangent_gradient[normal_position, along_position])
            else:
                normal_value = float(normal_gradient[along_position, normal_position])
                tangent_value = float(tangent_gradient[along_position, normal_position])
            magnitude = math.hypot(normal_value, tangent_value)
            alignment = abs(normal_value) / magnitude if magnitude else 0.0
            if (
                normal_value * rail_polarity[side] >= gradient_threshold
                and alignment >= tangent_normal_alignment
            ):
                tangent_fraction = abs(tangent_value) / magnitude if magnitude else 0.0
                matches.append(
                    {
                        "normal_position_px": int(normal_position),
                        "normal_value": normal_value,
                        "tangent_value": tangent_value,
                        "magnitude": magnitude,
                        "normal_alignment": alignment,
                        "tangent_fraction": tangent_fraction,
                    }
                )
        if not matches:
            return None
        return max(
            matches,
            key=lambda match: (
                float(match["normal_alignment"]),
                abs(float(match["normal_value"])),
            ),
        )

    def rail_aligned_at(side: str, along_position: int) -> bool:
        return rail_alignment_at(side, along_position) is not None

    corner_tangent_scan_specs = {
        "top_left": (("top", sides["left"], 1), ("left", sides["top"], 1)),
        "top_right": (("top", sides["right"], -1), ("right", sides["top"], 1)),
        "bottom_right": (
            ("bottom", sides["right"], -1),
            ("right", sides["bottom"], -1),
        ),
        "bottom_left": (
            ("bottom", sides["left"], 1),
            ("left", sides["bottom"], -1),
        ),
    }

    def tangent_transition_candidates(
        side: str,
        origin: int,
        direction: int,
    ) -> dict[str, Any]:
        along_limit = width if side in {"top", "bottom"} else height
        maximum_scan_distance = min(
            maximum_radius + tangent_bridge_length,
            origin if direction < 0 else along_limit - 1 - origin,
        )
        aligned = np.asarray(
            [
                rail_aligned_at(side, origin + direction * distance)
                for distance in range(maximum_scan_distance + 1)
            ],
            dtype=bool,
        )
        padded = np.concatenate(([False], aligned, [False])).astype(np.int8)
        changes = np.diff(padded)
        aligned_runs = [
            (int(start), int(stop - 1))
            for start, stop in zip(
                np.flatnonzero(changes == 1),
                np.flatnonzero(changes == -1),
                strict=True,
            )
        ]
        long_support_runs = [
            (start, stop)
            for start, stop in aligned_runs
            if stop - start + 1 >= tangent_bridge_length
        ]
        candidates: list[dict[str, Any]] = []
        for run_start, run_stop in aligned_runs:
            if run_stop - run_start + 1 < tangent_confirmation_run_length:
                continue
            confirmed_distance = run_start + tangent_confirmation_run_length - 1
            confirmation_samples: list[dict[str, Any]] = []
            for distance in range(run_start, confirmed_distance + 1):
                along_position = origin + direction * distance
                alignment = rail_alignment_at(side, along_position)
                if alignment is None:
                    continue
                tangent_fraction = float(alignment["tangent_fraction"])
                radius_from_orientation = float(distance) / max(
                    0.05,
                    1.0 - tangent_fraction,
                )
                confirmation_samples.append(
                    {
                        "distance_px": int(distance),
                        "along_position_px": int(along_position),
                        "normal_position_px": int(
                            alignment["normal_position_px"]
                        ),
                        "normal_alignment": float(
                            alignment["normal_alignment"]
                        ),
                        "tangent_fraction": tangent_fraction,
                        "orientation_radius_estimate_px": radius_from_orientation,
                    }
                )
            orientation_radius = float(
                np.median(
                    [
                        sample["orientation_radius_estimate_px"]
                        for sample in confirmation_samples
                    ]
                )
            )
            estimated_radius = int(
                round(max(float(confirmed_distance), orientation_radius))
            )
            bridge_stop = min(len(aligned), run_start + tangent_bridge_length)
            bridge = aligned[run_start:bridge_stop]
            missing_padded = np.concatenate(([False], ~bridge, [False])).astype(
                np.int8
            )
            missing_changes = np.diff(missing_padded)
            missing_runs = [
                [int(start), int(stop - 1)]
                for start, stop in zip(
                    np.flatnonzero(missing_changes == 1),
                    np.flatnonzero(missing_changes == -1),
                    strict=True,
                )
            ]
            maximum_gap = max(
                (stop - start + 1 for start, stop in missing_runs),
                default=0,
            )
            coverage = float(np.mean(bridge)) if len(bridge) else 0.0
            direct_bridge_passed = bool(
                len(bridge) >= tangent_bridge_length
                and coverage >= minimum_tangent_bridge_coverage
                and maximum_gap <= maximum_tangent_bridge_gap
            )
            later_support = next(
                (
                    (support_start, support_stop)
                    for support_start, support_stop in long_support_runs
                    if support_start >= run_start
                    and support_start - run_stop - 1 <= tangent_bridge_length
                ),
                None,
            )
            linked_to_long_run = later_support is not None
            candidates.append(
                {
                    "side": side,
                    "scan_origin_px": int(origin),
                    "scan_direction": int(direction),
                    "confirmation_run_distance_px": [
                        int(run_start),
                        int(confirmed_distance),
                    ],
                    "confirmation_run_px": [
                        int(origin + direction * run_start),
                        int(origin + direction * confirmed_distance),
                    ],
                        "threshold_confirmation_radius_px": int(
                            confirmed_distance
                        ),
                        "orientation_radius_estimate_px": orientation_radius,
                        "orientation_radius_samples": confirmation_samples,
                        "estimated_radius_px": estimated_radius,
                    "reference_rail": dict(rail_reference[side]),
                    "bridge_to_long_run_evidence": {
                        "run_start_distance_px": int(run_start),
                        "run_confirmed_distance_px": int(confirmed_distance),
                        "run_stop_distance_px": int(run_stop),
                        "bridge_stop_distance_px": int(bridge_stop - 1),
                        "bridge_coverage": coverage,
                        "bridge_missing_runs": missing_runs,
                        "maximum_bridge_gap_px": int(maximum_gap),
                        "direct_bridge_passed": direct_bridge_passed,
                        "support_run_distance_px": (
                            [int(later_support[0]), int(later_support[1])]
                            if later_support is not None
                            else None
                        ),
                        "support_gap_px": (
                            max(0, int(later_support[0] - run_stop - 1))
                            if later_support is not None
                            else None
                        ),
                        "linked_to_locked_long_run": linked_to_long_run,
                        "passed": bool(
                            direct_bridge_passed or linked_to_long_run
                        ),
                    },
                }
            )
        accepted = [
            candidate
            for candidate in candidates
            if candidate["bridge_to_long_run_evidence"]["passed"]
        ]
        if not accepted:
            raise DetectionError(
                f"locked outer {side} rail has no confirmed corner tangent transition"
            )
        return {
            "side": side,
            "scan_origin_px": int(origin),
            "scan_direction": int(direction),
            "reference_rail": dict(rail_reference[side]),
            "aligned_runs_distance_px": [list(run) for run in aligned_runs],
            "candidates": candidates,
        }

    tangent_pair_options: dict[str, list[dict[str, Any]]] = {}
    for corner, scans in corner_tangent_scan_specs.items():
        scans_by_side = {
            side: tangent_transition_candidates(side, int(origin), int(direction))
            for side, origin, direction in scans
        }
        first_side, second_side = scans_by_side
        candidate_pairs = [
            {
                "first": first,
                "second": second,
                "delta": abs(
                    int(first["estimated_radius_px"])
                    - int(second["estimated_radius_px"])
                ),
            }
            for first in scans_by_side[first_side]["candidates"]
            for second in scans_by_side[second_side]["candidates"]
            if first["bridge_to_long_run_evidence"]["passed"]
            and second["bridge_to_long_run_evidence"]["passed"]
        ]
        options: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(candidate_pairs):
            transitions = {
                first_side: pair["first"],
                second_side: pair["second"],
            }
            estimates = {
                side: int(transition["estimated_radius_px"])
                for side, transition in transitions.items()
            }
            estimate_values = list(estimates.values())
            delta = abs(estimate_values[0] - estimate_values[1])
            target = int(round(float(np.mean(estimate_values))))
            half_window = max(1, min(4, round(target * 0.08)))
            options.append(
                {
                    "pair_index": int(pair_index),
                    "transitions": transitions,
                    "estimates": estimates,
                    "delta": int(delta),
                    "target": int(target),
                    "candidate_window": [
                        max(minimum_radius, target - half_window),
                        min(maximum_radius, target + half_window),
                    ],
                    "passed": delta <= maximum_tangent_radius_delta,
                    "candidate_pair_count": len(candidate_pairs),
                    "side_scan_evidence": scans_by_side,
                }
            )
        tangent_pair_options[corner] = options
    corner_tangent_sides = {
        "top_left": ("left", "top"),
        "top_right": ("top", "right"),
        "bottom_right": ("right", "bottom"),
        "bottom_left": ("bottom", "left"),
    }
    mismatched_corner_polarity = [
        corner
        for corner, (first, second) in corner_tangent_sides.items()
        if rail_polarity[first] != rail_polarity[second]
    ]
    corner_polarity = {
        corner: np.asarray(
            [rail_polarity[first]] * 13 + [rail_polarity[second]] * 12,
            dtype=np.int8,
        )
        for corner, (first, second) in corner_tangent_sides.items()
    }
    angles_by_corner = {
        "top_left": np.linspace(180.0, 270.0, 25),
        "top_right": np.linspace(270.0, 360.0, 25),
        "bottom_right": np.linspace(0.0, 90.0, 25),
        "bottom_left": np.linspace(90.0, 180.0, 25),
    }

    def arc_points(
        corner: str,
        radius: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if corner == "top_left":
            center_x = sides["left"] + radius
            center_y = sides["top"] + radius
        elif corner == "top_right":
            center_x = sides["right"] - radius
            center_y = sides["top"] + radius
        elif corner == "bottom_right":
            center_x = sides["right"] - radius
            center_y = sides["bottom"] - radius
        else:
            center_x = sides["left"] + radius
            center_y = sides["bottom"] - radius
        radians = np.deg2rad(angles_by_corner[corner])
        xs = np.rint(center_x + radius * np.cos(radians)).astype(int)
        ys = np.rint(center_y + radius * np.sin(radians)).astype(int)
        return (
            np.clip(xs, 0, width - 1),
            np.clip(ys, 0, height - 1),
            np.cos(radians),
            np.sin(radians),
        )

    def oriented_hits(
        xs: np.ndarray,
        ys: np.ndarray,
        normal_x: np.ndarray,
        normal_y: np.ndarray,
        expected_polarity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hits = np.zeros(len(xs), dtype=bool)
        alignment_scores = np.zeros(len(xs), dtype=np.float64)
        outermost_offsets = np.full(len(xs), np.nan, dtype=np.float64)
        normal_offsets = np.arange(
            -probe_radius,
            outer_search_depth + 1,
            dtype=np.float64,
        )
        tangent_offsets = np.arange(
            -probe_radius,
            probe_radius + 1,
            dtype=np.float64,
        )
        for index, (x, y, expected_x, expected_y, point_polarity) in enumerate(
            zip(
                xs,
                ys,
                normal_x,
                normal_y,
                expected_polarity,
                strict=True,
            )
        ):
            tangent_x = -expected_y
            tangent_y = expected_x
            sample_x = np.rint(
                x
                + normal_offsets[:, None] * expected_x
                + tangent_offsets[None, :] * tangent_x
            ).astype(int)
            sample_y = np.rint(
                y
                + normal_offsets[:, None] * expected_y
                + tangent_offsets[None, :] * tangent_y
            ).astype(int)
            valid = (
                (sample_x >= 0)
                & (sample_x < width)
                & (sample_y >= 0)
                & (sample_y < height)
            )
            if not np.any(valid):
                continue
            clipped_x = np.clip(sample_x, 0, width - 1)
            clipped_y = np.clip(sample_y, 0, height - 1)
            magnitude = gradient[clipped_y, clipped_x]
            signed_dot = (
                gradient_x[clipped_y, clipped_x] * expected_x
                + gradient_y[clipped_y, clipped_x] * expected_y
            )
            alignment_values = np.zeros_like(magnitude, dtype=np.float32)
            strong = valid & (magnitude >= gradient_threshold)
            alignment_values[strong] = (
                np.abs(signed_dot[strong]) / magnitude[strong]
            )
            aligned = (
                strong
                & (signed_dot * int(point_polarity) > 0)
                & (alignment_values >= minimum_normal_alignment)
            )
            matched_offset_indexes = np.flatnonzero(np.any(aligned, axis=1))
            if not len(matched_offset_indexes):
                continue
            outermost_index = int(matched_offset_indexes[-1])
            outermost_offsets[index] = float(normal_offsets[outermost_index])
            local_indexes = matched_offset_indexes[
                normal_offsets[matched_offset_indexes] <= probe_radius
            ]
            if not len(local_indexes):
                continue
            local_outermost_index = int(local_indexes[-1])
            alignment_scores[index] = float(
                np.max(
                    alignment_values[local_outermost_index][
                        aligned[local_outermost_index]
                    ]
                )
            )
            hits[index] = True
        return hits, alignment_scores, outermost_offsets

    requested_radii = {
        name: float(geometry["radii"][name]) for name in angles_by_corner
    }
    source_connected_foreground, source_connected_foreground_evidence = (
        _source_connected_frame_foreground(
            rgb,
            geometry,
            anchors={side: int(sides[side]) for side in sides},
            probe_radius=max(2, round(min(width, height) * 0.006)),
        )
    )
    component_corridor_radius = max(
        probe_radius * 2 + 1,
        round(short_side * 0.006),
    )
    corridor_axis = np.arange(
        -component_corridor_radius,
        component_corridor_radius + 1,
    )
    corridor_y, corridor_x = np.meshgrid(
        corridor_axis,
        corridor_axis,
        indexing="ij",
    )
    corridor_structure = (
        corridor_x * corridor_x + corridor_y * corridor_y
        <= component_corridor_radius * component_corridor_radius
    )

    def source_component_arc_evidence(
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> dict[str, Any]:
        padding = component_corridor_radius + 1
        x0 = max(0, int(xs.min()) - padding)
        x1 = min(width, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(height, int(ys.max()) + padding + 1)
        local_xs = xs - x0
        local_ys = ys - y0
        arc_seed = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        arc_seed[local_ys, local_xs] = True
        corridor = ndimage.binary_dilation(
            arc_seed,
            structure=corridor_structure,
        )
        local_foreground = source_connected_foreground[y0:y1, x0:x1] & corridor
        labels, label_count = ndimage.label(
            local_foreground,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        endpoint_labels: list[list[int]] = []
        for endpoint_x, endpoint_y in (
            (int(local_xs[0]), int(local_ys[0])),
            (int(local_xs[-1]), int(local_ys[-1])),
        ):
            endpoint_seed = np.zeros_like(local_foreground)
            endpoint_seed[endpoint_y, endpoint_x] = True
            endpoint_probe = ndimage.binary_dilation(
                endpoint_seed,
                structure=corridor_structure,
            )
            endpoint_labels.append(
                [
                    int(label)
                    for label in np.unique(labels[endpoint_probe])
                    if label > 0
                ]
            )
        shared_labels = sorted(
            set(endpoint_labels[0]).intersection(endpoint_labels[1])
        )
        near_foreground = ndimage.binary_dilation(
            local_foreground,
            structure=corridor_structure,
        )
        point_support = near_foreground[local_ys, local_xs]
        point_coverage = float(np.mean(point_support))
        passed = bool(shared_labels and point_coverage >= 0.80)
        return {
            "passed": passed,
            "method": (
                "source-background-difference-component-in-narrow-arc-corridor"
            ),
            "corridor_radius_px": int(component_corridor_radius),
            "point_coverage": point_coverage,
            "minimum_point_coverage": 0.80,
            "endpoint_component_labels": endpoint_labels,
            "shared_endpoint_component_labels": shared_labels,
            "corridor_component_count": int(label_count),
            "source_foreground_method": source_connected_foreground_evidence[
                "method"
            ],
            "source_foreground_color_policy": source_connected_foreground_evidence[
                "color_policy"
            ],
        }

    def boolean_runs(values: np.ndarray, expected: bool) -> list[list[int]]:
        padded = np.concatenate(([False], values == expected, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return [
            [int(start), int(stop - 1)]
            for start, stop in zip(starts, stops, strict=True)
        ]

    normalized_radii: dict[str, float] = {}
    corner_evidence: dict[str, Any] = {}
    weak: list[str] = []
    arc_candidate_cache: dict[str, dict[int, dict[str, Any]]] = {
        corner: {} for corner in angles_by_corner
    }

    def evaluate_arc_candidate(corner: str, radius: int) -> dict[str, Any]:
        cached = arc_candidate_cache[corner].get(radius)
        if cached is not None:
            return cached
        xs, ys, normal_x, normal_y = arc_points(corner, radius)
        hits, normal_alignment, outermost_offsets = oriented_hits(
            xs,
            ys,
            normal_x,
            normal_y,
            corner_polarity[corner],
        )
        tangent_hits = [bool(hits[0]), bool(hits[-1])]
        arc_coverage = float(np.mean(hits))
        interior_coverage = float(np.mean(hits[2:-2]))
        normal_alignment_mean = (
            float(np.mean(normal_alignment[hits])) if np.any(hits) else 0.0
        )
        outward_competing = (
            np.isfinite(outermost_offsets)
            & (outermost_offsets > probe_radius)
        )
        outward_competing_fraction = float(np.mean(outward_competing))
        finite_offsets = outermost_offsets[np.isfinite(outermost_offsets)]
        hit_runs = boolean_runs(hits, True)
        missing_runs = boolean_runs(hits, False)
        internal_missing_runs = [
            run
            for run in missing_runs
            if run[0] > 0 and run[1] < len(hits) - 1
        ]
        maximum_internal_gap = max(
            (stop - start + 1 for start, stop in internal_missing_runs),
            default=0,
        )
        gradient_run_connects_both_tangents = bool(
            all(tangent_hits)
            and maximum_internal_gap <= maximum_connected_arc_gap_points
        )
        component_evidence = source_component_arc_evidence(xs, ys)
        connects_both_tangents = bool(
            all(tangent_hits) and component_evidence["passed"]
        )
        score = (
            arc_coverage * 0.65
            + interior_coverage * 0.20
            + normal_alignment_mean * 0.05
            + float(np.mean(tangent_hits)) * 0.10
            - outward_competing_fraction * 0.10
        )
        candidate = {
            "radius_px": radius,
            "score": score,
            "arc_coverage": arc_coverage,
            "interior_arc_coverage": interior_coverage,
            "normal_alignment_mean": normal_alignment_mean,
            "tangent_hits": tangent_hits,
            "outward_competing_gradient_fraction": (
                outward_competing_fraction
            ),
            "outermost_matched_offset_range_px": (
                [float(np.min(finite_offsets)), float(np.max(finite_offsets))]
                if len(finite_offsets)
                else None
            ),
            "connected_outer_arc_evidence": {
                "passed": connects_both_tangents,
                "connects_both_locked_tangents": connects_both_tangents,
                "single_source_component_required": True,
                "direct_gradient_run_passed": (
                    gradient_run_connects_both_tangents
                ),
                "hit_runs": hit_runs,
                "missing_runs": missing_runs,
                "maximum_internal_gap_points": int(maximum_internal_gap),
                "maximum_allowed_internal_gap_points": (
                    maximum_connected_arc_gap_points
                ),
                "source_component_corridor_evidence": component_evidence,
            },
        }
        arc_candidate_cache[corner][radius] = candidate
        return candidate

    def candidate_passes(candidate: dict[str, Any]) -> bool:
        source_component_arc = bool(
            candidate["connected_outer_arc_evidence"][
                "source_component_corridor_evidence"
            ]["passed"]
        )
        return bool(
            all(candidate["tangent_hits"])
            and candidate["outward_competing_gradient_fraction"]
            <= maximum_outward_competing_fraction
            and candidate["connected_outer_arc_evidence"]["passed"]
            and source_component_arc
        )

    for corner in angles_by_corner:
        evaluations: list[dict[str, Any]] = []
        for tangent_target in tangent_pair_options[corner]:
            if not tangent_target["passed"]:
                continue
            candidate_start, candidate_stop = tangent_target["candidate_window"]
            if candidate_start > candidate_stop:
                continue
            for radius in range(candidate_start, candidate_stop + 1):
                candidate = evaluate_arc_candidate(corner, radius)
                estimates = list(tangent_target["estimates"].values())
                evaluations.append(
                    {
                        "tangent_target": tangent_target,
                        "candidate": candidate,
                        "tangent_residual_px": int(
                            sum(abs(radius - estimate) for estimate in estimates)
                        ),
                        "passed": candidate_passes(candidate),
                    }
                )

        passing_evaluations = [
            evaluation for evaluation in evaluations if evaluation["passed"]
        ]
        if passing_evaluations:
            selected = min(
                passing_evaluations,
                key=lambda evaluation: (
                    int(evaluation["tangent_residual_px"]),
                    int(evaluation["tangent_target"]["delta"]),
                    -float(evaluation["candidate"]["score"]),
                    int(evaluation["tangent_target"]["pair_index"]),
                ),
            )
            passed = True
        elif evaluations:
            selected = max(
                evaluations,
                key=lambda evaluation: (
                    float(evaluation["candidate"]["score"]),
                    float(evaluation["candidate"]["arc_coverage"]),
                    -int(evaluation["tangent_residual_px"]),
                ),
            )
            passed = False
        else:
            raise DetectionError(
                f"corner {corner} has no two-side tangent pair within the radius delta gate"
            )
        tangent_target = selected["tangent_target"]
        best = selected["candidate"]
        if not passed:
            weak.append(corner)
        normalized_radii[corner] = float(best["radius_px"])
        corner_evidence[corner] = {
            "passed": passed,
            "requested_radius_px": requested_radii[corner],
            "normalized_radius_px": float(best["radius_px"]),
            "requested_to_normalized_delta_px": (
                float(best["radius_px"]) - requested_radii[corner]
            ),
            "arc_coverage": best["arc_coverage"],
            "interior_arc_coverage": best["interior_arc_coverage"],
            "normal_alignment_mean": best["normal_alignment_mean"],
            "tangent_hits": best["tangent_hits"],
            "outward_competing_gradient_fraction": best[
                "outward_competing_gradient_fraction"
            ],
            "outermost_matched_offset_range_px": best[
                "outermost_matched_offset_range_px"
            ],
            "locked_tangent_transition_evidence": dict(
                tangent_target["transitions"]
            ),
            "locked_tangent_radius_estimates_px": dict(
                tangent_target["estimates"]
            ),
            "tangent_radius_delta_px": int(tangent_target["delta"]),
            "tangent_target_radius_px": int(tangent_target["target"]),
            "candidate_radius_window_px": list(
                tangent_target["candidate_window"]
            ),
            "tangent_pair_candidate_count": int(
                tangent_target["candidate_pair_count"]
            ),
            "tangent_pairs_within_delta_gate": int(
                sum(option["passed"] for option in tangent_pair_options[corner])
            ),
            "selected_tangent_pair_index": int(
                tangent_target["pair_index"]
            ),
            "selected_tangent_residual_px": int(
                selected["tangent_residual_px"]
            ),
            "connected_outer_arc_evidence": best[
                "connected_outer_arc_evidence"
            ],
            "score": best["score"],
        }
    evidence = {
        "passed": not weak,
        "method": "source-corner-gradient-arc-and-tangent-normalization",
        "gradient_threshold": gradient_threshold,
        "minimum_normal_alignment": minimum_normal_alignment,
        "gradient_probe_radius_px": probe_radius,
        "outer_rail_direct_gradient_required": True,
        "outer_rail_signed_polarity_required": True,
        "outermost_gradient_continuity_required": True,
        "outer_gradient_search_depth_px": outer_search_depth,
        "maximum_outward_competing_gradient_fraction": (
            maximum_outward_competing_fraction
        ),
        "radius_selection_policy": (
            "locked-tangent-target-then-narrow-connected-outer-arc"
        ),
        "tangent_transition_policy": (
            "orientation-corrected-locked-band-tangent-pairs-validated-by-"
            "narrow-single-component-outer-arcs"
        ),
        "tangent_normal_alignment": tangent_normal_alignment,
        "tangent_normal_offset_tolerance_px": tangent_normal_offset_tolerance,
        "tangent_confirmation_run_length_px": tangent_confirmation_run_length,
        "maximum_tangent_radius_delta_px": maximum_tangent_radius_delta,
        "maximum_connected_arc_gap_points": maximum_connected_arc_gap_points,
        "locked_rail_reference_evidence": rail_reference,
        "locked_rail_outward_gradient_polarity": rail_polarity,
        "locked_rail_polarity_confidence": rail_polarity_confidence,
        "mixed_polarity_corners": mismatched_corner_polarity,
        "corner_tangent_side_polarity_policy": {
            corner: list(sides_for_corner)
            for corner, sides_for_corner in corner_tangent_sides.items()
        },
        "candidate_radius_range_px": [minimum_radius, maximum_radius],
        "minimum_arc_coverage": 0.60,
        "minimum_interior_arc_coverage": 0.55,
        "tangents_required": True,
        "corners": corner_evidence,
    }
    if weak:
        raise DetectionError(
            "corner radius gradient/tangent alignment failed on: "
            + ", ".join(weak)
        )
    normalized = dict(geometry)
    normalized["radii"] = normalized_radii
    normalized["radius_method"] = evidence["method"]
    return normalized, evidence


def _redirect_edge_hugging_geometry_from_gradients(
    rgb: np.ndarray,
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace clipped extrema with one safe pair of opposite frame gradients.

    Long artwork lines are common inside these cards.  Treating every side
    independently makes one unmatched line fatal, while selecting the strongest
    line can silently choose artwork.  This gate therefore accepts only
    approximately mirrored, similarly thick opposite bands.  Nested bands may
    describe layers of the same frame only when they remain close together; the
    outermost such layer is retained.  A missing *short* side may be reflected
    from its unique observed opposite side, but the two long sides must both be
    observed and paired.
    """
    gray = rgb.astype(np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    height, width = gray.shape
    normal_gradients = {
        "top": np.abs(ndimage.sobel(gray, axis=0)) / 4.0,
        "bottom": np.abs(ndimage.sobel(gray, axis=0)) / 4.0,
        "left": np.abs(ndimage.sobel(gray, axis=1)) / 4.0,
        "right": np.abs(ndimage.sobel(gray, axis=1)) / 4.0,
    }
    horizontal_span = np.concatenate(
        (
            np.arange(round(width * 0.14), round(width * 0.42)),
            np.arange(round(width * 0.58), round(width * 0.86)),
        )
    )
    vertical_span = np.arange(round(height * 0.15), round(height * 0.85))
    if not len(horizontal_span) or not len(vertical_span):
        raise DetectionError("source is too small for long outer-frame gradient probes")

    gradient_threshold = 18.0
    minimum_coverage = 0.70
    maximum_gap = 2

    def merged_bands(coverage: np.ndarray, start: int, stop: int) -> list[tuple[int, int]]:
        active = np.flatnonzero(coverage[start:stop] >= minimum_coverage) + start
        if not len(active):
            return []
        bands: list[tuple[int, int]] = []
        band_start = int(active[0])
        previous = int(active[0])
        for position_value in active[1:]:
            position = int(position_value)
            if position - previous - 1 > maximum_gap:
                bands.append((band_start, previous))
                band_start = position
            previous = position
        bands.append((band_start, previous))
        return bands

    horizontal_coverage = np.mean(
        normal_gradients["top"][:, horizontal_span] >= gradient_threshold,
        axis=1,
    )
    vertical_coverage = np.mean(
        normal_gradients["left"][vertical_span] >= gradient_threshold,
        axis=0,
    )
    edge_height = max(8, round(height * 0.18))
    edge_width = max(8, round(width * 0.18))
    side_data: dict[str, tuple[np.ndarray, int, int]] = {
        "top": (horizontal_coverage, 0, edge_height),
        "bottom": (horizontal_coverage, height - edge_height, height),
        "left": (vertical_coverage, 0, edge_width),
        "right": (vertical_coverage, width - edge_width, width),
    }
    candidates_by_side = {
        side: merged_bands(values, start, stop)
        for side, (values, start, stop) in side_data.items()
    }

    def select_opposite_pair(
        first_side: str,
        second_side: str,
        axis_length: int,
        *,
        allow_reflected_missing_side: bool,
    ) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
        minimum_thickness = max(4, round(axis_length * 0.004))
        symmetry_tolerance = max(4, round(axis_length * 0.020))
        thickness_tolerance = max(3, round(axis_length * 0.012))
        maximum_layer_depth = max(6, round(axis_length * 0.040))

        strong = {
            first_side: [
                band
                for band in candidates_by_side[first_side]
                if band[1] - band[0] + 1 >= minimum_thickness
            ],
            second_side: [
                band
                for band in candidates_by_side[second_side]
                if band[1] - band[0] + 1 >= minimum_thickness
            ],
        }
        paired: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for first in strong[first_side]:
            for second in strong[second_side]:
                first_inset = first[0]
                second_inset = axis_length - 1 - second[1]
                first_width = first[1] - first[0] + 1
                second_width = second[1] - second[0] + 1
                if abs(first_inset - second_inset) > symmetry_tolerance:
                    continue
                if abs(first_width - second_width) > thickness_tolerance:
                    continue
                if second[1] - first[0] < axis_length * 0.60:
                    continue
                paired.append((first, second))

        inferred_side: str | None = None
        if paired:
            paired.sort(key=lambda pair: pair[1][1] - pair[0][0], reverse=True)
            selected_first, selected_second = paired[0]
            for other_first, other_second in paired[1:]:
                if (
                    abs(other_first[0] - selected_first[0]) > maximum_layer_depth
                    or abs(other_second[1] - selected_second[1])
                    > maximum_layer_depth
                ):
                    raise DetectionError(
                        f"multiple plausible paired-gradient {first_side}/{second_side} "
                        "frame boundaries were detected"
                    )
        elif allow_reflected_missing_side:
            first_values = strong[first_side]
            second_values = strong[second_side]
            if len(first_values) == 1 and not second_values:
                selected_first = first_values[0]
                selected_second = (
                    axis_length - 1 - selected_first[1],
                    axis_length - 1 - selected_first[0],
                )
                inferred_side = second_side
            elif len(second_values) == 1 and not first_values:
                selected_second = second_values[0]
                selected_first = (
                    axis_length - 1 - selected_second[1],
                    axis_length - 1 - selected_second[0],
                )
                inferred_side = first_side
            else:
                raise DetectionError(
                    f"no unique paired-gradient {first_side}/{second_side} "
                    "frame boundary was detected"
                )
        else:
            raise DetectionError(
                f"no unique paired-gradient {first_side}/{second_side} "
                "frame boundary was detected"
            )

        selected = {
            first_side: selected_first,
            second_side: selected_second,
        }
        rejected_unmatched = {
            first_side: max(
                0,
                len(strong[first_side])
                - (0 if inferred_side == first_side else 1),
            ),
            second_side: max(
                0,
                len(strong[second_side])
                - (0 if inferred_side == second_side else 1),
            ),
        }
        return selected, {
            "sides": [first_side, second_side],
            "minimum_band_thickness_px": minimum_thickness,
            "maximum_mirrored_inset_delta_px": symmetry_tolerance,
            "maximum_band_thickness_delta_px": thickness_tolerance,
            "maximum_parallel_layer_depth_px": maximum_layer_depth,
            "strong_candidate_count": {
                first_side: len(strong[first_side]),
                second_side: len(strong[second_side]),
            },
            "paired_candidate_count": len(paired),
            "inferred_side": inferred_side,
            "rejected_unmatched_band_count": rejected_unmatched,
        }

    if height >= width:
        long_sides = ("left", "right")
        short_sides = ("top", "bottom")
        long_length, short_length = width, height
    else:
        long_sides = ("top", "bottom")
        short_sides = ("left", "right")
        long_length, short_length = height, width
    long_bands, long_pair_evidence = select_opposite_pair(
        *long_sides,
        long_length,
        allow_reflected_missing_side=False,
    )
    short_bands, short_pair_evidence = select_opposite_pair(
        *short_sides,
        short_length,
        allow_reflected_missing_side=True,
    )
    bands = {**long_bands, **short_bands}

    coverage: dict[str, float] = {}
    for side, (values, start, stop) in side_data.items():
        selected = bands[side]
        inferred = side in {
            long_pair_evidence["inferred_side"],
            short_pair_evidence["inferred_side"],
        }
        if inferred:
            opposite_side = (
                short_sides[0] if side == short_sides[1] else short_sides[1]
            )
            opposite = bands[opposite_side]
            opposite_values = side_data[opposite_side][0]
            coverage[side] = float(
                np.max(opposite_values[opposite[0] : opposite[1] + 1])
            )
        else:
            coverage[side] = float(
                np.max(values[selected[0] : selected[1] + 1])
            )

    redirected = dict(geometry)
    redirected.update(
        {
            "left": bands["left"][0] - 0.5,
            "right": bands["right"][1] + 0.5,
            "top": bands["top"][0] - 0.5,
            "bottom": bands["bottom"][1] + 0.5,
            "dark_sides_px": {
                "left": bands["left"][0],
                "right": bands["right"][1],
                "top": bands["top"][0],
                "bottom": bands["bottom"][1],
            },
            "anchor_method": "paired-long-edge-gradient-fallback",
            "gradient_bands_px": {
                side: [int(start), int(stop)]
                for side, (start, stop) in bands.items()
            },
        }
    )
    if redirected["right"] <= redirected["left"] or redirected["bottom"] <= redirected["top"]:
        raise DetectionError("long-gradient outer-frame anchors do not form a closed card")
    redirected, radius_alignment = _normalize_corner_radii_from_source_gradients(
        rgb,
        redirected,
    )
    evidence = {
        "method": "paired-long-edge-gradient-fallback",
        "selection_mode": "paired-long-edge-gradient-fallback",
        "paired_long_sides": list(long_sides),
        "paired_short_sides": list(short_sides),
        "normal_gradient_threshold": gradient_threshold,
        "minimum_long_line_coverage": minimum_coverage,
        "maximum_merged_gap_px": maximum_gap,
        "long_line_coverage": coverage,
        "bands_px": {
            side: [int(start), int(stop)] for side, (start, stop) in bands.items()
        },
        "candidate_bands_px": {
            side: [[int(start), int(stop)] for start, stop in candidates]
            for side, candidates in candidates_by_side.items()
        },
        "long_pair_evidence": long_pair_evidence,
        "short_pair_evidence": short_pair_evidence,
        "inferred_sides": [
            side
            for side in (
                long_pair_evidence["inferred_side"],
                short_pair_evidence["inferred_side"],
            )
            if side is not None
        ],
        "rejected_unmatched_band_count": {
            **long_pair_evidence["rejected_unmatched_band_count"],
            **short_pair_evidence["rejected_unmatched_band_count"],
        },
        "corner_radius_alignment": radius_alignment,
    }
    return redirected, evidence


def _frame_geometry_evidence(rgb: np.ndarray, filled: np.ndarray) -> dict[str, Any]:
    """Require the same four-side and frame-structure proof on every path."""
    height, width = filled.shape
    x0, y0, x1, y1 = _tight_bbox(filled)
    frame_width = x1 - x0
    frame_height = y1 - y0
    if frame_width < 12 or frame_height < 12:
        raise DetectionError("outer frame evidence is too small to validate")

    leftmost = np.full(frame_height, np.nan)
    rightmost = np.full(frame_height, np.nan)
    for offset, y in enumerate(range(y0, y1)):
        xs = np.flatnonzero(filled[y])
        if len(xs):
            leftmost[offset] = xs[0]
            rightmost[offset] = xs[-1]
    topmost = np.full(frame_width, np.nan)
    bottommost = np.full(frame_width, np.nan)
    for offset, x in enumerate(range(x0, x1)):
        ys = np.flatnonzero(filled[:, x])
        if len(ys):
            topmost[offset] = ys[0]
            bottommost[offset] = ys[-1]

    horizontal_offsets = np.concatenate(
        (
            np.arange(round(frame_width * 0.18), round(frame_width * 0.40)),
            np.arange(round(frame_width * 0.60), round(frame_width * 0.82)),
        )
    )
    vertical_offsets = np.arange(
        round(frame_height * 0.18), round(frame_height * 0.82)
    )
    if not len(horizontal_offsets) or not len(vertical_offsets):
        raise DetectionError("outer frame evidence has no continuous side probes")
    left = int(round(float(np.nanmedian(leftmost[vertical_offsets]))))
    right = int(round(float(np.nanmedian(rightmost[vertical_offsets]))))
    top = int(round(float(np.nanmedian(topmost[horizontal_offsets]))))
    bottom = int(round(float(np.nanmedian(bottommost[horizontal_offsets]))))
    side_tolerance = max(2, round(min(frame_width, frame_height) * 0.006))
    continuous = {
        "top": float(np.mean(np.abs(topmost[horizontal_offsets] - top) <= side_tolerance)),
        "right": float(
            np.mean(np.abs(rightmost[vertical_offsets] - right) <= side_tolerance)
        ),
        "bottom": float(
            np.mean(np.abs(bottommost[horizontal_offsets] - bottom) <= side_tolerance)
        ),
        "left": float(np.mean(np.abs(leftmost[vertical_offsets] - left) <= side_tolerance)),
    }

    corner_offset = max(1, min(frame_width - 1, round(frame_width * 0.02)))
    corner_delta = max(1, round(min(frame_width, frame_height) * 0.0025))
    corner_offsets = {
        "top_left": float(topmost[corner_offset] - top),
        "top_right": float(topmost[-corner_offset - 1] - top),
        "bottom_left": float(bottom - bottommost[corner_offset]),
        "bottom_right": float(bottom - bottommost[-corner_offset - 1]),
    }
    rounded_corner_evidence = sum(
        value >= corner_delta for value in corner_offsets.values()
    ) >= 3

    depth = max(8, round(min(frame_width, frame_height) * 0.08))

    def transition_groups(profile: np.ndarray) -> int:
        if len(profile) < 3:
            return 0
        changes = np.max(
            np.abs(np.diff(profile.astype(np.int16), axis=0)), axis=1
        ) >= 12
        return int(np.sum(changes & np.concatenate(([True], ~changes[:-1]))))

    def double_layer_coverage(side: str) -> float:
        groups: list[int] = []
        if side in {"top", "bottom"}:
            step = max(1, len(horizontal_offsets) // 160)
            for offset in horizontal_offsets[::step]:
                x = x0 + int(offset)
                if side == "top":
                    start = max(0, top - 3)
                    stop = min(height, top + depth)
                    profile = rgb[start:stop, x]
                else:
                    start = max(0, bottom - depth)
                    stop = min(height, bottom + 4)
                    profile = rgb[start:stop, x][::-1]
                groups.append(transition_groups(profile))
        else:
            step = max(1, len(vertical_offsets) // 160)
            for offset in vertical_offsets[::step]:
                y = y0 + int(offset)
                if side == "left":
                    start = max(0, left - 3)
                    stop = min(width, left + depth)
                    profile = rgb[y, start:stop]
                else:
                    start = max(0, right - depth)
                    stop = min(width, right + 4)
                    profile = rgb[y, start:stop][::-1]
                groups.append(transition_groups(profile))
        return float(np.mean(np.asarray(groups) >= 2)) if groups else 0.0

    double_layer = {
        side: double_layer_coverage(side)
        for side in ("top", "right", "bottom", "left")
    }
    continuous_pass = min(continuous.values()) >= 0.50
    structure_pass = rounded_corner_evidence or min(double_layer.values()) >= 0.35
    passed = continuous_pass and structure_pass
    evidence = {
        "passed": passed,
        "continuous_side_coverage": continuous,
        "double_layer_coverage": double_layer,
        "rounded_corner_evidence": rounded_corner_evidence,
        "rounded_corner_offsets_px": corner_offsets,
        "anchors_px": {"top": top, "right": right, "bottom": bottom, "left": left},
        "criteria": {
            "minimum_continuous_side_coverage": 0.50,
            "minimum_double_layer_coverage": 0.35,
            "rounded_or_double_layer_required": True,
        },
    }
    if not passed:
        raise DetectionError(
            "outer frame evidence failed: four continuous sides plus rounded corners "
            "or parallel border layers are required"
        )
    return evidence


def _refine_center_ornament_foreground(
    rgb: np.ndarray,
    source_foreground: np.ndarray,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Separate narrow top/bottom ornaments from a connected exterior matte.

    The global source/background component intentionally keeps the complete
    frame and enclosed artwork, but on edge-hugging cards it may also connect a
    broad exterior paper or matte region to that frame.  For the only permitted
    exterior continuation -- a narrow top/bottom center ornament -- compare the
    center pixels with robust same-row flanks from both sides.  This preserves
    color-independent local source evidence while preventing a broad component
    from inheriting ornament status merely through connectivity.
    """
    height, width = source_foreground.shape
    if rgb.shape[:2] != (height, width):
        raise DetectionError("ornament foreground shape does not match the source")
    sides = geometry["dark_sides_px"]
    left = max(0, int(sides["left"]))
    right = min(width - 1, int(sides["right"]))
    top = max(0, int(sides["top"]))
    bottom = min(height - 1, int(sides["bottom"]))
    frame_width = right - left + 1
    if frame_width < 24 or right <= left or bottom <= top:
        raise DetectionError("frame is too small for local center-ornament evidence")

    maximum_width = max(12, round(frame_width * 0.24))
    center = (left + right) / 2.0
    center_start = max(left, int(round(center - maximum_width / 2)))
    center_stop = min(right + 1, int(round(center + maximum_width / 2)))
    left_flank_start = left + round(frame_width * 0.06)
    left_flank_stop = left + round(frame_width * 0.34)
    right_flank_start = left + round(frame_width * 0.66)
    right_flank_stop = min(right + 1, left + round(frame_width * 0.94))
    if left_flank_stop - left_flank_start < 4 or right_flank_stop - right_flank_start < 4:
        raise DetectionError("insufficient same-row flanks for center-ornament evidence")

    palette_flank_width = max(4, round(frame_width * 0.12))
    left_palette_start = max(left, center_start - palette_flank_width)
    left_palette_stop = center_start
    right_palette_start = center_stop
    right_palette_stop = min(right + 1, center_stop + palette_flank_width)
    if (
        left_palette_stop - left_palette_start < 4
        or right_palette_stop - right_palette_start < 4
    ):
        raise DetectionError(
            "insufficient adjacent palette flanks for center-ornament evidence"
        )

    center_x = np.arange(center_start, center_stop, dtype=np.float32)
    left_reference_x = (left_flank_start + left_flank_stop - 1) / 2.0
    right_reference_x = (right_flank_start + right_flank_stop - 1) / 2.0
    interpolation = np.clip(
        (center_x - left_reference_x) / max(1.0, right_reference_x - left_reference_x),
        0.0,
        1.0,
    )[:, None]
    refined = source_foreground.astype(bool, copy=True)
    strong_refined = np.zeros_like(refined)
    base_weak_refined = np.zeros_like(refined)
    palette_distinct_refined = np.zeros_like(refined)
    palette_novelty_distance = np.zeros_like(refined, dtype=np.float32)
    side_evidence: dict[str, Any] = {}

    def refine_rows(side: str, rows: range) -> None:
        row_indices = list(rows)
        weak_thresholds: dict[str, float] = {}
        strong_thresholds: dict[str, float] = {}
        background_novelty_thresholds: dict[str, float] = {}
        background_novelty_p99: dict[str, float] = {}
        retained_before_connectivity = 0
        removed_by_local_background = 0
        for row_position, y in enumerate(row_indices):
            original = refined[y].copy()
            left_values = rgb[y, left_flank_start:left_flank_stop].astype(np.float32)
            right_values = rgb[y, right_flank_start:right_flank_stop].astype(np.float32)
            left_median = np.median(left_values, axis=0)
            right_median = np.median(right_values, axis=0)
            left_residual = np.max(np.abs(left_values - left_median), axis=1)
            right_residual = np.max(np.abs(right_values - right_median), axis=1)
            flank_residual = np.concatenate((left_residual, right_residual))
            residual_median = float(np.median(flank_residual))
            residual_sigma = float(
                1.4826 * np.median(np.abs(flank_residual - residual_median))
            )
            weak_threshold = float(
                np.clip(
                    max(
                        6.0,
                        float(np.quantile(flank_residual, 0.70)) + 3.0,
                        residual_median + residual_sigma,
                    ),
                    6.0,
                    24.0,
                )
            )
            strong_threshold = float(
                np.clip(
                    max(
                        weak_threshold + 4.0,
                        float(np.quantile(flank_residual, 0.99)) + 2.0,
                        residual_median + 4.0 * residual_sigma,
                    ),
                    10.0,
                    64.0,
                )
            )
            modeled_center = (
                left_median[None, :] * (1.0 - interpolation)
                + right_median[None, :] * interpolation
            )
            center_residual = np.max(
                np.abs(rgb[y, center_start:center_stop].astype(np.float32) - modeled_center),
                axis=1,
            )
            # A same-row median can mistake a gently varying or embossed matte
            # for foreground when the center happens to differ from the two
            # flanks.  Prove that a candidate color is also novel relative to
            # the actual nearby flank palette.  Sampling a small vertical
            # neighborhood handles low-frequency shading without importing a
            # global, card-specific color rule.
            neighbor_rows = row_indices[
                max(0, row_position - 3) : min(len(row_indices), row_position + 4)
            ]
            nearby_background = np.concatenate(
                (
                    rgb[
                        neighbor_rows, left_palette_start:left_palette_stop
                    ].reshape(-1, 3),
                    rgb[
                        neighbor_rows, right_palette_start:right_palette_stop
                    ].reshape(-1, 3),
                ),
                axis=0,
            ).astype(np.float32)
            center_values = rgb[y, center_start:center_stop].astype(np.float32)
            nearest_background_distance = cKDTree(nearby_background).query(
                center_values,
                k=1,
                p=np.inf,
            )[0]
            background_novelty_threshold = 3.0
            palette_distinct_threshold = 5.0
            median_weak_support = center_residual > weak_threshold
            novel_color_support = (
                nearest_background_distance > background_novelty_threshold
            )
            weak_support = median_weak_support & novel_color_support
            strong_support = (
                center_residual > strong_threshold
            ) & novel_color_support
            refined[y].fill(False)
            refined[y, center_start:center_stop] = (
                original[center_start:center_stop] & weak_support
            )
            strong_refined[y].fill(False)
            strong_refined[y, center_start:center_stop] = (
                original[center_start:center_stop] & strong_support
            )
            base_weak_refined[y, center_start:center_stop] = (
                original[center_start:center_stop] & median_weak_support
            )
            palette_distinct_refined[y, center_start:center_stop] = (
                original[center_start:center_stop]
                & (center_residual > weak_threshold)
                & (nearest_background_distance > palette_distinct_threshold)
            )
            palette_novelty_distance[
                y, center_start:center_stop
            ] = nearest_background_distance
            weak_thresholds[str(y)] = weak_threshold
            strong_thresholds[str(y)] = strong_threshold
            background_novelty_thresholds[str(y)] = background_novelty_threshold
            background_novelty_p99[str(y)] = float(
                np.quantile(nearest_background_distance, 0.99)
            )
            retained_before_connectivity += int(refined[y].sum())
            removed_by_local_background += int(
                np.count_nonzero(original & ~refined[y])
            )

        disconnected_pixels = 0
        connected_component_count = 0
        rail_connected_component_count = 0
        bridged_one_pixel_gap_count = 0
        selected_bridge_records: list[dict[str, Any]] = []
        ambiguous_gap_forks: list[dict[str, Any]] = []
        removed_low_novelty_outer_tail_pixels = 0
        restored_short_horizontal_gap_pixels = 0
        retained_outer_limits: list[int] = []
        preserved_narrow_low_contrast_branches: list[dict[str, Any]] = []
        rejected_broad_chain_records: list[dict[str, Any]] = []
        base_weak_component_count = 0
        if row_indices:
            row_start = row_indices[0]
            row_stop = row_indices[-1] + 1
            local_source_support = refined[
                row_start:row_stop, center_start:center_stop
            ].copy()
            local_strong_support = strong_refined[
                row_start:row_stop, center_start:center_stop
            ]
            local_palette_distinct = palette_distinct_refined[
                row_start:row_stop, center_start:center_stop
            ]
            local_palette_novelty = palette_novelty_distance[
                row_start:row_stop, center_start:center_stop
            ]
            local_base_weak_support = base_weak_refined[
                row_start:row_stop, center_start:center_stop
            ]
            local_raw_source_support = source_foreground[
                row_start:row_stop, center_start:center_stop
            ]

            def narrow_branch_spatial_evidence(
                base_component: np.ndarray,
                base_rows: np.ndarray,
            ) -> dict[str, Any]:
                """Measure whether strong evidence is distributed along a branch."""

                palette_rows = np.flatnonzero(
                    np.any(local_palette_distinct & base_component, axis=1)
                )
                strong_rows = np.flatnonzero(
                    np.any(local_strong_support & base_component, axis=1)
                )
                branch_span = int(base_rows.max() - base_rows.min() + 1)
                if strong_rows.size:
                    unsupported_outward_tail = (
                        int(strong_rows.min() - base_rows.min())
                        if side == "top"
                        else int(base_rows.max() - strong_rows.max())
                    )
                else:
                    unsupported_outward_tail = branch_span
                maximum_unsupported_outward_tail = max(
                    2,
                    math.floor(branch_span * 0.60),
                )
                return {
                    "palette_rows": palette_rows,
                    "strong_rows": strong_rows,
                    "strong_source_pixels": int(
                        np.count_nonzero(local_strong_support & base_component)
                    ),
                    "branch_row_span_px": branch_span,
                    "unsupported_outward_tail_rows_px": unsupported_outward_tail,
                    "maximum_unsupported_outward_tail_rows_px": (
                        maximum_unsupported_outward_tail
                    ),
                }

            base_weak_labels, base_weak_component_count = ndimage.label(
                local_base_weak_support,
                structure=np.ones((3, 3), dtype=bool),
            )
            labels, connected_component_count = ndimage.label(
                local_source_support,
                structure=np.ones((3, 3), dtype=bool),
            )
            rail_row = -1 if side == "top" else 0
            rail_labels = np.unique(labels[rail_row])
            rail_labels = rail_labels[rail_labels != 0]

            def component_rows(label_id: int) -> np.ndarray:
                return np.flatnonzero(np.any(labels == label_id, axis=1))

            def contains_strong(label_id: int) -> bool:
                return bool(np.any(local_strong_support & (labels == label_id)))

            def horizontally_aligned_across_gap(
                rail_label: int, outer_label: int, rail_face: int, outer_face: int
            ) -> bool:
                rail_x = np.flatnonzero(labels[rail_face] == rail_label)
                outer_x = np.flatnonzero(labels[outer_face] == outer_label)
                if rail_x.size == 0 or outer_x.size == 0:
                    return False
                return bool(
                    np.any(
                        np.abs(rail_x[:, None] - outer_x[None, :]) <= 1
                    )
                )

            component_ids = list(range(1, connected_component_count + 1))
            rows_by_label = {
                label_id: component_rows(label_id) for label_id in component_ids
            }
            bridge_graph: dict[int, set[int]] = {
                label_id: set() for label_id in component_ids
            }
            bridge_records: list[dict[str, Any]] = []
            for inner_label in component_ids:
                inner_rows = rows_by_label[inner_label]
                if inner_rows.size == 0:
                    continue
                inner_face = (
                    int(inner_rows.min())
                    if side == "top"
                    else int(inner_rows.max())
                )
                expected_outer_face = (
                    inner_face - 2 if side == "top" else inner_face + 2
                )
                outward_candidates: list[tuple[int, int]] = []
                for outer_label in component_ids:
                    if outer_label == inner_label:
                        continue
                    outer_rows = rows_by_label[outer_label]
                    if outer_rows.size == 0:
                        continue
                    outer_face = (
                        int(outer_rows.max())
                        if side == "top"
                        else int(outer_rows.min())
                    )
                    if outer_face != expected_outer_face:
                        continue
                    if not horizontally_aligned_across_gap(
                        inner_label,
                        outer_label,
                        inner_face,
                        outer_face,
                    ):
                        continue
                    outward_candidates.append((outer_label, outer_face))
                if len(outward_candidates) == 1:
                    outer_label, outer_face = outward_candidates[0]
                    bridge_graph[inner_label].add(outer_label)
                    bridge_graph[outer_label].add(inner_label)
                    bridge_records.append(
                        {
                            "inner_label": inner_label,
                            "outer_label": outer_label,
                            "inner_face_row": inner_face + row_start,
                            "missing_row": (
                                inner_face - 1 if side == "top" else inner_face + 1
                            )
                            + row_start,
                            "outer_face_row": outer_face + row_start,
                        }
                    )
                elif len(outward_candidates) > 1:
                    ambiguous_gap_forks.append(
                        {
                            "inner_label": inner_label,
                            "inner_face_row": inner_face + row_start,
                            "candidate_labels": [
                                outer_label
                                for outer_label, _outer_face in outward_candidates
                            ],
                        }
                    )

            selected_labels: set[int] = set()
            selected_source_support = np.zeros_like(local_source_support)
            preserved_base_label_ids: set[int] = set()
            visited_rail_labels: set[int] = set()
            for label_id in rail_labels:
                rail_label = int(label_id)
                if rail_label in visited_rail_labels:
                    continue
                chain_labels = {rail_label}
                pending = [rail_label]
                while pending:
                    current_label = pending.pop()
                    for connected_label in bridge_graph[current_label]:
                        if connected_label not in chain_labels:
                            chain_labels.add(connected_label)
                            pending.append(connected_label)
                visited_rail_labels.update(chain_labels)
                chain_rows = np.unique(
                    np.concatenate(
                        [rows_by_label[chain_label] for chain_label in chain_labels]
                    )
                )
                outward_span = int(chain_rows.max() - chain_rows.min() + 1)
                chain_source_pixels = int(
                    np.count_nonzero(
                        local_source_support & np.isin(labels, list(chain_labels))
                    )
                )
                if (
                    chain_rows.size < 3
                    or outward_span < 4
                    or chain_source_pixels < 4
                    or not any(contains_strong(chain_label) for chain_label in chain_labels)
                ):
                    continue
                chain_support = local_source_support & np.isin(
                    labels, list(chain_labels)
                )
                active_chain_row_widths = np.count_nonzero(
                    chain_support, axis=1
                )
                active_chain_row_widths = active_chain_row_widths[
                    active_chain_row_widths > 0
                ]
                median_chain_row_width = float(
                    np.median(active_chain_row_widths)
                )
                maximum_median_chain_row_width = max(
                    4, round(frame_width * 0.18)
                )
                if median_chain_row_width > maximum_median_chain_row_width:
                    rejected_broad_chain_records.append(
                        {
                            "chain_labels": sorted(chain_labels),
                            "median_active_row_width_px": median_chain_row_width,
                            "maximum_allowed_median_row_width_px": (
                                maximum_median_chain_row_width
                            ),
                        }
                    )
                    continue
                palette_components, palette_component_count = ndimage.label(
                    chain_support & local_palette_distinct,
                    structure=np.ones((3, 3), dtype=bool),
                )
                maximum_palette_chain_row_width = max(
                    4,
                    round(frame_width * 0.20),
                )
                qualifying_palette_rows: list[np.ndarray] = []
                broad_palette_component_records: list[dict[str, Any]] = []
                for palette_label in range(1, palette_component_count + 1):
                    palette_component = palette_components == palette_label
                    palette_rows = np.flatnonzero(np.any(palette_component, axis=1))
                    if palette_rows.size == 0:
                        continue
                    palette_span = int(
                        palette_rows.max() - palette_rows.min() + 1
                    )
                    maximum_palette_novelty = float(
                        np.max(local_palette_novelty[palette_component])
                    )
                    if palette_span >= 3:
                        chain_row_widths_on_palette_rows = np.count_nonzero(
                            chain_support[palette_rows],
                            axis=1,
                        )
                        observed_palette_chain_row_width = int(
                            chain_row_widths_on_palette_rows.max()
                        )
                        if (
                            observed_palette_chain_row_width
                            > maximum_palette_chain_row_width
                        ):
                            broad_palette_component_records.append(
                                {
                                    "palette_component_label": palette_label,
                                    "palette_source_row_range_px": [
                                        int(palette_rows.min() + row_start),
                                        int(palette_rows.max() + row_start),
                                    ],
                                    "maximum_chain_row_width_on_palette_rows_px": (
                                        observed_palette_chain_row_width
                                    ),
                                    "maximum_allowed_chain_row_width_px": (
                                        maximum_palette_chain_row_width
                                    ),
                                    "maximum_palette_novelty_linf_px": (
                                        maximum_palette_novelty
                                    ),
                                }
                            )
                        else:
                            qualifying_palette_rows.append(palette_rows)
                if broad_palette_component_records:
                    rejected_broad_chain_records.append(
                        {
                            "chain_labels": sorted(chain_labels),
                            "reason": "broad-chain-width-on-qualifying-palette-rows",
                            "palette_components": broad_palette_component_records,
                        }
                    )
                    continue
                if not qualifying_palette_rows:
                    continue
                if side == "top":
                    outer_limit = min(
                        int(rows.min()) for rows in qualifying_palette_rows
                    )
                    outward_gate = (
                        np.arange(chain_support.shape[0])[:, None] >= outer_limit
                    )
                else:
                    outer_limit = max(
                        int(rows.max()) for rows in qualifying_palette_rows
                    )
                    outward_gate = (
                        np.arange(chain_support.shape[0])[:, None] <= outer_limit
                    )
                trimmed_chain_support = chain_support & outward_gate
                retained_chain_row_widths = np.count_nonzero(
                    trimmed_chain_support,
                    axis=1,
                )
                maximum_retained_chain_row_width = int(
                    retained_chain_row_widths.max(initial=0)
                )
                if (
                    maximum_retained_chain_row_width
                    > maximum_palette_chain_row_width
                ):
                    rejected_broad_chain_records.append(
                        {
                            "chain_labels": sorted(chain_labels),
                            "reason": "broad-chain-width-after-outward-tail-trim",
                            "maximum_retained_chain_row_width_px": (
                                maximum_retained_chain_row_width
                            ),
                            "maximum_allowed_chain_row_width_px": (
                                maximum_palette_chain_row_width
                            ),
                            "retained_outer_limit_px": outer_limit + row_start,
                        }
                    )
                    continue
                maximum_narrow_branch_width = max(4, round(frame_width * 0.035))
                overlapping_base_labels = np.unique(base_weak_labels[chain_support])
                overlapping_base_labels = overlapping_base_labels[
                    overlapping_base_labels != 0
                ]
                for base_label in overlapping_base_labels:
                    base_component = base_weak_labels == int(base_label)
                    base_rows = np.flatnonzero(np.any(base_component, axis=1))
                    if base_rows.size == 0:
                        continue
                    touches_rail = bool(
                        np.any(base_component[-1 if side == "top" else 0])
                    )
                    touches_source_edge = bool(
                        np.any(base_component[0 if side == "top" else -1])
                    )
                    row_widths = np.count_nonzero(base_component, axis=1)
                    maximum_row_width = int(row_widths.max())
                    maximum_novelty = float(
                        np.max(local_palette_novelty[base_component])
                    )
                    spatial_evidence = narrow_branch_spatial_evidence(
                        base_component,
                        base_rows,
                    )
                    palette_distinct_rows = spatial_evidence["palette_rows"]
                    strong_rows = spatial_evidence["strong_rows"]
                    strong_source_pixels = spatial_evidence["strong_source_pixels"]
                    if not (
                        touches_rail
                        and touches_source_edge
                        and maximum_row_width <= maximum_narrow_branch_width
                        and maximum_novelty > 64.0
                        and palette_distinct_rows.size >= 2
                        and strong_source_pixels >= 4
                        and strong_rows.size >= 3
                        and spatial_evidence[
                            "unsupported_outward_tail_rows_px"
                        ]
                        <= spatial_evidence[
                            "maximum_unsupported_outward_tail_rows_px"
                        ]
                    ):
                        continue
                    trimmed_chain_support |= base_component
                    preserved_base_label_ids.add(int(base_label))
                    preserved_narrow_low_contrast_branches.append(
                        {
                            "base_weak_component_label": int(base_label),
                            "source_row_range_px": [
                                int(base_rows.min() + row_start),
                                int(base_rows.max() + row_start),
                            ],
                            "maximum_row_width_px": maximum_row_width,
                            "maximum_allowed_row_width_px": (
                                maximum_narrow_branch_width
                            ),
                            "maximum_palette_novelty_linf_px": maximum_novelty,
                            "palette_distinct_source_row_count": int(
                                palette_distinct_rows.size
                            ),
                            "strong_source_pixels": strong_source_pixels,
                            "strong_source_row_count": int(strong_rows.size),
                            "branch_row_span_px": spatial_evidence[
                                "branch_row_span_px"
                            ],
                            "unsupported_outward_tail_rows_px": spatial_evidence[
                                "unsupported_outward_tail_rows_px"
                            ],
                            "maximum_unsupported_outward_tail_rows_px": (
                                spatial_evidence[
                                    "maximum_unsupported_outward_tail_rows_px"
                                ]
                            ),
                        }
                    )
                removed_low_novelty_outer_tail_pixels += int(
                    np.count_nonzero(chain_support & ~trimmed_chain_support)
                )
                retained_outer_limits.append(outer_limit + row_start)
                selected_labels.update(chain_labels)
                selected_source_support |= trimmed_chain_support
                rail_connected_component_count += 1
                chain_bridge_records = [
                    record
                    for record in bridge_records
                    if record["inner_label"] in chain_labels
                    and record["outer_label"] in chain_labels
                ]
                selected_bridge_records.extend(chain_bridge_records)
                bridged_one_pixel_gap_count += len(chain_bridge_records)
            # A very thin antialiased branch can match the matte along most of
            # its outward length, leaving fewer than three palette-novel rows
            # for the gated topology above.  Restore such a branch only when
            # its complete median-residual component independently proves all
            # four constraints: rail contact, source-edge contact, narrowness,
            # and an extreme palette-novel source core.  Broad matte/haze
            # components therefore remain ineligible.
            maximum_narrow_branch_width = max(4, round(frame_width * 0.035))
            for base_label in range(1, base_weak_component_count + 1):
                if base_label in preserved_base_label_ids:
                    continue
                base_component = base_weak_labels == base_label
                base_rows = np.flatnonzero(np.any(base_component, axis=1))
                if base_rows.size < 4:
                    continue
                touches_rail = bool(
                    np.any(base_component[-1 if side == "top" else 0])
                )
                touches_source_edge = bool(
                    np.any(base_component[0 if side == "top" else -1])
                )
                row_widths = np.count_nonzero(base_component, axis=1)
                maximum_row_width = int(row_widths.max())
                maximum_novelty = float(
                    np.max(local_palette_novelty[base_component])
                )
                spatial_evidence = narrow_branch_spatial_evidence(
                    base_component,
                    base_rows,
                )
                palette_distinct_rows = spatial_evidence["palette_rows"]
                strong_rows = spatial_evidence["strong_rows"]
                strong_source_pixels = spatial_evidence["strong_source_pixels"]
                if not (
                    touches_rail
                    and touches_source_edge
                    and maximum_row_width <= maximum_narrow_branch_width
                    and maximum_novelty > 64.0
                    and palette_distinct_rows.size >= 2
                    and strong_source_pixels >= 4
                    and strong_rows.size >= 3
                    and spatial_evidence[
                        "unsupported_outward_tail_rows_px"
                    ]
                    <= spatial_evidence[
                        "maximum_unsupported_outward_tail_rows_px"
                    ]
                ):
                    continue
                selected_source_support |= base_component
                preserved_base_label_ids.add(base_label)
                rail_connected_component_count += 1
                preserved_narrow_low_contrast_branches.append(
                    {
                        "base_weak_component_label": base_label,
                        "source_row_range_px": [
                            int(base_rows.min() + row_start),
                            int(base_rows.max() + row_start),
                        ],
                        "maximum_row_width_px": maximum_row_width,
                        "maximum_allowed_row_width_px": (
                            maximum_narrow_branch_width
                        ),
                        "maximum_palette_novelty_linf_px": maximum_novelty,
                        "palette_distinct_source_row_count": int(
                            palette_distinct_rows.size
                        ),
                        "strong_source_pixels": strong_source_pixels,
                        "strong_source_row_count": int(strong_rows.size),
                        "branch_row_span_px": spatial_evidence[
                            "branch_row_span_px"
                        ],
                        "unsupported_outward_tail_rows_px": spatial_evidence[
                            "unsupported_outward_tail_rows_px"
                        ],
                        "maximum_unsupported_outward_tail_rows_px": (
                            spatial_evidence[
                                "maximum_unsupported_outward_tail_rows_px"
                            ]
                        ),
                        "restoration_path": (
                            "independent-narrow-edge-to-rail-base-weak-component"
                        ),
                    }
                )
            # Decorative flowers often contain a few source-supported pixels
            # inside short horizontal openings that remain connected only in
            # the continuous-tone source, especially where the flower meets a
            # canvas edge. Restore only original foreground pixels bracketed
            # by already-selected support on the same row; do not synthesize
            # the intervening gap and do not expand either outer endpoint.
            maximum_horizontal_gap = max(1, round(frame_width * 0.006))
            for local_y in range(selected_source_support.shape[0]):
                selected_x = np.flatnonzero(selected_source_support[local_y])
                if selected_x.size < 2:
                    continue
                runs_start = selected_x[
                    np.concatenate(([True], np.diff(selected_x) > 1))
                ]
                runs_stop = selected_x[
                    np.concatenate((np.diff(selected_x) > 1, [True]))
                ]
                for previous_stop, next_start in zip(
                    runs_stop[:-1], runs_start[1:], strict=True
                ):
                    gap_start = int(previous_stop) + 1
                    gap_stop = int(next_start)
                    if gap_stop - gap_start > maximum_horizontal_gap:
                        continue
                    restore = local_raw_source_support[
                        local_y, gap_start:gap_stop
                    ]
                    restored_short_horizontal_gap_pixels += int(
                        np.count_nonzero(
                            restore
                            & ~selected_source_support[
                                local_y, gap_start:gap_stop
                            ]
                        )
                    )
                    selected_source_support[
                        local_y, gap_start:gap_stop
                    ] |= restore
            connected_source_support = selected_source_support
            disconnected_pixels = int(
                np.count_nonzero(local_source_support & ~connected_source_support)
            )
            refined[row_start:row_stop, center_start:center_stop] = (
                connected_source_support
            )

        retained = int(
            refined[row_indices, center_start:center_stop].sum()
            if row_indices
            else 0
        )
        side_evidence[side] = {
            "rows_evaluated": len(row_indices),
            "row_threshold_linf_px": weak_thresholds,
            "weak_threshold_linf_px": weak_thresholds,
            "strong_threshold_linf_px": strong_thresholds,
            "background_novelty_threshold_linf_px": background_novelty_thresholds,
            "background_novelty_distance_p99_linf_px": background_novelty_p99,
            "retained_source_supported_pixels": retained,
            "retained_before_connectivity_pixels": retained_before_connectivity,
            "removed_broad_component_pixels": removed_by_local_background,
            "removed_disconnected_texture_pixels": disconnected_pixels,
            "removed_low_novelty_outer_tail_pixels": (
                removed_low_novelty_outer_tail_pixels
            ),
            "restored_short_horizontal_gap_pixels": (
                restored_short_horizontal_gap_pixels
            ),
            "maximum_restored_horizontal_gap_px": max(
                1, round(frame_width * 0.006)
            ),
            "retained_outer_limits_px": retained_outer_limits,
            "base_weak_component_count": int(base_weak_component_count),
            "preserved_narrow_low_contrast_branch_count": len(
                preserved_narrow_low_contrast_branches
            ),
            "preserved_narrow_low_contrast_branches": (
                preserved_narrow_low_contrast_branches
            ),
            "rejected_broad_chain_count": len(rejected_broad_chain_records),
            "rejected_broad_chains": rejected_broad_chain_records,
            "connectivity_component_count": int(connected_component_count),
            "rail_connected_component_count": rail_connected_component_count,
            "connectivity_dilation_px": 0,
            "bridged_one_pixel_gap_count": bridged_one_pixel_gap_count,
            "bridged_one_pixel_gaps": selected_bridge_records,
            "ambiguous_gap_fork_count": len(ambiguous_gap_forks),
            "ambiguous_gap_forks": ambiguous_gap_forks,
            "maximum_bridged_gap_px": 1,
            "gap_bridge_direction": "outward-normal-only",
            "minimum_rail_neck_source_rows": 3,
            "minimum_rail_neck_outward_span_px": 4,
            "minimum_rail_neck_source_pixels": 4,
            "palette_distinct_threshold_linf_px": 5.0,
            "minimum_palette_distinct_outward_span_px": 3,
            "maximum_median_chain_row_width_fraction": 0.18,
            "maximum_palette_evidence_chain_row_width_fraction": 0.20,
            "narrow_low_contrast_branch_maximum_width_fraction": 0.035,
            "narrow_low_contrast_branch_extreme_novelty_linf_px": 64.0,
            "narrow_low_contrast_branch_minimum_palette_distinct_rows": 2,
            "narrow_low_contrast_branch_minimum_strong_source_rows": 3,
            "narrow_low_contrast_branch_maximum_unsupported_outward_tail_fraction": (
                0.60
            ),
        }

    refine_rows("top", range(0, top))
    refine_rows("bottom", range(bottom + 1, height))
    evidence = {
        "policy": "same-row-flank-background-difference",
        "color_policy": "per-channel-local-background-distance",
        "center_zone_px": [center_start, center_stop],
        "maximum_center_ornament_width_px": maximum_width,
        "maximum_center_ornament_width_fraction": 0.24,
        "left_flank_px": [left_flank_start, left_flank_stop],
        "right_flank_px": [right_flank_start, right_flank_stop],
        "flank_policy": "same-row-left-and-right-away-from-center",
        "left_palette_flank_px": [left_palette_start, left_palette_stop],
        "right_palette_flank_px": [right_palette_start, right_palette_stop],
        "palette_flank_policy": (
            "immediately-adjacent-left-and-right-within-plus-or-minus-3-rows"
        ),
        "flank_residual_quantile": 0.70,
        "residual_margin_linf_px": 3.0,
        "minimum_threshold_linf_px": 6.0,
        "maximum_threshold_linf_px": 24.0,
        "threshold_policy": {
            "robust_sigma": "1.4826 * MAD(flank residuals)",
            "weak": "clip(max(6, Q70+3, median+sigma), 6, 24)",
            "strong": "clip(max(weak+4, Q99+2, median+4*sigma), 10, 64)",
            "local_palette_novelty": (
                "candidate L-inf distance to nearest color in adjacent flanks "
                "within +/-3 exterior rows must exceed 3"
            ),
        },
        "component_policy": "rail-neck-connected-weak-component-containing-strong-source-pixel",
        "connectivity_policy": (
            "retain-source-supported-components-connected-to-the-rail-adjacent-"
            "exterior-row-after-one-pixel-gap-bridging"
        ),
        "sides": side_evidence,
    }
    return refined, evidence


def _render_dark_frame_alpha(
    frame_component: np.ndarray,
    ornament_foreground: np.ndarray,
    geometry: dict[str, Any],
    *,
    supersample: int = 8,
    lock_to_source_support: bool = False,
) -> np.ndarray:
    height, width = frame_component.shape
    left = float(geometry["left"])
    right = float(geometry["right"])
    top = float(geometry["top"])
    bottom = float(geometry["bottom"])
    radii = geometry["radii"]
    top_left = float(radii["top_left"])
    top_right = float(radii["top_right"])
    bottom_left = float(radii["bottom_left"])
    bottom_right = float(radii["bottom_right"])

    points: list[tuple[float, float]] = [(left + top_left, top), (right - top_right, top)]
    points.extend(_arc(right - top_right, top + top_right, top_right, -90, 0)[1:])
    points.append((right, bottom - bottom_right))
    points.extend(_arc(right - bottom_right, bottom - bottom_right, bottom_right, 0, 90)[1:])
    points.append((left + bottom_left, bottom))
    points.extend(_arc(left + bottom_left, bottom - bottom_left, bottom_left, 90, 180)[1:])
    points.append((left, top + top_left))
    points.extend(_arc(left + top_left, top + top_left, top_left, 180, 270)[1:])

    high_resolution = Image.new("L", (width * supersample, height * supersample), 0)
    draw = ImageDraw.Draw(high_resolution)
    draw.polygon(
        [(round(x * supersample), round(y * supersample)) for x, y in points],
        fill=255,
    )
    def runs(row: np.ndarray) -> list[tuple[int, int]]:
        padded = np.concatenate(([False], row.astype(bool), [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]

    def draw_connected_ornament(rows: range) -> None:
        ordered_rows = list(rows)
        if not ordered_rows:
            return
        frame_center = (left + right) / 2.0
        maximum_width = max(12, round((right - left) * 0.24))
        center_start = int(round(frame_center - maximum_width / 2))
        center_stop = int(round(frame_center + maximum_width / 2))
        growth = max(2, round(maximum_width * 0.025))

        def central_envelope(intervals: list[tuple[int, int]]) -> tuple[int, int] | None:
            clipped = []
            for start, stop in intervals:
                if stop <= center_start or start >= center_stop:
                    continue
                # The confirmed horizontal frame rail itself is normally much
                # wider than the local ornament window.  It is a valid seed,
                # but only its centered intersection may seed the outward
                # continuation; subsequent rows remain subject to the narrow
                # local-run gate below.
                if stop - start > maximum_width:
                    if not start <= frame_center < stop:
                        continue
                    clipped.append((center_start, center_stop))
                else:
                    clipped.append(
                        (max(start, center_start), min(stop, center_stop))
                    )
            clipped = [interval for interval in clipped if interval[1] > interval[0]]
            if not clipped:
                return None
            return min(interval[0] for interval in clipped), max(
                interval[1] for interval in clipped
            )

        seed_interval: tuple[int, int] | None = None
        seed_index = 0
        for seed_index, y in enumerate(ordered_rows[:4]):
            seed_interval = central_envelope(runs(frame_component[y]))
            if seed_interval is not None:
                break
        if seed_interval is None:
            return
        left_points: list[tuple[int, int]] = []
        right_points: list[tuple[int, int]] = []
        previous = seed_interval
        # The seed belongs to the confirmed rail and is already covered by the
        # geometric fill.  Continue from the first exterior row so a wide
        # horizontal rail cannot be mistaken for a wide ornament run.
        continuation_rows = ordered_rows[seed_index + 1 :]
        skip_row_index: int | None = None
        for row_index, y in enumerate(continuation_rows):
            if row_index == skip_row_index:
                continue
            allowed_start = max(center_start, previous[0] - growth)
            allowed_stop = min(center_stop, previous[1] + growth)
            row_runs = runs(ornament_foreground[y])

            def locally_narrows_on_next_row(interval: tuple[int, int]) -> bool:
                if row_index != 0 or row_index + 1 >= len(continuation_rows):
                    return False
                next_y = continuation_rows[row_index + 1]
                return any(
                    next_stop - next_start <= maximum_width
                    and next_stop >= max(interval[0], allowed_start) - growth
                    and next_start <= min(interval[1], allowed_stop) + growth
                    for next_start, next_stop in runs(ornament_foreground[next_y])
                )

            candidates = [
                (max(interval[0], allowed_start), min(interval[1], allowed_stop))
                for interval in row_runs
                if interval[1] >= previous[0] - growth
                and interval[0] <= previous[1] + growth
                and (
                    interval[1] - interval[0] <= maximum_width
                    or locally_narrows_on_next_row(interval)
                )
            ]
            candidates = [interval for interval in candidates if interval[1] > interval[0]]
            if not candidates:
                if row_index + 1 < len(continuation_rows):
                    next_y = continuation_rows[row_index + 1]
                    bridge_candidates = [
                        (
                            max(interval[0], allowed_start),
                            min(interval[1], allowed_stop),
                        )
                        for interval in runs(ornament_foreground[next_y])
                        if interval[1] - interval[0] <= maximum_width
                        and interval[1] >= previous[0] - 1
                        and interval[0] <= previous[1] + 1
                    ]
                    bridge_candidates = [
                        interval
                        for interval in bridge_candidates
                        if interval[1] > interval[0]
                    ]
                    if len(bridge_candidates) == 1:
                        selected = bridge_candidates[0]
                        left_points.append((selected[0] - 1, next_y))
                        right_points.append((selected[1] + 1, next_y))
                        previous = selected
                        skip_row_index = row_index + 1
                        continue
                break
            selected = (
                min(interval[0] for interval in candidates),
                max(interval[1] for interval in candidates),
            )
            left_points.append((selected[0] - 1, y))
            right_points.append((selected[1] + 1, y))
            previous = selected
        polygon = left_points + list(reversed(right_points))
        if polygon:
            draw.polygon(
                [(x * supersample, y * supersample) for x, y in polygon],
                fill=255,
            )

    dark_sides = geometry.get("dark_sides_px", {})
    top_connection_y = int(dark_sides.get("top", math.ceil(top) - 1))
    bottom_connection_y = int(dark_sides.get("bottom", math.floor(bottom) + 1))
    draw_connected_ornament(
        range(min(height - 1, max(0, top_connection_y)), -1, -1)
    )
    draw_connected_ornament(
        range(min(height - 1, max(0, bottom_connection_y)), height)
    )
    alpha = np.asarray(
        high_resolution.resize((width, height), Image.Resampling.BOX),
        dtype=np.uint8,
    ).copy()
    if lock_to_source_support:
        # The confirmed anchors are inclusive source-pixel rails.  The
        # supersampled polygon is useful only for selecting the narrow,
        # connected top/bottom ornament continuation: it must not invent
        # antialiased coverage outside the source-connected frame evidence.
        # Fill that evidence's enclosed interior, clip the core to the four
        # inclusive rail indices, and merge only source-supported ornament
        # pixels selected by the narrow continuation mask.
        sides = geometry["dark_sides_px"]
        inside_rails = np.zeros((height, width), dtype=bool)
        inside_rails[
            max(0, int(sides["top"])) : min(height, int(sides["bottom"]) + 1),
            max(0, int(sides["left"])) : min(width, int(sides["right"]) + 1),
        ] = True
        source_core = (
            ndimage.binary_fill_holes(ornament_foreground)
            & inside_rails
            & (alpha >= 8)
        )
        selected_ornament_support = ornament_foreground.astype(bool)
        outside_ornament_support = selected_ornament_support & ~inside_rails
        outside_labels, outside_component_count = ndimage.label(
            outside_ornament_support,
            structure=np.ones((3, 3), dtype=bool),
        )
        filled_outside_ornament = outside_ornament_support.copy()
        for outside_label in range(1, outside_component_count + 1):
            outside_component = outside_labels == outside_label
            filled_outside_ornament |= ndimage.binary_fill_holes(
                outside_component
            ) & ~outside_component
        connected_ornament = (
            (alpha > 0) & filled_outside_ornament
        )
        alpha.fill(0)
        alpha[source_core | connected_ornament] = 255
    alpha[alpha < 8] = 0
    alpha[alpha > 247] = 255
    return alpha


def _detect_dark_closed_frame(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = rgb.astype(np.float32) @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    candidates, threshold, connected_component_count = _enumerate_dark_frame_candidates(gray)
    if not candidates:
        raise DetectionError("no plausible closed dark outer-frame candidate was detected")

    # Parallel lines separated by only a few pixels may describe one coherent
    # frame.  Distinct nested boundaries remain separate plausible candidates
    # and are unsafe to choose between automatically.
    cluster_tolerance = max(4.0, min(gray.shape) * 0.015)
    clusters: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        sides = candidate["geometry"]["dark_sides_px"]
        for cluster in clusters:
            reference = cluster[0]["geometry"]["dark_sides_px"]
            if max(abs(float(sides[side]) - float(reference[side])) for side in sides) <= cluster_tolerance:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    if len(clusters) != 1:
        raise DetectionError(
            f"multiple plausible closed frame candidates were detected ({len(clusters)})"
        )
    selected = max(clusters[0], key=lambda item: int(item["filled_pixels"]))
    component = selected["component"]
    filled = selected["filled"]
    geometry = selected["geometry"]
    dark_sides = geometry["dark_sides_px"]
    height, width = gray.shape
    vertical_edge_slack = max(2, round(height * 0.002))
    horizontal_edge_slack = max(2, round(width * 0.002))
    edge_hugging = (
        dark_sides["top"] <= vertical_edge_slack
        or dark_sides["bottom"] >= height - 1 - vertical_edge_slack
        or dark_sides["left"] <= horizontal_edge_slack
        or dark_sides["right"] >= width - 1 - horizontal_edge_slack
    )
    gradient_redirect: dict[str, Any] | None = None
    if edge_hugging:
        try:
            geometry, gradient_redirect = _redirect_edge_hugging_geometry_from_gradients(
                rgb, geometry
            )
        except DetectionError as error:
            clipped_sides = [
                side
                for side, clipped in {
                    "top": dark_sides["top"] <= vertical_edge_slack,
                    "right": dark_sides["right"] >= width - 1 - horizontal_edge_slack,
                    "bottom": dark_sides["bottom"] >= height - 1 - vertical_edge_slack,
                    "left": dark_sides["left"] <= horizontal_edge_slack,
                }.items()
                if clipped
            ]
            raise DetectionError(
                "continuous outer frame is clipped by the source canvas: "
                + ", ".join(clipped_sides)
            ) from error
        geometry_evidence: dict[str, Any] | None = None
    else:
        geometry_evidence = _frame_geometry_evidence(rgb, filled)
    ornament_foreground, ornament_evidence = _source_connected_frame_foreground(
        rgb,
        geometry,
        anchors={
            side: int(geometry["dark_sides_px"][side])
            for side in ("top", "right", "bottom", "left")
        },
        probe_radius=max(2, round(min(width, height) * 0.006)),
    )
    ornament_foreground, local_ornament_evidence = (
        _refine_center_ornament_foreground(rgb, ornament_foreground, geometry)
    )
    ornament_evidence["local_center_ornament_refinement"] = local_ornament_evidence
    alpha = _render_dark_frame_alpha(
        component,
        ornament_foreground,
        geometry,
        lock_to_source_support=True,
    )
    visible = alpha > 0
    if not np.any(visible):
        raise DetectionError("dark outer-frame subject mask is empty")
    if edge_hugging:
        geometry_evidence = _frame_geometry_evidence(rgb, visible)
        geometry_evidence["continuous_side_coverage"] = gradient_redirect[
            "long_line_coverage"
        ]
        geometry_evidence["anchors_px"] = dict(geometry["dark_sides_px"])
        geometry_evidence["edge_hugging_component_redirect"] = gradient_redirect
    assert geometry_evidence is not None
    height, width = visible.shape
    contacts = {
        "top": int(visible[0].sum()),
        "right": int(visible[:, -1].sum()),
        "bottom": int(visible[-1].sum()),
        "left": int(visible[:, 0].sum()),
    }
    continuous_contact = [
        side
        for side, count in contacts.items()
        if count
        > ({"top": width, "bottom": width, "left": height, "right": height}[side] * 0.25)
    ]
    if continuous_contact:
        raise DetectionError(
            "continuous outer frame is clipped by the source canvas: "
            + ", ".join(continuous_contact)
        )
    details = {
        "method": "continuous-dark-frame-plus-rounded-geometric-fill",
        "candidate_count": len(candidates),
        "candidate_cluster_count": len(clusters),
        "connected_dark_component_count": connected_component_count,
        "candidate_enumeration": "adaptive-border-contrast-then-closed-frame-geometry",
        "candidate_cluster_tolerance_px": cluster_tolerance,
        "dark_threshold": threshold,
        "dark_component_pixels": int(component.sum()),
        "subject_pixels_after_fill": int(visible.sum()),
        "source_canvas_contacts_px": contacts,
        "frame_geometry_px": geometry,
        "geometry_evidence": geometry_evidence,
        "source_connected_ornament_evidence": ornament_evidence,
        "edge_hugging_component_geometry_redirected": edge_hugging,
    }
    return alpha, details


def _boundary_connected(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, _ = ndimage.label(mask, structure=structure)
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[1:-1, 0], labels[1:-1, -1]))
    )
    boundary_labels = boundary_labels[boundary_labels > 0]
    return np.isin(labels, boundary_labels)


def _normalize_validated_frame_override(
    override: dict[str, Any],
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(override, dict):
        raise DetectionError("validated frame override must be a JSON object")
    required_sides = ("left", "right", "top", "bottom")
    missing = [name for name in (*required_sides, "radii") if name not in override]
    if missing:
        raise DetectionError(
            "validated frame override is missing: " + ", ".join(missing)
        )
    try:
        sides = {
            side: int(round(float(override[side]))) for side in required_sides
        }
    except (TypeError, ValueError, OverflowError) as error:
        raise DetectionError(
            "validated frame override sides must be finite pixel coordinates"
        ) from error
    if any(not math.isfinite(float(override[side])) for side in required_sides):
        raise DetectionError(
            "validated frame override sides must be finite pixel coordinates"
        )

    raw_radii = override["radii"]
    radius_names = ("top_left", "top_right", "bottom_left", "bottom_right")
    if isinstance(raw_radii, (int, float)):
        raw_radii = {name: raw_radii for name in radius_names}
    if not isinstance(raw_radii, dict) or any(
        name not in raw_radii for name in radius_names
    ):
        raise DetectionError(
            "validated frame override radii must be a number or four named corner radii"
        )
    try:
        radii = {name: float(raw_radii[name]) for name in radius_names}
    except (TypeError, ValueError, OverflowError) as error:
        raise DetectionError(
            "validated frame override radii must be finite positive pixels"
        ) from error
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii.values()):
        raise DetectionError(
            "validated frame override radii must be finite positive pixels"
        )

    left, right = sides["left"], sides["right"]
    top, bottom = sides["top"], sides["bottom"]
    if not (0 < left < right < width - 1 and 0 < top < bottom < height - 1):
        raise DetectionError(
            "validated frame override must leave boundary-connected exterior on all sides"
        )
    frame_width = right - left
    frame_height = bottom - top
    if not 0.42 <= frame_width / frame_height <= 0.90:
        raise DetectionError(
            "validated frame override does not have a plausible portrait card aspect"
        )
    canvas_area = width * height
    frame_area = frame_width * frame_height
    if not canvas_area * 0.20 <= frame_area <= canvas_area * 0.995:
        raise DetectionError(
            "validated frame override has an implausible card area"
        )
    maximum_radius = min(frame_width, frame_height) / 2.0
    if any(radius > maximum_radius for radius in radii.values()):
        raise DetectionError(
            "validated frame override radius exceeds half of the frame size"
        )

    corner_radius_validation = override.get("corner_radius_validation")
    allowed_reviewed_radius_policy = "operator-reviewed-source-corner-tangents"
    if (
        corner_radius_validation is not None
        and corner_radius_validation != allowed_reviewed_radius_policy
    ):
        raise DetectionError(
            "validated frame override corner_radius_validation is unsupported"
        )

    requested = {
        "left": override["left"],
        "right": override["right"],
        "top": override["top"],
        "bottom": override["bottom"],
        "radii": {name: raw_radii[name] for name in radius_names},
    }
    if corner_radius_validation is not None:
        requested["corner_radius_validation"] = corner_radius_validation
    geometry = {
        "left": left - 0.5,
        "right": right + 0.5,
        "top": top - 0.5,
        "bottom": bottom + 0.5,
        "radii": radii,
        "dark_sides_px": sides,
        "anchor_method": "explicit-validated-frame-override",
    }
    return geometry, requested


def _validated_frame_override_gradient_alignment(
    rgb: np.ndarray,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Prove an explicit override follows four real, continuous image edges."""
    gray = rgb.astype(np.float32) @ np.asarray(
        [0.2126, 0.7152, 0.0722], dtype=np.float32
    )
    height, width = gray.shape
    sides = geometry["dark_sides_px"]
    frame_width = sides["right"] - sides["left"]
    frame_height = sides["bottom"] - sides["top"]
    horizontal_span = np.concatenate(
        (
            np.arange(
                sides["left"] + round(frame_width * 0.14),
                sides["left"] + round(frame_width * 0.42),
            ),
            np.arange(
                sides["left"] + round(frame_width * 0.58),
                sides["left"] + round(frame_width * 0.86),
            ),
        )
    )
    vertical_span = np.arange(
        sides["top"] + round(frame_height * 0.15),
        sides["top"] + round(frame_height * 0.85),
    )
    horizontal_span = horizontal_span[
        (horizontal_span >= 0) & (horizontal_span < width)
    ]
    vertical_span = vertical_span[(vertical_span >= 0) & (vertical_span < height)]
    if not len(horizontal_span) or not len(vertical_span):
        raise DetectionError(
            "validated frame override has no usable four-side gradient probes"
        )

    gradient_threshold = 18.0
    minimum_coverage = 0.70
    probe_radius = max(3, round(min(width, height) * 0.006))
    search_radius = probe_radius * 2
    vertical_signed_gradient = ndimage.sobel(gray, axis=0) / 4.0
    horizontal_signed_gradient = ndimage.sobel(gray, axis=1) / 4.0
    vertical_gradient = np.abs(vertical_signed_gradient)
    horizontal_gradient = np.abs(horizontal_signed_gradient)
    side_coverage: dict[str, float] = {}
    best_anchor: dict[str, int] = {}
    anchor_delta: dict[str, int] = {}
    candidate_bands: dict[str, list[dict[str, Any]]] = {}
    selected_bands: dict[str, list[int] | None] = {}
    for side in ("top", "right", "bottom", "left"):
        requested = int(sides[side])
        axis_limit = height if side in {"top", "bottom"} else width
        start = max(0, requested - search_radius)
        stop = min(axis_limit, requested + search_radius + 1)
        if side in {"top", "bottom"}:
            signed_values = vertical_signed_gradient[start:stop][:, horizontal_span]
            strong = np.abs(signed_values) >= gradient_threshold
            values = np.mean(strong, axis=1)
        else:
            signed_values = horizontal_signed_gradient[vertical_span, start:stop]
            strong = np.abs(signed_values) >= gradient_threshold
            values = np.mean(strong, axis=0)
        positive = np.mean(strong & (signed_values > 0), axis=(1 if side in {"top", "bottom"} else 0))
        negative = np.mean(strong & (signed_values < 0), axis=(1 if side in {"top", "bottom"} else 0))
        polarity = np.where(positive >= negative, 1, -1)
        active = np.flatnonzero(values >= minimum_coverage)
        bands: list[dict[str, Any]] = []
        if len(active):
            band_start = int(active[0])
            previous = int(active[0])
            for raw_position in active[1:]:
                position = int(raw_position)
                if (
                    position != previous + 1
                    or int(polarity[position]) != int(polarity[previous])
                ):
                    absolute_start = start + band_start
                    absolute_stop = start + previous
                    bands.append(
                        {
                            "band_px": [absolute_start, absolute_stop],
                            "representative_anchor_px": (
                                absolute_stop
                                if side in {"top", "left"}
                                else absolute_start
                            )
                            ,
                            "maximum_coverage": float(
                                np.max(values[band_start : previous + 1])
                            ),
                            "gradient_polarity": int(polarity[band_start]),
                        }
                    )
                    band_start = position
                previous = position
            absolute_start = start + band_start
            absolute_stop = start + previous
            bands.append(
                {
                    "band_px": [absolute_start, absolute_stop],
                    "representative_anchor_px": (
                        absolute_stop
                        if side in {"top", "left"}
                        else absolute_start
                    )
                    ,
                    "maximum_coverage": float(
                        np.max(values[band_start : previous + 1])
                    ),
                    "gradient_polarity": int(polarity[band_start]),
                }
            )
        candidate_bands[side] = bands
        eligible = [
            band
            for band in bands
            if abs(int(band["representative_anchor_px"]) - requested)
            <= probe_radius
        ]
        if not eligible:
            side_coverage[side] = 0.0
            best_anchor[side] = requested
            anchor_delta[side] = 0
            selected_bands[side] = None
            continue
        selected = min(
            eligible,
            key=lambda band: int(band["representative_anchor_px"]),
        ) if side in {"top", "left"} else max(
            eligible,
            key=lambda band: int(band["representative_anchor_px"]),
        )
        best = int(selected["representative_anchor_px"])
        side_coverage[side] = float(selected["maximum_coverage"])
        best_anchor[side] = best
        anchor_delta[side] = best - requested
        selected_bands[side] = list(selected["band_px"])
    weak = [
        side for side, value in side_coverage.items() if value < minimum_coverage
    ]
    evidence = {
        "passed": not weak,
        "normal_gradient_threshold": gradient_threshold,
        "minimum_side_coverage": minimum_coverage,
        "anchor_probe_radius_px": probe_radius,
        "anchor_search_radius_px": search_radius,
        "rail_selection_policy": "outermost-continuous-gradient-band",
        "side_coverage": side_coverage,
        "candidate_gradient_bands_px": candidate_bands,
        "selected_outer_rail_bands_px": selected_bands,
        "requested_anchors_px": dict(sides),
        "best_aligned_anchors_px": best_anchor,
        "best_anchor_delta_px": anchor_delta,
    }
    if weak:
        raise DetectionError(
            "validated frame override gradient alignment failed on: "
            + ", ".join(weak)
        )
    return evidence


def _snap_validated_geometry_to_aligned_anchors(
    geometry: dict[str, Any],
    alignment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = dict(geometry["dark_sides_px"])
    aligned = {
        side: int(alignment["best_aligned_anchors_px"][side])
        for side in ("left", "right", "top", "bottom")
    }
    if aligned["right"] <= aligned["left"] or aligned["bottom"] <= aligned["top"]:
        raise DetectionError(
            "validated frame override aligned anchors do not form a closed frame"
        )
    normalized = dict(geometry)
    normalized.update(
        {
            "left": aligned["left"] - 0.5,
            "right": aligned["right"] + 0.5,
            "top": aligned["top"] - 0.5,
            "bottom": aligned["bottom"] + 0.5,
            "dark_sides_px": aligned,
            "anchor_method": "explicit-override-snapped-to-source-gradient",
            "gradient_bands_px": {
                side: list(alignment["selected_outer_rail_bands_px"][side])
                for side in ("top", "right", "bottom", "left")
            },
        }
    )
    evidence = {
        "status": (
            "already-source-aligned"
            if requested == aligned
            else "snapped-to-source-gradient"
        ),
        "requested_anchors_px": requested,
        "normalized_anchors_px": aligned,
        "normalization_delta_px": {
            side: aligned[side] - requested[side] for side in aligned
        },
    }
    return normalized, evidence


def _source_connected_frame_foreground(
    rgb: np.ndarray,
    geometry: dict[str, Any],
    *,
    anchors: dict[str, int],
    probe_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep source/background differences physically connected to the frame."""
    height, width = rgb.shape[:2]
    sides = geometry["dark_sides_px"]
    frame_width = sides["right"] - sides["left"]
    frame_height = sides["bottom"] - sides["top"]
    horizontal_span = np.concatenate(
        (
            np.arange(
                sides["left"] + round(frame_width * 0.14),
                sides["left"] + round(frame_width * 0.42),
            ),
            np.arange(
                sides["left"] + round(frame_width * 0.58),
                sides["left"] + round(frame_width * 0.86),
            ),
        )
    )
    vertical_span = np.arange(
        sides["top"] + round(frame_height * 0.15),
        sides["top"] + round(frame_height * 0.85),
    )
    horizontal_span = horizontal_span[
        (horizontal_span >= 0) & (horizontal_span < width)
    ]
    vertical_span = vertical_span[(vertical_span >= 0) & (vertical_span < height)]
    probe_radius = max(2, int(probe_radius))
    side_probes: dict[str, np.ndarray] = {}
    for side in ("top", "right", "bottom", "left"):
        anchor = int(anchors[side])
        probe = np.zeros((height, width), dtype=bool)
        if side in {"top", "bottom"}:
            start = max(0, anchor - probe_radius)
            stop = min(height, anchor + probe_radius + 1)
            probe[start:stop, horizontal_span] = True
        else:
            start = max(0, anchor - probe_radius)
            stop = min(width, anchor + probe_radius + 1)
            probe[vertical_span, start:stop] = True
        side_probes[side] = probe

    provisional_background = _render_background(
        _perimeter_model(rgb), width, height
    )
    foreground_distance = np.max(
        np.abs(rgb.astype(np.int16) - provisional_background.astype(np.int16)),
        axis=2,
    )
    foreground_tolerance = _automatic_tolerance(rgb, provisional_background)
    raw_foreground = foreground_distance > foreground_tolerance
    closed = ndimage.binary_closing(
        raw_foreground,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    source_foreground = raw_foreground | closed
    labels, label_count = ndimage.label(
        source_foreground,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    selected_labels: list[int] = []
    selected_side_hits: dict[str, list[str]] = {}
    for label_id in range(1, label_count + 1):
        component = labels == label_id
        hits = [
            side for side, probe in side_probes.items() if np.any(component & probe)
        ]
        if len(hits) >= 2:
            selected_labels.append(label_id)
            selected_side_hits[str(label_id)] = hits
    connected = np.isin(labels, selected_labels)
    evidence = {
        "method": (
            "source-background-difference-components-connected-to-multiple-"
            "confirmed-frame-side-probes"
        ),
        "color_policy": "source-vs-modeled-exterior-difference",
        "foreground_tolerance": foreground_tolerance,
        "source_foreground_pixels": int(source_foreground.sum()),
        "source_connected_component_count": len(selected_labels),
        "source_connected_pixels": int(connected.sum()),
        "selected_component_side_hits": selected_side_hits,
    }
    return connected, evidence


def _validated_override_source_connected_foreground(
    rgb: np.ndarray,
    geometry: dict[str, Any],
    alignment: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    return _source_connected_frame_foreground(
        rgb,
        geometry,
        anchors={
            side: int(alignment["best_aligned_anchors_px"][side])
            for side in ("top", "right", "bottom", "left")
        },
        probe_radius=int(alignment["anchor_probe_radius_px"]),
    )


def _detect_validated_frame_override(
    rgb: np.ndarray,
    override: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = rgb.shape[:2]
    geometry, requested = _normalize_validated_frame_override(
        override,
        width,
        height,
    )
    alignment = _validated_frame_override_gradient_alignment(rgb, geometry)
    geometry, anchor_normalization = _snap_validated_geometry_to_aligned_anchors(
        geometry,
        alignment,
    )
    reviewed_radius_policy = override.get("corner_radius_validation")
    if reviewed_radius_policy == "operator-reviewed-source-corner-tangents":
        radius_alignment = {
            "status": "operator-reviewed",
            "passed": True,
            "policy": reviewed_radius_policy,
            "radii_px": dict(geometry["radii"]),
            "automatic_radius_normalization": "not-used",
        }
    else:
        geometry, radius_alignment = _normalize_corner_radii_from_source_gradients(
            rgb,
            geometry,
        )
    connected_foreground, ornament_evidence = (
        _validated_override_source_connected_foreground(rgb, geometry, alignment)
    )
    connected_foreground, local_ornament_evidence = (
        _refine_center_ornament_foreground(rgb, connected_foreground, geometry)
    )
    ornament_evidence["local_center_ornament_refinement"] = local_ornament_evidence
    alpha = _render_dark_frame_alpha(
        connected_foreground,
        connected_foreground,
        geometry,
        lock_to_source_support=True,
    )
    subject = alpha > 0
    geometry_evidence = _frame_geometry_evidence(rgb, subject)
    contacts = {
        "top": int(subject[0].sum()),
        "right": int(subject[:, -1].sum()),
        "bottom": int(subject[-1].sum()),
        "left": int(subject[:, 0].sum()),
    }
    ornament_evidence["retained_contact_pixels"] = contacts
    details = {
        "method": "explicit-validated-frame-override",
        "candidate_count": 1,
        "source_canvas_contacts_px": contacts,
        "frame_geometry_px": geometry,
        "geometry_evidence": geometry_evidence,
        "validated_frame_override": {
            "status": "applied",
            "validation": (
                "four-side-gradient-snap-plus-operator-reviewed-corner-radius-"
                "and-frame-structure"
                if reviewed_radius_policy
                == "operator-reviewed-source-corner-tangents"
                else "four-side-gradient-snap-plus-corner-radius-and-frame-structure"
            ),
            "requested_geometry_px": requested,
            "normalized_geometry_px": geometry,
            "gradient_alignment": alignment,
            "anchor_normalization": anchor_normalization,
            "corner_radius_alignment": radius_alignment,
            "source_connected_ornament_evidence": ornament_evidence,
        },
    }
    return subject, alpha, details


def _detect_closed_subject(
    rgb: np.ndarray,
    background_tolerance: int | None,
    *,
    validated_frame_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    """Find the unique large central region enclosed from the source exterior.

    The fill is geometric: pixels enclosed by the detected outer frame remain
    in the subject even when their RGB is identical to the source exterior.
    """
    height, width = rgb.shape[:2]
    provisional = _perimeter_model(rgb)
    predicted = _render_background(provisional, width, height)
    tolerance = (
        _automatic_tolerance(rgb, predicted)
        if background_tolerance is None
        else int(background_tolerance)
    )
    if not 1 <= tolerance <= 96:
        raise DetectionError(f"background tolerance must be in 1..96, got {tolerance}")
    if validated_frame_override is not None:
        subject, alpha, details = _detect_validated_frame_override(
            rgb,
            validated_frame_override,
        )
        details["background_tolerance"] = tolerance
        exterior = _boundary_connected(~subject)
        if np.any(exterior & subject):
            raise DetectionError("subject and exterior masks overlap")
        if int(exterior.sum()) < width * height * 0.03:
            raise DetectionError(
                "validated frame override leaves insufficient exterior for bleed sampling"
            )
        return subject, alpha, exterior.astype(bool), tolerance, details
    distance = np.max(np.abs(rgb.astype(np.int16) - predicted.astype(np.int16)), axis=2)
    background_candidate = distance <= tolerance
    exterior = _boundary_connected(background_candidate)
    non_exterior = ~exterior

    labels, count = ndimage.label(non_exterior, structure=np.ones((3, 3), dtype=np.uint8))
    candidates: list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]] = []
    canvas_area = width * height
    for label_id in range(1, count + 1):
        component = labels == label_id
        component_area = int(component.sum())
        if component_area < canvas_area * 0.05:
            continue
        filled = ndimage.binary_fill_holes(component)
        if not filled[height // 2, width // 2]:
            continue
        x0, y0, x1, y1 = _tight_bbox(filled)
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        aspect = bbox_width / bbox_height
        filled_area = int(filled.sum())
        if not 0.42 <= aspect <= 0.90:
            continue
        if not canvas_area * 0.20 <= filled_area <= canvas_area * 0.96:
            continue
        candidates.append((filled_area, component_area, filled, (x0, y0, x1, y1)))
    if len(candidates) == 1:
        filled_area, component_area, subject, _ = candidates[0]
        alpha = subject.astype(np.uint8) * 255
        touches = {
            "top": int(subject[0].sum()),
            "right": int(subject[:, -1].sum()),
            "bottom": int(subject[-1].sum()),
            "left": int(subject[:, 0].sum()),
        }
        continuous_contact = [
            side
            for side, contact_count in touches.items()
            if contact_count
            > ({"top": width, "bottom": width, "left": height, "right": height}[side] * 0.25)
        ]
        if continuous_contact:
            try:
                alpha, details = _detect_dark_closed_frame(rgb)
            except DetectionError as error:
                raise DetectionError(
                    "continuous outer frame is clipped by the source canvas: "
                    + ", ".join(continuous_contact)
                ) from error
            subject = alpha > 0
            details["color_model_candidate_count"] = len(candidates)
            details["color_model_rejected_reason"] = (
                "candidate merged the outer frame with a continuous canvas edge: "
                + ", ".join(continuous_contact)
            )
            details["color_model_rejected_contacts_px"] = touches
            details["background_tolerance"] = tolerance
        else:
            geometry_evidence = _frame_geometry_evidence(rgb, subject)
            details = {
                "method": "boundary-connected-exterior-plus-geometric-enclosure-fill",
                "candidate_count": len(candidates),
                "background_tolerance": tolerance,
                "component_pixels_before_fill": component_area,
                "subject_pixels_after_fill": filled_area,
                "geometrically_filled_pixels": filled_area - component_area,
                "source_canvas_contacts_px": touches,
                "geometry_evidence": geometry_evidence,
            }
    else:
        alpha, details = _detect_dark_closed_frame(rgb)
        subject = alpha > 0
        details["color_model_candidate_count"] = len(candidates)
        details["background_tolerance"] = tolerance

    exterior = _boundary_connected(~subject)
    if np.any(exterior & subject):
        raise DetectionError("subject and exterior masks overlap")
    if int(exterior.sum()) < canvas_area * 0.03:
        raise DetectionError("insufficient boundary-connected exterior for bleed sampling")
    return subject.astype(bool), alpha, exterior.astype(bool), tolerance, details


def _median_selected(values: np.ndarray, selected: np.ndarray, label: str) -> np.ndarray:
    pixels = values[selected]
    if len(pixels) < 3:
        raise DetectionError(f"insufficient exterior-only samples for {label}")
    return np.median(pixels.reshape(-1, 3), axis=0)


def _fit_reviewed_flat_exterior_background(
    rgb: np.ndarray,
    exterior: np.ndarray,
    subject: np.ndarray,
) -> tuple[BackgroundModel, np.ndarray, dict[str, Any]]:
    """Fit one continuous color from the current card's connected exterior.

    This is an explicit reviewed alternative for cards whose narrow exterior
    contains real texture that is unsuitable for a four-side gradient model.
    It never samples the protected subject and never stretches source edge
    rows or columns into the bleed.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DetectionError("reviewed flat exterior source must be RGB")
    if exterior.shape != rgb.shape[:2] or subject.shape != rgb.shape[:2]:
        raise DetectionError("reviewed flat exterior masks do not match the source")
    exterior = exterior.astype(bool, copy=False)
    subject = subject.astype(bool, copy=False)
    if np.any(exterior & subject):
        raise DetectionError("reviewed flat exterior sampling overlaps the subject")

    sample_mask = exterior.copy()
    sample_count = int(sample_mask.sum())
    minimum_samples = max(16, round(sample_mask.size * 0.005))
    if sample_count < minimum_samples:
        raise DetectionError("insufficient connected exterior for reviewed flat sampling")

    flat_color = np.rint(np.median(rgb[sample_mask], axis=0)).astype(np.uint8)
    coefficients = flat_color.astype(np.float64)[:, None]
    model = BackgroundModel(
        top=coefficients.copy(),
        right=coefficients.copy(),
        bottom=coefficients.copy(),
        left=coefficients.copy(),
        corners=np.repeat(flat_color.astype(np.float64)[None, :], 4, axis=0),
    )

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
        selected = sample_mask[ys, xs]
        count = int(selected.sum())
        if count < 3:
            raise DetectionError(
                f"reviewed flat exterior lacks source samples on the {side} side"
            )
        side_counts[side] = count
        denominator = selected.size
        side_coverage[side] = float(count / denominator)
        side_medians[side] = [
            int(round(value))
            for value in np.median(rgb[ys, xs][selected], axis=0)
        ]

    corner_counts: dict[str, int] = {}
    corner_coverage: dict[str, float] = {}
    corner_medians: dict[str, list[int]] = {}
    for corner, (ys, xs) in corner_regions.items():
        selected = sample_mask[ys, xs]
        count = int(selected.sum())
        if count < 3:
            raise DetectionError(
                f"reviewed flat exterior lacks source samples at {corner}"
            )
        corner_counts[corner] = count
        corner_coverage[corner] = float(count / selected.size)
        corner_medians[corner] = [
            int(round(value))
            for value in np.median(rgb[ys, xs][selected], axis=0)
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
    details: dict[str, Any] = {
        "sampling_method": "reviewed-flat-exterior-median-v1",
        "region_policy": "exterior-only",
        "flat_background_rgb": flat_color.tolist(),
        "flat_color_statistic": "per-channel-median-of-boundary-connected-exterior",
        "source_edge_extension": "forbidden",
        "subject_overlap_pixels": int(np.count_nonzero(sample_mask & subject)),
        "non_exterior_sample_pixels": int(np.count_nonzero(sample_mask & ~exterior)),
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
        "side_interpolation_method": "not-applicable-reviewed-flat-median",
        "side_interpolated_coordinate_count": dict(zeros_by_side),
        "side_interpolation_maximum_gap_px": dict(zeros_by_side),
        "side_interpolation_maximum_allowed_gap_px": dict(zeros_by_side),
        "side_interpolation_exceptions": not_needed_by_side,
    }
    return model, sample_mask, details


def _fit_exterior_background(
    rgb: np.ndarray,
    exterior: np.ndarray,
    subject: np.ndarray,
) -> tuple[BackgroundModel, np.ndarray, dict[str, Any]]:
    height, width = exterior.shape
    band = max(4, round(min(width, height) * 0.06))
    sample_mask = np.zeros_like(exterior)

    def interpolate_missing_same_side(
        result: np.ndarray,
        valid: np.ndarray,
        label: str,
        *,
        side: str,
        source_edge_contact: np.ndarray,
    ) -> tuple[np.ndarray, int, int, int, dict[str, Any]]:
        valid_coordinates = np.flatnonzero(valid)
        missing_coordinates = np.flatnonzero(~valid)
        maximum_allowed_gap = max(12, round(len(result) * 0.10))
        exception_evidence: dict[str, Any] = {
            "status": "not-needed",
            "reason": None,
            "base_maximum_gap_px": maximum_allowed_gap,
            "missing_run_px": None,
            "source_edge_contact_overlap_px": 0,
            "allowed_center_zone_px": None,
            "bracketing_direct_sample_coordinates_px": None,
        }
        if not len(missing_coordinates):
            return result, 0, 0, maximum_allowed_gap, exception_evidence
        if len(valid_coordinates) < 2:
            raise DetectionError(
                f"insufficient same-side exterior samples to interpolate {label}"
            )
        if (
            missing_coordinates[0] < valid_coordinates[0]
            or missing_coordinates[-1] > valid_coordinates[-1]
        ):
            raise DetectionError(
                f"same-side exterior samples do not bracket every missing {label} coordinate"
            )
        padded = np.concatenate(([False], (~valid).astype(bool), [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        maximum_gap = int(np.max(stops - starts)) if len(starts) else 0
        oversized_runs = [
            (int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
            if stop - start > maximum_allowed_gap
        ]
        if oversized_runs:
            center_zone_width = max(12, round(len(result) * 0.24))
            center_zone_start = math.floor((len(result) - center_zone_width) / 2)
            center_zone_stop = center_zone_start + center_zone_width
            allowed_exception: tuple[int, int] | None = None
            if len(oversized_runs) == 1 and side in {"top", "bottom"}:
                start, stop = oversized_runs[0]
                contact_overlap = int(source_edge_contact[start:stop].sum())
                directly_bracketed = (
                    start > 0
                    and stop < len(valid)
                    and bool(valid[start - 1])
                    and bool(valid[stop])
                )
                centered = start >= center_zone_start and stop <= center_zone_stop
                if contact_overlap > 0 and directly_bracketed and centered:
                    allowed_exception = (start, stop)
                    exception_evidence = {
                        "status": "allowed",
                        "reason": "centered-source-edge-ornament-contact",
                        "base_maximum_gap_px": maximum_allowed_gap,
                        "missing_run_px": [start, stop],
                        "source_edge_contact_overlap_px": contact_overlap,
                        "allowed_center_zone_px": [
                            center_zone_start,
                            center_zone_stop,
                        ],
                        "bracketing_direct_sample_coordinates_px": [
                            start - 1,
                            stop,
                        ],
                    }
            if allowed_exception is None:
                raise DetectionError(
                    f"same-side {label} interpolation gap of {maximum_gap} px exceeds "
                    f"the local evidence limit of {maximum_allowed_gap} px"
                )
        coordinate = np.arange(len(result), dtype=np.float64)
        for channel in range(3):
            result[~valid, channel] = np.interp(
                coordinate[~valid],
                coordinate[valid],
                result[valid, channel],
            )
        return (
            result,
            int(len(missing_coordinates)),
            maximum_gap,
            maximum_allowed_gap,
            exception_evidence,
        )

    def horizontal_samples(
        top: bool,
    ) -> tuple[np.ndarray, int, np.ndarray, int, int, int, dict[str, Any]]:
        result = np.full((width, 3), np.nan, dtype=np.float64)
        valid = np.zeros(width, dtype=bool)
        count = 0
        selected_mask = np.zeros_like(exterior)
        for x in range(width):
            ys = np.arange(0, band) if top else np.arange(height - band, height)
            allowed = exterior[ys, x]
            if not np.any(allowed):
                continue
            ys = ys[allowed]
            result[x] = np.median(rgb[ys, x], axis=0)
            valid[x] = True
            sample_mask[ys, x] = True
            selected_mask[ys, x] = True
            count += len(ys)
        side = "top" if top else "bottom"
        result, interpolated, maximum_gap, maximum_allowed_gap, exception = (
            interpolate_missing_same_side(
                result,
                valid,
                f"{side} image-column",
                side=side,
                source_edge_contact=subject[0] if top else subject[-1],
            )
        )
        return (
            result,
            count,
            selected_mask,
            interpolated,
            maximum_gap,
            maximum_allowed_gap,
            exception,
        )

    def vertical_samples(
        left: bool,
    ) -> tuple[np.ndarray, int, np.ndarray, int, int, int, dict[str, Any]]:
        result = np.full((height, 3), np.nan, dtype=np.float64)
        valid = np.zeros(height, dtype=bool)
        count = 0
        selected_mask = np.zeros_like(exterior)
        for y in range(height):
            xs = np.arange(0, band) if left else np.arange(width - band, width)
            allowed = exterior[y, xs]
            if not np.any(allowed):
                continue
            xs = xs[allowed]
            result[y] = np.median(rgb[y, xs], axis=0)
            valid[y] = True
            sample_mask[y, xs] = True
            selected_mask[y, xs] = True
            count += len(xs)
        side = "left" if left else "right"
        result, interpolated, maximum_gap, maximum_allowed_gap, exception = (
            interpolate_missing_same_side(
                result,
                valid,
                f"{side} image-row",
                side=side,
                source_edge_contact=subject[:, 0] if left else subject[:, -1],
            )
        )
        return (
            result,
            count,
            selected_mask,
            interpolated,
            maximum_gap,
            maximum_allowed_gap,
            exception,
        )

    (
        top,
        top_count,
        top_mask,
        top_interpolated,
        top_maximum_gap,
        top_maximum_allowed_gap,
        top_interpolation_exception,
    ) = horizontal_samples(True)
    (
        bottom,
        bottom_count,
        bottom_mask,
        bottom_interpolated,
        bottom_maximum_gap,
        bottom_maximum_allowed_gap,
        bottom_interpolation_exception,
    ) = horizontal_samples(False)
    (
        left,
        left_count,
        left_mask,
        left_interpolated,
        left_maximum_gap,
        left_maximum_allowed_gap,
        left_interpolation_exception,
    ) = vertical_samples(True)
    (
        right,
        right_count,
        right_mask,
        right_interpolated,
        right_maximum_gap,
        right_maximum_allowed_gap,
        right_interpolation_exception,
    ) = vertical_samples(False)

    corner_extent = max(band, round(min(width, height) * 0.08))
    corner_regions = (
        (slice(0, corner_extent), slice(0, corner_extent)),
        (slice(0, corner_extent), slice(width - corner_extent, width)),
        (slice(height - corner_extent, height), slice(width - corner_extent, width)),
        (slice(height - corner_extent, height), slice(0, corner_extent)),
    )
    corner_names = ("top_left", "top_right", "bottom_right", "bottom_left")
    corners: list[np.ndarray] = []
    corner_counts: dict[str, int] = {}
    for name, (ys, xs) in zip(corner_names, corner_regions, strict=True):
        selected = exterior[ys, xs]
        values = rgb[ys, xs]
        corners.append(_median_selected(values, selected, name))
        region = sample_mask[ys, xs]
        region[selected] = True
        sample_mask[ys, xs] = region
        corner_counts[name] = int(selected.sum())

    overlap = int(np.logical_and(sample_mask, subject).sum())
    outside_exterior = int(np.logical_and(sample_mask, ~exterior).sum())
    if overlap or outside_exterior:
        raise DetectionError("exterior sampling mask overlaps the subject or a non-exterior region")
    side_values = {"top": top, "right": right, "bottom": bottom, "left": left}
    side_masks = {
        "top": top_mask,
        "right": right_mask,
        "bottom": bottom_mask,
        "left": left_mask,
    }
    side_counts = {
        "top": top_count,
        "right": right_count,
        "bottom": bottom_count,
        "left": left_count,
    }
    coefficients = {side: _polyfit_robust(values) for side, values in side_values.items()}

    def longest_true_run(values: np.ndarray) -> int:
        padded = np.concatenate(([False], values.astype(bool), [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        return int(np.max(stops - starts)) if len(starts) else 0

    second_difference_limit = 24.0
    bad_fraction_limit = 0.15
    bad_run_limit = 0.10
    minimum_side_coverage = 0.08
    minimum_corner_coverage = 0.12
    coverage: dict[str, float] = {}
    dispersion: dict[str, dict[str, float]] = {}
    model_residual: dict[str, float] = {}
    polluted_sides: list[str] = []
    for side in ("top", "right", "bottom", "left"):
        mask = side_masks[side]
        medians = side_values[side]
        horizontal = side in {"top", "bottom"}
        axis_length = width if horizontal else height
        coverage[side] = float(side_counts[side] / (axis_length * band))
        coordinate_roughness = np.zeros(axis_length, dtype=np.float64)
        for coordinate in range(axis_length):
            if horizontal:
                positions = np.flatnonzero(mask[:, coordinate])
                pixels = rgb[positions, coordinate]
            else:
                positions = np.flatnonzero(mask[coordinate])
                pixels = rgb[coordinate, positions]
            if len(pixels) < 3:
                coordinate_roughness[coordinate] = 0.0
                continue
            second_difference = np.diff(pixels.astype(np.float64), n=2, axis=0)
            high_frequency = np.max(np.abs(second_difference), axis=1)
            coordinate_roughness[coordinate] = float(
                np.percentile(high_frequency, 90)
            )
        bad = coordinate_roughness > second_difference_limit
        bad_fraction = float(np.mean(bad))
        bad_run_fraction = float(longest_true_run(bad) / axis_length)
        dispersion[side] = {
            "coordinate_p90_second_difference_linf_p90": float(
                np.percentile(coordinate_roughness, 90)
            ),
            "bad_coordinate_fraction": bad_fraction,
            "longest_bad_run_fraction": bad_run_fraction,
        }
        fitted = _evaluate(coefficients[side], axis_length)
        model_residual[side] = float(
            np.percentile(np.max(np.abs(medians - fitted), axis=1), 90)
        )
        if (
            coverage[side] < minimum_side_coverage
            or bad_fraction > bad_fraction_limit
            or bad_run_fraction > bad_run_limit
        ):
            polluted_sides.append(side)

    corner_coverage = {
        name: float(corner_counts[name] / (corner_extent * corner_extent))
        for name in corner_names
    }
    weak_corners = [
        name for name, value in corner_coverage.items() if value < minimum_corner_coverage
    ]
    if polluted_sides or weak_corners:
        reasons: list[str] = []
        if polluted_sides:
            reasons.append("polluted/insufficient side texture: " + ", ".join(polluted_sides))
        if weak_corners:
            reasons.append("insufficient corner coverage: " + ", ".join(weak_corners))
        raise DetectionError("exterior sampling is unsafe; " + "; ".join(reasons))

    model = BackgroundModel(
        top=coefficients["top"],
        right=coefficients["right"],
        bottom=coefficients["bottom"],
        left=coefficients["left"],
        corners=np.stack(corners),
    )
    details = {
        "region_policy": "exterior-only",
        "subject_overlap_pixels": overlap,
        "non_exterior_sample_pixels": outside_exterior,
        "exterior_sample_pixels": int(sample_mask.sum()),
        "side_sample_pixels": {
            "top": top_count,
            "right": right_count,
            "bottom": bottom_count,
            "left": left_count,
        },
        "side_interpolated_coordinate_count": {
            "top": top_interpolated,
            "right": right_interpolated,
            "bottom": bottom_interpolated,
            "left": left_interpolated,
        },
        "side_interpolation_method": "same-side-valid-coordinate-linear",
        "side_interpolation_maximum_gap_px": {
            "top": top_maximum_gap,
            "right": right_maximum_gap,
            "bottom": bottom_maximum_gap,
            "left": left_maximum_gap,
        },
        "side_interpolation_maximum_allowed_gap_px": {
            "top": top_maximum_allowed_gap,
            "right": right_maximum_allowed_gap,
            "bottom": bottom_maximum_allowed_gap,
            "left": left_maximum_allowed_gap,
        },
        "side_interpolation_exceptions": {
            "top": top_interpolation_exception,
            "right": right_interpolation_exception,
            "bottom": bottom_interpolation_exception,
            "left": left_interpolation_exception,
        },
        "corner_sample_pixels": corner_counts,
        "side_coverage_fraction": coverage,
        "corner_coverage_fraction": corner_coverage,
        "side_texture_dispersion": dispersion,
        "side_model_residual_p90": model_residual,
        "sampling_quality_gate": {
            "status": "passed",
            "coordinate_second_difference_linf_p90_limit": second_difference_limit,
            "maximum_bad_coordinate_fraction": bad_fraction_limit,
            "maximum_bad_run_fraction": bad_run_limit,
            "minimum_side_coverage_fraction": minimum_side_coverage,
            "minimum_corner_coverage_fraction": minimum_corner_coverage,
        },
        "side_median_rgb": {
            "top": [int(round(value)) for value in np.median(top, axis=0)],
            "right": [int(round(value)) for value in np.median(right, axis=0)],
            "bottom": [int(round(value)) for value in np.median(bottom, axis=0)],
            "left": [int(round(value)) for value in np.median(left, axis=0)],
        },
        "corner_median_rgb": {
            name: [int(round(value)) for value in color]
            for name, color in zip(corner_names, corners, strict=True)
        },
        "corner_method": "direct-exterior-only-source-patch-median",
    }
    return model, sample_mask, details


def _fraction(value: float) -> Fraction:
    return Fraction(str(float(value))).limit_denominator(10000)


def _fixed_raster_geometry(
    subject_width: int,
    subject_height: int,
    trim_width_mm: float,
    trim_height_mm: float,
    bleed_left_mm: float,
    bleed_right_mm: float,
    bleed_top_mm: float,
    bleed_bottom_mm: float,
    minimum_effective_ppi: float,
) -> dict[str, Any]:
    media_width = _fraction(trim_width_mm + bleed_left_mm + bleed_right_mm)
    media_height = _fraction(trim_height_mm + bleed_top_mm + bleed_bottom_mm)
    ratio = media_width / media_height
    base_width, base_height = ratio.numerator, ratio.denominator
    scale_per_multiplier = float(Fraction(base_width, 1) / media_width)
    minimum_pixels_per_mm = minimum_effective_ppi / 25.4
    required_scale = max(
        subject_width / trim_width_mm,
        subject_height / trim_height_mm,
        minimum_pixels_per_mm,
    )
    minimum_multiplier = max(1, math.ceil(required_scale / scale_per_multiplier - 1e-12))

    def centering_penalty(multiplier: int) -> int:
        width = base_width * multiplier
        height = base_height * multiplier
        return (width - subject_width) % 2 + (height - subject_height) % 2

    candidates = range(minimum_multiplier, minimum_multiplier + 3)
    multiplier = min(candidates, key=lambda value: (centering_penalty(value), value))
    media_width_px = base_width * multiplier
    media_height_px = base_height * multiplier
    pixels_per_mm = media_width_px / float(media_width)
    trim_center_x_px = (bleed_left_mm + trim_width_mm / 2) * pixels_per_mm
    trim_center_y_px = (bleed_top_mm + trim_height_mm / 2) * pixels_per_mm
    placement_x = int(math.floor(trim_center_x_px - subject_width / 2 + 0.5))
    placement_y = int(math.floor(trim_center_y_px - subject_height / 2 + 0.5))
    center_error_x = placement_x + subject_width / 2 - trim_center_x_px
    center_error_y = placement_y + subject_height / 2 - trim_center_y_px
    subject_width_mm = subject_width / pixels_per_mm
    subject_height_mm = subject_height / pixels_per_mm
    x_mm = placement_x / pixels_per_mm
    y_mm = placement_y / pixels_per_mm
    trim_box = [
        bleed_left_mm,
        bleed_top_mm,
        bleed_left_mm + trim_width_mm,
        bleed_top_mm + trim_height_mm,
    ]
    epsilon = 1e-9
    if (
        x_mm < trim_box[0] - epsilon
        or y_mm < trim_box[1] - epsilon
        or x_mm + subject_width_mm > trim_box[2] + epsilon
        or y_mm + subject_height_mm > trim_box[3] + epsilon
    ):
        raise DetectionError("complete subject cannot fit inside the fixed trim without cropping")
    return {
        "media_size_px": [media_width_px, media_height_px],
        "raster_density_policy": "minimum-effective-ppi-floor-without-resampling",
        "pixels_per_mm": pixels_per_mm,
        "effective_ppi": pixels_per_mm * 25.4,
        "placement_px": [placement_x, placement_y],
        "placement_mm": {
            "trim_box_mm": [float(value) for value in trim_box],
            "x_mm": x_mm,
            "y_mm": y_mm,
            "width_mm": subject_width_mm,
            "height_mm": subject_height_mm,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "center_error_px": [center_error_x, center_error_y],
            "center_anchor": "trim-box",
            "cropped": False,
        },
    }


def _artifact(path: Path, source_sha256: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source_sha256": source_sha256,
    }


def prepare_subject_first(
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
    reviewed_flat_exterior: bool = False,
) -> dict[str, Any]:
    """Run the five-stage subject-first preparation and write audited artifacts."""
    dimensions = {
        "trim_width_mm": trim_width_mm,
        "trim_height_mm": trim_height_mm,
        "bleed_left_mm": bleed_left_mm,
        "bleed_right_mm": bleed_right_mm,
        "bleed_top_mm": bleed_top_mm,
        "bleed_bottom_mm": bleed_bottom_mm,
    }
    if trim_width_mm <= 0 or trim_height_mm <= 0:
        raise DetectionError("trim dimensions must be positive")
    if any(value < 0 for name, value in dimensions.items() if name.startswith("bleed_")):
        raise DetectionError("bleed dimensions must be non-negative")
    if not math.isfinite(minimum_effective_ppi) or minimum_effective_ppi < 300.0:
        raise DetectionError("minimum effective PPI must be at least the 300 PPI Skill floor")

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    with Image.open(source_path) as image:
        source_mode = image.mode
        rgb = np.asarray(image.convert("RGB")).copy()
    source_hash = _sha256(source_path)

    subject_mask, source_alpha, exterior, tolerance, detection = _detect_closed_subject(
        rgb,
        background_tolerance,
        validated_frame_override=validated_frame_override,
    )
    x0, y0, x1, y1 = _tight_bbox(subject_mask)
    crop_mask = subject_mask[y0:y1, x0:x1]
    crop_rgb = rgb[y0:y1, x0:x1]
    subject_height, subject_width = crop_mask.shape
    canonical_rgb = np.zeros_like(crop_rgb)
    canonical_rgb[crop_mask] = crop_rgb[crop_mask]
    alpha = source_alpha[y0:y1, x0:x1]
    binary_alpha = crop_mask.astype(np.uint8) * 255
    subject_rgba = np.dstack((canonical_rgb, alpha))

    if reviewed_flat_exterior:
        model, sampling_mask, sampling = _fit_reviewed_flat_exterior_background(
            rgb,
            exterior,
            subject_mask,
        )
    else:
        model, sampling_mask, sampling = _fit_exterior_background(
            rgb,
            exterior,
            subject_mask,
        )
    sampling.update(
        {
            "source_sha256": source_hash,
            "background_tolerance": tolerance,
            "sampling_mask_bbox_px": list(_tight_bbox(sampling_mask)),
        }
    )

    geometry = _fixed_raster_geometry(
        subject_width,
        subject_height,
        trim_width_mm,
        trim_height_mm,
        bleed_left_mm,
        bleed_right_mm,
        bleed_top_mm,
        bleed_bottom_mm,
        minimum_effective_ppi,
    )
    effective_ppi = float(geometry["effective_ppi"])
    if effective_ppi + 1e-9 < minimum_effective_ppi:
        raise DetectionError(
            f"effective PPI {effective_ppi:.3f} is below the required {minimum_effective_ppi:.3f}"
        )
    media_width_px, media_height_px = geometry["media_size_px"]
    placement_x, placement_y = geometry["placement_px"]
    background = _render_background(model, media_width_px, media_height_px)
    media = background.copy()
    destination = media[
        placement_y : placement_y + subject_height,
        placement_x : placement_x + subject_width,
    ]
    alpha_float = alpha.astype(np.float32)[..., None] / 255.0
    composite = np.rint(
        crop_rgb.astype(np.float32) * alpha_float
        + destination.astype(np.float32) * (1.0 - alpha_float)
    )
    destination[:] = np.clip(composite, 0, 255).astype(np.uint8)
    opaque = alpha == 255
    destination[opaque] = crop_rgb[opaque]

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "subject-rgba.png"
    mask_path = output_dir / "subject-mask.png"
    sampling_mask_path = output_dir / "exterior-sampling-mask.png"
    sampling_provenance_path = output_dir / "sampling-provenance.json"
    bleed_background_path = output_dir / "bleed-background-rgb.png"
    media_path = output_dir / "media-rgb.png"
    overlay_path = output_dir / "detection-overlay.png"
    sampling_overlay_path = output_dir / "exterior-sampling-overlay.png"
    layout_guide_path = output_dir / "layout-guide.png"
    manifest_path = output_dir / "manifest.json"
    Image.fromarray(subject_rgba, "RGBA").save(subject_path)
    Image.fromarray(binary_alpha, "L").save(mask_path)
    Image.fromarray(sampling_mask.astype(np.uint8) * 255, "L").save(sampling_mask_path)
    Image.fromarray(background, "RGB").save(bleed_background_path)
    Image.fromarray(media, "RGB").save(media_path)
    overlay = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 255), width=2)
    overlay.save(overlay_path)

    sampling_overlay = rgb.astype(np.float32)
    sampling_overlay[sampling_mask] = (
        sampling_overlay[sampling_mask] * 0.45
        + np.asarray((0, 255, 80), dtype=np.float32) * 0.55
    )
    subject_boundary = subject_mask & ~ndimage.binary_erosion(subject_mask)
    sampling_overlay[subject_boundary] = np.asarray((255, 0, 255), dtype=np.float32)
    Image.fromarray(np.clip(np.rint(sampling_overlay), 0, 255).astype(np.uint8), "RGB").save(
        sampling_overlay_path
    )

    layout_guide = Image.fromarray(media, "RGB")
    layout_draw = ImageDraw.Draw(layout_guide)
    pixels_per_mm = float(geometry["pixels_per_mm"])
    trim_left_px = int(round(bleed_left_mm * pixels_per_mm))
    trim_top_px = int(round(bleed_top_mm * pixels_per_mm))
    trim_right_px = int(round((bleed_left_mm + trim_width_mm) * pixels_per_mm)) - 1
    trim_bottom_px = int(round((bleed_top_mm + trim_height_mm) * pixels_per_mm)) - 1
    layout_draw.rectangle(
        (trim_left_px, trim_top_px, trim_right_px, trim_bottom_px),
        outline=(255, 0, 255),
        width=2,
    )
    layout_draw.rectangle(
        (
            placement_x,
            placement_y,
            placement_x + subject_width - 1,
            placement_y + subject_height - 1,
        ),
        outline=(0, 255, 255),
        width=2,
    )
    layout_guide.save(layout_guide_path)

    artifacts = {
        "subject_rgba": _artifact(subject_path, source_hash),
        "subject_mask": _artifact(mask_path, source_hash),
        "exterior_sampling_mask": _artifact(sampling_mask_path, source_hash),
        "bleed_background_rgb": _artifact(bleed_background_path, source_hash),
        "media_rgb": _artifact(media_path, source_hash),
    }
    sampling["sampling_mask_sha256"] = artifacts["exterior_sampling_mask"]["sha256"]
    provenance_payload = _sampling_provenance_artifact_payload(
        source_sha256=source_hash,
        sampling_mask_sha256=artifacts["exterior_sampling_mask"]["sha256"],
        sampling=sampling,
    )
    sampling_provenance_path.write_text(
        json.dumps(provenance_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts["sampling_provenance"] = _artifact(
        sampling_provenance_path,
        source_hash,
    )
    placed = media[
        placement_y : placement_y + subject_height,
        placement_x : placement_x + subject_width,
    ]
    exact_opaque = int(np.all(placed[opaque] == crop_rgb[opaque], axis=1).sum())
    media_mm = [
        float(trim_width_mm + bleed_left_mm + bleed_right_mm),
        float(trim_height_mm + bleed_top_mm + bleed_bottom_mm),
    ]
    placement_mm = dict(geometry["placement_mm"])
    placement_mm["source_bbox_px"] = [x0, y0, x1, y1]
    report: dict[str, Any] = {
        "pipeline_version": CURRENT_PIPELINE_VERSION,
        "sampling_provenance_schema_version": SAMPLING_PROVENANCE_SCHEMA_VERSION,
        "sampling_provenance_sha256": artifacts["sampling_provenance"]["sha256"],
        "pipeline_stages": [
            {"name": name, "status": "completed"} for name in PIPELINE_STAGE_NAMES
        ],
        "source": str(source_path.resolve()),
        "source_sha256": source_hash,
        "source_mode": source_mode,
        "source_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
        "subject_bbox_px": [x0, y0, x1, y1],
        "source_crop_bbox_px": [x0, y0, x1, y1],
        "subject_size_px": [subject_width, subject_height],
        "subject_image": str(subject_path.resolve()),
        "subject_mask": str(mask_path.resolve()),
        "exterior_sampling_mask": str(sampling_mask_path.resolve()),
        "sampling_provenance_artifact": str(sampling_provenance_path.resolve()),
        "bleed_background_image": str(bleed_background_path.resolve()),
        "media_image": str(media_path.resolve()),
        "detection_overlay": str(overlay_path.resolve()),
        "exterior_sampling_overlay": str(sampling_overlay_path.resolve()),
        "layout_guide": str(layout_guide_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "artifacts": artifacts,
        "subject_extraction": {
            "source_sha256": source_hash,
            "boundary_method": detection["method"],
            "subject_bbox_px": [x0, y0, x1, y1],
            "subject_mask_sha256": artifacts["subject_mask"]["sha256"],
            "subject_rgba_sha256": artifacts["subject_rgba"]["sha256"],
            "complete_closed_frame_and_enclosed_content": True,
        },
        "frame_detection": detection,
        "background_sampling": sampling,
        "trim_mm": [float(trim_width_mm), float(trim_height_mm)],
        "bleed_mm": {
            "left": float(bleed_left_mm),
            "right": float(bleed_right_mm),
            "top": float(bleed_top_mm),
            "bottom": float(bleed_bottom_mm),
        },
        "media_mm": media_mm,
        "media_size_px": geometry["media_size_px"],
        "raster_density_policy": geometry["raster_density_policy"],
        "placement_px": geometry["placement_px"],
        "subject_placement_mm": placement_mm,
        "frame_spacing": {
            "mode": "fixed-trim-and-bleed",
            "measurement_anchor": "complete-subject-bounds",
        },
        "frame_margins_px": {
            "top": placement_y,
            "right": media_width_px - placement_x - subject_width,
            "bottom": media_height_px - placement_y - subject_height,
            "left": placement_x,
        },
        "bleed_policy": (
            "current-card-exterior-only-reviewed-flat-median"
            if reviewed_flat_exterior
            else "current-card-exterior-only-side-and-corner-sampling"
        ),
        "transform_policy": "integer-translation-no-resample",
        "cropped": False,
        "resampled": False,
        "minimum_effective_ppi": float(minimum_effective_ppi),
        "effective_ppi": effective_ppi,
        "opaque_subject_pixels": int(opaque.sum()),
        "opaque_subject_exact_pixels": exact_opaque,
        "semi_transparent_edge_pixels": int(((alpha > 0) & (alpha < 255)).sum()),
        "media_sha256": artifacts["media_rgb"]["sha256"],
        "bleed_background_sha256": artifacts["bleed_background_rgb"]["sha256"],
    }
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


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
        "--reviewed-flat-exterior",
        action="store_true",
        help=(
            "use one continuous median color sampled only from this card's "
            "boundary-connected exterior; requires operator review"
        ),
    )
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
                    override_path = Path(raw_override)
                    validated_frame_override = json.loads(
                        override_path.read_text(encoding="utf-8")
                    )
            except (OSError, json.JSONDecodeError) as error:
                raise DetectionError(
                    "validated frame override must be valid JSON text or a readable JSON file"
                ) from error
        report = prepare_subject_first(
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
            reviewed_flat_exterior=args.reviewed_flat_exterior,
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
