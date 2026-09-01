from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest
from scipy import ndimage


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

import subject_first_pipeline as subject_pipeline  # noqa: E402
from subject_first_pipeline import (  # noqa: E402
    DetectionError,
    _detect_closed_subject,
    _detect_dark_closed_frame,
    _fit_exterior_background,
    _redirect_edge_hugging_geometry_from_gradients,
    _render_dark_frame_alpha,
    _source_connected_frame_foreground,
    prepare_subject_first,
)


EXTERIOR_GOLD = (197, 145, 67)
REAL_SOURCE_ENV = os.environ.get("FATE_TAROT_SOURCE_DIR")
REAL_SOURCE_ROOT = Path(
    REAL_SOURCE_ENV or str(SKILL_ROOT / ".missing-real-sources")
).expanduser()
DEVIL_SOURCE = REAL_SOURCE_ROOT / "0EFD5AE0-6E88-4D6D-9A5C-DC1643EC172E.png"
E0C8_SOURCE = DEVIL_SOURCE.parent / "E0C8E863-F59F-4C96-B39C-A2665E8F48A8.png"
STAR_SOURCE = DEVIL_SOURCE.parent / "0A8D6B2A-05AD-4C2C-9DC8-68096007CB7A.png"
SHARD1_MATTE_SOURCE = DEVIL_SOURCE.parent / "0E4CCED8-A13A-45C9-9F90-C757B74AD081.png"
SHARD1_GREEN_MATTE_SOURCE = (
    DEVIL_SOURCE.parent / "36CACAE2-9E29-41EE-9882-C806E59E8817.png"
)
SHARD1_BLUE_HAZE_SOURCE = (
    DEVIL_SOURCE.parent / "D0C4D41C-A84E-4D75-83EE-6463265B51E7.png"
)
LONG_EDGE_FALLBACK_SOURCES = (
    "00050EA0-90A6-4F13-820E-679917D7922D",
    "163CEE6C-49D1-4B06-963C-CEE7C3A840B6",
    "250B7328-9850-4B3B-8B41-263AFDDCC885",
    "304D62D7-0368-40E1-BCE0-9CBF0426A03F",
    "3FCAD3EE-97D3-4696-A880-37D969607FD4",
    "657D7FA7-291C-4E29-AE6B-0FD9B06492D6",
    "AE3F9094-A2B0-4AFE-ABE4-D4B10D979FF3",
    "D703AC8C-473C-4B5C-AE6E-740388A54B74",
    "E0C8E863-F59F-4C96-B39C-A2665E8F48A8",
    "F164A283-86C3-4C49-94E7-DEF5350D751F",
)
SAFE_REJECTION_OCCLUDED_BOTTOM_RAIL_SOURCE = (
    "C26E179C-2BF2-4DFD-8958-2B0E58E9EF93"
)
REAL_VALIDATED_FRAME_CASES = (
    (
        "DD2B56B5-2B53-4FC6-8BE0-C4E9225B7F50",
        {"top": 86, "right": 926, "bottom": 1439, "left": 98},
        40.0,
    ),
    (
        "F626A521-1B30-4DCE-A141-C535D9396221",
        {"top": 52, "right": 945, "bottom": 1497, "left": 77},
        42.0,
    ),
    (
        "C294A793-EE21-4E08-AE4E-79A38FEEED88",
        {"top": 49, "right": 954, "bottom": 1501, "left": 68},
        42.0,
    ),
)


def _require_real_source(source: Path, message: str) -> None:
    if source.is_file():
        return
    if REAL_SOURCE_ENV:
        pytest.fail(f"configured {message}: {source}")
    pytest.skip(f"{message}: {source}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_reviewed_flat_exterior_background_uses_only_current_card_exterior() -> None:
    """A reviewed flat bleed must not turn source edge details into stripes."""

    rgb = np.full((12, 16, 3), (198, 24, 20), dtype=np.uint8)
    subject = np.zeros((12, 16), dtype=bool)
    subject[2:10, 3:13] = True
    exterior = ~subject

    # Deliberately contaminate complete source-edge rows/columns.  A clamp or
    # edge-stretch implementation would reproduce these as visible bands.
    rgb[0, 5] = (0, 0, 0)
    rgb[-1, 9] = (255, 255, 255)
    rgb[4, 0] = (0, 120, 255)
    rgb[7, -1] = (255, 220, 0)
    rgb[subject] = (40, 80, 120)

    model, sampling_mask, details = (
        subject_pipeline._fit_reviewed_flat_exterior_background(
            rgb,
            exterior,
            subject,
        )
    )
    rendered = subject_pipeline._render_background(model, 90, 130)

    expected_color = np.rint(np.median(rgb[exterior], axis=0)).astype(np.uint8)
    assert details["sampling_method"] == "reviewed-flat-exterior-median-v1"
    assert details["region_policy"] == "exterior-only"
    assert details["flat_background_rgb"] == expected_color.tolist()
    assert np.array_equal(sampling_mask, exterior)
    assert not np.any(sampling_mask & subject)
    assert np.unique(rendered.reshape(-1, 3), axis=0).tolist() == [
        expected_color.tolist()
    ]


def _make_closed_frame_card(
    path: Path,
    *,
    exterior_rgb: tuple[int, int, int] = EXTERIOR_GOLD,
    interior_variant: int = 0,
    matching_exterior_patch: bool = True,
) -> None:
    """Create a closed rounded frame with connected top/bottom ornaments."""
    image = Image.new("RGB", (360, 540), exterior_rgb)
    draw = ImageDraw.Draw(image)

    # The complete subject is the closed frame, everything enclosed by it, and
    # the two ornaments that physically overlap the frame. Nothing else on the
    # exterior canvas belongs to the subject.
    draw.rounded_rectangle(
        (48, 64, 312, 476),
        radius=30,
        fill=(43, 91, 129),
        outline=(75, 39, 10),
        width=9,
    )
    draw.rounded_rectangle(
        (57, 73, 303, 467),
        radius=23,
        outline=(244, 202, 91),
        width=4,
    )
    draw.ellipse((166, 49, 194, 79), fill=(238, 176, 24), outline=(75, 39, 10), width=4)
    draw.ellipse((166, 461, 194, 491), fill=(238, 176, 24), outline=(75, 39, 10), width=4)

    # Deliberately use the exterior color inside the closed frame. A detector
    # based only on color difference will punch a hole here; a border-first
    # detector must preserve it because the region is enclosed by the frame.
    patch_rgb = exterior_rgb if matching_exterior_patch else (197, 145, 67)
    draw.rectangle((87, 150, 139, 214), fill=patch_rgb)
    draw.ellipse((154, 205, 272, 323), fill=(181, 58, 83), outline=(249, 220, 135), width=5)
    draw.rectangle(
        (111, 346, 249, 414),
        fill=(92, 45, 134) if interior_variant == 0 else (29, 163, 116),
    )
    image.save(path)


def _source_space_mask(report: dict[str, object]) -> np.ndarray:
    source_width, source_height = (int(value) for value in report["source_size_px"])
    x0, y0, x1, y1 = (int(value) for value in report["subject_bbox_px"])
    cropped = np.asarray(Image.open(str(report["subject_mask"])).convert("L")) > 0
    reconstructed = np.zeros((source_height, source_width), dtype=bool)
    assert cropped.shape == (y1 - y0, x1 - x0)
    reconstructed[y0:y1, x0:x1] = cropped
    return reconstructed


def _artifact_rgb(report: dict[str, object], key: str) -> np.ndarray:
    return np.asarray(Image.open(str(report[key])).convert("RGB"))


def test_extracts_closed_frame_contents_and_connected_ornaments(tmp_path: Path) -> None:
    """Catches rectangular/color-key crops that lose ornaments or hollow same-color interiors."""
    source = tmp_path / "closed-frame.png"
    _make_closed_frame_card(source, matching_exterior_patch=True)

    report = prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)
    mask = _source_space_mask(report)

    # Exact, independently known points from the synthetic drawing.
    assert mask[270, 48]  # outer frame's continuous left run
    assert mask[50, 180]  # connected top ornament beyond the frame
    assert mask[490, 180]  # connected bottom ornament beyond the frame
    assert mask[180, 110]  # enclosed patch exactly matching the exterior color
    assert not mask[20, 20]  # exterior canvas is excluded
    assert not mask[270, 20]
    assert report["subject_bbox_px"] == [48, 49, 313, 492]


def test_interior_changes_cannot_change_exterior_only_bleed(tmp_path: Path) -> None:
    """Catches bleed sampling that accidentally reads card artwork inside the subject mask."""
    source_a = tmp_path / "interior-a.png"
    source_b = tmp_path / "interior-b.png"
    _make_closed_frame_card(source_a, interior_variant=0)
    _make_closed_frame_card(source_b, interior_variant=1)

    report_a = prepare_subject_first(source_a, tmp_path / "out-a", minimum_effective_ppi=300)
    report_b = prepare_subject_first(source_b, tmp_path / "out-b", minimum_effective_ppi=300)

    bleed_a = _artifact_rgb(report_a, "bleed_background_image")
    bleed_b = _artifact_rgb(report_b, "bleed_background_image")
    assert np.array_equal(bleed_a, bleed_b)
    assert report_a["artifacts"]["bleed_background_rgb"]["sha256"] == report_b["artifacts"][
        "bleed_background_rgb"
    ]["sha256"]
    assert report_a["background_sampling"]["subject_overlap_pixels"] == 0
    assert report_b["background_sampling"]["subject_overlap_pixels"] == 0


def test_prepare_reviewed_flat_exterior_writes_a_uniform_audited_bleed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reviewed-flat.png"
    _make_closed_frame_card(source, exterior_rgb=(198, 24, 20))

    report = prepare_subject_first(
        source,
        tmp_path / "prepared-flat",
        minimum_effective_ppi=300,
        reviewed_flat_exterior=True,
    )
    background = _artifact_rgb(report, "bleed_background_image")
    sampling = report["background_sampling"]

    assert sampling["sampling_method"] == "reviewed-flat-exterior-median-v1"
    assert sampling["region_policy"] == "exterior-only"
    assert sampling["subject_overlap_pixels"] == 0
    assert report["bleed_policy"] == (
        "current-card-exterior-only-reviewed-flat-median"
    )
    assert np.unique(background.reshape(-1, 3), axis=0).tolist() == [
        sampling["flat_background_rgb"]
    ]


def test_exterior_changes_only_bleed_not_subject_or_mask(tmp_path: Path) -> None:
    """Catches subject contamination by exterior pixels and bleed copied from another card."""
    source_gold = tmp_path / "gold-exterior.png"
    source_blue = tmp_path / "blue-exterior.png"
    _make_closed_frame_card(
        source_gold,
        exterior_rgb=(197, 145, 67),
        matching_exterior_patch=False,
    )
    _make_closed_frame_card(
        source_blue,
        exterior_rgb=(37, 104, 176),
        matching_exterior_patch=False,
    )

    gold = prepare_subject_first(source_gold, tmp_path / "gold", minimum_effective_ppi=300)
    blue = prepare_subject_first(source_blue, tmp_path / "blue", minimum_effective_ppi=300)

    assert np.array_equal(
        np.asarray(Image.open(str(gold["subject_image"])).convert("RGBA")),
        np.asarray(Image.open(str(blue["subject_image"])).convert("RGBA")),
    )
    assert np.array_equal(
        np.asarray(Image.open(str(gold["subject_mask"])).convert("L")),
        np.asarray(Image.open(str(blue["subject_mask"])).convert("L")),
    )
    assert not np.array_equal(
        _artifact_rgb(gold, "bleed_background_image"),
        _artifact_rgb(blue, "bleed_background_image"),
    )
    assert gold["background_sampling"]["subject_overlap_pixels"] == 0
    assert blue["background_sampling"]["subject_overlap_pixels"] == 0


def test_defaults_are_fixed_five_mm_bleed_and_proportional_centered_placement(
    tmp_path: Path,
) -> None:
    """Catches dynamic side bleed, cropping, off-center placement, and non-uniform stretching."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)

    report = prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)
    placement = report["subject_placement_mm"]

    assert report["trim_mm"] == [80.0, 120.0]
    assert report["bleed_mm"] == {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0}
    assert report["media_mm"] == [90.0, 130.0]
    assert placement["trim_box_mm"] == [5.0, 5.0, 85.0, 125.0]
    assert placement["cropped"] is False
    assert placement["source_bbox_px"] == report["subject_bbox_px"]
    assert placement["scale_x"] == pytest.approx(placement["scale_y"], abs=1e-12)
    assert placement["x_mm"] + placement["width_mm"] / 2 == pytest.approx(45.0, abs=1e-9)
    assert placement["y_mm"] + placement["height_mm"] / 2 == pytest.approx(65.0, abs=1e-9)
    assert placement["x_mm"] >= 5.0 - 1e-9
    assert placement["y_mm"] >= 5.0 - 1e-9
    assert placement["x_mm"] + placement["width_mm"] <= 85.0 + 1e-9
    assert placement["y_mm"] + placement["height_mm"] <= 125.0 + 1e-9


def test_manifest_audits_subject_first_stages_hashes_and_same_source_sampling(
    tmp_path: Path,
) -> None:
    """Catches an unauditable one-pass composite or sampling from a different source/artifact."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)

    report = prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest = json.loads(Path(str(report["manifest"])).read_text(encoding="utf-8"))

    assert [stage["name"] for stage in manifest["pipeline_stages"]] == [
        "detect-subject-frame",
        "extract-subject",
        "sample-exterior",
        "render-bleed-background",
        "place-subject",
    ]
    assert all(stage["status"] == "completed" for stage in manifest["pipeline_stages"])
    assert manifest["source_sha256"] == _sha256(source)
    assert manifest["subject_extraction"]["source_sha256"] == manifest["source_sha256"]
    assert manifest["background_sampling"]["source_sha256"] == manifest["source_sha256"]
    assert manifest["background_sampling"]["region_policy"] == "exterior-only"
    assert manifest["background_sampling"]["subject_overlap_pixels"] == 0
    assert manifest["background_sampling"]["exterior_sample_pixels"] > 0

    expected_artifacts = {
        "subject_rgba": "subject_image",
        "subject_mask": "subject_mask",
        "exterior_sampling_mask": "exterior_sampling_mask",
        "sampling_provenance": "sampling_provenance_artifact",
        "bleed_background_rgb": "bleed_background_image",
        "media_rgb": "media_image",
    }
    assert manifest["pipeline_version"] == 3
    assert manifest["sampling_provenance_schema_version"] == 1
    assert manifest["sampling_provenance_sha256"] == manifest["artifacts"][
        "sampling_provenance"
    ]["sha256"]
    assert set(manifest["artifacts"]) == set(expected_artifacts)
    for artifact_name, report_key in expected_artifacts.items():
        artifact = manifest["artifacts"][artifact_name]
        assert Path(artifact["path"]).resolve() == Path(str(report[report_key])).resolve()
        assert artifact["sha256"] == _sha256(artifact["path"])
        assert len(artifact["sha256"]) == 64
        assert artifact["source_sha256"] == manifest["source_sha256"]


def test_writes_source_space_sampling_mask_overlay_and_layout_guide(tmp_path: Path) -> None:
    """Catches unverifiable exterior sampling or a layout without visible trim/subject bounds."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)

    report = prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)
    subject = _source_space_mask(report)
    sampling_path = Path(str(report["exterior_sampling_mask"]))
    sampling = np.asarray(Image.open(sampling_path).convert("L")) > 0

    assert sampling.shape == subject.shape
    assert not np.any(sampling & subject)
    assert int(sampling.sum()) == report["background_sampling"]["exterior_sample_pixels"]
    assert report["artifacts"]["exterior_sampling_mask"]["path"] == str(
        sampling_path.resolve()
    )
    assert report["artifacts"]["exterior_sampling_mask"]["sha256"] == _sha256(
        sampling_path
    )
    assert report["artifacts"]["exterior_sampling_mask"]["source_sha256"] == report[
        "source_sha256"
    ]

    overlay_path = Path(str(report["exterior_sampling_overlay"]))
    layout_path = Path(str(report["layout_guide"]))
    assert overlay_path.is_file()
    assert layout_path.is_file()
    assert Image.open(overlay_path).size == tuple(report["source_size_px"])
    assert Image.open(layout_path).size == tuple(report["media_size_px"])


def test_asymmetric_bleed_centers_subject_on_trim_not_media(tmp_path: Path) -> None:
    """Catches placement around the page center when unequal bleeds offset the TrimBox."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)

    report = prepare_subject_first(
        source,
        tmp_path / "prepared",
        bleed_left_mm=3,
        bleed_right_mm=7,
        bleed_top_mm=4,
        bleed_bottom_mm=6,
        minimum_effective_ppi=300,
    )
    placement = report["subject_placement_mm"]
    pixels_per_mm = report["effective_ppi"] / 25.4
    half_pixel_mm = 0.5 / pixels_per_mm + 1e-12
    center_x = placement["x_mm"] + placement["width_mm"] / 2
    center_y = placement["y_mm"] + placement["height_mm"] / 2

    assert report["bleed_mm"] == {"left": 3.0, "right": 7.0, "top": 4.0, "bottom": 6.0}
    assert placement["trim_box_mm"] == [3.0, 4.0, 83.0, 124.0]
    assert center_x == pytest.approx(43.0, abs=half_pixel_mm)
    assert center_y == pytest.approx(64.0, abs=half_pixel_mm)
    assert center_x != pytest.approx(45.0, abs=half_pixel_mm)
    assert center_y != pytest.approx(65.0, abs=half_pixel_mm)
    assert placement["x_mm"] >= 3.0 - half_pixel_mm
    assert placement["y_mm"] >= 4.0 - half_pixel_mm
    assert placement["x_mm"] + placement["width_mm"] <= 83.0 + half_pixel_mm
    assert placement["y_mm"] + placement["height_mm"] <= 124.0 + half_pixel_mm


def test_rejects_large_central_solid_rectangle_without_frame_evidence(tmp_path: Path) -> None:
    """Catches treating any large central foreground block as a bordered card design."""
    source = tmp_path / "solid-rectangle.png"
    image = Image.new("RGB", (360, 540), EXTERIOR_GOLD)
    ImageDraw.Draw(image).rectangle((50, 70, 310, 470), fill=(43, 91, 129))
    image.save(source)

    with pytest.raises(DetectionError, match="frame evidence|border layers|rounded frame"):
        prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)


def test_successful_detection_records_shared_closed_frame_geometry_gate(tmp_path: Path) -> None:
    """Catches success paths that bypass four-side continuity and frame-structure proof."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)

    report = prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)
    evidence = report["frame_detection"]["geometry_evidence"]

    assert evidence["passed"] is True
    assert set(evidence["continuous_side_coverage"]) == {"top", "right", "bottom", "left"}
    assert min(evidence["continuous_side_coverage"].values()) >= 0.50
    assert evidence["rounded_corner_evidence"] or min(
        evidence["double_layer_coverage"].values()
    ) >= 0.35


def test_dark_fallback_stops_when_multiple_closed_frame_candidates_are_plausible() -> None:
    """Catches selecting the largest dark contour while silently ignoring another valid frame."""
    image = Image.new("RGB", (360, 540), (232, 201, 124))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 355, 535), radius=32, outline=(18, 12, 8), width=5)
    draw.rounded_rectangle((34, 48, 325, 491), radius=26, outline=(18, 12, 8), width=5)
    rgb = np.asarray(image)

    with pytest.raises(DetectionError, match="multiple plausible.*frame"):
        _detect_dark_closed_frame(rgb)


def test_dark_fallback_keeps_only_ornaments_connected_to_confirmed_frame() -> None:
    """Catches absorbing unrelated central dark marks solely because they sit above the frame."""
    image = Image.new("RGB", (360, 540), (232, 201, 124))
    draw = ImageDraw.Draw(image)
    dark = (18, 12, 8)
    draw.rounded_rectangle((20, 40, 340, 500), radius=30, outline=dark, width=6)
    draw.rounded_rectangle((28, 48, 332, 492), radius=24, outline=dark, width=4)
    for start, end in (((180, 40), (180, 48)), ((180, 492), (180, 500))):
        draw.line((start, end), fill=dark, width=3)
    draw.ellipse((165, 25, 195, 55), fill=(238, 176, 24), outline=dark, width=4)
    draw.ellipse((171, 5, 189, 18), fill=dark)  # disconnected decoy

    alpha, details = _detect_dark_closed_frame(np.asarray(image))

    assert alpha[32, 180] > 0  # connected ornament is protected
    assert alpha[10, 180] == 0  # disconnected dark decoy stays exterior
    geometry = details["frame_geometry_px"]
    sides = geometry["dark_sides_px"]
    assert geometry["left"] == pytest.approx(sides["left"] - 0.5)
    assert geometry["right"] == pytest.approx(sides["right"] + 0.5)
    assert geometry["top"] == pytest.approx(sides["top"] - 0.5)
    assert geometry["bottom"] == pytest.approx(sides["bottom"] + 0.5)


def test_automatic_frame_keeps_bright_connected_ornaments_and_excludes_decoy() -> None:
    """Catches the automatic path retaining ornaments only through a dark threshold."""
    image = Image.new("RGB", (360, 540), (232, 201, 124))
    draw = ImageDraw.Draw(image)
    dark = (18, 12, 8)
    draw.polygon(((173, 42), (180, 0), (187, 42)), fill=(255, 255, 245))
    draw.polygon(((173, 498), (180, 539), (187, 498)), fill=(170, 255, 255))
    draw.ellipse((12, 248, 22, 258), fill=(255, 255, 245))
    draw.rounded_rectangle(
        (20, 40, 340, 500),
        radius=30,
        outline=dark,
        width=6,
    )
    draw.rounded_rectangle((28, 48, 332, 492), radius=24, outline=dark, width=4)

    alpha, details = _detect_dark_closed_frame(np.asarray(image))

    assert alpha[0, 180] > 0
    assert alpha[-1, 180] > 0
    assert alpha[253, 17] == 0
    evidence = details["source_connected_ornament_evidence"]
    assert evidence["color_policy"] == "source-vs-modeled-exterior-difference"


def test_gradient_redirect_does_not_extend_a_broad_matte_as_a_center_ornament() -> None:
    """Catches a broad exterior matte becoming a 24%-wide center alpha strip."""
    height, width = 96, 160
    frame_component = np.zeros((height, width), dtype=bool)
    ornament_foreground = np.zeros((height, width), dtype=bool)
    frame_component[30, 20:140] = True

    # The real ornament is the narrow source-connected run immediately above
    # the confirmed top rail.  A much wider exterior matte touches its outward
    # tip but must not inherit ornament status or continue to the canvas edge.
    ornament_foreground[18:30, 77:83] = True
    ornament_foreground[0:18, 30:130] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        frame_component,
        ornament_foreground,
        geometry,
        lock_to_source_support=True,
    )

    assert alpha[24, 80] == 255
    assert np.count_nonzero(alpha[:18]) == 0


def test_renderer_stops_before_an_ambiguous_gap_fork() -> None:
    """Catches rendering two branches when a one-row gap has no unique continuation."""
    height, width = 96, 160
    source_support = np.zeros((height, width), dtype=bool)
    source_support[30:81, 20:140] = True
    source_support[25:30, 80] = True
    source_support[20:24, 79] = True
    source_support[20:24, 81] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        source_support,
        source_support,
        geometry,
        lock_to_source_support=True,
    )

    assert alpha[27, 80] == 255
    assert alpha[24, 80] == 0
    assert alpha[22, 79] == 0
    assert alpha[22, 81] == 0


def test_source_locked_ornament_fills_only_its_closed_local_flower() -> None:
    """Catches source locking turning an enclosed flower center transparent."""
    height, width = 96, 160
    source_support = np.zeros((height, width), dtype=bool)
    source_support[30:81, 20:140] = True
    source_support[24:30, 78:83] = True
    source_support[14:17, 74:87] = True
    source_support[21:24, 74:87] = True
    source_support[17:21, 74:77] = True
    source_support[17:21, 84:87] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        source_support,
        source_support,
        geometry,
        lock_to_source_support=True,
    )

    assert alpha[19, 80] == 255
    assert alpha[10, 80] == 0


def test_source_locked_ornament_does_not_fill_an_open_c_shape() -> None:
    """Catches local hole filling sealing an open branch into invented subject."""
    height, width = 96, 160
    source_support = np.zeros((height, width), dtype=bool)
    source_support[30:81, 20:140] = True
    source_support[24:30, 78:83] = True
    source_support[14:17, 74:87] = True
    source_support[21:24, 74:87] = True
    source_support[17:21, 74:77] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        source_support,
        source_support,
        geometry,
        lock_to_source_support=True,
    )

    assert alpha[19, 75] == 255
    assert alpha[19, 80] == 0


def test_source_locked_ornament_does_not_fill_between_an_open_arch_and_the_rail() -> None:
    """Catches the filled card core closing an otherwise open ornament arch."""
    height, width = 96, 160
    source_support = np.zeros((height, width), dtype=bool)
    source_support[30:81, 20:140] = True
    source_support[20, 75:86] = True
    source_support[20:30, 75] = True
    source_support[20:30, 85] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        source_support,
        source_support,
        geometry,
        lock_to_source_support=True,
    )

    assert alpha[20, 80] == 255
    assert alpha[25, 75] == 255
    assert alpha[25, 85] == 255
    assert alpha[25, 80] == 0


def test_local_exterior_model_separates_a_true_ornament_from_overlapping_matte() -> None:
    """Catches deleting a real ornament when its connected component also contains matte."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), (154, 27, 8), dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[:30, 20:140] = True  # over-broad global foreground component
    source_foreground[:30, 77:83] = True
    rgb[:30, 77:83] = (244, 191, 46)
    rgb[4:6, 69:71] = (20, 220, 220)  # disconnected high-contrast matte texture
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )
    alpha = _render_dark_frame_alpha(
        source_foreground,
        refined,
        geometry,
        lock_to_source_support=True,
    )

    assert evidence["policy"] == "same-row-flank-background-difference"
    assert not refined[4, 70]
    assert alpha[0, 80] == 255
    assert alpha[0, 60] == 0
    assert np.count_nonzero(alpha[0]) == 6


def test_local_ornament_rejects_an_adjacent_strong_texture_without_a_rail_neck() -> None:
    """Catches a short high-contrast texture cluster becoming an ornament neck."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), (154, 27, 8), dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[28:30, 70] = True
    rgb[28:30, 70] = (20, 220, 220)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[28:30, 70])
    assert evidence["sides"]["top"]["retained_source_supported_pixels"] == 0


def test_local_ornament_hysteresis_keeps_a_weak_branch_to_a_strong_flower() -> None:
    """Catches a single high threshold deleting a subtle rail-connected branch."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[20:30, 80] = True
    source_foreground[16:20, 78:83] = True
    rgb[20:30, 80] = (107, 100, 100)
    rgb[16:20, 78:83] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert refined[25, 80]
    assert refined[17, 80]
    assert evidence["sides"]["top"]["weak_threshold_linf_px"]["25"] == 6.0
    assert evidence["sides"]["top"]["strong_threshold_linf_px"]["17"] == 10.0


def test_sparse_flank_texture_does_not_raise_the_weak_gate_above_a_branch() -> None:
    """Catches Q90 flank outliers erasing a branch that is distinct from the matte."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[20:30, 80] = True
    source_foreground[16:20, 78:83] = True
    rgb[20:30, 80] = (107, 100, 100)
    rgb[16:20, 78:83] = (130, 100, 100)
    rgb[:30, 27:61:5] = (115, 100, 100)
    rgb[:30, 99:133:5] = (115, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert refined[25, 80]
    assert refined[17, 80]
    assert evidence["sides"]["top"]["weak_threshold_linf_px"]["25"] == 6.0


def test_flank_dispersion_uses_one_mad_so_topology_can_recover_a_weak_branch() -> None:
    """Catches a 2.5-MAD weak gate deleting source-supported antialiased branches."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[20:30, 80] = True
    source_foreground[16:20, 78:83] = True
    rgb[20:30, 80] = (108, 100, 100)
    rgb[16:20, 78:83] = (130, 100, 100)
    flank_pattern = np.resize(np.asarray((96, 100, 100, 104), dtype=np.uint8), 34)
    rgb[:30, 27:61, 0] = flank_pattern
    rgb[:30, 99:133, 0] = flank_pattern
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert refined[25, 80]
    assert refined[17, 80]
    assert evidence["sides"]["top"]["weak_threshold_linf_px"]["25"] == 7.0


def test_local_ornament_does_not_use_lateral_texture_to_promote_a_weak_branch() -> None:
    """Catches all-direction dilation borrowing a strong pixel from nearby texture."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[20:30, 80] = True
    source_foreground[20:23, 82] = True
    rgb[20:30, 80] = (107, 100, 100)
    rgb[20:23, 82] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not refined[25, 80]
    assert not refined[21, 82]
    assert evidence["sides"]["top"]["rail_connected_component_count"] == 0


def test_local_ornament_bridges_one_vertical_gap_without_inventing_alpha() -> None:
    """Catches fail-closed component labeling deleting a one-row branch gap."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[25:30, 80] = True
    source_foreground[20:24, 80] = True
    source_foreground[16:20, 78:83] = True
    rgb[25:30, 80] = (107, 100, 100)
    rgb[20:24, 80] = (107, 100, 100)
    rgb[16:20, 78:83] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )
    alpha = _render_dark_frame_alpha(
        source_foreground,
        refined,
        geometry,
        lock_to_source_support=True,
    )

    assert refined[27, 80]
    assert refined[22, 80]
    assert refined[17, 80]
    assert not refined[24, 80]
    assert alpha[22, 80] == 255
    assert alpha[17, 80] == 255
    assert alpha[24, 80] == 0
    assert evidence["sides"]["top"]["bridged_one_pixel_gap_count"] == 1


def test_local_ornament_keeps_a_weak_tip_beyond_strong_support_across_one_gap() -> None:
    """Catches stopping hysteresis at the strong flower before its weak outer tip."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[25:30, 80] = True
    source_foreground[20:25, 78:83] = True
    source_foreground[16:19, 80] = True
    rgb[25:30, 80] = (107, 100, 100)
    rgb[20:25, 78:83] = (130, 100, 100)
    rgb[16:19, 80] = (107, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert refined[17, 80]
    assert not refined[19, 80]
    assert evidence["sides"]["top"]["bridged_one_pixel_gap_count"] == 1


def test_local_ornament_rejects_an_ambiguous_fork_across_a_gap() -> None:
    """Catches one rail neck absorbing two equally plausible outer texture branches."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[25:30, 80] = True
    source_foreground[20:24, 79] = True
    source_foreground[20:24, 81] = True
    rgb[25:30, 80] = (107, 100, 100)
    rgb[20:24, 79] = (130, 100, 100)
    rgb[20:24, 81] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[25:30, 80])
    assert not np.any(refined[20:24, 79])
    assert not np.any(refined[20:24, 81])
    assert evidence["sides"]["top"]["ambiguous_gap_fork_count"] == 1


def test_local_palette_novelty_prunes_only_the_low_novelty_outward_tail() -> None:
    """Catches moving the palette gate onto the full rail-to-flower branch."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:30, 80] = True
    source_foreground[16:20, 78:83] = True
    rgb[0:30, 80] = (107, 100, 100)
    rgb[16:20, 78:83] = (130, 100, 100)
    # The outward tail is locally only four levels away from the adjacent
    # matte, while the same-row distant model still proves a weak branch.
    rgb[0:16, 51:65] = (103, 100, 100)
    rgb[0:16, 94:108] = (103, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not refined[0, 80]
    assert not refined[15, 80]
    assert refined[17, 80]
    assert refined[25, 80]
    top = evidence["sides"]["top"]
    assert top["removed_low_novelty_outer_tail_pixels"] == 16
    assert top["retained_outer_limits_px"] == [16]


def test_narrow_extreme_branch_restores_antialias_without_restoring_broad_matte() -> None:
    """Catches losing a real narrow edge branch or reopening a broad matte strip."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:30, 73:77] = True
    source_foreground[0:30, 84:90] = True
    rgb[0:16, 73:77] = (107, 100, 100)
    rgb[16:30, 73:77] = (200, 100, 100)
    rgb[0:16, 84:90] = (107, 100, 100)
    rgb[16:30, 84:90] = (200, 100, 100)
    # Make both antialiased tails locally non-novel. Only the four-pixel-wide
    # branch is eligible for the narrow, extreme-core topology exception.
    rgb[0:16, 51:65] = (104, 100, 100)
    rgb[0:16, 94:108] = (104, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert np.all(refined[0:30, 73:77])
    assert not np.any(refined[0:16, 84:90])
    assert np.all(refined[16:30, 84:90])
    top = evidence["sides"]["top"]
    assert top["preserved_narrow_low_contrast_branch_count"] == 1
    branch = top["preserved_narrow_low_contrast_branches"][0]
    assert branch["maximum_row_width_px"] == 4
    assert branch["maximum_allowed_row_width_px"] == 4
    assert branch["maximum_palette_novelty_linf_px"] > 64.0


@pytest.mark.parametrize("palette_distinct_edge_rows", (1, 3))
def test_palette_outlier_cannot_promote_a_full_width_matte_chain(
    palette_distinct_edge_rows: int,
) -> None:
    """Catches one texture seed promoting a 24%-wide center matte as ornament."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:30, 65:94] = True
    rgb[0:30, 65:94] = (108, 100, 100)
    rgb[0:30, 51:65] = (104, 100, 100)
    rgb[0:30, 94:108] = (104, 100, 100)
    rgb[0:palette_distinct_edge_rows, 65:94] = (110, 100, 100)
    rgb[min(1, palette_distinct_edge_rows - 1), 80] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[0:30, 65:94])
    assert evidence["sides"]["top"]["retained_outer_limits_px"] == []


def test_narrow_neck_cannot_hide_a_broad_matte_cap_from_the_width_gate() -> None:
    """Catches a long one-pixel neck diluting a broad cap's median row width."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:14, 65:94] = True
    source_foreground[14:30, 80] = True
    rgb[0:14, 65:94] = (108, 100, 100)
    rgb[14:30, 80] = (108, 100, 100)
    rgb[0:30, 51:65] = (104, 100, 100)
    rgb[0:30, 94:108] = (104, 100, 100)
    rgb[0:3, 65:94] = (110, 100, 100)
    rgb[1, 80] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[0:14, 65:94])
    assert evidence["sides"]["top"]["rejected_broad_chain_count"] == 1


def test_narrow_palette_tip_cannot_promote_a_broad_rail_facing_matte() -> None:
    """Catches narrow outer evidence retaining a broad matte nearer the rail."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:17, 80] = True
    source_foreground[17:30, 65:94] = True
    rgb[0:17, 80] = (108, 100, 100)
    rgb[17:30, 65:94] = (108, 100, 100)
    rgb[0:30, 51:65] = (104, 100, 100)
    rgb[0:30, 94:108] = (104, 100, 100)
    rgb[0:3, 80] = (110, 100, 100)
    rgb[1, 80] = (130, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[17:30, 65:94])
    assert evidence["sides"]["top"]["rejected_broad_chain_count"] == 1


def test_rail_end_evidence_cannot_restore_a_long_low_novelty_tail() -> None:
    """Catches two rail-end evidence rows promoting a mostly unsupported branch."""
    height, width = 96, 160
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[30:81, 20:140] = True
    source_foreground[0:30, 78:82] = True
    rgb[0:28, 78:82] = (107, 100, 100)
    rgb[0:28, 51:65] = (104, 100, 100)
    rgb[0:28, 94:108] = (104, 100, 100)
    rgb[28, 78:82] = (200, 100, 100)
    rgb[29, 78:82] = (110, 100, 100)
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": 29.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 30, "bottom": 80},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert not np.any(refined[0:28, 78:82])
    assert (
        evidence["sides"]["top"]["preserved_narrow_low_contrast_branch_count"]
        == 0
    )


def test_source_locked_core_clips_connected_paper_to_the_rounded_frame() -> None:
    """Catches the rectangular rail box absorbing source-connected corner paper."""
    height = width = 100
    support_image = Image.new("L", (width, height), 0)
    support_draw = ImageDraw.Draw(support_image)
    support_draw.rounded_rectangle((20, 20, 80, 80), radius=20, fill=255)
    source_support = np.asarray(support_image) > 0
    source_support = source_support.copy()
    source_support[20:40, 20:40] = True  # connected paper/shadow outside the arc
    source_support[10:20, 48:52] = True  # real narrow top-center ornament
    geometry = {
        "left": 19.5,
        "right": 80.5,
        "top": 19.5,
        "bottom": 80.5,
        "dark_sides_px": {"left": 20, "right": 80, "top": 20, "bottom": 80},
        "radii": {
            "top_left": 20.0,
            "top_right": 20.0,
            "bottom_left": 20.0,
            "bottom_right": 20.0,
        },
    }

    alpha = _render_dark_frame_alpha(
        source_support,
        source_support,
        geometry,
        lock_to_source_support=True,
    )
    geometric_alpha = _render_dark_frame_alpha(
        np.zeros_like(source_support),
        np.zeros_like(source_support),
        geometry,
        lock_to_source_support=False,
    )
    inside_rails = np.zeros_like(source_support)
    inside_rails[20:81, 20:81] = True

    assert alpha[21, 21] == 0
    assert alpha[20, 50] == 255
    assert alpha[50, 50] == 255
    assert alpha[12, 50] == 255
    assert np.count_nonzero((alpha > 0) & (geometric_alpha == 0) & inside_rails) == 0


def test_center_ornament_refinement_handles_frame_flush_with_source_edges() -> None:
    """Catches empty top/bottom exterior rows leaving evidence variables unbound."""
    height, width = 120, 160
    rgb = np.full((height, width, 3), (186, 142, 82), dtype=np.uint8)
    source_foreground = np.zeros((height, width), dtype=bool)
    source_foreground[:, 20:140] = True
    geometry = {
        "left": 19.5,
        "right": 139.5,
        "top": -0.5,
        "bottom": 119.5,
        "dark_sides_px": {"left": 20, "right": 139, "top": 0, "bottom": 119},
        "radii": {
            "top_left": 12.0,
            "top_right": 12.0,
            "bottom_left": 12.0,
            "bottom_right": 12.0,
        },
    }

    refined, evidence = subject_pipeline._refine_center_ornament_foreground(
        rgb,
        source_foreground,
        geometry,
    )

    assert np.array_equal(refined, source_foreground)
    for side in ("top", "bottom"):
        assert evidence["sides"][side]["rows_evaluated"] == 0
        assert evidence["sides"][side]["bridged_one_pixel_gap_count"] == 0
        assert evidence["sides"][side]["bridged_one_pixel_gaps"] == []
        assert evidence["sides"][side]["ambiguous_gap_fork_count"] == 0
        assert evidence["sides"][side]["ambiguous_gap_forks"] == []


@pytest.mark.parametrize(
    ("source", "retained_ornament_xy"),
    (
        (SHARD1_GREEN_MATTE_SOURCE, (527, 20)),
        (SHARD1_BLUE_HAZE_SOURCE, (527, 15)),
    ),
)
def test_real_center_ornament_excludes_adjacent_exterior_matte(
    tmp_path: Path,
    source: Path,
    retained_ornament_xy: tuple[int, int],
) -> None:
    """Catches a distant flank model attaching green/blue matte to the ornament."""
    _require_real_source(source, "real regression source unavailable")

    report = prepare_subject_first(
        source,
        tmp_path / source.stem,
        minimum_effective_ppi=300,
    )
    source_mask = _source_space_mask(report)
    retained_x, retained_y = retained_ornament_xy

    assert source_mask[retained_y, retained_x]
    assert np.count_nonzero(source_mask[0]) == 0
    assert report["background_sampling"]["subject_overlap_pixels"] == 0
    assert report["cropped"] is False
    assert report["resampled"] is False


def test_rejects_obviously_textured_exterior_sampling_pollution(tmp_path: Path) -> None:
    """Catches robust fitting that hides a visibly polluted side instead of stopping."""
    source = tmp_path / "polluted.png"
    _make_closed_frame_card(source)
    rgb = np.asarray(Image.open(source).convert("RGB")).copy()
    patch = rgb[100:440, 3:40]
    y, x = np.indices(patch.shape[:2])
    checker = ((x // 2 + y // 2) % 2) == 0
    patch[checker] = (100, 50, 20)
    patch[~checker] = (245, 220, 160)
    rgb[100:440, 3:40] = patch
    Image.fromarray(rgb, "RGB").save(source)

    with pytest.raises(DetectionError, match="polluted|texture|dispersion|residual"):
        prepare_subject_first(source, tmp_path / "prepared", minimum_effective_ppi=300)


def test_accepts_strong_smooth_normal_exterior_gradient() -> None:
    """Catches treating legitimate low-frequency normal shading as texture pollution."""
    height, width = 270, 180
    y, x = np.indices((height, width))
    distance_to_edge = np.minimum.reduce((x, width - 1 - x, y, height - 1 - y))
    rgb = np.stack(
        (
            24 + 6 * distance_to_edge,
            38 + 5 * distance_to_edge,
            61 + 4 * distance_to_edge,
        ),
        axis=2,
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    subject = np.zeros((height, width), dtype=bool)
    subject[58:212, 42:138] = True
    exterior = ~subject

    _, sampling_mask, details = _fit_exterior_background(rgb, exterior, subject)

    assert not np.any(sampling_mask & subject)
    assert details["sampling_quality_gate"]["status"] == "passed"
    assert max(
        side["coordinate_p90_second_difference_linf_p90"]
        for side in details["side_texture_dispersion"].values()
    ) <= details["sampling_quality_gate"]["coordinate_second_difference_linf_p90_limit"]


def test_pairs_opposite_long_gradient_bands_and_ignores_an_unmatched_decoy() -> None:
    """Catches rejecting a unique mirrored frame because one side has an extra long line."""
    width, height = 360, 540
    image = Image.new("RGB", (width, height), (224, 190, 120))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (18, 18, 342, 522),
        radius=30,
        fill=(42, 83, 122),
        outline=(20, 12, 8),
        width=4,
    )
    # This high-coverage line is deliberately inside the left probe region but
    # has no mirrored partner on the right. The old independent-side gate sees
    # two left bands and rejects the otherwise unique outer frame.
    draw.rectangle((52, 80, 58, 460), fill=(240, 220, 160))
    draw.line((55, 80, 55, 460), fill=(20, 12, 8), width=3)
    provisional = {
        "left": -0.5,
        "right": width - 0.5,
        "top": -0.5,
        "bottom": height - 0.5,
        "radii": {
            "top_left": 90.0,
            "top_right": 90.0,
            "bottom_left": 90.0,
            "bottom_right": 90.0,
        },
        "dark_sides_px": {
            "left": 0,
            "right": width - 1,
            "top": 0,
            "bottom": height - 1,
        },
    }

    redirected, evidence = _redirect_edge_hugging_geometry_from_gradients(
        np.asarray(image), provisional
    )

    assert redirected["dark_sides_px"] == {
        "left": 17,
        "right": 343,
        "top": 17,
        "bottom": 523,
    }
    assert redirected["left"] == pytest.approx(16.5)
    assert redirected["right"] == pytest.approx(343.5)
    assert redirected["top"] == pytest.approx(16.5)
    assert redirected["bottom"] == pytest.approx(523.5)
    assert redirected["anchor_method"] == "paired-long-edge-gradient-fallback"
    assert evidence["selection_mode"] == "paired-long-edge-gradient-fallback"
    assert evidence["paired_long_sides"] == ["left", "right"]
    assert evidence["rejected_unmatched_band_count"]["left"] >= 1
    assert evidence["corner_radius_alignment"]["passed"] is True
    assert all(
        22.0 <= radius <= 38.0 for radius in redirected["radii"].values()
    )


def _smooth_exterior_rgb(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        (
            32 + x * 0.35 + y * 0.04,
            58 + x * 0.22 + y * 0.06,
            81 + x * 0.18 + y * 0.03,
        ),
        axis=2,
    ).round().clip(0, 255).astype(np.uint8)


def test_interpolates_missing_columns_only_from_valid_samples_on_each_same_side() -> None:
    """Catches crossing the subject to sample another side when whole columns are occluded."""
    height, width = 180, 120
    rgb = _smooth_exterior_rgb(height, width)
    subject = np.zeros((height, width), dtype=bool)
    subject[:, 56:64] = True
    exterior = ~subject

    model, sampling_mask, details = _fit_exterior_background(rgb, exterior, subject)

    assert not sampling_mask[:, 56:64].any()
    assert not np.any(sampling_mask & subject)
    assert details["subject_overlap_pixels"] == 0
    assert details["non_exterior_sample_pixels"] == 0
    assert details["side_interpolated_coordinate_count"] == {
        "top": 8,
        "right": 0,
        "bottom": 8,
        "left": 0,
    }
    assert details["side_interpolation_method"] == "same-side-valid-coordinate-linear"
    assert details["side_interpolation_maximum_gap_px"]["top"] == 8
    assert details["side_interpolation_maximum_gap_px"]["bottom"] == 8
    assert details["side_interpolation_maximum_allowed_gap_px"]["top"] >= 8
    assert np.isfinite(model.top).all()
    assert np.isfinite(model.bottom).all()


def test_interpolates_missing_rows_only_from_valid_samples_on_each_same_side() -> None:
    """Catches crossing the subject to sample another side when whole rows are occluded."""
    height, width = 180, 120
    rgb = _smooth_exterior_rgb(height, width)
    subject = np.zeros((height, width), dtype=bool)
    subject[84:96, :] = True
    exterior = ~subject

    model, sampling_mask, details = _fit_exterior_background(rgb, exterior, subject)

    assert not sampling_mask[84:96, :].any()
    assert not np.any(sampling_mask & subject)
    assert details["subject_overlap_pixels"] == 0
    assert details["non_exterior_sample_pixels"] == 0
    assert details["side_interpolated_coordinate_count"] == {
        "top": 0,
        "right": 12,
        "bottom": 0,
        "left": 12,
    }
    assert details["side_interpolation_method"] == "same-side-valid-coordinate-linear"
    assert details["side_interpolation_maximum_gap_px"]["left"] == 12
    assert details["side_interpolation_maximum_gap_px"]["right"] == 12
    assert details["side_interpolation_maximum_allowed_gap_px"]["left"] >= 12
    assert np.isfinite(model.left).all()
    assert np.isfinite(model.right).all()


def test_rejects_a_large_same_side_interpolation_gap_without_local_evidence() -> None:
    """Catches bridging a wide subject occlusion with two distant side samples."""
    height, width = 180, 120
    rgb = _smooth_exterior_rgb(height, width)
    subject = np.zeros((height, width), dtype=bool)
    subject[:, 36:84] = True
    exterior = ~subject

    with pytest.raises(DetectionError, match="same-side.*interpolation gap"):
        _fit_exterior_background(rgb, exterior, subject)


@pytest.mark.parametrize("gap_kind", ["no-subject-contact", "off-center-contact"])
def test_rejects_an_eighteen_percent_gap_without_center_ornament_evidence(
    gap_kind: str,
) -> None:
    """Catches a globally relaxed threshold accepting unsupported 15-20% gaps."""
    height, width = 180, 120
    rgb = _smooth_exterior_rgb(height, width)
    band = max(4, round(min(width, height) * 0.06))
    subject = np.zeros((height, width), dtype=bool)
    exterior = ~subject
    if gap_kind == "no-subject-contact":
        exterior[:band, 49:71] = False
    else:
        subject[:band, 10:32] = True
        exterior = ~subject

    with pytest.raises(DetectionError, match="same-side.*interpolation gap"):
        _fit_exterior_background(rgb, exterior, subject)


def test_allows_only_an_audited_centered_source_edge_ornament_gap() -> None:
    """Catches a legitimate center ornament being rejected or silently exempted."""
    height, width = 180, 120
    rgb = _smooth_exterior_rgb(height, width)
    band = max(4, round(min(width, height) * 0.06))
    subject = np.zeros((height, width), dtype=bool)
    subject[:band, 49:71] = True
    exterior = ~subject

    _, sampling_mask, details = _fit_exterior_background(rgb, exterior, subject)

    assert not np.any(sampling_mask & subject)
    assert details["side_interpolation_maximum_allowed_gap_px"]["top"] == 12
    evidence = details["side_interpolation_exceptions"]["top"]
    assert evidence["status"] == "allowed"
    assert evidence["reason"] == "centered-source-edge-ornament-contact"
    assert evidence["missing_run_px"] == [49, 71]
    assert evidence["source_edge_contact_overlap_px"] == 22
    assert evidence["bracketing_direct_sample_coordinates_px"] == [48, 71]


@pytest.mark.parametrize("stem", LONG_EDGE_FALLBACK_SOURCES)
def test_real_edge_hugging_frames_use_audited_paired_long_edge_fallback(stem: str) -> None:
    """Catches regressions on batch cards whose dark component merges into a canvas edge."""
    source = REAL_SOURCE_ROOT / f"{stem}.png"
    _require_real_source(source, "local batch regression source unavailable")
    rgb = np.asarray(Image.open(source).convert("RGB"))

    alpha, details = _detect_dark_closed_frame(rgb)

    geometry = details["frame_geometry_px"]
    redirect = details["geometry_evidence"]["edge_hugging_component_redirect"]
    contacts = details["source_canvas_contacts_px"]
    height, width = alpha.shape
    assert geometry["anchor_method"] == "paired-long-edge-gradient-fallback"
    assert redirect["selection_mode"] == "paired-long-edge-gradient-fallback"
    assert redirect["paired_long_sides"] == ["left", "right"]
    assert geometry["left"] > 0
    assert geometry["right"] < width - 1
    assert geometry["top"] > 0
    assert geometry["bottom"] < height - 1
    assert contacts["left"] < height * 0.25
    assert contacts["right"] < height * 0.25
    assert contacts["top"] < width * 0.25
    assert contacts["bottom"] < width * 0.25


def test_real_occluded_bottom_rail_requires_hash_bound_validated_override() -> None:
    """Catches inventing a radius where the lower physical rail is occluded."""
    source = REAL_SOURCE_ROOT / (
        f"{SAFE_REJECTION_OCCLUDED_BOTTOM_RAIL_SOURCE}.png"
    )
    _require_real_source(source, "local batch regression source unavailable")
    rgb = np.asarray(Image.open(source).convert("RGB"))

    with pytest.raises(
        DetectionError,
        match="continuous outer frame is clipped by the source canvas: bottom",
    ):
        _detect_dark_closed_frame(rgb)


def _uniform_radii(radius: float) -> dict[str, float]:
    return {
        "top_left": radius,
        "top_right": radius,
        "bottom_left": radius,
        "bottom_right": radius,
    }


@pytest.mark.parametrize("stem,sides,radius", REAL_VALIDATED_FRAME_CASES)
def test_real_paper_edge_cases_accept_only_an_explicit_validated_design_frame_override(
    stem: str,
    sides: dict[str, int],
    radius: float,
) -> None:
    """Catches silently retaining photographed paper, shadow, or an exterior white block."""
    source = REAL_SOURCE_ROOT / f"{stem}.png"
    _require_real_source(source, "local batch regression source unavailable")
    rgb = np.asarray(Image.open(source).convert("RGB"))
    override = {**sides, "radii": _uniform_radii(radius)}

    subject, _, exterior, _, details = _detect_closed_subject(
        rgb,
        None,
        validated_frame_override=override,
    )

    audit = details["validated_frame_override"]
    assert details["method"] == "explicit-validated-frame-override"
    assert audit["status"] == "applied"
    assert audit["validation"] == (
        "four-side-gradient-snap-plus-corner-radius-and-frame-structure"
    )
    assert audit["normalized_geometry_px"]["dark_sides_px"] == audit[
        "gradient_alignment"
    ]["best_aligned_anchors_px"]
    assert max(
        abs(value)
        for value in audit["anchor_normalization"][
            "normalization_delta_px"
        ].values()
    ) <= audit["gradient_alignment"]["anchor_probe_radius_px"]
    assert audit["corner_radius_alignment"]["passed"] is True
    assert details["geometry_evidence"]["passed"] is True
    assert not np.any(subject & exterior)
    assert not subject[0, 0]
    assert not subject[0, -1]
    assert not subject[-1, 0]
    assert not subject[-1, -1]


def test_validated_frame_override_is_recorded_in_the_manifest(tmp_path: Path) -> None:
    """Catches an unauditable manual frame that cannot be traced to explicit geometry."""
    source = tmp_path / "paper-and-design-frame.png"
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 20, 342, 520), radius=34, fill=(226, 205, 165))
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    image.save(source)
    override = {
        "left": 41,
        "right": 319,
        "top": 53,
        "bottom": 487,
        "radii": _uniform_radii(25.0),
    }

    report = prepare_subject_first(
        source,
        tmp_path / "prepared",
        minimum_effective_ppi=300,
        validated_frame_override=override,
    )
    manifest = json.loads(Path(str(report["manifest"])).read_text(encoding="utf-8"))

    assert report["frame_detection"]["validated_frame_override"]["status"] == "applied"
    assert manifest["frame_detection"]["validated_frame_override"] == report[
        "frame_detection"
    ]["validated_frame_override"]
    assert report["frame_detection"]["validated_frame_override"][
        "requested_geometry_px"
    ] == override
    assert report["background_sampling"]["subject_overlap_pixels"] == 0
    assert report["background_sampling"]["non_exterior_sample_pixels"] == 0


def test_validated_frame_override_snaps_mask_to_best_aligned_source_anchors() -> None:
    """Catches validating a nearby edge but rendering at the stale requested box."""
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (46, 58, 314, 482),
        radius=26,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    requested = {
        "left": 43,
        "right": 317,
        "top": 55,
        "bottom": 485,
        "radii": _uniform_radii(26.0),
    }

    _, _, _, _, details = _detect_closed_subject(
        np.asarray(image),
        None,
        validated_frame_override=requested,
    )

    audit = details["validated_frame_override"]
    best = audit["gradient_alignment"]["best_aligned_anchors_px"]
    normalized = audit["normalized_geometry_px"]["dark_sides_px"]
    assert best != {side: requested[side] for side in ("left", "right", "top", "bottom")}
    assert normalized == best
    assert details["frame_geometry_px"]["dark_sides_px"] == best
    assert audit["anchor_normalization"]["status"] == "snapped-to-source-gradient"


def test_validated_frame_override_can_use_explicit_operator_reviewed_corner_radii(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "reviewed-radii.png"
    _make_closed_frame_card(source)
    rgb = np.asarray(Image.open(source).convert("RGB"))

    def reject_automatic_radius_normalization(*_args, **_kwargs):
        raise DetectionError("automatic corner tangent normalization is unavailable")

    monkeypatch.setattr(
        subject_pipeline,
        "_normalize_corner_radii_from_source_gradients",
        reject_automatic_radius_normalization,
    )
    subject, _alpha, details = subject_pipeline._detect_validated_frame_override(
        rgb,
        {
            "left": 48,
            "right": 312,
            "top": 64,
            "bottom": 476,
            "radii": 30,
            "corner_radius_validation": (
                "operator-reviewed-source-corner-tangents"
            ),
        },
    )

    radius_evidence = details["validated_frame_override"][
        "corner_radius_alignment"
    ]
    assert radius_evidence["status"] == "operator-reviewed"
    assert radius_evidence["policy"] == (
        "operator-reviewed-source-corner-tangents"
    )
    assert details["frame_geometry_px"]["radii"] == {
        "top_left": 30.0,
        "top_right": 30.0,
        "bottom_left": 30.0,
        "bottom_right": 30.0,
    }
    assert np.any(subject)


def test_validated_override_normalization_is_stable_for_probe_near_requests() -> None:
    """Catches request-window-dependent snapping between inner and outer rails."""
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (46, 58, 314, 482),
        radius=26,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    requests = (
        {
            "left": 46,
            "right": 314,
            "top": 58,
            "bottom": 482,
            "radii": _uniform_radii(26.0),
        },
        {
            "left": 47,
            "right": 315,
            "top": 59,
            "bottom": 483,
            "radii": _uniform_radii(26.0),
        },
        {
            "left": 45,
            "right": 313,
            "top": 57,
            "bottom": 481,
            "radii": _uniform_radii(26.0),
        },
        {
            "left": 48,
            "right": 316,
            "top": 60,
            "bottom": 484,
            "radii": _uniform_radii(26.0),
        },
    )

    detections = [
        _detect_closed_subject(
            np.asarray(image),
            None,
            validated_frame_override=request,
        )
        for request in requests
    ]
    subjects = [result[0] for result in detections]
    alphas = [result[1] for result in detections]
    audits = [result[4]["validated_frame_override"] for result in detections]

    assert all(
        audit["normalized_geometry_px"] == audits[0]["normalized_geometry_px"]
        for audit in audits[1:]
    )
    assert all(np.array_equal(subject, subjects[0]) for subject in subjects[1:])
    assert all(np.array_equal(alpha, alphas[0]) for alpha in alphas[1:])
    assert audits[0]["gradient_alignment"]["rail_selection_policy"] == (
        "outermost-continuous-gradient-band"
    )


def test_validated_frame_mask_preserves_locked_source_design_pixel_cells() -> None:
    """Catches asymmetric half-pixel bounds cropping real right/bottom rails."""
    exterior_rgb = np.asarray((238, 229, 207), dtype=np.uint8)
    image = Image.new("RGB", (360, 540), tuple(int(value) for value in exterior_rgb))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    rgb = np.asarray(image)
    override = {
        "left": 42,
        "right": 318,
        "top": 54,
        "bottom": 486,
        "radii": _uniform_radii(24.0),
    }

    subject, alpha, _, _, details = _detect_closed_subject(
        rgb,
        None,
        validated_frame_override=override,
    )

    source_design = np.any(rgb != exterior_rgb, axis=2)
    geometry = details["frame_geometry_px"]
    assert geometry["left"] == pytest.approx(41.5)
    assert geometry["right"] == pytest.approx(318.5)
    assert geometry["top"] == pytest.approx(53.5)
    assert geometry["bottom"] == pytest.approx(486.5)
    assert np.all(alpha[source_design] == 255)
    assert not np.any(subject & ~source_design)

    locked_rails = {
        "left": (source_design[:, 42], alpha[:, 42]),
        "right": (source_design[:, 318], alpha[:, 318]),
        "top": (source_design[54], alpha[54]),
        "bottom": (source_design[486], alpha[486]),
    }
    for source_rail, alpha_rail in locked_rails.values():
        assert np.any(source_rail)
        assert np.all(alpha_rail[source_rail] == 255)

    corner_boxes = (
        (slice(54, 82), slice(42, 70)),
        (slice(54, 82), slice(291, 319)),
        (slice(459, 487), slice(291, 319)),
        (slice(459, 487), slice(42, 70)),
    )
    for corner in corner_boxes:
        assert np.any(source_design[corner])
        assert np.all(alpha[corner][source_design[corner]] == 255)

    ys, xs = np.nonzero(subject)
    tight_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    assert tight_bbox == (42, 54, 319, 487)
    assert subject[54:487, 42:319].shape == (433, 277)
    assert np.array_equal(subject, alpha > 0)

    assert not np.any(subject[:, 41])
    assert not np.any(subject[:, 319])
    assert not np.any(subject[53])
    assert not np.any(subject[487])


def test_validated_frame_artifacts_keep_the_exact_inclusive_source_crop(
    tmp_path: Path,
) -> None:
    """Catches a correct source mask being shifted again during artifact cropping."""
    source = tmp_path / "inclusive-rails.png"
    exterior_rgb = np.asarray((238, 229, 207), dtype=np.uint8)
    image = Image.new("RGB", (360, 540), tuple(int(value) for value in exterior_rgb))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    image.save(source)

    report = prepare_subject_first(
        source,
        tmp_path / "prepared",
        minimum_effective_ppi=300,
        validated_frame_override={
            "left": 42,
            "right": 318,
            "top": 54,
            "bottom": 486,
            "radii": _uniform_radii(24.0),
        },
    )

    rgba = np.asarray(Image.open(str(report["subject_image"])).convert("RGBA"))
    mask = np.asarray(Image.open(str(report["subject_mask"])).convert("L"))
    source_design = np.any(np.asarray(image) != exterior_rgb, axis=2)[54:487, 42:319]
    assert report["subject_bbox_px"] == [42, 54, 319, 487]
    assert rgba.shape == (433, 277, 4)
    assert mask.shape == (433, 277)
    assert np.array_equal(mask > 0, source_design)
    assert np.array_equal(rgba[..., 3] > 0, source_design)
    assert np.all(rgba[..., 3][source_design] == 255)


def test_corner_radius_stays_on_locked_outer_rail_when_inner_arc_has_larger_radius() -> None:
    """Catches a nearby decorative arc replacing the true outer rounded corner."""
    images: list[Image.Image] = []
    for add_inner_arc in (False, True):
        image = Image.new("RGB", (360, 540), (238, 229, 207))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            (42, 54, 318, 486),
            radius=24,
            fill=(52, 91, 128),
            outline=(34, 20, 9),
            width=4,
        )
        if add_inner_arc:
            draw.rounded_rectangle(
                (44, 56, 316, 484),
                radius=40,
                outline=(250, 205, 80),
                width=3,
            )
        images.append(image)
    override = {
        "left": 42,
        "right": 318,
        "top": 54,
        "bottom": 486,
        "radii": _uniform_radii(24.0),
    }

    detections = [
        _detect_closed_subject(
            np.asarray(image),
            None,
            validated_frame_override=override,
        )
        for image in images
    ]
    subjects = [result[0] for result in detections]
    geometries = [result[4]["frame_geometry_px"] for result in detections]
    evidence = detections[1][4]["validated_frame_override"][
        "corner_radius_alignment"
    ]

    assert geometries[0]["dark_sides_px"] == geometries[1]["dark_sides_px"]
    assert geometries[0]["radii"] == geometries[1]["radii"]
    assert all(
        radius == pytest.approx(24.0, abs=1.0)
        for radius in geometries[0]["radii"].values()
    )
    assert np.array_equal(subjects[0], subjects[1])
    assert evidence["outer_rail_direct_gradient_required"] is True
    assert evidence["outermost_gradient_continuity_required"] is True
    assert evidence["radius_selection_policy"] == (
        "locked-tangent-target-then-narrow-connected-outer-arc"
    )
    assert all(
        corner["outward_competing_gradient_fraction"] == 0.0
        for corner in evidence["corners"].values()
    )
    for corner in evidence["corners"].values():
        estimates = corner["locked_tangent_radius_estimates_px"]
        assert all(value == pytest.approx(24.0, abs=1.0) for value in estimates.values())
        assert corner["tangent_radius_delta_px"] <= evidence[
            "maximum_tangent_radius_delta_px"
        ]
        assert corner["tangent_target_radius_px"] == pytest.approx(24.0, abs=1.0)
        assert corner["connected_outer_arc_evidence"]["passed"] is True
        assert corner["connected_outer_arc_evidence"][
            "connects_both_locked_tangents"
        ] is True


def test_validated_frame_override_normalizes_arbitrary_radii_to_corner_evidence() -> None:
    """Catches a caller-selected radius silently inventing a source corner shape."""
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    requested = {
        "left": 41,
        "right": 319,
        "top": 53,
        "bottom": 487,
        "radii": _uniform_radii(90.0),
    }

    _, _, _, _, details = _detect_closed_subject(
        np.asarray(image),
        None,
        validated_frame_override=requested,
    )

    audit = details["validated_frame_override"]
    normalized = audit["normalized_geometry_px"]["radii"]
    assert audit["corner_radius_alignment"]["passed"] is True
    assert audit["corner_radius_alignment"]["method"] == (
        "source-corner-gradient-arc-and-tangent-normalization"
    )
    assert all(18.0 <= radius <= 32.0 for radius in normalized.values())
    assert all(radius != 90.0 for radius in normalized.values())


def test_validated_frame_override_keeps_only_source_connected_center_ornaments(
    tmp_path: Path,
) -> None:
    """Catches an override replacing real connected ornaments with a bare shape."""
    source = tmp_path / "connected-ornaments.png"
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    draw.polygon(((173, 56), (180, 0), (187, 56)), fill=(34, 20, 9))
    draw.polygon(((173, 484), (180, 539), (187, 484)), fill=(34, 20, 9))
    draw.ellipse((12, 248, 22, 258), fill=(34, 20, 9))
    image.save(source)
    override = {
        "left": 41,
        "right": 319,
        "top": 53,
        "bottom": 487,
        "radii": _uniform_radii(25.0),
    }

    rgb = np.asarray(image)
    subject, _, exterior, _, details = _detect_closed_subject(
        rgb,
        None,
        validated_frame_override=override,
    )

    assert subject[0, 180]
    assert subject[-1, 180]
    assert not subject[253, 17]
    assert not np.any(subject & exterior)
    evidence = details["validated_frame_override"][
        "source_connected_ornament_evidence"
    ]
    assert evidence["source_connected_component_count"] == 1
    assert evidence["retained_contact_pixels"]["top"] > 0
    assert evidence["retained_contact_pixels"]["bottom"] > 0


def test_validated_frame_override_keeps_bright_and_colored_connected_ornaments() -> None:
    """Catches a luminance-only mask dropping physical gold or colored ornaments."""
    image = Image.new("RGB", (360, 540), (238, 229, 207))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (42, 54, 318, 486),
        radius=24,
        fill=(52, 91, 128),
        outline=(34, 20, 9),
        width=4,
    )
    draw.polygon(((173, 56), (180, 0), (187, 56)), fill=(255, 190, 30))
    draw.polygon(((173, 484), (180, 539), (187, 484)), fill=(25, 220, 235))
    draw.ellipse((12, 248, 22, 258), fill=(255, 190, 30))
    override = {
        "left": 41,
        "right": 319,
        "top": 53,
        "bottom": 487,
        "radii": _uniform_radii(25.0),
    }

    subject, _, exterior, _, details = _detect_closed_subject(
        np.asarray(image),
        None,
        validated_frame_override=override,
    )

    assert subject[0, 180]
    assert subject[-1, 180]
    assert not subject[253, 17]
    assert not np.any(subject & exterior)
    evidence = details["validated_frame_override"][
        "source_connected_ornament_evidence"
    ]
    assert evidence["color_policy"] == "source-vs-modeled-exterior-difference"


def test_validated_frame_override_rejects_geometry_not_aligned_to_all_four_edges(
    tmp_path: Path,
) -> None:
    """Catches treating an explicit rectangle as trusted without image-edge validation."""
    source = tmp_path / "card.png"
    _make_closed_frame_card(source)
    invalid = {
        "left": 92,
        "right": 268,
        "top": 140,
        "bottom": 400,
        "radii": _uniform_radii(20.0),
    }

    with pytest.raises(DetectionError, match="validated frame override.*gradient|alignment"):
        prepare_subject_first(
            source,
            tmp_path / "prepared",
            minimum_effective_ppi=300,
            validated_frame_override=invalid,
        )


@pytest.mark.skipif(not DEVIL_SOURCE.is_file(), reason="local Devil regression source unavailable")
def test_real_devil_uses_outer_gradient_frame_and_keeps_local_bottom_ornament(
    tmp_path: Path,
) -> None:
    """Catches dark artwork extrema replacing the four long outer-frame anchors."""
    report = prepare_subject_first(
        DEVIL_SOURCE,
        tmp_path / "prepared",
        minimum_effective_ppi=300,
    )
    geometry = report["frame_detection"]["frame_geometry_px"]
    contacts = report["frame_detection"]["source_canvas_contacts_px"]
    source_width, source_height = report["source_size_px"]

    assert geometry["left"] == pytest.approx(21.5, abs=2.0)
    assert geometry["right"] == pytest.approx(1029.5, abs=2.0)
    assert geometry["top"] == pytest.approx(16.5, abs=3.0)
    assert geometry["bottom"] == pytest.approx(1484.5, abs=3.0)
    assert report["subject_bbox_px"][0] <= 22
    assert report["subject_bbox_px"][2] >= 1030
    assert report["subject_bbox_px"][3] == source_height
    assert 0 < contacts["bottom"] < source_width * 0.20
    assert contacts["top"] < source_width * 0.20
    assert contacts["left"] < source_height * 0.20
    assert contacts["right"] < source_height * 0.20
    assert report["background_sampling"]["sampling_quality_gate"]["status"] == "passed"


@pytest.mark.skipif(not DEVIL_SOURCE.is_file(), reason="local Devil regression source unavailable")
def test_real_devil_automatic_path_keeps_bright_bottom_center_ornament() -> None:
    """Catches the gold source-edge ornament disappearing behind a luminance gate."""
    rgb = np.asarray(Image.open(DEVIL_SOURCE).convert("RGB"))

    alpha, details = _detect_dark_closed_frame(rgb)

    center_x = rgb.shape[1] // 2
    bottom_contact = alpha[-1] > 0
    assert alpha[-1, center_x] > 0
    assert 50 <= int(bottom_contact.sum()) < rgb.shape[1] * 0.25
    evidence = details["source_connected_ornament_evidence"]
    assert evidence["color_policy"] == "source-vs-modeled-exterior-difference"


@pytest.mark.skipif(
    not SHARD1_MATTE_SOURCE.is_file(),
    reason="local broad-matte regression source unavailable",
)
def test_real_gradient_redirect_keeps_gold_support_without_the_red_matte_strip() -> None:
    """Catches fixing the broad strip by either retaining it or deleting the ornament."""
    rgb = np.asarray(Image.open(SHARD1_MATTE_SOURCE).convert("RGB"))

    alpha, details = _detect_dark_closed_frame(rgb)

    top_support = np.flatnonzero(alpha[0] > 0)
    assert 0 < len(top_support) < rgb.shape[1] * 0.05
    assert alpha[0, 476] == 255  # visible source-supported gold branch
    assert alpha[0, rgb.shape[1] // 2] == 0  # surrounding red matte stays exterior
    refinement = details["source_connected_ornament_evidence"][
        "local_center_ornament_refinement"
    ]
    assert refinement["policy"] == "same-row-flank-background-difference"


@pytest.mark.parametrize(
    ("source", "required_contacts"),
    (
        (DEVIL_SOURCE, ("bottom",)),
        (E0C8_SOURCE, ("top", "bottom")),
        (STAR_SOURCE, ("top", "bottom")),
    ),
)
def test_real_automatic_frame_never_absorbs_pixels_outside_source_connected_support(
    source: Path,
    required_contacts: tuple[str, ...],
) -> None:
    """Catches semantic half-pixel geometry absorbing exterior around automatic frames."""
    _require_real_source(
        source,
        f"local real-card regression source unavailable: {source.name}",
    )
    rgb = np.asarray(Image.open(source).convert("RGB"))

    alpha, details = _detect_dark_closed_frame(rgb)
    geometry = details["frame_geometry_px"]
    connected, _ = _source_connected_frame_foreground(
        rgb,
        geometry,
        anchors={
            side: int(geometry["dark_sides_px"][side])
            for side in ("top", "right", "bottom", "left")
        },
        probe_radius=max(2, round(min(rgb.shape[:2]) * 0.006)),
    )
    allowed_support = ndimage.binary_fill_holes(connected)
    subject = alpha > 0

    assert int(np.count_nonzero(subject & ~allowed_support)) == 0
    contacts = {
        "top": int(subject[0].sum()),
        "right": int(subject[:, -1].sum()),
        "bottom": int(subject[-1].sum()),
        "left": int(subject[:, 0].sum()),
    }
    assert all(contacts[side] > 0 for side in required_contacts)
    assert all(contacts[side] < subject.shape[1] * 0.25 for side in {"top", "bottom"})
