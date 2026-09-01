from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from prepare_equal_frame import (  # noqa: E402
    DetectionError,
    _fit_background,
    _render_background,
    main as prepare_equal_frame_main,
    prepare_equal_frame,
)


def _optional_real_source(stem: str) -> Path:
    source_root = os.environ.get("FATE_TAROT_SOURCE_DIR")
    if not source_root:
        pytest.skip("FATE_TAROT_SOURCE_DIR is not configured")
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        pytest.fail("FATE_TAROT_SOURCE_DIR does not point to an available directory")
    source = root / f"{stem}.png"
    if not source.is_file():
        pytest.fail(f"configured real regression source is unavailable: {stem}.png")
    return source


def _gradient_background(
    size: tuple[int, int],
    corners: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    width, height = size
    tl, tr, br, bl = (np.asarray(color, dtype=np.float32) for color in corners)
    u = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    rgb = (
        (1 - u) * (1 - v) * tl
        + u * (1 - v) * tr
        + u * v * br
        + (1 - u) * v * bl
    )
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _make_card(
    path: Path,
    *,
    size: tuple[int, int],
    offset: tuple[int, int],
    background_corners: tuple[tuple[int, int, int], ...],
    clipped: bool = False,
) -> None:
    width, height = size
    rgb = _gradient_background(size, background_corners)
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    left = (0 if clipped else round(width * 0.075)) + dx
    top = (0 if clipped else round(height * 0.055)) + dy
    right = round(width * 0.925) + dx
    bottom = round(height * 0.945) + dy
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(7, width // 24),
        fill=(225, 190, 92),
        outline=(92, 52, 12),
        width=3,
    )
    draw.rounded_rectangle(
        (left + 4, top + 4, right - 4, bottom - 4),
        radius=max(5, width // 28),
        fill=(42, 91, 128),
        outline=(252, 224, 132),
        width=2,
    )
    center_x = (left + right) // 2
    draw.ellipse(
        (center_x - 6, max(0, top - 7), center_x + 6, top + 7),
        fill=(236, 179, 25),
        outline=(92, 52, 12),
        width=2,
    )
    draw.ellipse(
        (center_x - 6, bottom - 7, center_x + 6, min(height - 1, bottom + 7)),
        fill=(236, 179, 25),
        outline=(92, 52, 12),
        width=2,
    )
    draw.rectangle(
        (left + width // 5, top + height // 5, right - width // 5, bottom - height // 5),
        fill=(174, 49, 83),
    )
    image.save(path)


def test_public_api_rejects_minimum_effective_ppi_below_skill_floor(
    tmp_path: Path,
) -> None:
    """Catches callers weakening the Skill's non-negotiable 300 PPI floor."""
    with pytest.raises(DetectionError, match="300 PPI"):
        prepare_equal_frame(
            tmp_path / "source-need-not-be-opened.png",
            tmp_path / "out",
            minimum_effective_ppi=299.99,
        )


def test_cli_rejects_minimum_effective_ppi_below_skill_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches the compatibility CLI accepting a lower density than the API contract."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_equal_frame.py",
            "--source",
            str(tmp_path / "source-need-not-be-opened.png"),
            "--output-dir",
            str(tmp_path / "out"),
            "--minimum-effective-ppi",
            "299.99",
        ],
    )

    with pytest.raises(SystemExit) as error:
        prepare_equal_frame_main()

    assert error.value.code == 2
    assert "300 PPI" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("size", "offset", "corners"),
    [
        (
            (220, 320),
            (2, -1),
            ((188, 24, 28), (215, 35, 31), (172, 18, 23), (201, 29, 34)),
        ),
        (
            (228, 328),
            (-3, 2),
            ((18, 82, 158), (26, 126, 181), (10, 62, 132), (34, 105, 169)),
        ),
    ],
)
def test_processes_each_card_geometry_and_color_independently(
    tmp_path: Path,
    size: tuple[int, int],
    offset: tuple[int, int],
    corners: tuple[tuple[int, int, int], ...],
) -> None:
    """Catches fixed crop coordinates and a bleed color copied from another card."""
    source = tmp_path / f"card-{size[0]}.png"
    output_dir = tmp_path / f"out-{size[0]}"
    _make_card(source, size=size, offset=offset, background_corners=corners)

    report = prepare_equal_frame(
        source,
        output_dir,
        trim_width_mm=80,
        trim_height_mm=120,
        bleed_top_mm=5,
        bleed_bottom_mm=5,
        minimum_effective_ppi=300,
    )

    placement = report["subject_placement_mm"]
    assert report["media_mm"] == [90.0, 130.0]
    assert report["bleed_mm"] == {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0}
    assert placement["cropped"] is False
    assert placement["scale_x"] == placement["scale_y"] == 1.0
    assert abs(placement["center_error_px"][0]) <= 0.5
    assert abs(placement["center_error_px"][1]) <= 0.5
    media = np.asarray(Image.open(report["media_image"]).convert("RGB"))
    assert np.max(np.abs(media[0, 0].astype(int) - np.asarray(corners[0], dtype=int))) <= 12
    assert report["source_size_px"] == [size[0], size[1]]
    assert report["background_sampling"]["source_sha256"]
    assert report["background_sampling"]["region_policy"] == "exterior-only"
    assert report["background_sampling"]["subject_overlap_pixels"] == 0


def test_samples_each_corner_directly_instead_of_inventing_it_from_side_fits() -> None:
    """Catches corner colors being inferred only from adjacent side curves."""
    width, height = 240, 340
    rgb = np.full((height, width, 3), (72, 83, 94), dtype=np.uint8)
    expected = np.asarray(
        [
            (188, 31, 42),
            (28, 119, 187),
            (39, 151, 79),
            (151, 62, 173),
        ],
        dtype=np.uint8,
    )
    extent = 14
    rgb[:extent, :extent] = expected[0]
    rgb[:extent, width - extent :] = expected[1]
    rgb[height - extent :, width - extent :] = expected[2]
    rgb[height - extent :, :extent] = expected[3]

    model = _fit_background(rgb)
    rendered = _render_background(model, width, height)
    actual_rendered = np.stack(
        [rendered[0, 0], rendered[0, -1], rendered[-1, -1], rendered[-1, 0]]
    )

    assert np.max(np.abs(model.corners.astype(int) - expected.astype(int))) <= 2
    assert np.max(np.abs(actual_rendered.astype(int) - expected.astype(int))) <= 2


def test_preserves_every_fully_opaque_subject_pixel_without_resampling(tmp_path: Path) -> None:
    """Catches interpolation, scaling, color conversion, or non-integer placement."""
    source = tmp_path / "card.png"
    output_dir = tmp_path / "out"
    corners = ((37, 118, 72), (55, 151, 84), (27, 101, 59), (45, 132, 76))
    _make_card(source, size=(224, 324), offset=(1, 1), background_corners=corners)

    report = prepare_equal_frame(
        source,
        output_dir,
        trim_width_mm=80,
        trim_height_mm=120,
        bleed_top_mm=5,
        bleed_bottom_mm=5,
        minimum_effective_ppi=300,
    )

    source_rgb = np.asarray(Image.open(source).convert("RGB"))
    subject = np.asarray(Image.open(report["subject_image"]).convert("RGBA"))
    media = np.asarray(Image.open(report["media_image"]).convert("RGB"))
    x0, y0, x1, y1 = report["source_crop_bbox_px"]
    px, py = report["placement_px"]
    opaque = subject[..., 3] == 255
    source_crop = source_rgb[y0:y1, x0:x1]
    placed = media[py : py + subject.shape[0], px : px + subject.shape[1]]

    assert np.array_equal(placed[opaque], source_crop[opaque])
    assert report["opaque_subject_pixels"] == int(opaque.sum())
    assert report["opaque_subject_exact_pixels"] == int(opaque.sum())
    assert report["resampled"] is False


def test_rejects_clipped_design_and_raises_canvas_density_without_resampling(
    tmp_path: Path,
) -> None:
    """Catches treating a PPI floor as a reason to resize or reject original pixels."""
    corners = ((160, 30, 30), (180, 36, 36), (150, 22, 22), (170, 28, 28))
    clipped = tmp_path / "clipped.png"
    _make_card(clipped, size=(220, 320), offset=(0, 0), background_corners=corners, clipped=True)
    with pytest.raises(DetectionError, match="clipped by the source canvas|touches the source canvas"):
        prepare_equal_frame(clipped, tmp_path / "clipped-out", minimum_effective_ppi=300)

    low_resolution = tmp_path / "low-resolution.png"
    _make_card(low_resolution, size=(220, 320), offset=(0, 0), background_corners=corners)
    low = prepare_equal_frame(
        low_resolution,
        tmp_path / "low-out",
        minimum_effective_ppi=300,
    )
    subject = np.asarray(Image.open(low["subject_image"]).convert("RGBA"))
    media = np.asarray(Image.open(low["media_image"]).convert("RGB"))
    placement_x, placement_y = low["placement_px"]
    placed = media[
        placement_y : placement_y + subject.shape[0],
        placement_x : placement_x + subject.shape[1],
    ]
    opaque = subject[..., 3] == 255

    assert low["effective_ppi"] >= 300
    assert low["raster_density_policy"] == (
        "minimum-effective-ppi-floor-without-resampling"
    )
    assert low["resampled"] is False
    assert low["subject_placement_mm"]["scale_x"] == 1.0
    assert low["subject_placement_mm"]["scale_y"] == 1.0
    assert low["subject_placement_mm"]["width_mm"] < 80.0
    assert low["subject_placement_mm"]["height_mm"] < 120.0
    assert np.array_equal(placed[opaque], subject[..., :3][opaque])


def test_writes_machine_readable_manifest(tmp_path: Path) -> None:
    """Catches missing audit evidence needed before PDF export."""
    source = tmp_path / "card.png"
    corners = ((115, 57, 151), (137, 72, 177), (95, 43, 132), (124, 64, 161))
    _make_card(source, size=(226, 326), offset=(0, 0), background_corners=corners)

    report = prepare_equal_frame(
        source,
        tmp_path / "out",
        minimum_effective_ppi=300,
    )
    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))

    assert manifest["transform_policy"] == "integer-translation-no-resample"
    assert manifest["frame_spacing"]["mode"] == "fixed-trim-and-bleed"
    assert manifest["frame_spacing"]["measurement_anchor"] == "complete-subject-bounds"
    assert manifest["bleed_policy"] == "current-card-exterior-only-side-and-corner-sampling"
    assert manifest["background_sampling"]["corner_method"] == "direct-exterior-only-source-patch-median"
    assert manifest["background_sampling"]["region_policy"] == "exterior-only"
    assert manifest["background_sampling"]["subject_overlap_pixels"] == 0
    assert [stage["name"] for stage in manifest["pipeline_stages"]] == [
        "detect-subject-frame",
        "extract-subject",
        "sample-exterior",
        "render-bleed-background",
        "place-subject",
    ]
    assert set(manifest["background_sampling"]["corner_median_rgb"]) == {
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    }


def test_edge_hugging_card_uses_paired_frame_evidence_and_keeps_local_ornaments(
    tmp_path: Path,
) -> None:
    """Catches dropping local ornaments or sampling them after a paired gradient redirect."""
    source = _optional_real_source("E0C8E863-F59F-4C96-B39C-A2665E8F48A8")

    report = prepare_equal_frame(source, tmp_path / "paired-gradient")
    rgba = np.asarray(Image.open(report["subject_image"]).convert("RGBA"))
    alpha = rgba[..., 3]
    left, _, _, _ = report["subject_bbox_px"]
    source_center_x = report["source_size_px"][0] // 2
    redirect = report["frame_detection"]["geometry_evidence"][
        "edge_hugging_component_redirect"
    ]
    contacts = report["frame_detection"]["source_canvas_contacts_px"]

    assert report["frame_detection"]["frame_geometry_px"]["dark_sides_px"] == {
        "left": 22,
        "right": 1030,
        "top": 16,
        "bottom": 1484,
    }
    assert redirect["selection_mode"] == "paired-long-edge-gradient-fallback"
    assert redirect["paired_long_sides"] == ["left", "right"]
    assert set(redirect["long_line_coverage"]) == {"top", "right", "bottom", "left"}
    assert min(redirect["long_line_coverage"].values()) >= 0.70
    assert int(alpha[0, source_center_x - left]) > 0
    assert int(alpha[-1, source_center_x - left]) > 0
    assert 0 < contacts["top"] < report["source_size_px"][0] * 0.25
    assert 0 < contacts["bottom"] < report["source_size_px"][0] * 0.25
    assert contacts["left"] == contacts["right"] == 0
    sampling = np.asarray(
        Image.open(report["exterior_sampling_mask"]).convert("L")
    ) > 0
    source_subject = np.zeros(sampling.shape, dtype=bool)
    x0, y0, x1, y1 = report["subject_bbox_px"]
    source_subject[y0:y1, x0:x1] = alpha > 0
    assert not np.any(sampling & source_subject)
    assert report["background_sampling"]["subject_overlap_pixels"] == 0
    assert report["background_sampling"]["non_exterior_sample_pixels"] == 0


def test_compatibility_api_forwards_validated_frame_override(tmp_path: Path) -> None:
    """Catches the legacy entry point silently dropping an audited frame override."""
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
        "radii": 25.0,
    }

    report = prepare_equal_frame(
        source,
        tmp_path / "compat-override",
        minimum_effective_ppi=300,
        validated_frame_override=override,
    )

    audit = report["frame_detection"]["validated_frame_override"]
    assert audit["status"] == "applied"
    assert audit["requested_geometry_px"] == {
        **{side: override[side] for side in ("left", "right", "top", "bottom")},
        "radii": {
            "top_left": 25.0,
            "top_right": 25.0,
            "bottom_left": 25.0,
            "bottom_right": 25.0,
        },
    }


def test_real_star_preserves_local_edge_ornament_and_fixed_geometry(tmp_path: Path) -> None:
    """Catches treating a narrow center ornament contact as a continuously clipped frame side."""
    source = _optional_real_source("0A8D6B2A-05AD-4C2C-9DC8-68096007CB7A")

    report = prepare_equal_frame(source, tmp_path / "star")
    rgba = np.asarray(Image.open(report["subject_image"]).convert("RGBA"))
    original = np.asarray(Image.open(source).convert("RGB"))
    left, top, right, bottom = report["subject_bbox_px"]
    alpha = rgba[..., 3]
    opaque = alpha == 255

    assert report["frame_detection"]["method"] == "continuous-dark-frame-plus-rounded-geometric-fill"
    assert report["frame_detection"]["frame_geometry_px"]["dark_sides_px"] == {
        "left": 16,
        "right": 1037,
        "top": 12,
        "bottom": 1487,
    }
    # The confirmed right rail is source pixel x=1037, so its half-open crop
    # endpoint is 1038.  The old 1039 assertion retained one exterior column.
    assert report["subject_bbox_px"] == [15, 0, 1038, 1492]
    assert np.array_equal(rgba[..., :3][opaque], original[top:bottom, left:right][opaque])
    assert int(alpha[-1, 520 - left]) > 0  # bottom-center ornament touches locally and remains.
    contacts = report["frame_detection"]["source_canvas_contacts_px"]
    assert 0 < contacts["top"] < report["source_size_px"][0] * 0.25
    assert contacts["right"] == contacts["left"] == 0
    assert 0 < contacts["bottom"] < report["source_size_px"][0] * 0.25
    assert report["media_size_px"] == [1152, 1664]
    assert report["placement_px"] == [65, 86]
    assert report["background_sampling"]["subject_overlap_pixels"] == 0
    assert report["effective_ppi"] == pytest.approx(325.12)
    assert report["resampled"] is False
