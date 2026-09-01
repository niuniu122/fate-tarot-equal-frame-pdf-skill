from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import fitz
import numpy as np
from PIL import Image, ImageDraw
import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
import export_from_manifest as export_module  # noqa: E402
from export_from_manifest import (  # noqa: E402
    _boundary_connected,
    _find_print_skill_root,
    _manifest_contract_evidence,
    _recompute_v2_sampling_semantics,
    _validate_subject_first_manifest,
    export_from_manifest,
)
from prepare_equal_frame import prepare_equal_frame  # noqa: E402
from subject_first_pipeline import prepare_subject_first  # noqa: E402


def _optional_fixture_root(environment_variable: str) -> Path:
    value = os.environ.get(environment_variable)
    if not value:
        pytest.skip(f"{environment_variable} is not configured")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        pytest.fail(f"{environment_variable} does not point to an available directory")
    return root


def _optional_approved_manifest_paths() -> list[Path]:
    value = os.environ.get("FATE_TAROT_APPROVED_MANIFESTS")
    if not value:
        pytest.skip("FATE_TAROT_APPROVED_MANIFESTS is not configured")
    paths = [
        Path(item).expanduser().resolve()
        for item in value.split(os.pathsep)
        if item.strip()
    ]
    if not paths:
        pytest.fail("FATE_TAROT_APPROVED_MANIFESTS contains no manifest paths")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        pytest.fail(f"configured approved manifests are unavailable: {missing}")
    return paths


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_print_skill_discovery_never_loads_from_the_current_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevents an untrusted project from shadowing the installed PDF dependency."""
    project_root = tmp_path / "project"
    shadow_scripts = (
        project_root / ".codex" / "skills" / "card-artwork-print-pdf" / "scripts"
    )
    shadow_scripts.mkdir(parents=True)
    (shadow_scripts / "build_print_pdf.py").write_text("RAISED = True\n", encoding="utf-8")
    (shadow_scripts / "verify_artwork.py").write_text("RAISED = True\n", encoding="utf-8")

    monkeypatch.chdir(project_root)
    monkeypatch.delenv("CARD_ARTWORK_PRINT_PDF_SKILL", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "isolated-codex-home"))

    discovered = _find_print_skill_root(None)

    assert discovered == (SKILL_ROOT / "vendor" / "card-artwork-print-pdf").resolve()
    assert project_root not in discovered.parents


def _fake_print_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    render_nonblank: bool = True,
) -> dict[str, str]:
    """Install deterministic in-process builder/verifier doubles."""
    observations: dict[str, str] = {}
    skill_root = tmp_path / "trusted-print-skill"

    def write_print_pdf(*, image_path, output_pdf, **kwargs):
        observations["build_image_sha256"] = _sha256(image_path)
        output = Path(output_pdf)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"%PDF-1.4\n% deterministic test double\n")
        return {
            "pdf": str(output),
            "source_image": str(image_path),
            "resampled": False,
        }

    def verify_pdf(pdf_path, expected_image, **kwargs):
        observations["verify_image_sha256"] = _sha256(expected_image)
        return {
            "pdf": str(pdf_path),
            "direct_image_xobject": True,
            "embedded_pixels_equal_approved_image": True,
            "renderers": ["test-double"] if render_nonblank else [],
            "render_nonblank": render_nonblank,
        }

    builder = SimpleNamespace(write_print_pdf=write_print_pdf)
    verifier = SimpleNamespace(verify_pdf=verify_pdf)

    def load_module(_name: str, path: Path):
        if path.name == "build_print_pdf.py":
            return builder
        if path.name == "verify_artwork.py":
            return verifier
        raise AssertionError(f"unexpected print module: {path}")

    monkeypatch.setattr(export_module, "_find_print_skill_root", lambda _explicit: skill_root)
    monkeypatch.setattr(export_module, "_load_module", load_module)
    return observations


def test_export_uses_verified_media_snapshot_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches replacing approved media between validation and PDF construction."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    media_path = Path(manifest["artifacts"]["media_rgb"]["path"])
    approved_sha256 = manifest["artifacts"]["media_rgb"]["sha256"]
    observations = _fake_print_modules(monkeypatch, tmp_path)
    original_find = export_module._find_print_skill_root

    def mutate_after_validation(explicit):
        media = np.asarray(Image.open(media_path).convert("RGB")).copy()
        media[0, 0] = np.bitwise_xor(media[0, 0], np.asarray((255, 255, 255), dtype=np.uint8))
        Image.fromarray(media, "RGB").save(media_path)
        assert _sha256(media_path) != approved_sha256
        return original_find(explicit)

    monkeypatch.setattr(export_module, "_find_print_skill_root", mutate_after_validation)
    export_from_manifest(manifest_path, tmp_path / "card.pdf")

    assert observations == {
        "build_image_sha256": approved_sha256,
        "verify_image_sha256": approved_sha256,
    }


def test_export_rejects_output_path_that_overwrites_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    original = manifest_path.read_bytes()
    monkeypatch.setattr(
        export_module,
        "_find_print_skill_root",
        lambda _explicit: (_ for _ in ()).throw(AssertionError("dependency lookup must not run")),
    )

    with pytest.raises(ValueError, match="collid|overwrite|input"):
        export_from_manifest(manifest_path, manifest_path)

    assert manifest_path.read_bytes() == original


def test_export_rejects_external_report_path_that_overwrites_pdf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    output_pdf = tmp_path / "card.pdf"

    with pytest.raises(ValueError, match="collid|output|report"):
        export_from_manifest(
            prepared["manifest"],
            output_pdf,
            report_path=output_pdf,
        )

    assert not output_pdf.exists()


def test_export_requires_independent_nonblank_render_and_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    output_pdf = tmp_path / "card.pdf"
    original = b"existing approved PDF"
    output_pdf.write_bytes(original)
    _fake_print_modules(monkeypatch, tmp_path, render_nonblank=False)

    with pytest.raises(RuntimeError, match="independent.*render|renderer"):
        export_from_manifest(prepared["manifest"], output_pdf)

    assert output_pdf.read_bytes() == original


def _sync_artifact_hash(manifest: dict, name: str) -> None:
    artifact = manifest["artifacts"][name]
    digest = _sha256(artifact["path"])
    artifact["sha256"] = digest
    if name == "subject_rgba":
        manifest["subject_extraction"]["subject_rgba_sha256"] = digest
    elif name == "subject_mask":
        manifest["subject_extraction"]["subject_mask_sha256"] = digest
    elif name == "exterior_sampling_mask":
        manifest["background_sampling"]["sampling_mask_sha256"] = digest
    elif name == "media_rgb":
        manifest["media_sha256"] = digest
    elif name == "bleed_background_rgb":
        manifest["bleed_background_sha256"] = digest
    elif name == "sampling_provenance":
        manifest["sampling_provenance_sha256"] = digest


def _write_sampling_provenance_artifact(
    manifest: dict,
    payload: object,
) -> None:
    path = Path(manifest["artifacts"]["sampling_provenance"]["path"])
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _sync_artifact_hash(manifest, "sampling_provenance")


def _tamper_subject_alpha(manifest: dict) -> None:
    path = Path(manifest["artifacts"]["subject_rgba"]["path"])
    rgba = np.asarray(Image.open(path).convert("RGBA")).copy()
    y, x = np.argwhere(rgba[..., 3] == 255)[0]
    rgba[y, x, 3] = 0
    Image.fromarray(rgba, "RGBA").save(path)
    manifest["opaque_subject_pixels"] -= 1
    manifest["opaque_subject_exact_pixels"] -= 1
    _sync_artifact_hash(manifest, "subject_rgba")


def _tamper_nonbinary_subject_mask(manifest: dict) -> None:
    path = Path(manifest["artifacts"]["subject_mask"]["path"])
    mask = np.asarray(Image.open(path).convert("L")).copy()
    y, x = np.argwhere(mask == 255)[0]
    mask[y, x] = 128
    Image.fromarray(mask, "L").save(path)
    _sync_artifact_hash(manifest, "subject_mask")


def _tamper_sampling_into_subject(manifest: dict) -> None:
    subject = np.asarray(
        Image.open(manifest["artifacts"]["subject_mask"]["path"]).convert("L")
    )
    subject_y, subject_x = np.argwhere(subject > 0)[0]
    left, top, _, _ = manifest["subject_bbox_px"]
    sampling_path = Path(manifest["artifacts"]["exterior_sampling_mask"]["path"])
    sampling = np.asarray(Image.open(sampling_path).convert("L")).copy()
    sampling[top + subject_y, left + subject_x] = 255
    Image.fromarray(sampling, "L").save(sampling_path)
    manifest["background_sampling"]["exterior_sample_pixels"] = int((sampling > 0).sum())
    manifest["background_sampling"]["subject_overlap_pixels"] = 0
    _sync_artifact_hash(manifest, "exterior_sampling_mask")


def _tamper_opaque_media_pixel(manifest: dict) -> None:
    rgba = np.asarray(
        Image.open(manifest["artifacts"]["subject_rgba"]["path"]).convert("RGBA")
    )
    subject_y, subject_x = np.argwhere(rgba[..., 3] == 255)[0]
    placement_x, placement_y = manifest["placement_px"]
    media_path = Path(manifest["artifacts"]["media_rgb"]["path"])
    media = np.asarray(Image.open(media_path).convert("RGB")).copy()
    y = placement_y + subject_y
    x = placement_x + subject_x
    media[y, x] = np.bitwise_xor(media[y, x], np.asarray((255, 255, 255), dtype=np.uint8))
    Image.fromarray(media, "RGB").save(media_path)
    _sync_artifact_hash(manifest, "media_rgb")


def _tamper_background_dimensions(manifest: dict) -> None:
    path = Path(manifest["artifacts"]["bleed_background_rgb"]["path"])
    background = np.asarray(Image.open(path).convert("RGB"))[:-1].copy()
    Image.fromarray(background, "RGB").save(path)
    _sync_artifact_hash(manifest, "bleed_background_rgb")


def _tamper_source_and_all_declared_source_hashes(manifest: dict) -> None:
    rgba = np.asarray(
        Image.open(manifest["artifacts"]["subject_rgba"]["path"]).convert("RGBA")
    )
    subject_y, subject_x = np.argwhere(rgba[..., 3] == 255)[0]
    left, top, _, _ = manifest["subject_bbox_px"]
    source_path = Path(manifest["source"])
    source = np.asarray(Image.open(source_path).convert("RGB")).copy()
    source[top + subject_y, left + subject_x] = np.bitwise_xor(
        source[top + subject_y, left + subject_x],
        np.asarray((255, 255, 255), dtype=np.uint8),
    )
    Image.fromarray(source, "RGB").save(source_path)
    digest = _sha256(source_path)
    manifest["source_sha256"] = digest
    manifest["subject_extraction"]["source_sha256"] = digest
    manifest["background_sampling"]["source_sha256"] = digest
    for artifact in manifest["artifacts"].values():
        artifact["source_sha256"] = digest


def _tamper_declared_effective_ppi_below_minimum(manifest: dict) -> None:
    manifest["minimum_effective_ppi"] = 300.0
    manifest["effective_ppi"] = 72.0


def _tamper_minimum_effective_ppi_below_skill_floor(manifest: dict) -> None:
    manifest["minimum_effective_ppi"] = 72.0


def _forge_passed_sampling_and_geometry_metrics(manifest: dict) -> None:
    geometry = manifest["frame_detection"]["geometry_evidence"]
    geometry["passed"] = True
    geometry["continuous_side_coverage"] = {
        side: 0.0 for side in ("top", "right", "bottom", "left")
    }
    sampling = manifest["background_sampling"]
    sampling["non_exterior_sample_pixels"] = 999
    sampling["side_coverage_fraction"] = {
        side: 0.0 for side in ("top", "right", "bottom", "left")
    }
    sampling["corner_coverage_fraction"] = {
        corner: 0.0
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    }
    sampling["side_texture_dispersion"] = {
        side: {
            "coordinate_p90_second_difference_linf_p90": 999.0,
            "bad_coordinate_fraction": 1.0,
            "longest_bad_run_fraction": 1.0,
        }
        for side in ("top", "right", "bottom", "left")
    }
    sampling["sampling_quality_gate"]["status"] = "passed"


def _forge_plausible_but_wrong_geometry_metrics(manifest: dict) -> None:
    geometry = manifest["frame_detection"]["geometry_evidence"]
    geometry["continuous_side_coverage"] = {
        side: 0.51 for side in ("top", "right", "bottom", "left")
    }
    geometry["double_layer_coverage"] = {
        side: 0.36 for side in ("top", "right", "bottom", "left")
    }


def _forge_plausible_but_wrong_sampling_metrics(manifest: dict) -> None:
    sampling = manifest["background_sampling"]
    sampling["side_coverage_fraction"] = {
        side: 0.50 for side in ("top", "right", "bottom", "left")
    }
    sampling["corner_coverage_fraction"] = {
        corner: 0.50
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    }
    sampling["side_texture_dispersion"] = {
        side: {
            "coordinate_p90_second_difference_linf_p90": 0.0,
            "bad_coordinate_fraction": 0.0,
            "longest_bad_run_fraction": 0.0,
        }
        for side in ("top", "right", "bottom", "left")
    }


def _forge_coordinated_plausible_sampling_counts_and_ratios(manifest: dict) -> None:
    """Keep forged ratios internally consistent with equally forged counts."""
    sampling = manifest["background_sampling"]
    source_width, source_height = manifest["source_size_px"]
    band = max(4, round(min(source_width, source_height) * 0.06))
    side_counts: dict[str, int] = {}
    side_coverage: dict[str, float] = {}
    for side in ("top", "right", "bottom", "left"):
        axis_length = source_width if side in {"top", "bottom"} else source_height
        denominator = axis_length * band
        side_counts[side] = round(denominator * 0.50)
        side_coverage[side] = side_counts[side] / denominator
    sampling["side_sample_pixels"] = side_counts
    sampling["side_coverage_fraction"] = side_coverage

    corner_extent = max(band, round(min(source_width, source_height) * 0.08))
    corner_denominator = corner_extent * corner_extent
    corner_count = round(corner_denominator * 0.50)
    sampling["corner_sample_pixels"] = {
        corner: corner_count
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    }
    sampling["corner_coverage_fraction"] = {
        corner: corner_count / corner_denominator
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    }


_V3_SAMPLING_PROVENANCE_FIELDS = (
    "side_interpolated_coordinate_count",
    "side_interpolation_maximum_gap_px",
    "side_interpolation_maximum_allowed_gap_px",
    "side_interpolation_exceptions",
)


def _delete_all_sampling_provenance(manifest: dict) -> None:
    sampling = manifest["background_sampling"]
    for field in _V3_SAMPLING_PROVENANCE_FIELDS:
        sampling.pop(field, None)


def _strip_every_v3_manifest_marker(manifest: dict) -> None:
    manifest["pipeline_version"] = 2
    manifest.pop("sampling_provenance_schema_version", None)
    manifest.pop("sampling_provenance_artifact", None)
    manifest.pop("sampling_provenance_sha256", None)
    manifest["artifacts"].pop("sampling_provenance", None)
    manifest["background_sampling"].pop("side_interpolation_method", None)
    _delete_all_sampling_provenance(manifest)


def _forge_absurd_sampling_provenance(manifest: dict) -> None:
    sampling = manifest["background_sampling"]
    side_names = ("top", "right", "bottom", "left")
    sampling["side_interpolated_coordinate_count"] = {
        side: 999_999 for side in side_names
    }
    sampling["side_interpolation_maximum_gap_px"] = {
        side: 999_999 for side in side_names
    }
    sampling["side_interpolation_maximum_allowed_gap_px"] = {
        side: 0 for side in side_names
    }
    sampling["side_interpolation_exceptions"] = "forged"


def _increment_bottom_count_and_ratio(manifest: dict) -> None:
    sampling = manifest["background_sampling"]
    source_width = manifest["source_size_px"][0]
    band = max(4, round(min(manifest["source_size_px"]) * 0.06))
    sampling["side_sample_pixels"]["bottom"] += 1
    sampling["side_coverage_fraction"]["bottom"] = (
        sampling["side_sample_pixels"]["bottom"] / (source_width * band)
    )


def _tamper_sampling_into_enclosed_non_subject_hole(manifest: dict) -> None:
    mask_path = Path(manifest["artifacts"]["subject_mask"]["path"])
    rgba_path = Path(manifest["artifacts"]["subject_rgba"]["path"])
    sampling_path = Path(manifest["artifacts"]["exterior_sampling_mask"]["path"])
    background_path = Path(manifest["artifacts"]["bleed_background_rgb"]["path"])
    media_path = Path(manifest["artifacts"]["media_rgb"]["path"])

    mask = np.asarray(Image.open(mask_path).convert("L")).copy()
    rgba = np.asarray(Image.open(rgba_path).convert("RGBA")).copy()
    sampling = np.asarray(Image.open(sampling_path).convert("L")).copy()
    background = np.asarray(Image.open(background_path).convert("RGB"))
    media = np.asarray(Image.open(media_path).convert("RGB")).copy()
    center_y, center_x = np.asarray(mask.shape) // 2
    candidates = np.argwhere(mask == 255)
    subject_y, subject_x = min(
        candidates,
        key=lambda coordinate: (
            int(coordinate[0] - center_y) ** 2
            + int(coordinate[1] - center_x) ** 2
        ),
    )

    mask[subject_y, subject_x] = 0
    rgba[subject_y, subject_x] = 0
    left, top, _, _ = manifest["subject_bbox_px"]
    sampling[top + subject_y, left + subject_x] = 255
    placement_x, placement_y = manifest["placement_px"]
    media[placement_y + subject_y, placement_x + subject_x] = background[
        placement_y + subject_y,
        placement_x + subject_x,
    ]

    Image.fromarray(mask, "L").save(mask_path)
    Image.fromarray(rgba, "RGBA").save(rgba_path)
    Image.fromarray(sampling, "L").save(sampling_path)
    Image.fromarray(media, "RGB").save(media_path)
    manifest["opaque_subject_pixels"] -= 1
    manifest["opaque_subject_exact_pixels"] -= 1
    manifest["background_sampling"]["exterior_sample_pixels"] = int(
        (sampling > 0).sum()
    )
    manifest["background_sampling"]["subject_overlap_pixels"] = 0
    manifest["background_sampling"]["non_exterior_sample_pixels"] = 0
    _sync_artifact_hash(manifest, "subject_mask")
    _sync_artifact_hash(manifest, "subject_rgba")
    _sync_artifact_hash(manifest, "exterior_sampling_mask")
    _sync_artifact_hash(manifest, "media_rgb")


def _make_source(path: Path) -> None:
    width, height = 240, 350
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 24 + (x * 18 // width)
    rgb[..., 1] = 76 + (y * 22 // height)
    rgb[..., 2] = 148 + ((x + y) * 24 // (width + height))
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    frame = (18, 19, 222, 331)
    draw.rounded_rectangle(frame, radius=12, fill=(230, 194, 88), outline=(84, 49, 15), width=3)
    draw.rounded_rectangle((22, 23, 218, 327), radius=9, fill=(53, 99, 126), outline=(250, 222, 126), width=2)
    draw.ellipse((114, 11, 126, 27), fill=(240, 179, 24), outline=(84, 49, 15), width=2)
    draw.ellipse((114, 323, 126, 339), fill=(240, 179, 24), outline=(84, 49, 15), width=2)
    image.save(path)


def test_prepared_media_builds_and_verifies_as_print_pdf(tmp_path: Path) -> None:
    """Catches a manifest whose dimensions cannot survive the real PDF builder/verifier."""
    source = tmp_path / "blue-card.png"
    _make_source(source)

    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    pdf = tmp_path / "card.pdf"
    report = export_from_manifest(prepared["manifest"], pdf)
    document = fitz.open(pdf)
    pixmap = document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    rendered = np.frombuffer(pixmap.samples, dtype=np.uint8)
    document.close()

    assert report["pdf"]["direct_image_xobject"] is True
    assert report["pdf"]["embedded_pixels_equal_approved_image"] is True
    assert report["manifest_validation"] == {
        "pipeline_version": 3,
        "sampling_provenance_schema_version": 1,
        "validation_mode": "v3-exact-decoded-provenance",
    }
    assert Path(report["build_report"]).exists()
    assert Path(report["verify_report"]).exists()
    build_report = json.loads(
        Path(report["build_report"]).read_text(encoding="utf-8")
    )
    assert build_report["manifest_validation"] == report["manifest_validation"]
    assert rendered.size > 0
    assert np.any(rendered < 250)


def test_export_gate_accepts_reviewed_flat_exterior_and_recomputes_its_color(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reviewed-flat-card.png"
    _make_source(source)
    prepared = prepare_subject_first(
        source,
        tmp_path / "prepared-flat",
        minimum_effective_ppi=300,
        reviewed_flat_exterior=True,
    )
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    media_path, trim_width_mm, trim_height_mm, bleed = (
        _validate_subject_first_manifest(manifest, manifest_path.parent)
    )

    assert media_path == Path(manifest["media_image"]).resolve()
    assert (trim_width_mm, trim_height_mm) == (80.0, 120.0)
    assert bleed == {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0}


def test_export_gate_rejects_a_stripe_in_reviewed_flat_exterior(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reviewed-flat-card.png"
    _make_source(source)
    prepared = prepare_subject_first(
        source,
        tmp_path / "prepared-flat",
        minimum_effective_ppi=300,
        reviewed_flat_exterior=True,
    )
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    background_path = Path(manifest["bleed_background_image"])
    media_path = Path(manifest["media_image"])
    background = np.asarray(Image.open(background_path).convert("RGB")).copy()
    media = np.asarray(Image.open(media_path).convert("RGB")).copy()
    replacement = np.asarray((0, 255, 255), dtype=np.uint8)
    background[0, 0] = replacement
    media[0, 0] = replacement
    Image.fromarray(background, "RGB").save(background_path)
    Image.fromarray(media, "RGB").save(media_path)
    _sync_artifact_hash(manifest, "bleed_background_rgb")
    _sync_artifact_hash(manifest, "media_rgb")

    with pytest.raises(ValueError, match="reviewed flat.*uniform|uniform.*reviewed flat"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["media_mm"].__setitem__(0, 91.0),
            "media geometry",
        ),
        (
            lambda manifest: manifest["pipeline_stages"][2].__setitem__("status", "skipped"),
            "not all completed",
        ),
        (
            lambda manifest: manifest["background_sampling"].__setitem__(
                "subject_overlap_pixels", 1
            ),
            "overlaps the protected subject",
        ),
        (
            lambda manifest: manifest["subject_placement_mm"].__setitem__("scale_y", 1.01),
            "unequal horizontal and vertical scale",
        ),
        (
            lambda manifest: manifest["frame_detection"]["geometry_evidence"].__setitem__(
                "passed", False
            ),
            "frame geometry evidence",
        ),
        (
            lambda manifest: manifest["background_sampling"][
                "sampling_quality_gate"
            ].__setitem__("status", "failed"),
            "sampling quality gate",
        ),
        (
            _tamper_declared_effective_ppi_below_minimum,
            "effective PPI",
        ),
    ],
)
def test_export_gate_rejects_tampered_subject_first_evidence(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    """Catches PDF export after fixed geometry, stage, sampling, or transform proof is altered."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_tamper_subject_alpha, "alpha.*subject mask|subject mask.*alpha"),
        (_tamper_nonbinary_subject_mask, "binary subject mask"),
        (_tamper_sampling_into_subject, "sampling mask.*overlap|overlaps.*subject"),
        (_tamper_opaque_media_pixel, "opaque subject pixels"),
        (_tamper_background_dimensions, "background.*dimensions|media dimensions"),
        (_tamper_source_and_all_declared_source_hashes, "subject RGB.*source|source.*subject RGB"),
    ],
)
def test_export_gate_rejects_coordinated_file_hash_and_manifest_tampering(
    tmp_path: Path,
    tamper,
    message: str,
) -> None:
    """Catches coordinated file+hash edits that satisfy self-reported fields but not decoded pixels."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    tamper(manifest)

    with pytest.raises(ValueError, match=message):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_accepts_existing_approved_top_level_raster_schema() -> None:
    """Keeps the exporter compatible with manifests already approved in this batch."""
    manifest_path = (
        _optional_fixture_root("FATE_TAROT_BATCH_CARDS_DIR")
        / "05614033-A60E-4462-956A-D48CD619DBB7"
        / "manifest.json"
    )
    if not manifest_path.is_file():
        pytest.fail("configured approved batch manifest is not available in this workspace")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    media_path, trim_width_mm, trim_height_mm, bleed = _validate_subject_first_manifest(
        manifest,
        manifest_path.parent,
    )

    assert media_path == Path(manifest["artifacts"]["media_rgb"]["path"]).resolve()
    assert (trim_width_mm, trim_height_mm) == (80.0, 120.0)
    assert bleed == {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0}


def test_export_gate_accepts_all_existing_approved_batch_manifests() -> None:
    """Recomputes every manifest in an explicitly approved regression set."""
    manifest_paths = _optional_approved_manifest_paths()
    assert manifest_paths
    assert all(path.is_file() for path in manifest_paths)

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_v2_replay_exactly_reconstructs_0efd_fallback_counts_and_union() -> None:
    """Catches replacing the historical whole-column fallback with a tolerance."""
    manifest_path = (
        _optional_fixture_root("FATE_TAROT_BATCH_CARDS_DIR")
        / "0EFD5AE0-6E88-4D6D-9A5C-DC1643EC172E"
        / "manifest.json"
    )
    if not manifest_path.is_file():
        pytest.fail("configured 0EFD legacy regression artifact is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = np.asarray(Image.open(manifest["source"]).convert("RGB"))
    source_height, source_width = source.shape[:2]
    left, top, right, bottom = manifest["subject_bbox_px"]
    subject = np.zeros((source_height, source_width), dtype=bool)
    subject[top:bottom, left:right] = np.asarray(
        Image.open(manifest["artifacts"]["subject_mask"]["path"]).convert("L")
    ) == 255
    exterior = _boundary_connected(~subject)

    replayed_mask, replayed = _recompute_v2_sampling_semantics(
        source,
        exterior,
        subject,
    )
    decoded_mask = np.asarray(
        Image.open(
            manifest["artifacts"]["exterior_sampling_mask"]["path"]
        ).convert("L")
    ) == 255

    assert (source_width, source_height, replayed["band_px"]) == (1055, 1491, 63)
    assert replayed["side_sample_pixels"]["bottom"] == 10_806
    assert replayed["side_coverage_fraction"]["bottom"] == pytest.approx(
        0.1625818099751749,
        abs=1e-15,
    )
    assert replayed["exterior_sample_pixels"] == 92_830
    assert np.array_equal(replayed_mask, decoded_mask)


def test_export_gate_rejects_plus_one_coordinated_v2_bottom_counter() -> None:
    """Catches accepting a plausible v2 +1 count/ratio under a broad tolerance."""
    manifest_path = (
        _optional_fixture_root("FATE_TAROT_BATCH_CARDS_DIR")
        / "0EFD5AE0-6E88-4D6D-9A5C-DC1643EC172E"
        / "manifest.json"
    )
    if not manifest_path.is_file():
        pytest.fail("configured 0EFD legacy regression artifact is unavailable")
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _increment_bottom_count_and_ratio(manifest)

    with pytest.raises(ValueError, match="v2 sampling replay side_sample_pixels"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_declared_minimum_below_skill_floor(tmp_path: Path) -> None:
    """Catches lowering the manifest minimum while leaving a high-density raster."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _tamper_minimum_effective_ppi_below_skill_floor(manifest)

    with pytest.raises(ValueError, match="minimum effective PPI.*300|300.*minimum"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_recomputes_geometry_and_sampling_quality_metrics(
    tmp_path: Path,
) -> None:
    """Catches forged passed strings paired with impossible coverage/texture counters."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _forge_passed_sampling_and_geometry_metrics(manifest)

    with pytest.raises(ValueError, match="recomputed|geometry|sampling"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_plausible_but_wrong_geometry_counters(
    tmp_path: Path,
) -> None:
    """Catches threshold-passing geometry ratios that contradict decoded pixels."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _forge_plausible_but_wrong_geometry_metrics(manifest)

    with pytest.raises(ValueError, match="geometry.*counter|geometry.*recomputed"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_plausible_but_wrong_sampling_counters(
    tmp_path: Path,
) -> None:
    """Catches threshold-passing sampling ratios that contradict decoded pixels."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _forge_plausible_but_wrong_sampling_metrics(manifest)

    with pytest.raises(ValueError, match="sampling.*counter|sampling.*recomputed"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_coordinated_plausible_sampling_count_forgery(
    tmp_path: Path,
) -> None:
    """Catches synchronized count+ratio edits that preserve internal arithmetic."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _forge_coordinated_plausible_sampling_counts_and_ratios(manifest)

    with pytest.raises(ValueError, match="sampling.*counter|sampling.*recomputed"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_preparer_emits_explicit_v3_sampling_provenance_contract(
    tmp_path: Path,
) -> None:
    """Catches new manifests silently remaining on the unauditable v2 schema."""
    source = tmp_path / "card.png"
    _make_source(source)

    prepared = prepare_equal_frame(
        source,
        tmp_path / "prepared",
        minimum_effective_ppi=300,
    )

    assert prepared["pipeline_version"] == 3
    sampling = prepared["background_sampling"]
    assert all(field in sampling for field in _V3_SAMPLING_PROVENANCE_FIELDS)


def test_export_gate_accepts_a_fresh_exact_v3_manifest_without_export(
    tmp_path: Path,
) -> None:
    """Catches drift between the current preparer, sidecar, and decoded exporter replay."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    media_path, trim_width_mm, trim_height_mm, bleed = (
        _validate_subject_first_manifest(manifest, manifest_path.parent)
    )

    assert media_path == Path(manifest["artifacts"]["media_rgb"]["path"]).resolve()
    assert (trim_width_mm, trim_height_mm) == (80.0, 120.0)
    assert bleed == {"left": 5.0, "right": 5.0, "top": 5.0, "bottom": 5.0}


def test_export_gate_rejects_v3_with_all_sampling_provenance_deleted(
    tmp_path: Path,
) -> None:
    """Catches deleting every v3 provenance field to trigger an implicit legacy path."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _delete_all_sampling_provenance(manifest)

    with pytest.raises(ValueError, match="version 3|v3|sampling provenance"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_v3_with_partially_deleted_sampling_provenance(
    tmp_path: Path,
) -> None:
    """Catches a current manifest with one provenance field silently omitted."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["background_sampling"].pop("side_interpolation_exceptions")

    with pytest.raises(ValueError, match="sampling provenance"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_absurd_v3_sampling_provenance_contents_and_types(
    tmp_path: Path,
) -> None:
    """Catches presence-only provenance validation that accepts impossible bodies."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _forge_absurd_sampling_provenance(manifest)

    with pytest.raises(ValueError, match="sampling provenance|interpolat"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["provenance"][
            "side_interpolated_coordinate_count"
        ].__setitem__("bottom", 999_999),
        lambda payload: payload["provenance"].__setitem__(
            "side_interpolation_exceptions",
            "forged",
        ),
    ],
)
def test_export_gate_rejects_coordinated_v3_provenance_artifact_forgery(
    tmp_path: Path,
    mutation,
) -> None:
    """Catches rehashing an artifact whose contents/types contradict the manifest."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    provenance_path = Path(manifest["artifacts"]["sampling_provenance"]["path"])
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_sampling_provenance_artifact(manifest, payload)

    with pytest.raises(ValueError, match="sampling provenance artifact"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_matching_absurd_manifest_and_provenance_artifact(
    tmp_path: Path,
) -> None:
    """Catches rehashing the same forged provenance into both v3 evidence copies."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _forge_absurd_sampling_provenance(manifest)
    provenance_path = Path(manifest["artifacts"]["sampling_provenance"]["path"])
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["provenance"]["side_interpolated_coordinate_count"] = copy.deepcopy(
        manifest["background_sampling"]["side_interpolated_coordinate_count"]
    )
    payload["provenance"]["side_interpolation_maximum_gap_px"] = copy.deepcopy(
        manifest["background_sampling"]["side_interpolation_maximum_gap_px"]
    )
    payload["provenance"][
        "side_interpolation_maximum_allowed_gap_px"
    ] = copy.deepcopy(
        manifest["background_sampling"][
            "side_interpolation_maximum_allowed_gap_px"
        ]
    )
    payload["provenance"]["side_interpolation_exceptions"] = "forged"
    _write_sampling_provenance_artifact(manifest, payload)

    with pytest.raises(ValueError, match="sampling provenance|interpolat"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_current_manifest_relabelled_as_v2(
    tmp_path: Path,
) -> None:
    """Catches changing only the explicit version to bypass the current exact contract."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["pipeline_version"] = 2

    with pytest.raises(ValueError, match="version 2|v2|sampling provenance"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_coordinated_count_forgery_after_provenance_deletion(
    tmp_path: Path,
) -> None:
    """Catches a +1 count/ratio forgery combined with deleting every provenance field."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _increment_bottom_count_and_ratio(manifest)
    _delete_all_sampling_provenance(manifest)

    with pytest.raises(ValueError, match="version 3|v3|sampling provenance|sampling.*counter"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_v3_workdir_downgraded_to_v2_manifest_shape(
    tmp_path: Path,
) -> None:
    """Catches stripping all v3 JSON markers while its fixed provenance sibling remains."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    _strip_every_v3_manifest_marker(manifest)

    with pytest.raises(ValueError, match="version 2|v2|sampling provenance|sibling"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


@pytest.mark.parametrize("schema_mutation", ["missing", "unexpected"])
def test_export_gate_enforces_the_rigid_v2_sampling_schema(
    tmp_path: Path,
    schema_mutation: str,
) -> None:
    """Catches a legacy branch that accepts arbitrary field deletion or injection."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    provenance_path = Path(manifest["artifacts"]["sampling_provenance"]["path"])
    _strip_every_v3_manifest_marker(manifest)
    provenance_path.unlink()
    if schema_mutation == "missing":
        manifest["background_sampling"].pop("side_model_residual_p90")
    else:
        manifest["background_sampling"]["unexpected_legacy_field"] = 0

    with pytest.raises(ValueError, match="version 2 sampling schema mismatch"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_manifest_contract_evidence_names_the_explicit_validation_mode(
    tmp_path: Path,
) -> None:
    """Catches exporter reports that conceal whether v2 or v3 validation ran."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)

    assert _manifest_contract_evidence(prepared) == {
        "pipeline_version": 3,
        "sampling_provenance_schema_version": 1,
        "validation_mode": "v3-exact-decoded-provenance",
    }

    legacy = copy.deepcopy(prepared)
    legacy["pipeline_version"] = 2
    legacy.pop("sampling_provenance_schema_version")
    assert _manifest_contract_evidence(legacy) == {
        "pipeline_version": 2,
        "sampling_provenance_schema_version": None,
        "validation_mode": "v2-exact-deterministic-replay",
    }


def test_export_gate_rejects_v2_with_null_v3_schema_marker() -> None:
    """Catches treating an explicitly present null v3-only key as legacy absence."""
    manifest = {
        "pipeline_version": 2,
        "sampling_provenance_schema_version": None,
    }

    with pytest.raises(ValueError, match="version 2|provenance schema"):
        _manifest_contract_evidence(manifest)


@pytest.mark.parametrize("pipeline_version", [None, "3", True, 3.0, 1, 4])
def test_export_gate_rejects_missing_mistyped_or_unknown_pipeline_version(
    tmp_path: Path,
    pipeline_version,
) -> None:
    """Catches any implicit/default version path or JSON type substitution."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = copy.deepcopy(json.loads(manifest_path.read_text(encoding="utf-8")))
    if pipeline_version is None:
        manifest.pop("pipeline_version")
    else:
        manifest["pipeline_version"] = pipeline_version

    with pytest.raises(ValueError, match="pipeline_version|pipeline version"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)


def test_export_gate_rejects_sampling_from_enclosed_non_exterior_hole(
    tmp_path: Path,
) -> None:
    """Catches a zero-overlap sample that is not boundary-connected exterior."""
    source = tmp_path / "card.png"
    _make_source(source)
    prepared = prepare_equal_frame(source, tmp_path / "prepared", minimum_effective_ppi=300)
    manifest_path = Path(prepared["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _tamper_sampling_into_enclosed_non_subject_hole(manifest)

    with pytest.raises(ValueError, match="boundary-connected exterior|non-exterior"):
        _validate_subject_first_manifest(manifest, manifest_path.parent)
