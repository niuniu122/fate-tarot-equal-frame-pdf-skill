---
name: fate-tarot-equal-frame-pdf
description: Use when bordered raster tarot, oracle, or game-card artwork has small canvas, frame-position, or exterior-color differences and must become fixed-size print PDFs with preserved artwork, exact trim, and fixed bleed. Do not use for redesign, unrelated templates, or ordinary image-to-PDF conversion.
---

# Fate Tarot Subject-First Print PDF

Process every card independently. The protected subject is the detected outer
design frame, everything enclosed by that frame, and ornaments physically
connected to it. Pixels outside that boundary are source material for this
card's bleed, never part of the protected subject.

## Required contract

- Detect one unique, closed outer design frame before calculating the page.
  Preserve rounded corners, parallel frame lines, and attached top/bottom
  ornaments. Interior pixels remain subject even when their color matches the
  exterior canvas.
- An edge-hugging dark component may be redirected only when opposite long-edge
  gradient bands form one mirrored, thickness-compatible pair. Unmatched long
  artwork lines are rejected from the pair, not selected. A missing short edge
  may be reflected from its unique observed opposite edge, and that inference
  must be explicit in the report.
- A visually reviewed card may use `--validated-frame-override` with explicit
  top/right/bottom/left pixel anchors and one or four corner radii. The override
  does not bypass safety: all four anchors must align to continuous normal
  gradient bands, equivalent probe-near requests must resolve to the same
  outermost design rail, and the actual geometry and mask must snap to those
  stable source anchors. By default, every radius must be normalized from one
  source corner arc that follows the signed outer gradients and both locked
  side tangents.
  Estimate each tangent only inside its locked rail band, orientation-correct
  the transition, pair the two side estimates under the radius-delta gate, and
  test only that pair's narrow radius window against one source-connected outer
  arc. Evaluate every audited pair before rejecting, but never fall back to a
  global radius search; a larger nested decorative arc is not valid corner
  evidence. When a full-size review establishes the four source corner tangents
  but a source edge or connected ornament makes automatic radius normalization
  unsafe, the override may declare
  `corner_radius_validation: operator-reviewed-source-corner-tangents`. Record
  the reviewed policy and exact radii, and still rerun the shared
  frame-structure gate. Record candidate bands, tangent evidence, selected
  policy, and requested versus normalized anchors and radii in the manifest.
- Treat every locked top/right/bottom/left anchor as the inclusive center of a
  real source rail pixel. Its continuous cell boundary is therefore
  `left-0.5/right+0.5/top-0.5/bottom+0.5`, and its source crop is the half-open
  box `[left, top, right+1, bottom+1]`. Apply this convention identically to
  automatic paired-gradient recovery and validated overrides; never shift the
  right/bottom rails inward or absorb an adjacent exterior row/column.
- Preserve attached ornaments by source-versus-modeled-exterior difference and
  physical connectivity to the confirmed automatic or validated frame,
  regardless of whether the ornament is dark, bright gold, or colored. Exclude
  disconnected exterior marks and broad paper/matte edges; never redraw a
  missing ornament. Intersect the hole-filled source-connected core with the
  final thresholded rounded geometry inside the inclusive rails so the rail box
  cannot absorb corner paper or shadow. Outside those rails, merge only audited
  source pixels from a narrow top/bottom rail-neck component that contains both
  weak and strong local-background evidence. A one-pixel gap may prove a unique
  outward continuation but never creates subject pixels; reject ambiguous
  forks, broad matte runs, and components without a strong source seed. Read
  [the detailed workflow](references/workflow.md) for the exact thresholds and
  topology gates.
- Extract and approve an independent subject RGBA and binary subject mask before
  layout. Crop and integer-translate only; never crop the detected subject,
  rotate, stretch, resample, sharpen, recolor, or redraw it.
- Build the bleed model only from boundary-connected pixels outside the subject
  mask in the same source image. Require sampling/subject overlap to equal zero.
  Sample the four sides and four corners independently; never hardcode a color
  or borrow another card's sampling result. A locally missing row/column may be
  interpolated only from bracketing valid samples on that same side and only
  within the audited maximum local-gap gate.
- When full-size review confirms that the intended bleed is one continuous
  exterior color, `--reviewed-flat-exterior` may use the per-channel median of
  this card's complete boundary-connected exterior sample set. It must still
  sample all four sides and corners, retain zero subject overlap, record the
  exact flat RGB and sampling mask, and pass independent exporter replay. Never
  infer this mode from another card.
- Keep subject RGBA, binary mask, exterior sampling mask, bleed-background RGB,
  composed media RGB, and the version-bound sampling-provenance JSON as separate
  artifacts with SHA-256 hashes. The provenance sidecar binds pipeline/schema
  version, source hash, sampling-mask hash, interpolation evidence, and sampling
  metrics. Compose only after the extraction and sampling gates pass.
- Default to an 80 x 120 mm `TrimBox` plus 5 mm on every side: a 90 x 130 mm
  `MediaBox`/`BleedBox`. Other dimensions must come from explicit user values.
  Center the complete subject proportionally inside the trim. Remaining trim
  area uses the same current-card exterior model.
- The public API and CLI enforce a minimum job floor of 300 effective PPI. If
  filling the trim with the unchanged source pixels would fall below that floor,
  use a 300-PPI-or-higher page raster, keep the subject pixels one-to-one, and
  accept additional current-card background inside the trim. Never upscale or
  resample the subject to satisfy density.

## Workflow

The scripts require Python 3.10 or newer and the packages in
`requirements.txt`.

1. Use the pinned conservative PDF builder and verifier under
   `vendor/card-artwork-print-pdf/`. Do not replace them with project-local code.
   An explicitly audited alternative may be supplied with `--print-skill-root`.
2. Create a per-card directory under `tmp/pdfs/`, then run:

   ```text
   python <this-skill>/scripts/subject_first_pipeline.py --source <card.png> --output-dir <work-dir> --trim-width-mm 80 --trim-height-mm 120 --bleed-left-mm 5 --bleed-right-mm 5 --bleed-top-mm 5 --bleed-bottom-mm 5 --minimum-effective-ppi 300 --report <work-dir>/prepare-report.json
   ```

   When automatic detection safely stops on a photographed paper edge or matte,
   first inspect and measure the true design frame, then pass either a JSON
   object or a path to a JSON file:

   ```text
   --validated-frame-override '{"left":98,"right":926,"top":86,"bottom":1439,"radii":{"top_left":40,"top_right":40,"bottom_left":40,"bottom_right":40}}'
   ```

   Do not create an override from an unreviewed detector guess.

   If the same review confirms one continuous exterior color, add
   `--reviewed-flat-exterior`. If automatic corner-radius normalization is not
   safe but the source corner tangents were reviewed, add
   `"corner_radius_validation":"operator-reviewed-source-corner-tangents"`
   to the override object.

3. Before PDF export, inspect the source, detection overlay, binary mask,
   subject RGBA on light and dark backgrounds, exterior-sampling overlay,
   bleed-only background, layout guide, and composed media. Read
   [the detailed workflow](references/workflow.md) for the acceptance gates.
4. Require all pipeline stages to be complete; artifact hashes to match;
   the shared four-side/rounded-or-parallel-frame geometry gate to report
   `passed: true`; the sampling-quality gate to report `status: passed`;
   `region_policy == exterior-only`; sampling overlap to be `0`; the source SHA
   to match extraction and sampling; `resampled == false`; all opaque subject
   pixels to remain exact; the requested trim/media/bleed geometry to be exact;
   horizontal/vertical physical scale to match; decoded raster dimensions and
   `media_mm` to recompute the declared effective PPI at or above the declared
   minimum; and every side/corner sampling coverage and texture-pollution gate
   to pass. The export gate reopens the source and artifacts, rebuilds the
   boundary-connected exterior in source space, proves the sampling mask is its
   exact permitted subset with zero subject overlap, and independently
   recomputes four-side/four-corner coverage, texture/bad-run, frame geometry,
   density, placement, and composite relationships instead of trusting manifest
   counters or approval strings. A declared minimum below the hard 300-PPI Skill
   floor is rejected even when the raster itself happens to exceed it. Require
   an integer `pipeline_version`. Version 3 requires
   `sampling_provenance_schema_version == 1`, every typed interpolation field,
   and the hashed fixed sibling `sampling-provenance.json`; all must match the
   independently decoded evidence exactly. Version 2 is accepted only under its
   rigid historical schema, with all v3 fields/sidecar absent and its original
   whole-row/whole-column fallback sampler replayed exactly. Never select a
   validation mode from optional-field presence or use a broad legacy tolerance.
   For tamper-evident version history beyond these local semantic checks,
   preserve the manifest hash in an external signed or otherwise trusted record.
5. Export exactly once from the approved manifest. The exporter snapshots the
   hash-approved media, builds and verifies a temporary candidate, requires a
   nonblank independent render, and only then atomically replaces the requested
   output:

   ```text
   python <this-skill>/scripts/export_from_manifest.py --manifest <work-dir>/manifest.json --output <output.pdf> --report <work-dir>/export-report.json
   ```

6. Verify the PDF boxes, direct RGB image embedding, embedded-pixel equality,
   and a nonblank independent render. Inspect the full render, four edges, four
   corners, and top/bottom ornaments for seams, halos, white gaps, clipping,
   color steps, repeated patches, or distortion.

## Stop conditions

Stop without layout or export when the outer frame is missing, open, clipped,
rotated, or ambiguous; more than one boundary is plausible; an ornament cannot
be classified safely; the exterior sampling area is insufficient or polluted;
sampling overlaps the subject; effective PPI is below the job minimum; the
complete subject cannot fit the trim without cropping/resampling; or visual QA
fails. Never resolve these cases by guessing, using another card, or forcing
four equal visible margins when the aspect ratios conflict.

## Delivery

Deliver the final PDF and a compact report containing trim/media/bleed sizes,
subject placement and scale, source and artifact hashes, exterior-only sampling
evidence, the explicit manifest validation mode, opaque-pixel equality, PDF
structure, renderer evidence, and QA paths.
Say "structure-compatible with Illustrator" unless Illustrator actually opened
the exact final PDF and displayed it correctly.
