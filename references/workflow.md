# Subject-first fixed-bleed workflow

Use this reference when reviewing a frame detection, approving a subject,
checking exterior sampling, calculating the fixed print page, or deciding
whether a card is safe to export.

## Five ordered stages

The order is a data contract, not merely a presentation preference:

1. **Detect subject frame.** Locate one closed outer design frame around the card
   center. Use geometric continuity and enclosure, not a fixed color, crop, or
   mask copied from another card.
2. **Extract subject.** Fill the closed boundary so every enclosed pixel remains
   subject, then include only ornaments physically connected to the frame.
   Produce a binary source-space mask and a cropped RGBA. Do not let an interior
   region disappear merely because it matches the exterior color.
3. **Sample exterior.** Define the sample population as pixels outside the
   subject mask that remain connected to the source canvas boundary. Four side
   and four corner samples must come only from that population. Record the
   sample count and prove zero overlap with the subject.
4. **Render fixed bleed and place subject.** Render one continuous low-frequency
   current-card background across the complete media raster. Place the approved
   RGBA by integer translation inside the trim, centered to within one pixel.
   Do not reopen the source to redefine the subject during this stage.
5. **Export and verify PDF.** Export the approved media raster with exact PDF
   boxes, then independently render and inspect it.

Each stage must be marked complete in the manifest. The subject mask, subject
RGBA, sampling mask, bleed-only background, composed media image, and current
version's sampling-provenance JSON are separate artifacts with SHA-256 hashes
tied to the current source SHA-256.

Before export, reopen those files and recompute their semantic relationships:
binary mask validity, RGBA alpha support, source-pixel equality, source-space
sampling overlap, media placement, and the final composite. Matching hashes and
self-reported counters alone are not approval evidence.

## Frame acceptance

A valid automatic candidate must:

- enclose the image center;
- provide continuous top, right, bottom, and left frame evidence;
- have a plausible card aspect ratio and rounded-corner geometry;
- preserve outer and parallel inner frame lines;
- preserve attached center ornaments without absorbing unrelated exterior
  canvas; and
- be unique at the detector's confidence threshold.

When a connected dark component reaches a canvas edge, its extrema are not
trusted as frame anchors. The recovery path must observe both long sides as one
approximately mirrored, thickness-compatible gradient pair. It may discard a
long line only because that line has no valid opposite partner. Multiple
spatially distinct pairs remain ambiguous. If exactly one short side is absent,
the detector may reflect the unique observed opposite short band; the manifest
  must list that side in `inferred_sides`. This recovery never bypasses the unique
  closed-candidate gate or the final four-side/rounded-or-parallel structure gate.
Every selected rail anchor is an inclusive source-pixel center. Convert its
semantic boundary symmetrically to `left-0.5/right+0.5/top-0.5/bottom+0.5`, and
convert the source crop to `[left, top, right+1, bottom+1]`. The automatic
paired-gradient and validated-override paths use the same convention; no path
may move the right/bottom rails inward or include the adjacent exterior cells.
Its corner radii are normalized from the same source's oriented corner-gradient
arc and both locked side tangents instead of inheriting provisional component
radii. Within each already locked gradient band, the detector records aligned
transition runs, orientation-corrects their radius estimates, and audits any
short transition against later long straight support on that same rail. It
pairs both side estimates under the radius-delta gate, then tests only the
pair-specific narrow radius window. Every audited pair may be considered, but
there is no global radius search. The selected arc must join both locked
tangents through one source-versus-background connected component, while each
direct gradient point keeps the signed polarity of its adjacent outer rail.
This prevents a stronger or larger nested decorative arc from replacing the
design corner. If an outer side is too occluded to provide locked straight and
tangent evidence, automatic detection stops; original artwork status does not
authorize inventing that missing geometry.
A narrow top/bottom center ornament that is physically seeded from the confirmed
frame may cross one broader first exterior row only inside its audited center
envelope and only when the next outward row immediately narrows with direct
overlap. A continuing broad paper/matte run cannot seed an ornament.

For a visually inspected source where automatic detection selects a physical
paper edge, matte, shadow, or other wrong boundary, a reviewer may supply a
`validated_frame_override` containing `left`, `right`, `top`, `bottom`, and
`radii` in source pixels. `radii` may be one number or the four named corner
radii. The pipeline must reject malformed, out-of-bounds, implausible, or
off-edge geometry. It must group continuous signed-gradient responses into
bands across an expanded search, select the outermost eligible design rail, and
produce the same normalized rail for equivalent probe-near requests. It snaps
the actual mask geometry to those four stable anchors and retains candidate
bands plus requested and normalized anchors. It must also normalize all four
radii from one outer corner arc tied to both locked rail tangents and signed
polarities rather than silently accepting caller-selected radii or a nested
decorative arc. If full-size operator review establishes the four source corner
tangents but a source edge or connected ornament makes automatic normalization
unsafe, the override may declare
`corner_radius_validation: operator-reviewed-source-corner-tangents`. In that
mode the supplied radii are the reviewed source-tangent measurements; record
them exactly, mark automatic radius normalization as unused, and keep rail
snapping and frame-structure checks mandatory. It then renders the rounded mask, reruns frame-structure
validation, and records requested versus normalized geometry in
`frame_detection.validated_frame_override`. An override is auditable reviewed
input, not permission to force a failing image through.

Connected-ornament evidence is color-independent on both automatic and
validated paths: compare the source against a modeled exterior/background,
retain only difference components that physically connect to multiple confirmed
frame-side probes, and then follow their narrow top/bottom center envelopes.
This preserves dark, bright-gold, and colored ornaments while excluding
disconnected decoys and broad physical paper edges.

Within each top/bottom center envelope (nominally 24% of the locked frame width,
computed as `max(12 px, round(0.24 * frame width))`), use two independent,
current-card local tests. The first is the distant same-row background model:
take robust medians from the left 6%-34% and right 66%-94% portions of the
locked frame width and
interpolate those medians across the center envelope. Let `d` be the
per-pixel L-infinity RGB residual from that model, `m` the median residual in
both distant flanks, and `sigma = 1.4826 * MAD`. Use these literal gates:

```text
Tweak   = clip(max(6, Q70(flank d)+3, m+sigma), 6, 24)
Tstrong = clip(max(Tweak+4, Q99(flank d)+2, m+4*sigma), 10, 64)
```

The second test prevents a gently varying matte or haze from passing merely
because its center differs from the distant medians. Build one adjacent palette
from the immediately neighboring left and right flanks, each 12% of the locked
frame width (four pixels minimum), using only the current side's exterior rows
within plus or minus three rows of the candidate. For every candidate pixel,
compute its nearest L-infinity RGB distance to any color in that local palette.
The ordinary weak and strong supports both require that nearest distance to be
strictly greater than `3`; this is a per-pixel novelty test, not a card-color
key. The more selective palette-distinct support requires both `d > Tweak` and
nearest palette distance strictly greater than `5`.

The retained ornament is an 8-connected weak-support component seeded from the
rail-adjacent exterior row, containing at least one strong-support source pixel,
at least three source rows, four source pixels, and four pixels of outward span.
Both residual support tests use strict comparison (`d > Tweak` and
`d > Tstrong`). A rail-connected chain qualifies for outward retention only
when all of these gates pass:

- the median source-support width across the chain's active rows is no greater
  than `max(4 px, round(18% of locked frame width))`; and
- at least one connected palette-distinct component spans three or more
  outward rows; and
- for every palette-distinct component with that qualifying span, the maximum
  chain source-support width on its evidence rows is no greater than
  `max(4 px, round(20% of locked frame width))`.

A single row or an extreme novelty value never promotes the main chain by
itself. Record and reject a chain that exceeds the 18% active-row median-width
gate as broad matte/haze. Also reject the complete chain when any
three-row-or-longer palette-evidence component exceeds the separate 20%
maximum-width gate; the 20% evidence-row limit is intentionally distinct from
the 18% whole-chain median gate. Once a chain qualifies, trim only the weak,
low-novelty tail farther outward than the outermost qualifying evidence; retain
the source-supported portion from that limit back toward the confirmed rail.
Do not symmetrically erode the component or trim its rail-facing portion. After
that outward-tail trim, apply the same
`max(4 px, round(20% of locked frame width))` maximum-width gate to every row
that remains in `trimmed_chain_support`. If any retained row exceeds it, reject
the complete chain with
`reason == "broad-chain-width-after-outward-tail-trim"`; palette-evidence rows
cannot hide a broader retained matte closer to the rail.

There is no all-direction dilation. A single missing row may connect two
components only in the outward-normal direction, with horizontal alignment
within one pixel and exactly one eligible continuation. The missing row is
connectivity evidence only and remains transparent. Reject competing
continuations as an ambiguous fork.

A narrow low-contrast `base_weak` branch is the only exception to the ordinary
per-pixel novelty gate. Preserve it only when the original median-residual
component runs continuously from the source edge to the rail, its maximum row
width is no greater than 3.5% of the locked frame width (four pixels minimum),
its maximum nearest-palette L-infinity novelty is strictly greater than `64`,
it contains palette-distinct source pixels on at least two rows, and it contains
at least four strong-support source pixels distributed across at least three
source rows. From the source edge inward, the unsupported outward tail before
the first strong row must not exceed 60% of the branch's row span; measure the
bottom side symmetrically from the source bottom edge back to its last strong
row. The integer cap is
`max(2 rows, floor(0.60 * branch row span))`. These requirements apply whether
the branch overlaps an already qualified chain or is restored independently;
chain-level evidence cannot substitute for branch-local evidence. Broad matte
or haze components fail this narrowness/novelty/distributed-support gate.

After component selection, a short horizontal opening may recover only pixels
that were already present in the original source-supported foreground. The gap
must lie on one row, be bracketed on both sides by already selected support, and
be no wider than `max(1 px, round(0.6% of locked frame width))`. Never fill the
whole gap, extend an endpoint, synthesize a color, or convert palette similarity
into alpha. Fill holes separately inside each already selected exterior
ornament component; never use the card core or rail to close an open arch, and
never fill a hole touching the envelope or source boundary.

The extraction mask is binary, not residual-weighted alpha: retained weak-source
pixels, explicitly restored original source pixels, and valid per-component
filled holes become `255`; rejected pixels and connectivity-only bridge rows
remain `0`. The rounded guide is supersampled only to locate the accepted
boundary. After the `alpha >= 8` core intersection, the approved core is
likewise written as binary `255`. Never convert residual or palette distance
into semitransparency, invent antialiased ornament pixels absent from the
source-supported mask, or use a hardcoded gold, silver, green, blue, or other
color key.

The manifest's
`frame_detection.source_connected_ornament_evidence.local_center_ornament_refinement`
must make this decision replayable. Its top-level evidence records:

```text
policy, color_policy
center_zone_px
maximum_center_ornament_width_px
maximum_center_ornament_width_fraction == 0.24
left_flank_px, right_flank_px
flank_policy == "same-row-left-and-right-away-from-center"
left_palette_flank_px, right_palette_flank_px
palette_flank_policy ==
  "immediately-adjacent-left-and-right-within-plus-or-minus-3-rows"
flank_residual_quantile, residual_margin_linf_px
minimum_threshold_linf_px, maximum_threshold_linf_px
threshold_policy
component_policy, connectivity_policy
sides.top, sides.bottom
```

For each side, record all row thresholds and palette evidence plus the topology
and removal/restoration accounting:

```text
rows_evaluated
row_threshold_linf_px, weak_threshold_linf_px, strong_threshold_linf_px
background_novelty_threshold_linf_px
background_novelty_distance_p99_linf_px
retained_source_supported_pixels, retained_before_connectivity_pixels
removed_broad_component_pixels, removed_disconnected_texture_pixels
removed_low_novelty_outer_tail_pixels
retained_outer_limits_px
restored_short_horizontal_gap_pixels, maximum_restored_horizontal_gap_px
base_weak_component_count
preserved_narrow_low_contrast_branch_count
preserved_narrow_low_contrast_branches
rejected_broad_chain_count, rejected_broad_chains
connectivity_component_count, rail_connected_component_count
connectivity_dilation_px == 0
bridged_one_pixel_gap_count, bridged_one_pixel_gaps
ambiguous_gap_fork_count, ambiguous_gap_forks
maximum_bridged_gap_px, gap_bridge_direction
minimum_rail_neck_source_rows
minimum_rail_neck_outward_span_px
minimum_rail_neck_source_pixels
palette_distinct_threshold_linf_px
minimum_palette_distinct_outward_span_px
maximum_median_chain_row_width_fraction
maximum_palette_evidence_chain_row_width_fraction
narrow_low_contrast_branch_maximum_width_fraction
narrow_low_contrast_branch_extreme_novelty_linf_px
narrow_low_contrast_branch_minimum_palette_distinct_rows
narrow_low_contrast_branch_minimum_strong_source_rows
narrow_low_contrast_branch_maximum_unsupported_outward_tail_fraction
```

Each preserved low-contrast branch record includes its component label, source
row range, measured and allowed maximum row width, and maximum palette novelty;
every preserved branch also records `palette_distinct_source_row_count` and
`strong_source_pixels`, `strong_source_row_count`, `branch_row_span_px`,
`unsupported_outward_tail_rows_px`, and
`maximum_unsupported_outward_tail_rows_px`, while an independent restoration
additionally records `restoration_path`. A whole-chain median-width rejection
records the chain labels, measured `median_active_row_width_px`, and
`maximum_allowed_median_row_width_px`. A palette-evidence-row rejection records
`reason == "broad-chain-width-on-qualifying-palette-rows"` plus each rejected
entry under `palette_components` with
`palette_component_label`, `palette_source_row_range_px`,
`maximum_chain_row_width_on_palette_rows_px`,
`maximum_allowed_chain_row_width_px`, and
`maximum_palette_novelty_linf_px`. The side-level width fractions are
`maximum_median_chain_row_width_fraction == 0.18` and
`maximum_palette_evidence_chain_row_width_fraction == 0.20`; its narrow-branch
limits also record three minimum strong rows and a `0.60` maximum unsupported
outward-tail fraction. A post-trim width rejection records
`reason == "broad-chain-width-after-outward-tail-trim"`,
`maximum_retained_chain_row_width_px`,
`maximum_allowed_chain_row_width_px`, and `retained_outer_limit_px` with the
chain labels. Missing, contradictory, non-finite, or visually
implausible evidence is a fail-closed condition: stop before layout or export
instead of relaxing a threshold, choosing a card-specific color, or absorbing
the exterior matte.

To obtain the enclosed subject core, hole-fill the confirmed source-connected
support inside the inclusive rail box and intersect it with the final rounded
geometry at the same accepted alpha threshold (`>= 8`) used for output. This
extra intersection prevents rectangular rail bounds or sub-threshold BOX fringe
from absorbing photographed paper, shadow, or corner matte. Outside the rails,
merge only the selected source-supported ornament components described above.

Review the detection overlay and binary mask at full size and high zoom. Reject
a missing pale line, cut corner, lost ornament, exterior matte retained as
subject, open contour, or competing frame candidate. A tolerance override is
diagnostic only; never use it merely to force an uncertain card through.

## Exterior-only sampling

The exterior mask is the boundary-connected complement of the approved binary
subject mask. Sampling functions must consume this mask; merely reporting an
`exterior-only` policy string is insufficient.

Fit the top, right, bottom, and left trends independently and take each corner
from its own valid exterior pixels. Reconcile the four side trends to the four
corner samples before rendering one continuous surface. The manifest must show:

```text
background_sampling.source_sha256 == source_sha256
background_sampling.region_policy == "exterior-only"
background_sampling.subject_overlap_pixels == 0
background_sampling.exterior_sample_pixels > 0
```

After full-size review confirms that the intended bleed is one continuous
color, `reviewed-flat-exterior-median-v1` may render one flat RGB value instead
of four side trends. Compute it as the per-channel median of this card's
complete boundary-connected exterior sample set. Sampling must still contain
nonzero evidence from all four sides and four corners, preserve the exact
exterior-only mask with zero subject overlap, and record
`flat_background_rgb`, `flat_color_statistic`, and the reviewed-flat quality
gate. The exporter independently rebuilds the exterior population, recomputes
the median, and requires every bleed-background pixel to equal that RGB value.

If a whole image row or column contains no exterior-only pixel because a local
approved ornament spans that coordinate, do not cross the subject to borrow a
pixel from another side. Leave that coordinate unselected in the sampling mask
and linearly interpolate its color only from valid, bracketing coordinates on
the same side. Record the per-side interpolated-coordinate count, maximum
missing run, and maximum allowed gap. The conservative base gate permits at
most 10% of that side's coordinate length (with a 12-pixel minimum allowance).
The sole larger-gap exception is one top/bottom run wholly inside the audited
24%-wide center-ornament zone, directly bracketed by valid same-side samples,
and overlapping actual subject contact pixels on that corresponding source
edge. Record its reason, run, center zone, direct brackets, and contact overlap
under `side_interpolation_exceptions`. All other larger, unbracketed, or
under-supported gaps remain a hard failure. Interpolated coordinates never
count as sampled pixels.

New preparations use integer `pipeline_version: 3` and
`sampling_provenance_schema_version: 1`. Their fixed sibling
`sampling-provenance.json` binds those versions, the source SHA-256, sampling
mask SHA-256, every same-side interpolation field, and the sampling metrics.
The exporter requires the manifest fields and sidecar bodies to have the exact
decoded/recomputed values and JSON types. Deleting all fields is not a legacy
signal. Relabeling a v3 work directory as v2 is rejected while that fixed
sidecar remains.

Historical integer `pipeline_version: 2` manifests have one rigid schema and no
v3 provenance fields or sidecar. Validate them by replaying the original
sampler, not by tolerances: for each side coordinate first use exterior pixels
in that side band; only when that coordinate has none, take at most `band`
boundary-connected exterior pixels from the entire same column or row, using
the first pixels for top/left and last pixels for bottom/right. Keep four
per-side attribution masks, add direct exterior corner squares to their union,
and recompute counts, coverage, medians, texture/bad-run, and model residuals
exactly. The resulting union must equal the decoded sampling-mask artifact.
This local unsigned contract cannot prove historical version identity after an
attacker deletes the fixed sidecar and every v3-only marker; in that case exact
v2 replay still enforces v2 pixel semantics. Jobs that require tamper-evident
version history must preserve an external signed manifest hash or equivalent
trusted registry record.

Reject the card when clean exterior pixels are too sparse, when detailed artwork
pollutes the sample population, or when the rendered background develops a
visible color step, repeated patch, halo, or corner seam.

## Fixed physical geometry

For the default job:

```text
TrimBox        = 80 x 120 mm
bleed          = 5 mm on top/right/bottom/left
MediaBox       = 90 x 130 mm
BleedBox       = MediaBox
TrimBox origin = (5 mm, 5 mm)
```

Choose an integer raster grid that fits the complete extracted subject inside
the trim while meeting the minimum effective PPI and keeping the requested page
aspect exact. The API and both CLIs reject a requested minimum below the Skill's
hard 300 PPI floor. When printing the unchanged subject at full trim would fall
below 300 PPI, increase only the page raster density, keep the subject pixel grid
one-to-one at the center, and leave additional current-card background inside
the trim. Do not resize the extracted pixel grid. Opposite padding values may
differ by at most one pixel. The physical placement must report equal horizontal
and vertical scale and `cropped: false`.

Before export, recompute effective PPI from the decoded `media_rgb` pixel
dimensions and top-level `media_mm`; verify it matches the declared
`effective_ppi` and meets `minimum_effective_ppi`. Also require
`frame_detection.geometry_evidence.passed == true` and
`background_sampling.sampling_quality_gate.status == "passed"`. These are
semantic gates, not trusted approval strings. Enforce 300 PPI as an independent
Skill floor even if the manifest declares a smaller minimum. Rebuild the
boundary-connected complement of the decoded source-space subject mask and
prove the decoded sampling mask is exactly the permitted side/corner subset,
with no enclosed-hole or subject pixels. Independently recompute the shared
four-side/rounded-or-parallel geometry evidence plus four-side/four-corner
sampling coverage, second-difference texture fraction, and longest bad run;
reject contradictory stored counters even when their status strings say
`passed`. Version 3 manifests must make each side/corner coverage ratio
reproduce its declared integer sample count, match decoded-mask counts and
integer bad-run lengths exactly, and match the typed hashed provenance sidecar.
Version 2 uses the exact historical fallback replay described above. No version
may enter a compatibility path because optional fields are absent, and no broad
counter tolerance is an acceptable substitute for deterministic replay.

If the subject and trim have different aspect ratios, the remaining trim area
is intentional current-card background. Do not crop, distort, or force four
equal visible-frame margins to hide the mismatch.

## Pixel preservation

The RGBA crop retains source RGB wherever alpha is nonzero. Fully transparent
RGB may be normalized for deterministic hashing. Every fully opaque subject
pixel in the composed media must equal the corresponding subject RGB exactly.
Only antialiased boundary pixels may change through mathematical compositing
over the current-card background.

## Required inspection set

- source plus frame-detection and exterior-sampling overlays;
- binary subject mask;
- subject RGBA composited on white, grey, black, and a contrasting color;
- bleed-background RGB before subject placement;
- layout guide showing trim and subject bounds;
- full final PDF render;
- top, right, bottom, and left edge bands;
- all four corners; and
- top-center and bottom-center ornaments.

Numeric equality never overrides a visible defect.
