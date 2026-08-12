# Video bubble detection by temporal background subtraction

Notes for the Individual Project write-up. Everything below was measured on
`20260713_bigbubbles_900mA_2264mV_42deg.mp4` (90 frames, 640×480, 200 fps
camera, 900 mA cm⁻², 2264 mV, 42 °C).

## 1. Motivation — why single-frame detection has a hard ceiling

On one still frame, a scratch on the electrode and the rim of a bubble are
the same kind of local feature: a curved intensity edge. They cannot be
separated by any threshold, morphological filter, or Hough parameter,
because the information required to distinguish them is not present in a
single frame. This is a property of the data, not a deficiency of the
algorithm — and it is the reason iterative parameter tuning kept failing to
converge.

Across time the two are trivially separable: the electrode is static, the
bubbles are not.

## 2. Method

**Background model.** The bare electrode is reconstructed as the per-pixel
temporal *median* of the frame stack. Median rather than mean, because a
bubble covering a pixel in a minority of frames does not shift the median,
whereas it would drag the mean toward the bubble brightness and leave ghost
haloes.

    B(x,y) = median_t  I(x,y,t)

**Detection.** Per frame, |I − B| is thresholded, cleaned with a
morphological open/close, split with a distance-transform watershed (to
separate touching bubbles), and each region is accepted if it passes an
area and roundness test.

**Roundness** is computed as a fill ratio, `area / (π r²)` against the
minimum enclosing circle, *not* the usual `4πA/P²`. The perimeter form is
hypersensitive to the one-pixel jaggedness of segmented masks and rejects
genuinely round bubbles for having a ragged boundary.

## 3. Auto-calibration of the detection threshold

The threshold is derived from the video's own noise floor rather than tuned
by hand, so the method transfers to new recordings without re-tuning.

Each pixel's standard deviation over time mixes two populations: static
background pixels (std = sensor noise) and pixels that bubbles pass over
(std = real signal). Taking the 25th percentile of the per-pixel temporal
std selects predominantly the first population:

    noise_floor = percentile_25( std_t I(x,y,t) )
    threshold   = 2 × noise_floor          (2σ)

Measured: `noise_floor = 5.37` → `threshold = 10.75`, which reproduces the
value that had previously been hand-tuned to 10 on this clip.

Verification that it adapts rather than memorising one clip:

| Video | noise floor | auto threshold |
|---|---|---|
| Lab, 900 mA cm⁻² | 5.37 | 10.75 |
| Unrelated stock clip, different optics/lighting | 6.95 | 13.91 |

**Otsu was tested and rejected.** Otsu assumes a bimodal histogram; the
difference image is overwhelmingly near-zero because most pixels are
unchanged background, so Otsu chose 47 — keeping only the brightest ~4 % of
pixels and discarding most real bubbles (counts fell to ≈3/frame, worse
than the single-frame method it was meant to replace).

## 4. Temporal persistence filter

A real bubble is present in consecutive frames; a single-frame blip is
noise. Detections not belonging to a track seen in ≥ 2 frames are discarded.

Measured: 1825 raw detections → 286 removed (15.7 %) as non-persistent.

This filter is only available with video and is the main precision gain over
single-frame analysis.

## 5. Results

| | single-frame detector | background subtraction |
|---|---|---|
| mean bubbles / frame | 7.3 | **17.1** |
| range | 3 – 13 | 3 – 35 |

Tracking additionally yields quantities no single-frame method can produce,
and which the micro-region model of Wagner et al. (2025) is actually about:

| quantity | value |
|---|---|
| tracks with sufficient history | 192 |
| mean bubble lifetime | 0.358 s |
| mean growth rate | 0.0067 mm s⁻¹ |
| mean maximum diameter | 0.309 mm |

The mean maximum diameter of 0.309 mm falls inside the 0.1–1.1 mm range
reported for the GDE electrode in Wagner et al. (2025, Fig. 4).

## 6. Honest limitations

- **Requires a static camera.** If the camera or cell moves during the
  recording, the median background is invalid. For long recordings with slow
  drift, use `use_rolling_bg=True` (sliding-window median) at higher cost.
- **Requires video, not stills.** The method cannot be applied to isolated
  images; those still use the single-frame detector in `bubble_analysis.py`.
- **A bubble that never moves for the whole clip is absorbed into the
  background** and will be missed. In practice adhering bubbles grow, so
  they still produce a difference signal — but a completely static bubble in
  a short clip is a genuine blind spot.
- **Validated on one recording.** The auto-calibration is designed to
  generalise and was checked on a second, very different video, but bubble
  *counts* have only been validated against one lab clip. More recordings at
  other current densities would strengthen this considerably.
- **No ground-truth labels exist**, so precision and recall are not
  quantified. The comparison above is against the previous method, not
  against truth. Hand-labelling even 20–30 frames would allow reporting real
  precision/recall figures and is the single most valuable next step.
