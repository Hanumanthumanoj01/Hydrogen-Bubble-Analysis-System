"""
train_from_videos.py
====================
Trains the Sprint-7 baseline ML model on the four ANS video recordings.

WHAT THIS TRAINS (and what it does not)
---------------------------------------
This trains a model to predict OPERATING CONDITIONS (current density) from a
feature vector describing the bubble population in a frame:

    frame -> [bubble count, mean diameter, coverage, entropy, curvature, ...]
          -> Random Forest / Gradient Boosting
          -> predicted current density

This is exactly the deliverable in exposé Sprint 7. It is NOT a bubble
detector: the detector that produces the feature vector is still the
classical CV pipeline. Training a *detector* (exposé Sprint 9, YOLOv8) is a
separate task requiring hand-drawn bubble labels, which do not exist.

VALIDATION: LEAVE-ONE-VIDEO-OUT
--------------------------------
The ~370 frames come from only 4 recordings. Frames within one recording are
highly correlated -- consecutive frames of the same electrode at the same
current density are near-duplicates. A random train/test split would put
near-identical frames in both sets and report an accuracy near 100% that
means nothing.

We therefore hold out an ENTIRE video at a time and predict it from the
other three. This is the honest protocol, and it makes the effective sample
size 4, not 370. Results must be reported as indicative.

Because each video is a different current density, holding one out means its
target value is absent from training. Classification is therefore impossible
(the class does not exist in training) and the task is posed as REGRESSION,
which tests genuine interpolation/extrapolation.
"""
from __future__ import annotations

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_analysis import (
    load_video_gray, compute_background, auto_threshold,
    detect_bubbles_bgsub, shannon_entropy, MM_PER_PIXEL,
)

import re
import glob

# Videos are DISCOVERED from the folder, not hardcoded. Any .mp4/.mov/.avi
# whose filename contains a current density like "900mA" is picked up, so new
# recordings work without editing this file. Voltage ("2264mV") and temperature
# ("42deg") are also parsed when present, but are optional.
#
# Examples that all parse correctly:
#   20260713_bigbubbles_900mA_2264mV_42deg.mp4  -> 900 mA/cm2, 2264 mV, 42 C
#   run3_450mA.mp4                              -> 450 mA/cm2
#   GDE_1000mA_2400mV.mov                       -> 1000 mA/cm2, 2400 mV
VIDEO_EXTS = ('*.mp4', '*.mov', '*.avi', '*.MP4', '*.MOV')


def parse_video_name(path):
    """Extract (current_density, voltage_mV, temp_C) from a filename.
    Returns None for current density if the filename has no 'NNNmA' pattern."""
    name = os.path.basename(path)
    cd = re.search(r'(\d{2,4})\s*mA', name, re.I)
    mv = re.search(r'(\d{3,5})\s*mV', name, re.I)
    tc = re.search(r'(\d{1,3})\s*deg', name, re.I)
    return (int(cd.group(1)) if cd else None,
            int(mv.group(1)) if mv else None,
            int(tc.group(1)) if tc else None)


def discover_videos(folder):
    """Find all usable videos in `folder`, keyed by current density."""
    paths = []
    for pat in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    found = {}
    for p in sorted(paths):
        cd, mv, tc = parse_video_name(p)
        if cd is None:
            print(f'  ignored (no "NNNmA" in name): {os.path.basename(p)}')
            continue
        if cd in found:
            print(f'  ignored (duplicate {cd} mA/cm2): {os.path.basename(p)}')
            continue
        found[cd] = (p, mv, tc)
    return found

FEATURE_NAMES = [
    'bubble_count', 'mean_diameter_mm', 'std_diameter_mm', 'max_diameter_mm',
    'mean_curvature', 'coverage_pct', 'shannon_H', 'H_norm',
    'bubble_density_per_kpx', 'small_frac', 'large_frac',
]


def frame_features(gray, bubbles):
    h, w = gray.shape
    H, H_norm = shannon_entropy(gray)
    d = np.array([2 * b[2] * MM_PER_PIXEL for b in bubbles]) if bubbles else np.array([0.0])
    import cv2
    mask = np.zeros((h, w), np.uint8)
    for x, y, r in bubbles:
        cv2.circle(mask, (int(x), int(y)), max(1, int(r)), 1, -1)
    coverage = mask.sum() / (h * w) * 100
    mean_d = float(d.mean())
    return [
        len(bubbles),
        mean_d,
        float(d.std()),
        float(d.max()),
        1.0 / (mean_d / 2) if mean_d > 0 else 0.0,
        coverage,
        H,
        H_norm * 100,
        len(bubbles) / (h * w / 1000.0),
        float((d < 0.15).mean()),
        float((d > 0.30).mean()),
    ]


def build_dataset(upload_dir=None):
    """Build the feature dataset from the videos in `upload_dir`.

    Pass the folder as the first command-line argument, e.g.
        python3 train_from_videos.py ~/Downloads/videos
    If omitted, looks in ./videos next to this script.
    """
    if upload_dir is None:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
    if not os.path.isdir(upload_dir):
        print(f'ERROR: folder not found: {upload_dir}')
        print('Put the 4 .mp4 files in a folder and pass it as an argument:')
        print('    python3 train_from_videos.py /path/to/your/videos')
        sys.exit(1)
    print(f'Looking for videos in: {upload_dir}')
    videos = discover_videos(upload_dir)
    if len(videos) < 2:
        print(f'\nERROR: found {len(videos)} usable video(s). Need at least 2 to train.')
        print('Videos must be .mp4/.mov/.avi with the current density in the name,')
        print('e.g. 20260713_bigbubbles_900mA_2264mV_42deg.mp4  or  run3_450mA.mp4')
        sys.exit(1)

    print(f'Found {len(videos)} videos at: {sorted(videos.keys())} mA/cm2\n')
    X, y, groups = [], [], []
    for cd in sorted(videos):
        path, mv, tc = videos[cd]
        frames, fps = load_video_gray(path)
        bg = compute_background(frames)
        thr = auto_threshold(frames, bg=bg)
        for g in frames:
            bubbles = detect_bubbles_bgsub(g, bg, thr)
            X.append(frame_features(g, bubbles))
            y.append(cd)
            groups.append(cd)
        extra = []
        if mv: extra.append(f'{mv} mV')
        if tc: extra.append(f'{tc} C')
        suffix = f'  [{", ".join(extra)}]' if extra else ''
        print(f'  {cd:>4} mA/cm2: {len(frames)} frames, threshold={thr:.1f}{suffix}')
    return np.array(X, dtype=float), np.array(y, dtype=float), np.array(groups)


def main():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import mean_absolute_error, r2_score

    print('Extracting features from videos...')
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    X, y, groups = build_dataset(folder)
    print(f'\nDataset: {X.shape[0]} frames x {X.shape[1]} features, '
          f'{len(np.unique(groups))} operating points\n')

    logo = LeaveOneGroupOut()
    preds, truths, held = [], [], []
    for tr, te in logo.split(X, y, groups):
        model = RandomForestRegressor(n_estimators=300, random_state=0, min_samples_leaf=2)
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        preds.append(p.mean()); truths.append(y[te][0]); held.append(int(y[te][0]))
        print(f'  held out {int(y[te][0]):>4} mA/cm2 -> predicted mean {p.mean():7.1f} '
              f'(per-frame std {p.std():5.1f})')

    preds, truths = np.array(preds), np.array(truths)
    mae = mean_absolute_error(truths, preds)
    print(f'\nLEAVE-ONE-VIDEO-OUT  MAE = {mae:.1f} mA/cm2   '
          f'(range of targets: {truths.min():.0f}-{truths.max():.0f})')
    print('NOTE: effective n = 4. Indicative only, not a reportable accuracy.')

    # In-distribution reference: train on all data, report feature importance.
    model = RandomForestRegressor(n_estimators=400, random_state=0, min_samples_leaf=2)
    model.fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    print('\nFeature importance (trained on all frames):')
    for n, v in imp:
        print(f'  {n:<26} {v:.4f}  {"#" * int(v * 60)}')

    try:
        import joblib
        joblib.dump({'model': model, 'features': FEATURE_NAMES}, 'bubble_rf_model.pkl')
        print('\nSaved trained model -> bubble_rf_model.pkl')
    except Exception as e:
        print(f'\n(could not save model: {e})')

    json.dump({'mae_leave_one_video_out': float(mae),
               'held_out': held,
               'predicted': [float(p) for p in preds],
               'truth': [float(t) for t in truths],
               'feature_importance': {n: float(v) for n, v in imp},
               'n_frames': int(X.shape[0])},
              open('training_report.json', 'w'), indent=1)
    print('Saved training_report.json')


if __name__ == '__main__':
    main()
