"""
Heuristic candidate selector for FITS frames.

Usage (in Jupyter or as a script):

# 1) compute features + candidate ranking CSV
from heuristic_candidates import compute_features_for_folder, rank_candidates
df = compute_features_for_folder('/path/to/TOI_folder', out_features='features.csv')
ranked = rank_candidates(df, out_candidates='candidates.csv')

# 2) launch labeling but show candidate files first (uses label_folder from previous script)
from label_fits import label_folder
from heuristic_candidates import ranked_candidates_list
candidates = ranked_candidates_list('candidates.csv')   # returns list ordered by likelihood
# pass to labeling tool by copying candidate files into temp folder or modify label_folder to accept list.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.filters import sobel
import datetime
import math

# ----------------------
# Low-level helpers
# ----------------------
def read_fits_image(path):
    """Read FITS primary HDU as a 2D numpy array. If 3D, take first slice."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data in primary HDU: {path}")
        if data.ndim == 3:
            data = data[0]
        return np.array(data, dtype=float), hdul[0].header

def centroid(frame):
    """Flux-weighted centroid (x, y)."""
    f = np.nan_to_num(frame)
    Y, X = np.indices(f.shape)
    tot = f.sum() + 1e-8
    xc = (f * X).sum() / tot
    yc = (f * Y).sum() / tot
    return float(xc), float(yc)

def simple_fwhm_proxy(frame, bright_frac=0.05):
    """A quick proxy for FWHM: compute second-moment width of top bright_frac fraction of pixels.
    Returns NaN if insufficient signal.
    """
    f = np.nan_to_num(frame)
    flat = f.ravel()
    if flat.size == 0:
        return np.nan
    thr = np.percentile(flat, 100 * (1 - bright_frac))
    mask = f >= thr
    if mask.sum() < 5:
        # fallback: use std of all pixels
        return float(np.nanstd(f))
    ys, xs = np.where(mask)
    # approximate radius std dev
    rx = np.std(xs)
    ry = np.std(ys)
    # convert to FWHM-like proxy (2.355*sigma)
    return float(2.355 * (rx + ry) / 2.0)

def streak_score(frame, edge_thresh=None):
    """Edge-based fraction of high-gradient pixels (streakness)."""
    im = np.nan_to_num(frame)
    # Normalize robustly for edge detection
    med = np.median(im)
    std = np.std(im) + 1e-8
    imn = (im - med) / std
    edges = sobel(imn)
    if edge_thresh is None:
        # set threshold at fairly high gradient (tunable)
        edge_thresh = np.percentile(edges.ravel(), 90)
    return float((edges > edge_thresh).sum() / edges.size)

# ----------------------
# Feature computation for folder
# ----------------------
def compute_features_for_folder(folder, exts=('.fits','.fit','.fts'), out_features=None, verbose=True):
    """
    Walk `folder`, compute per-file features, return a DataFrame.
    Saves CSV to out_features if provided.
    Features:
      - median, mean, std, max
      - centroid_x, centroid_y
      - disp (displacement from previous frame centroid)
      - fwhm_proxy
      - streak (edge fraction)
      - flux_jump (abs difference of median from previous)
      - time (if DATE-OBS present)
    """
    folder = os.path.abspath(folder)
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])
    rows = []
    prev_cent = None
    prev_med = None
    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        try:
            img, hdr = read_fits_image(path)
        except Exception as e:
            print(f"Error reading {fname}: {e}; skipping")
            continue
        med = float(np.nanmedian(img))
        mean = float(np.nanmean(img))
        std = float(np.nanstd(img))
        mx = float(np.nanmax(img))
        xc, yc = centroid(img)
        if prev_cent is None:
            disp = 0.0
        else:
            disp = float(math.hypot(xc - prev_cent[0], yc - prev_cent[1]))
        prev_cent = (xc, yc)
        fwhm = simple_fwhm_proxy(img)
        streak = streak_score(img)
        if prev_med is None:
            flux_jump = 0.0
        else:
            flux_jump = float(abs(med - prev_med))
        prev_med = med
        date_obs = hdr.get('DATE-OBS', None)
        rows.append({
            'file': fname,
            'index': i,
            'median': med,
            'mean': mean,
            'std': std,
            'max': mx,
            'xc': xc,
            'yc': yc,
            'disp': disp,
            'fwhm_proxy': fwhm,
            'streak': streak,
            'flux_jump': flux_jump,
            'date_obs': date_obs
        })
        if verbose and (i+1) % 100 == 0:
            print(f"Processed {i+1} files")
    df = pd.DataFrame(rows)
    if out_features:
        os.makedirs(os.path.dirname(out_features), exist_ok=True)
        df.to_csv(out_features, index=False)
        if verbose:
            print(f"Wrote features to {out_features}")
    return df

# ----------------------
# Candidate scoring + ranking
# ----------------------
def robust_scale_series(s):
    """Scale a pandas Series to robust 0-1 using median and IQR; clip between 0 and 1."""
    med = s.median()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # fallback to min-max
        mn, mx = s.min(), s.max()
        if mx - mn == 0:
            return s * 0.0
        return ((s - mn) / (mx - mn)).clip(0,1)
    # scale such that med maps to 0, q3 maps to 0.75 etc.
    scaled = (s - med) / (1.5 * iqr)  # approx scale
    # map negative values to 0, positive values to a soft cap
    scaled = scaled.clip(0, 3.0) / 3.0
    return scaled.clip(0,1)

def rank_candidates(df, out_candidates=None, weights=None, verbose=True):
    """
    Build a composite candidate score and return DataFrame sorted by score desc.
    weights: dict of weights for features. If None, default weights used.
    By default, we consider: disp, streak, flux_jump, fwhm_proxy, std.
    Composite score is weighted sum of robust-scaled features (0..1).
    """
    dfc = df.copy().reset_index(drop=True)
    # features used
    features = ['disp','streak','flux_jump','fwhm_proxy','std']
    # default weights (tuneable)
    if weights is None:
        weights = {'disp':1.0, 'streak':2.0, 'flux_jump':1.0, 'fwhm_proxy':1.0, 'std':0.5}
    # if a feature missing, ignore
    present = [f for f in features if f in dfc.columns]
    for f in present:
        dfc[f'_rs_{f}'] = robust_scale_series(dfc[f])
    # compute weighted sum
    total_weight = sum(weights.get(f,0) for f in present)
    if total_weight == 0:
        total_weight = 1.0
    score = np.zeros(len(dfc), dtype=float)
    for f in present:
        w = weights.get(f, 0.0)
        score += w * dfc[f'_rs_{f}'].values
    score = score / total_weight
    dfc['candidate_score'] = score
    # also compute a "flag" by thresholding each robust feature (useful for quick filters)
    # e.g., flagged if any robust feature > 0.6
    dfc['flag_any'] = (dfc[[f'_rs_{f}' for f in present]] > 0.6).any(axis=1)
    # sort
    dfc = dfc.sort_values('candidate_score', ascending=False).reset_index(drop=True)
    if out_candidates:
        os.makedirs(os.path.dirname(out_candidates), exist_ok=True)
        dfc.to_csv(out_candidates, index=False)
        if verbose:
            print(f"Wrote ranked candidates to {out_candidates}")
    return dfc

def ranked_candidates_list(candidates_csv, top_n=None):
    """Utility: returns list of filenames from a candidates CSV sorted by score desc."""
    df = pd.read_csv(candidates_csv)
    if top_n:
        return df['file'].tolist()[:top_n]
    return df['file'].tolist()

# ----------------------
# Integration helper: create a prioritized labeling folder (optional)
# ----------------------
def create_prioritized_folder(original_folder, out_folder, candidates_csv, top_n=None):
    """
    Make a new folder where candidate bad frames (top_n most-likely-bad) are copied first,
    followed by the rest. This is useful if your labeling tool only labels folders sequentially.
    """
    import shutil
    os.makedirs(out_folder, exist_ok=True)
    df = pd.read_csv(candidates_csv)
    files_order = df['file'].tolist()
    if top_n:
        # put top_n first, then the rest excluding those top_n
        top = set(files_order[:top_n])
        rest = [f for f in files_order if f not in top]
        ordered = list(files_order[:top_n]) + rest
    else:
        ordered = files_order
    # copy files in this order into out_folder, prefix with index to preserve order
    for i, fname in enumerate(ordered, start=1):
        src = os.path.join(original_folder, fname)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_folder, f"{i:05d}_{fname}")
        shutil.copy2(src, dst)
    print(f"Copied {len(ordered)} files to prioritized folder {out_folder}")

# ----------------------
# Quick tuning guide (printed helper)
# ----------------------
def print_tuning_guide():
    print("""
Heuristic tuning guide:
- disp (centroid displacement): sensitive to tracking/jitter/target-off. Units: pixels.
  * Typical tuning: check distribution and pick a percentile threshold (e.g., frames with disp > 90th percentile).
- streak: edge-based fraction [0..1]. Values near 0.0 normal; large streaks push this up.
  * Typical tuning: use robust scaling in rank_candidates then inspect top frames to adjust weight.
- fwhm_proxy: larger values indicate blur; inspect distribution per-night.
- flux_jump: sudden jumps point to clouds, shutter issues, or saturation.
- Use rank_candidates(df) then inspect top 50-200 rows visually to calibrate weights and thresholds.
- Use create_prioritized_folder(...) to copy top candidates into a new directory for rapid labeling.
""")

# ----------------------
# If run as script, do a simple demo (not executed in import)
# ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute heuristic candidates from a folder of FITS files.")
    parser.add_argument('folder', help='Folder with FITS files')
    parser.add_argument('--out-features', default='features.csv', help='CSV to write features')
    parser.add_argument('--out-candidates', default='candidates.csv', help='CSV to write ranked candidates')
    parser.add_argument('--top-n', type=int, default=100, help='Top N candidates to consider (for copying)')
    parser.add_argument('--prior-folder', default='prioritized', help='Optional folder to create with top candidates first')
    args = parser.parse_args()

    df = compute_features_for_folder(args.folder, out_features=args.out_features)
    dfc = rank_candidates(df, out_candidates=args.out_candidates)
    print(dfc.head(20))
    if args.prior_folder:
        create_prioritized_folder(args.folder, args.prior_folder, args.out_candidates, top_n=args.top_n)
        print("Prioritized folder created. Use your labeling tool on that folder first.")
    print_tuning_guide()