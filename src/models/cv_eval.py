"""
CV Feature Representation Benchmark — cv_eval.py

Comprehensive benchmark of CV feature representations for predicting
DA/AA/UA concentrations from cyclic voltammetry data.

Families:
  1  Raw waveform baseline (100-pt sampled waveform)
  2  PCA features (various component counts and variance thresholds)
  3  Electrochemical peak features
  4  Area-based features
  5  Derivative features
  6  Statistical shape features
  7  Spline coefficient features
  8  Wavelet features (requires pywavelets)
  9  Feature selection on raw waveform
  10 Combined feature sets
  11 Autoencoder bottleneck features
  12 Preprocessing ablation study

Usage:
    python cv_eval.py
    python cv_eval.py --data-dir outputs_dataset_full --train-electrodes 1 5 --test-electrodes 3
    python cv_eval.py --ann-epochs 2000 --n-seeds 2 --skip-autoencoder --skip-wavelets
"""

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

# --- repo data/results anchors (auto-added during repo reorg) ---
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
_DATA_DIR = str(_REPO_ROOT / "data" / "outputs_dataset_full")
_RESULTS_DIR = str(_REPO_ROOT / "results" / "cv_eval_results")
# --- end repo anchors ---


import argparse
import copy
import glob
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.interpolate import splrep
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import MultiTaskLassoCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from feature_extract_cond import (
    load_cv_n, parse_labels, ALL_ELECTRODES, asls, sg, nearest_idx,
    peak_in_window, peak_shape_features, WINDOWS, V_BASE_LOW, V_BASE_HIGH,
)
from model import make_model, ANN

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("WARNING: pywavelets not installed — Family 8 (wavelet) experiments will be skipped.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CV feature representation benchmark.')
parser.add_argument('--data-dir',         default=_DATA_DIR)
parser.add_argument('--train-electrodes', nargs='+', default=['1', '5'], metavar='N')
parser.add_argument('--test-electrodes',  nargs='+', default=['3'],      metavar='N')
parser.add_argument('--ann-epochs',       type=int,  default=5000)
parser.add_argument('--n-seeds',          type=int,  default=3)
parser.add_argument('--cv-folds',         type=int,  default=5)
parser.add_argument('--wavelet',          default='db4')
parser.add_argument('--skip-autoencoder', action='store_true')
parser.add_argument('--skip-wavelets',    action='store_true')
parser.add_argument('--skip-ann',         action='store_true')
parser.add_argument('--families',         nargs='+', default=None, metavar='FAM',
                    help='Only run these feature families. Choices: raw, pca, electrochemical, '
                         'area, derivative, statistical, spline, wavelet, feature_selection, '
                         'combined, autoencoder, ablation')
args = parser.parse_args()

VALID_ELEC = {'1': 'i1', '3': 'i3', '5': 'i5', 'i1': 'i1', 'i3': 'i3', 'i5': 'i5'}
train_elecs = [VALID_ELEC[e] for e in args.train_electrodes]
test_elecs  = [VALID_ELEC[e] for e in args.test_electrodes]
LABEL_COLS  = ['DA_uM', 'AA_uM', 'UA_uM']

OUT_DIR = _RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n{'='*70}")
print(f"CV Feature Representation Benchmark")
print(f"  Train electrodes : {train_elecs}")
print(f"  Test electrodes  : {test_elecs}")
print(f"  ANN epochs       : {args.ann_epochs}")
print(f"  Seeds per trial  : {args.n_seeds}")
print(f"  CV folds         : {args.cv_folds}")
print(f"  Wavelet          : {args.wavelet}")
print(f"  Skip autoencoder : {args.skip_autoencoder}")
print(f"  Skip wavelets    : {args.skip_wavelets or not HAS_PYWT}")
print(f"  Skip ANN         : {args.skip_ann}")
print(f"{'='*70}\n")

# ── Data loading & preprocessing ───────────────────────────────────────────────
print("Loading CV data from condition directories...")

cond_dirs = sorted(
    glob.glob(os.path.join(args.data_dir, '*/')),
    key=lambda x: float(os.path.basename(x.rstrip('/\\')))
)
if not cond_dirs:
    raise FileNotFoundError(f"No condition directories found in {args.data_dir!r}")

CV_N_POINTS = 50  # points per sweep direction for raw waveform sampling

# Collect all processed traces as list of dicts
all_traces = []

for cond_dir in cond_dirs:
    base = cond_dir if cond_dir.endswith(os.sep) else cond_dir + os.sep
    try:
        da, aa, ua = parse_labels(base)
        V_cv, cv_I = load_cv_n(base)
    except Exception as exc:
        print(f"  WARNING: skipped {cond_dir}: {exc}")
        continue

    for elec in ALL_ELECTRODES:
        try:
            I_raw_A = cv_I[elec] * 1e6  # convert to µA
            I_smooth = sg(I_raw_A)       # Savitzky-Golay smoothing

            # Split into forward (ascending V) and reverse sweeps
            split = int(np.argmax(V_cv))
            V_fwd_full = V_cv[:split + 1]
            I_fwd_full = I_smooth[:split + 1]
            V_rev_full = V_cv[split:][::-1]
            I_rev_full = I_smooth[split:][::-1]

            # Trim forward sweep edges by 0.05V
            trim = (V_fwd_full >= V_fwd_full[0] + 0.05) & (V_fwd_full <= V_fwd_full[-1] - 0.05)
            V_fwd = V_fwd_full[trim]
            I_fwd = I_fwd_full[trim]
            V_rev = V_rev_full.copy()
            I_rev = I_rev_full.copy()

            # Apply asls separately to forward and reverse
            I_corr_fwd, bl_fwd = asls(I_fwd)
            I_corr_rev, bl_rev = asls(I_rev)

            # Build 100-point voltage grid: 50 fwd + 50 rev
            v_lo = max(V_fwd.min(), V_rev.min())
            v_hi = min(V_fwd.max(), V_rev.max())
            v_grid = np.linspace(v_lo, v_hi, CV_N_POINTS)

            raw_fwd = np.array([I_corr_fwd[nearest_idx(V_fwd, v)] for v in v_grid], dtype=np.float32)
            raw_rev = np.array([I_corr_rev[nearest_idx(V_rev, v)] for v in v_grid], dtype=np.float32)

            trace = {
                'electrode': elec,
                'DA_uM': da, 'AA_uM': aa, 'UA_uM': ua,
                'V_fwd': V_fwd,
                'I_fwd_raw': I_fwd,
                'I_corr_fwd': I_corr_fwd,
                'V_rev': V_rev,
                'I_rev_raw': I_rev,
                'I_corr_rev': I_corr_rev,
                'v_grid': v_grid,
                'raw_fwd': raw_fwd,
                'raw_rev': raw_rev,
                # preserve smoothed raw before asls for ablations
                'I_smooth_fwd': I_fwd.copy(),
                'I_smooth_rev': I_rev.copy(),
                'I_raw_fwd': I_raw_A[:split + 1][trim].copy(),
                'I_raw_rev': I_raw_A[split:][::-1].copy(),
            }
            all_traces.append(trace)
        except Exception as exc:
            print(f"  WARNING: {cond_dir} {elec}: {exc}")
            continue

print(f"  Loaded {len(all_traces)} traces ({len(cond_dirs)} conditions × {len(ALL_ELECTRODES)} electrodes)")

# Build common voltage grid across all samples for spline/wavelet
v_min_all = min(t['V_fwd'].min() for t in all_traces)
v_max_all = max(t['V_fwd'].max() for t in all_traces)
V_COMMON = np.linspace(v_min_all, v_max_all, 100)

# Split into train/test
train_traces = [t for t in all_traces if t['electrode'] in train_elecs]
test_traces  = [t for t in all_traces if t['electrode'] in test_elecs]

print(f"  Train traces: {len(train_traces)}   Test traces: {len(test_traces)}\n")

# Build label arrays
y_tr = np.array([[t['DA_uM'], t['AA_uM'], t['UA_uM']] for t in train_traces], dtype=np.float32)
y_te = np.array([[t['DA_uM'], t['AA_uM'], t['UA_uM']] for t in test_traces],  dtype=np.float32)


# ── Feature extraction functions ───────────────────────────────────────────────

def feats_raw_waveform(trace_list):
    """100-point raw waveform: 50 fwd + 50 rev. Shape: (N, 100)"""
    rows = []
    for t in trace_list:
        row = np.concatenate([t['raw_fwd'], t['raw_rev']]).astype(np.float32)
        rows.append(row)
    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_electrochemical(trace_list):
    """Electrochemical peak features. Shape: (N, ~35)"""
    rows = []
    for t in trace_list:
        row = []
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        V_rev = t['V_rev']
        I_rev = t['I_corr_rev']

        da_Ep_a = np.nan
        da_Ipa  = np.nan

        # Per-analyte anodic peaks from forward sweep
        for analyte, (v_lo, v_hi) in WINDOWS.items():
            try:
                pidx, Ep, Ipa = peak_in_window(V_fwd, I_fwd, v_lo, v_hi)
                if pidx is None or np.isnan(Ep):
                    row.extend([0.0, 0.0])
                else:
                    row.extend([float(Ipa), float(Ep)])
                    if analyte == 'DA':
                        da_Ep_a = float(Ep)
                        da_Ipa  = float(Ipa)
            except Exception:
                row.extend([0.0, 0.0])

        # DA cathodic peak from reverse sweep
        da_cathodic_feats = [0.0] * 7  # Ipc, Epc, deltaEp, E12, Ipa_Ipc_ratio, Ipa_Ipc_diff, found_flag
        if not np.isnan(da_Ep_a):
            try:
                I_cat_inv = -I_rev
                cat_peaks, _ = find_peaks(I_cat_inv, prominence=max(I_cat_inv.max() * 0.05, 1e-9))
                best = None
                for cp in cat_peaks:
                    if abs(V_rev[cp] - da_Ep_a) < 0.15:
                        if best is None or abs(V_rev[cp] - da_Ep_a) < abs(V_rev[best] - da_Ep_a):
                            best = cp
                if best is not None:
                    Ep_c = float(V_rev[best])
                    Ipc  = float(I_rev[best])
                    dEp  = float(da_Ep_a - Ep_c)
                    E12  = float((da_Ep_a + Ep_c) / 2.0)
                    ratio = float(abs(da_Ipa / Ipc)) if Ipc != 0 else 0.0
                    diff  = float(da_Ipa - Ipc)
                    da_cathodic_feats = [Ipc, Ep_c, dEp, E12, ratio, diff, 1.0]
            except Exception:
                pass
        row.extend(da_cathodic_feats)

        # Forward sweep global stats (7 features)
        try:
            row.extend([
                float(np.max(I_fwd)),
                float(np.min(I_fwd)),
                float(V_fwd[np.argmax(I_fwd)]),
                float(V_fwd[np.argmin(I_fwd)]),
                float(np.mean(I_fwd)),
                float(np.std(I_fwd)),
                float(np.ptp(I_fwd)),
            ])
        except Exception:
            row.extend([0.0] * 7)

        # Reverse sweep global stats (7 features)
        try:
            row.extend([
                float(np.max(I_rev)),
                float(np.min(I_rev)),
                float(V_rev[np.argmax(I_rev)]),
                float(V_rev[np.argmin(I_rev)]),
                float(np.mean(I_rev)),
                float(np.std(I_rev)),
                float(np.ptp(I_rev)),
            ])
        except Exception:
            row.extend([0.0] * 7)

        # Cross-sweep ratios (2 features)
        try:
            fwd_max = float(np.max(np.abs(I_fwd)))
            rev_max = float(np.max(np.abs(I_rev)))
            fwd_rev_max_ratio  = fwd_max / rev_max if rev_max != 0 else 0.0
            fwd_mean = float(np.mean(np.abs(I_fwd)))
            rev_mean = float(np.mean(np.abs(I_rev)))
            fwd_rev_mean_ratio = fwd_mean / rev_mean if rev_mean != 0 else 0.0
            row.extend([fwd_rev_max_ratio, fwd_rev_mean_ratio])
        except Exception:
            row.extend([0.0, 0.0])

        rows.append(np.array(row, dtype=np.float32))

    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_area(trace_list):
    """Area-based features. Shape: (N, 10)"""
    rows = []
    for t in trace_list:
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        V_rev = t['V_rev']
        I_rev = t['I_corr_rev']
        row = []

        try:
            fwd_signed_area = float(trapezoid(I_fwd, V_fwd))
        except Exception:
            fwd_signed_area = 0.0
        row.append(fwd_signed_area)

        try:
            rev_signed_area = float(trapezoid(I_rev, V_rev))
        except Exception:
            rev_signed_area = 0.0
        row.append(rev_signed_area)

        try:
            fwd_abs_area = float(trapezoid(np.abs(I_fwd), V_fwd))
        except Exception:
            fwd_abs_area = 0.0
        row.append(fwd_abs_area)

        try:
            rev_abs_area = float(trapezoid(np.abs(I_rev[::-1]), V_rev[::-1]))
        except Exception:
            rev_abs_area = 0.0
        row.append(rev_abs_area)

        try:
            fwd_pos_area = float(trapezoid(np.clip(I_fwd, 0, None), V_fwd))
        except Exception:
            fwd_pos_area = 0.0
        row.append(fwd_pos_area)

        try:
            fwd_neg_area = float(trapezoid(np.clip(I_fwd, None, 0), V_fwd))
        except Exception:
            fwd_neg_area = 0.0
        row.append(fwd_neg_area)

        try:
            rev_pos_area = float(trapezoid(np.clip(I_rev[::-1], 0, None), V_rev[::-1]))
        except Exception:
            rev_pos_area = 0.0
        row.append(rev_pos_area)

        try:
            rev_neg_area = float(trapezoid(np.clip(I_rev[::-1], None, 0), V_rev[::-1]))
        except Exception:
            rev_neg_area = 0.0
        row.append(rev_neg_area)

        # area_ratio_fwd_rev
        try:
            area_ratio = fwd_abs_area / rev_abs_area if rev_abs_area != 0 else np.nan
        except Exception:
            area_ratio = np.nan
        row.append(float(area_ratio) if not np.isnan(area_ratio) else 0.0)

        # hysteresis_area: interpolate rev onto fwd grid, integrate |I_fwd - I_rev_interp|
        try:
            f_rev = interp1d(V_rev[::-1], I_rev[::-1], kind='linear', bounds_error=False, fill_value='extrapolate')
            I_rev_interp = f_rev(V_fwd)
            hysteresis_area = float(trapezoid(np.abs(I_fwd - I_rev_interp), V_fwd))
        except Exception:
            hysteresis_area = 0.0
        row.append(hysteresis_area)

        rows.append(np.array(row, dtype=np.float32))

    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_derivative(trace_list):
    """Derivative-based features from forward sweep. Shape: (N, 8)"""
    rows = []
    for t in trace_list:
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        row = []

        try:
            dI_dV = np.gradient(I_fwd, V_fwd)
            row.append(float(np.max(dI_dV)))
            row.append(float(np.min(dI_dV)))
            row.append(float(V_fwd[np.argmax(dI_dV)]))
            row.append(float(V_fwd[np.argmin(dI_dV)]))
            row.append(float(np.mean(np.abs(dI_dV))))
            row.append(float(np.std(dI_dV)))
            row.append(float(np.max(np.abs(dI_dV))))

            # n_slope_extrema: count peaks in |dI/dV| above 5% of max
            abs_dIdV = np.abs(dI_dV)
            threshold = 0.05 * abs_dIdV.max()
            peaks, _ = find_peaks(abs_dIdV, height=threshold)
            row.append(float(len(peaks)))
        except Exception:
            row.extend([0.0] * 8)

        rows.append(np.array(row, dtype=np.float32))

    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_statistical(trace_list):
    """Statistical shape features. Shape: (N, 14)"""
    rows = []
    for t in trace_list:
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        I_rev = t['I_corr_rev']
        row = []

        try:
            I_all = np.concatenate([I_fwd, I_rev])
            q25, q75 = np.percentile(I_all, [25, 75])
            row.extend([
                float(np.mean(I_all)),
                float(np.std(I_all)),
                float(np.median(I_all)),
                float(q75 - q25),
                float(np.min(I_all)),
                float(np.max(I_all)),
                float(np.ptp(I_all)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                row.append(float(sp_skew(I_all)))
                row.append(float(sp_kurtosis(I_all)))
            row.append(float(np.sqrt(np.mean(I_all ** 2))))  # RMS
            row.append(float(np.linalg.norm(I_all)))          # L2 norm
            row.append(float(np.sum(np.abs(np.diff(I_all)))))  # total variation

            # zero_crossings: sign changes in I_corr_fwd after subtracting mean
            I_centered = I_fwd - np.mean(I_fwd)
            zero_crossings = int(np.sum(np.diff(np.sign(I_centered)) != 0))
            row.append(float(zero_crossings))

            # n_local_extrema: peaks + valleys in I_fwd above 5% of range
            I_range = float(np.ptp(I_fwd))
            if I_range > 0:
                prom_thresh = 0.05 * I_range
                peaks_up, _   = find_peaks(I_fwd, prominence=prom_thresh)
                peaks_dn, _   = find_peaks(-I_fwd, prominence=prom_thresh)
                n_extrema = len(peaks_up) + len(peaks_dn)
            else:
                n_extrema = 0
            row.append(float(n_extrema))
        except Exception:
            row.extend([0.0] * 14)

        rows.append(np.array(row, dtype=np.float32))

    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _spline_coeffs(V, I, v_grid, n_knots):
    """Fit a cubic spline to (V, I) on v_grid and return trimmed/padded coefficients."""
    target_len = n_knots + 4
    try:
        f_interp = interp1d(V, I, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_on_grid = f_interp(v_grid)
        interior_knots = np.linspace(v_grid[1], v_grid[-2], n_knots)
        tck = splrep(v_grid, I_on_grid, t=interior_knots, k=3)
        coeffs = np.array(tck[1], dtype=np.float32)
        if len(coeffs) >= target_len:
            return coeffs[:target_len]
        return np.pad(coeffs, (0, target_len - len(coeffs)), constant_values=0.0)
    except Exception:
        return np.zeros(target_len, dtype=np.float32)


def feats_spline(trace_list, n_knots, v_grid):
    """Spline coefficient features from both forward and reverse sweeps. Shape: (N, 2*(n_knots+4))"""
    rows = []
    for t in trace_list:
        fwd = _spline_coeffs(t['V_fwd'], t['I_corr_fwd'], v_grid, n_knots)
        rev = _spline_coeffs(t['V_rev'], t['I_corr_rev'], v_grid, n_knots)
        rows.append(np.concatenate([fwd, rev]))

    X = np.array(rows, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_wavelet(trace_list, wavelet='db4', top_n=None, energy_only=False, top_n_idx=None):
    """
    Wavelet features. Shape: (N, d).

    top_n_idx: pre-fitted indices for top_n (fitted on training data).
    """
    n_interp = 128
    rows = []
    for t in trace_list:
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        try:
            f_interp = interp1d(V_fwd, I_fwd, kind='linear', bounds_error=False, fill_value='extrapolate')
            v_interp = np.linspace(V_fwd.min(), V_fwd.max(), n_interp)
            signal_128 = f_interp(v_interp)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coeffs = pywt.wavedec(signal_128, wavelet)

            if energy_only:
                row = np.array([float(np.sum(c ** 2)) for c in coeffs], dtype=np.float32)
            elif top_n is not None and top_n_idx is not None:
                flat = np.concatenate(coeffs).astype(np.float32)
                # Pad if necessary
                if len(flat) < max(top_n_idx) + 1:
                    flat = np.pad(flat, (0, max(top_n_idx) + 1 - len(flat)))
                row = flat[top_n_idx].astype(np.float32)
            else:
                # energy per level + all coefficients flattened
                energy = np.array([float(np.sum(c ** 2)) for c in coeffs], dtype=np.float32)
                flat_coeffs = np.concatenate(coeffs).astype(np.float32)
                # cap to consistent length: use 256 (should cover all levels for 128-pt signal)
                max_coeff_len = 256
                if len(flat_coeffs) > max_coeff_len:
                    flat_coeffs = flat_coeffs[:max_coeff_len]
                elif len(flat_coeffs) < max_coeff_len:
                    flat_coeffs = np.pad(flat_coeffs, (0, max_coeff_len - len(flat_coeffs)))
                row = np.concatenate([energy, flat_coeffs]).astype(np.float32)
        except Exception:
            if energy_only:
                # determine expected size from a dummy signal
                try:
                    dummy = pywt.wavedec(np.zeros(n_interp), wavelet)
                    row = np.zeros(len(dummy), dtype=np.float32)
                except Exception:
                    row = np.zeros(8, dtype=np.float32)
            elif top_n is not None and top_n_idx is not None:
                row = np.zeros(len(top_n_idx), dtype=np.float32)
            else:
                row = np.zeros(256 + 8, dtype=np.float32)

        rows.append(row)

    if not rows:
        return np.zeros((0, 1), dtype=np.float32)

    # Ensure consistent shape
    max_len = max(len(r) for r in rows)
    padded = [np.pad(r, (0, max_len - len(r))) if len(r) < max_len else r for r in rows]
    X = np.array(padded, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def get_wavelet_top_n_idx(trace_list_tr, n, wavelet='db4'):
    """Fit top-N wavelet coefficient indices on training data."""
    n_interp = 128
    all_flat = []
    for t in trace_list_tr:
        V_fwd = t['V_fwd']
        I_fwd = t['I_corr_fwd']
        try:
            f_interp = interp1d(V_fwd, I_fwd, kind='linear', bounds_error=False, fill_value='extrapolate')
            v_interp = np.linspace(V_fwd.min(), V_fwd.max(), n_interp)
            signal_128 = f_interp(v_interp)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                coeffs = pywt.wavedec(signal_128, wavelet)
            flat = np.concatenate(coeffs)
            all_flat.append(flat)
        except Exception:
            pass

    if not all_flat:
        return np.arange(n)

    # Pad to same length
    max_len = max(len(f) for f in all_flat)
    padded  = np.array([np.pad(f, (0, max_len - len(f))) for f in all_flat])
    mean_abs = np.mean(np.abs(padded), axis=0)
    top_idx = np.argsort(mean_abs)[-n:][::-1]
    return top_idx.astype(int)


# ── Autoencoder ────────────────────────────────────────────────────────────────

class NumpyAutoencoder:
    """Simple fully-connected autoencoder with ReLU hidden layers, linear output."""

    def __init__(self, layer_sizes):
        """
        layer_sizes: full architecture e.g. [100, 64, 32, 8, 32, 64, 100]
        The encoder is the first ceil(len/2) layers.
        """
        self.layer_sizes = layer_sizes
        self.n_encoder_layers = len(layer_sizes) // 2  # number of weight matrices in encoder
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        cache = []
        a = X
        n_layers = len(self.weights)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            # Linear at bottleneck (middle layer) and output
            if i == n_layers - 1:
                a_next = z  # linear output
            elif i == self.n_encoder_layers - 1:
                a_next = z  # linear bottleneck
            else:
                a_next = self.relu(z)
            cache.append((z, a, a_next))
            a = a_next
        return a, cache

    def backward(self, X, y_true, cache):
        n_layers = len(self.weights)
        m = X.shape[0]
        gw = [None] * n_layers
        gb = [None] * n_layers

        y_pred = cache[-1][2]
        delta = 2 * (y_pred - y_true) / y_true.size

        for i in reversed(range(n_layers)):
            z, a_in, a_out = cache[i]
            a_prev = cache[i - 1][2] if i > 0 else X

            # Apply relu deriv except for linear output and linear bottleneck
            if i != n_layers - 1 and i != self.n_encoder_layers - 1:
                delta = delta * self.relu_deriv(z)

            gw[i] = a_prev.T @ delta / m
            gb[i] = delta.mean(axis=0, keepdims=True)
            delta  = delta @ self.weights[i].T

        return gw, gb

    def train(self, X, epochs=1000, lr=0.001):
        for epoch in range(epochs):
            y_pred, cache = self.forward(X)
            gw, gb = self.backward(X, X, cache)
            for i in range(len(self.weights)):
                self.weights[i] -= lr * gw[i]
                self.biases[i]  -= lr * gb[i]

    def encode(self, X):
        """Forward pass through encoder layers only."""
        a = X
        for i in range(self.n_encoder_layers):
            z = a @ self.weights[i] + self.biases[i]
            if i == self.n_encoder_layers - 1:
                a = z  # linear bottleneck
            else:
                a = self.relu(z)
        return a


# ── Model evaluation ───────────────────────────────────────────────────────────

def sanity_check(X_tr, X_te, name, y_tr_ref=None, y_te_ref=None):
    """Check for NaN/inf and shape consistency. Returns True if all pass."""
    ok = True
    if np.any(~np.isfinite(X_tr)):
        print(f"  WARN [{name}]: NaN/inf in X_tr — skipping")
        ok = False
    if np.any(~np.isfinite(X_te)):
        print(f"  WARN [{name}]: NaN/inf in X_te — skipping")
        ok = False
    if y_tr_ref is not None and X_tr.shape[0] != y_tr_ref.shape[0]:
        print(f"  WARN [{name}]: X_tr rows {X_tr.shape[0]} != y_tr rows {y_tr_ref.shape[0]} — skipping")
        ok = False
    if y_te_ref is not None and X_te.shape[0] != y_te_ref.shape[0]:
        print(f"  WARN [{name}]: X_te rows {X_te.shape[0]} != y_te rows {y_te_ref.shape[0]} — skipping")
        ok = False
    if X_tr.shape[1] == 0:
        print(f"  WARN [{name}]: n_features == 0 — skipping")
        ok = False
    return ok


def eval_sklearn(model, X_tr, X_te, y_tr, y_te, cv_folds):
    """Evaluate sklearn model with KFold CV and full train/test evaluation."""
    t_start = time.time()
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # KFold CV on training data
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_rmse_per_fold = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr_sc)):
        X_fold_tr = X_tr_sc[tr_idx]
        X_fold_va = X_tr_sc[va_idx]
        y_fold_tr = y_tr[tr_idx]
        y_fold_va = y_tr[va_idx]
        try:
            m_fold = copy.deepcopy(model)
            m_fold.fit(X_fold_tr, y_fold_tr)
            y_fold_pred = m_fold.predict(X_fold_va)
            rmse = np.sqrt(np.mean((y_fold_va - y_fold_pred) ** 2, axis=0))
            cv_rmse_per_fold.append(rmse)
        except Exception:
            pass

    if cv_rmse_per_fold:
        cv_rmse_arr = np.array(cv_rmse_per_fold)
        da_cv_rmse = float(cv_rmse_arr[:, 0].mean())
        aa_cv_rmse = float(cv_rmse_arr[:, 1].mean())
        ua_cv_rmse = float(cv_rmse_arr[:, 2].mean())
        mean_cv_rmse = float(np.mean([da_cv_rmse, aa_cv_rmse, ua_cv_rmse]))
    else:
        da_cv_rmse = aa_cv_rmse = ua_cv_rmse = mean_cv_rmse = np.nan

    # Full fit on training data, predict test
    try:
        m_full = copy.deepcopy(model)
        m_full.fit(X_tr_sc, y_tr)
        y_pred = m_full.predict(X_te_sc)
    except Exception as exc:
        y_pred = np.zeros_like(y_te)

    fit_time = time.time() - t_start

    da_test_rmse = float(np.sqrt(np.mean((y_te[:, 0] - y_pred[:, 0]) ** 2)))
    aa_test_rmse = float(np.sqrt(np.mean((y_te[:, 1] - y_pred[:, 1]) ** 2)))
    ua_test_rmse = float(np.sqrt(np.mean((y_te[:, 2] - y_pred[:, 2]) ** 2)))
    mean_test_rmse = float(np.mean([da_test_rmse, aa_test_rmse, ua_test_rmse]))

    da_test_mae = float(mean_absolute_error(y_te[:, 0], y_pred[:, 0]))
    aa_test_mae = float(mean_absolute_error(y_te[:, 1], y_pred[:, 1]))
    ua_test_mae = float(mean_absolute_error(y_te[:, 2], y_pred[:, 2]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        da_r2 = float(r2_score(y_te[:, 0], y_pred[:, 0]))
        aa_r2 = float(r2_score(y_te[:, 1], y_pred[:, 1]))
        ua_r2 = float(r2_score(y_te[:, 2], y_pred[:, 2]))

    return {
        'DA_cv_rmse':    da_cv_rmse,
        'AA_cv_rmse':    aa_cv_rmse,
        'UA_cv_rmse':    ua_cv_rmse,
        'mean_cv_rmse':  mean_cv_rmse,
        'DA_test_rmse':  da_test_rmse,
        'AA_test_rmse':  aa_test_rmse,
        'UA_test_rmse':  ua_test_rmse,
        'mean_test_rmse': mean_test_rmse,
        'DA_test_mae':   da_test_mae,
        'AA_test_mae':   aa_test_mae,
        'UA_test_mae':   ua_test_mae,
        'DA_test_r2':    da_r2,
        'AA_test_r2':    aa_r2,
        'UA_test_r2':    ua_r2,
        'fit_time':      fit_time,
    }


def eval_ann(X_tr, X_te, y_tr, y_te, n_seeds, epochs):
    """Evaluate ANN with multiple random seeds, averaging predictions."""
    t_start = time.time()
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_tr_sc = scaler_X.fit_transform(X_tr).astype(np.float32)
    y_tr_sc = scaler_y.fit_transform(y_tr).astype(np.float32)
    X_te_sc = scaler_X.transform(X_te).astype(np.float32)

    n_features = X_tr_sc.shape[1]
    all_preds = []

    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        model = make_model(n_features)
        model.train(X_tr_sc, y_tr_sc, epochs=epochs, lr=0.001, verbose=False)
        y_pred_sc = model.predict(X_te_sc)
        y_pred = scaler_y.inverse_transform(y_pred_sc)
        all_preds.append(y_pred)

    y_pred_avg = np.mean(all_preds, axis=0)
    fit_time = time.time() - t_start

    da_test_rmse = float(np.sqrt(np.mean((y_te[:, 0] - y_pred_avg[:, 0]) ** 2)))
    aa_test_rmse = float(np.sqrt(np.mean((y_te[:, 1] - y_pred_avg[:, 1]) ** 2)))
    ua_test_rmse = float(np.sqrt(np.mean((y_te[:, 2] - y_pred_avg[:, 2]) ** 2)))
    mean_test_rmse = float(np.mean([da_test_rmse, aa_test_rmse, ua_test_rmse]))

    da_test_mae = float(mean_absolute_error(y_te[:, 0], y_pred_avg[:, 0]))
    aa_test_mae = float(mean_absolute_error(y_te[:, 1], y_pred_avg[:, 1]))
    ua_test_mae = float(mean_absolute_error(y_te[:, 2], y_pred_avg[:, 2]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        da_r2 = float(r2_score(y_te[:, 0], y_pred_avg[:, 0]))
        aa_r2 = float(r2_score(y_te[:, 1], y_pred_avg[:, 1]))
        ua_r2 = float(r2_score(y_te[:, 2], y_pred_avg[:, 2]))

    return {
        'DA_cv_rmse':    np.nan,
        'AA_cv_rmse':    np.nan,
        'UA_cv_rmse':    np.nan,
        'mean_cv_rmse':  np.nan,
        'DA_test_rmse':  da_test_rmse,
        'AA_test_rmse':  aa_test_rmse,
        'UA_test_rmse':  ua_test_rmse,
        'mean_test_rmse': mean_test_rmse,
        'DA_test_mae':   da_test_mae,
        'AA_test_mae':   aa_test_mae,
        'UA_test_mae':   ua_test_mae,
        'DA_test_r2':    da_r2,
        'AA_test_r2':    aa_r2,
        'UA_test_r2':    ua_r2,
        'fit_time':      fit_time,
    }


# ── Preprocessing for ablation variants ────────────────────────────────────────

def preprocess_traces_ablation(trace_list, mode='default'):
    """
    Return a list of trace-like dicts with modified preprocessing.
    Modes:
      'default'         — use stored I_corr_fwd/I_corr_rev (sg + asls)
      'no_smooth'       — raw * 1e6 directly, then asls
      'no_baseline'     — sg only, no asls
      'per_curve_norm'  — default then divide each trace by its L2 norm
      'global_norm'     — default then divide by train-set mean absolute current
      'fwd_only'        — default but mark as fwd-only
    """
    processed = []
    for t in trace_list:
        new_t = dict(t)  # shallow copy — V arrays are shared (read-only usage)

        if mode == 'default':
            pass  # already has I_corr_fwd, I_corr_rev

        elif mode == 'no_smooth':
            # Use raw * 1e6, then asls
            I_raw_fwd = t['I_raw_fwd'].copy()
            I_raw_rev = t['I_raw_rev'].copy()
            try:
                I_corr_fwd, _ = asls(I_raw_fwd)
                I_corr_rev, _ = asls(I_raw_rev)
            except Exception:
                I_corr_fwd = I_raw_fwd
                I_corr_rev = I_raw_rev
            new_t['I_corr_fwd'] = I_corr_fwd
            new_t['I_corr_rev'] = I_corr_rev
            # Rebuild raw_fwd/raw_rev for waveform consistency
            v_grid = t['v_grid']
            V_fwd  = t['V_fwd']
            V_rev  = t['V_rev']
            new_t['raw_fwd'] = np.array([I_corr_fwd[nearest_idx(V_fwd, v)] for v in v_grid], dtype=np.float32)
            new_t['raw_rev'] = np.array([I_corr_rev[nearest_idx(V_rev, v)] for v in v_grid], dtype=np.float32)

        elif mode == 'no_baseline':
            # sg only, no asls
            I_smooth_fwd = t['I_smooth_fwd'].copy()
            I_smooth_rev = t['I_smooth_rev'].copy()
            new_t['I_corr_fwd'] = I_smooth_fwd
            new_t['I_corr_rev'] = I_smooth_rev
            v_grid = t['v_grid']
            V_fwd  = t['V_fwd']
            V_rev  = t['V_rev']
            new_t['raw_fwd'] = np.array([I_smooth_fwd[nearest_idx(V_fwd, v)] for v in v_grid], dtype=np.float32)
            new_t['raw_rev'] = np.array([I_smooth_rev[nearest_idx(V_rev, v)] for v in v_grid], dtype=np.float32)

        elif mode == 'per_curve_norm':
            I_fwd = t['I_corr_fwd'].copy()
            I_rev = t['I_corr_rev'].copy()
            norm = np.linalg.norm(np.concatenate([I_fwd, I_rev]))
            if norm > 0:
                I_fwd = I_fwd / norm
                I_rev = I_rev / norm
            new_t['I_corr_fwd'] = I_fwd
            new_t['I_corr_rev'] = I_rev
            v_grid = t['v_grid']
            V_fwd  = t['V_fwd']
            V_rev  = t['V_rev']
            new_t['raw_fwd'] = np.array([I_fwd[nearest_idx(V_fwd, v)] for v in v_grid], dtype=np.float32)
            new_t['raw_rev'] = np.array([I_rev[nearest_idx(V_rev, v)] for v in v_grid], dtype=np.float32)

        elif mode == 'fwd_only':
            # Zero out reverse sweep
            new_t['I_corr_rev'] = np.zeros_like(t['I_corr_rev'])
            new_t['raw_rev']    = np.zeros_like(t['raw_rev'])

        processed.append(new_t)

    return processed


def preprocess_global_norm(train_list, test_list):
    """Divide by train-set mean absolute current."""
    all_I = np.concatenate([np.concatenate([t['I_corr_fwd'], t['I_corr_rev']]) for t in train_list])
    global_mean = float(np.mean(np.abs(all_I)))
    if global_mean == 0:
        global_mean = 1.0

    def normalize_list(trace_list):
        result = []
        for t in trace_list:
            new_t = dict(t)
            I_fwd = t['I_corr_fwd'] / global_mean
            I_rev = t['I_corr_rev'] / global_mean
            new_t['I_corr_fwd'] = I_fwd
            new_t['I_corr_rev'] = I_rev
            v_grid = t['v_grid']
            V_fwd  = t['V_fwd']
            V_rev  = t['V_rev']
            new_t['raw_fwd'] = np.array([I_fwd[nearest_idx(V_fwd, v)] for v in v_grid], dtype=np.float32)
            new_t['raw_rev'] = np.array([I_rev[nearest_idx(V_rev, v)] for v in v_grid], dtype=np.float32)
            result.append(new_t)
        return result

    return normalize_list(train_list), normalize_list(test_list)


# ── Compute base feature matrices ───────────────────────────────────────────────
print("Extracting feature matrices...")

# Raw waveform (used across many families)
X_raw_tr = feats_raw_waveform(train_traces)
X_raw_te = feats_raw_waveform(test_traces)
print(f"  raw_waveform: {X_raw_tr.shape}")

# Electrochemical
X_echem_tr = feats_electrochemical(train_traces)
X_echem_te = feats_electrochemical(test_traces)
print(f"  electrochemical: {X_echem_tr.shape}")

# Area
X_area_tr = feats_area(train_traces)
X_area_te = feats_area(test_traces)
print(f"  area: {X_area_tr.shape}")

# Derivative
X_deriv_tr = feats_derivative(train_traces)
X_deriv_te = feats_derivative(test_traces)
print(f"  derivative: {X_deriv_tr.shape}")

# Statistical
X_stat_tr = feats_statistical(train_traces)
X_stat_te = feats_statistical(test_traces)
print(f"  statistical: {X_stat_tr.shape}")

# All handcrafted
X_all_hc_tr = np.concatenate([X_echem_tr, X_area_tr, X_deriv_tr, X_stat_tr], axis=1)
X_all_hc_te = np.concatenate([X_echem_te, X_area_te, X_deriv_te, X_stat_te], axis=1)
print(f"  all_handcrafted: {X_all_hc_tr.shape}")

# ── PCA (fit on train only) ────────────────────────────────────────────────────
print("\nFitting PCA on training raw waveform...")
pca_scaler = StandardScaler()
X_raw_tr_sc = pca_scaler.fit_transform(X_raw_tr)
X_raw_te_sc = pca_scaler.transform(X_raw_te)

pca_full = PCA().fit(X_raw_tr_sc)
cum_var  = np.cumsum(pca_full.explained_variance_ratio_)

def fit_pca(n_components, X_tr_sc, X_te_sc):
    pca = PCA(n_components=n_components).fit(X_tr_sc)
    return pca.transform(X_tr_sc).astype(np.float32), pca.transform(X_te_sc).astype(np.float32), pca

pca_cache = {}
for n_comp in [2, 3, 5, 10, 15]:
    if n_comp <= X_raw_tr_sc.shape[1]:
        Xtr, Xte, p = fit_pca(n_comp, X_raw_tr_sc, X_raw_te_sc)
        pca_cache[f'pca_{n_comp}'] = (Xtr, Xte, p)

# Variance threshold PCA
for var_thresh, tag in [(0.90, 'var90'), (0.95, 'var95'), (0.99, 'var99')]:
    n_comp = int(np.searchsorted(cum_var, var_thresh)) + 1
    n_comp = min(n_comp, X_raw_tr_sc.shape[1])
    Xtr, Xte, p = fit_pca(n_comp, X_raw_tr_sc, X_raw_te_sc)
    pca_cache[f'pca_{tag}'] = (Xtr, Xte, p)
    print(f"  PCA {tag}: {n_comp} components ({cum_var[n_comp-1]:.4f} variance)")

# ── Spline features (V_COMMON grid, fit labels on train) ──────────────────────
print("\nComputing spline features...")
spline_cache = {}
v_spline_grid = V_COMMON
for n_knots in [5, 8, 10, 15, 20]:
    Xtr = feats_spline(train_traces, n_knots, v_spline_grid)
    Xte = feats_spline(test_traces,  n_knots, v_spline_grid)
    spline_cache[n_knots] = (Xtr, Xte)
    print(f"  spline_{n_knots}: {Xtr.shape}")

# ── Wavelet features ────────────────────────────────────────────────────────────
wavelet_cache = {}
if HAS_PYWT and not args.skip_wavelets:
    print("\nComputing wavelet features...")
    # Energy only
    Xtr_we = feats_wavelet(train_traces, wavelet=args.wavelet, energy_only=True)
    Xte_we = feats_wavelet(test_traces,  wavelet=args.wavelet, energy_only=True)
    wavelet_cache['energy'] = (Xtr_we, Xte_we)
    print(f"  wavelet_energy: {Xtr_we.shape}")

    for top_n in [5, 10, 20]:
        top_idx = get_wavelet_top_n_idx(train_traces, top_n, wavelet=args.wavelet)
        Xtr_wt = feats_wavelet(train_traces, wavelet=args.wavelet, top_n=top_n, top_n_idx=top_idx)
        Xte_wt = feats_wavelet(test_traces,  wavelet=args.wavelet, top_n=top_n, top_n_idx=top_idx)
        wavelet_cache[f'top{top_n}'] = (Xtr_wt, Xte_wt, top_idx)
        print(f"  wavelet_top{top_n}: {Xtr_wt.shape}")

    # energy + top10
    top_idx_10 = wavelet_cache['top10'][2]
    Xtr_e10 = np.concatenate([Xtr_we, wavelet_cache['top10'][0]], axis=1)
    Xte_e10 = np.concatenate([Xte_we, wavelet_cache['top10'][1]], axis=1)
    wavelet_cache['energy_plus_top10'] = (Xtr_e10, Xte_e10)
    print(f"  wavelet_energy_plus_top10: {Xtr_e10.shape}")

# ── Feature selection on raw waveform ─────────────────────────────────────────
print("\nFitting feature selectors on raw waveform training data...")
sel_cache = {}

# F-statistic SelectKBest (average across 3 targets)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    f_scores_list = []
    for col in range(y_tr.shape[1]):
        f_scores, _ = f_regression(X_raw_tr_sc, y_tr[:, col])
        f_scores_list.append(f_scores)
    mean_f_scores = np.mean(f_scores_list, axis=0)

for k in [5, 10, 20]:
    top_idx = np.argsort(mean_f_scores)[-k:][::-1]
    Xtr = X_raw_tr[:, top_idx].astype(np.float32)
    Xte = X_raw_te[:, top_idx].astype(np.float32)
    sel_cache[f'fstat_k{k}'] = (Xtr, Xte)
    print(f"  select_k{k} (f-stat): {Xtr.shape}")

# Mutual information (average across 3 targets)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mi_scores_list = []
    for col in range(y_tr.shape[1]):
        mi = mutual_info_regression(X_raw_tr_sc, y_tr[:, col], random_state=42)
        mi_scores_list.append(mi)
    mean_mi_scores = np.mean(mi_scores_list, axis=0)

for k in [5, 10, 20]:
    top_idx = np.argsort(mean_mi_scores)[-k:][::-1]
    Xtr = X_raw_tr[:, top_idx].astype(np.float32)
    Xte = X_raw_te[:, top_idx].astype(np.float32)
    sel_cache[f'mi_k{k}'] = (Xtr, Xte)
    print(f"  mi_k{k}: {Xtr.shape}")

# Lasso feature selection
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    lasso = MultiTaskLassoCV(cv=5, max_iter=10000, n_jobs=-1)
    lasso.fit(X_raw_tr_sc, y_tr)
    lasso_mask = np.any(lasso.coef_ != 0, axis=0)
    if lasso_mask.sum() == 0:
        lasso_mask = np.ones(X_raw_tr.shape[1], dtype=bool)
    Xtr_lasso = X_raw_tr[:, lasso_mask].astype(np.float32)
    Xte_lasso = X_raw_te[:, lasso_mask].astype(np.float32)
    sel_cache['lasso'] = (Xtr_lasso, Xte_lasso)
    print(f"  lasso_selected: {Xtr_lasso.shape} ({lasso_mask.sum()} features)")

# ElasticNet: per-target, union of selected features
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.linear_model import ElasticNetCV
    en_mask = np.zeros(X_raw_tr_sc.shape[1], dtype=bool)
    for col in range(y_tr.shape[1]):
        en = ElasticNetCV(cv=5, max_iter=5000, n_jobs=-1)
        en.fit(X_raw_tr_sc, y_tr[:, col])
        en_mask |= (en.coef_ != 0)
    if en_mask.sum() == 0:
        en_mask = np.ones(X_raw_tr.shape[1], dtype=bool)
    Xtr_en = X_raw_tr[:, en_mask].astype(np.float32)
    Xte_en = X_raw_te[:, en_mask].astype(np.float32)
    sel_cache['elasticnet'] = (Xtr_en, Xte_en)
    print(f"  elasticnet_selected: {Xtr_en.shape} ({en_mask.sum()} features)")

# ── Autoencoder (fit on train only) ───────────────────────────────────────────
ae_cache = {}
if not args.skip_autoencoder:
    print("\nTraining autoencoders on raw waveform training data...")
    X_ae_tr = X_raw_tr.astype(np.float32)
    X_ae_te = X_raw_te.astype(np.float32)
    ae_scaler = StandardScaler()
    X_ae_tr_sc = ae_scaler.fit_transform(X_ae_tr).astype(np.float32)
    X_ae_te_sc = ae_scaler.transform(X_ae_te).astype(np.float32)

    n_ae_input = X_ae_tr_sc.shape[1]
    for bottleneck in [2, 3, 5, 8, 10]:
        np.random.seed(123)
        ae = NumpyAutoencoder([n_ae_input, 64, 32, bottleneck, 32, 64, n_ae_input])
        ae.train(X_ae_tr_sc, epochs=2000, lr=0.001)
        encoded_tr = ae.encode(X_ae_tr_sc).astype(np.float32)
        encoded_te = ae.encode(X_ae_te_sc).astype(np.float32)
        ae_cache[bottleneck] = (encoded_tr, encoded_te)
        print(f"  autoencoder_{bottleneck}: {encoded_tr.shape}")

# ── Preprocessing ablation ─────────────────────────────────────────────────────
print("\nBuilding preprocessing ablation variants...")
ablation_cache = {}

ablation_modes = [
    ('default',      'default'),
    ('no_smooth',    'no_smooth'),
    ('no_baseline',  'no_baseline'),
    ('per_curve_norm', 'per_curve_norm'),
    ('fwd_only',     'fwd_only'),
]

for abl_name, mode in ablation_modes:
    tr_abl = preprocess_traces_ablation(train_traces, mode=mode)
    te_abl = preprocess_traces_ablation(test_traces,  mode=mode)
    X_abl_tr = feats_electrochemical(tr_abl)
    X_abl_te = feats_electrochemical(te_abl)
    ablation_cache[abl_name] = (X_abl_tr, X_abl_te)
    print(f"  ablation_{abl_name}: {X_abl_tr.shape}")

# global_norm is special: normalization factor depends on training set
tr_gnorm, te_gnorm = preprocess_global_norm(train_traces, test_traces)
X_gnorm_tr = feats_electrochemical(tr_gnorm)
X_gnorm_te = feats_electrochemical(te_gnorm)
ablation_cache['global_norm'] = (X_gnorm_tr, X_gnorm_te)
print(f"  ablation_global_norm: {X_gnorm_tr.shape}")

# ── Build experiment registry ───────────────────────────────────────────────────
# Tuple: (exp_name, family_num, family_name, feature_set_name, X_tr, X_te, n_features, meta_str)
experiments = []

def add_exp(name, fam_num, fam_name, feat_name, X_tr, X_te, meta=''):
    if args.families and not any(f in fam_name for f in args.families):
        return
    n_feats = X_tr.shape[1] if X_tr.ndim == 2 else 0
    experiments.append((name, fam_num, fam_name, feat_name, X_tr, X_te, n_feats, meta))

# Family 1 — Raw waveform baseline
add_exp('raw_100', 1, 'raw_waveform', 'raw_100', X_raw_tr, X_raw_te, 'raw 100-pt waveform (50 fwd + 50 rev)')

# Family 2 — PCA features
for key, (Xtr, Xte, pca_obj) in pca_cache.items():
    var_sum = float(np.sum(pca_obj.explained_variance_ratio_))
    meta = f"n_components={Xtr.shape[1]}, explained_var={var_sum:.4f}"
    add_exp(key, 2, 'pca', key, Xtr, Xte, meta)

# Family 3 — Electrochemical peak features
add_exp('echem_peaks', 3, 'electrochemical', 'echem_peaks', X_echem_tr, X_echem_te, 'electrochemical peak features')

# Family 4 — Area features
add_exp('area_only', 4, 'area', 'area_only', X_area_tr, X_area_te, 'area features only')
add_exp('echem_plus_area', 4, 'area', 'echem_plus_area',
        np.concatenate([X_echem_tr, X_area_tr], axis=1),
        np.concatenate([X_echem_te, X_area_te], axis=1), 'echem + area')

# Family 5 — Derivative features
add_exp('deriv_only', 5, 'derivative', 'deriv_only', X_deriv_tr, X_deriv_te, 'derivative features only')
add_exp('echem_plus_deriv', 5, 'derivative', 'echem_plus_deriv',
        np.concatenate([X_echem_tr, X_deriv_tr], axis=1),
        np.concatenate([X_echem_te, X_deriv_te], axis=1), 'echem + derivative')
add_exp('echem_plus_area_plus_deriv', 5, 'derivative', 'echem_plus_area_plus_deriv',
        np.concatenate([X_echem_tr, X_area_tr, X_deriv_tr], axis=1),
        np.concatenate([X_echem_te, X_area_te, X_deriv_te], axis=1), 'echem + area + derivative')

# Family 6 — Statistical shape features
add_exp('statistical_only', 6, 'statistical', 'statistical_only', X_stat_tr, X_stat_te, 'statistical features only')
add_exp('echem_plus_statistical', 6, 'statistical', 'echem_plus_statistical',
        np.concatenate([X_echem_tr, X_stat_tr], axis=1),
        np.concatenate([X_echem_te, X_stat_te], axis=1), 'echem + statistical')
add_exp('all_handcrafted', 6, 'statistical', 'all_handcrafted', X_all_hc_tr, X_all_hc_te, 'echem + area + derivative + statistical')

# Family 7 — Spline features
for n_knots, (Xtr, Xte) in spline_cache.items():
    add_exp(f'spline_{n_knots}', 7, 'spline', f'spline_{n_knots}', Xtr, Xte, f'{n_knots} interior knots')

# Family 8 — Wavelet features
if HAS_PYWT and not args.skip_wavelets:
    Xtr_we, Xte_we = wavelet_cache['energy']
    add_exp('wavelet_energy', 8, 'wavelet', 'wavelet_energy', Xtr_we, Xte_we, f'wavelet={args.wavelet}, energy only')
    for top_n in [5, 10, 20]:
        Xtr_wt, Xte_wt = wavelet_cache[f'top{top_n}'][:2]
        add_exp(f'wavelet_top{top_n}', 8, 'wavelet', f'wavelet_top{top_n}', Xtr_wt, Xte_wt, f'wavelet={args.wavelet}, top {top_n} coefficients')
    Xtr_e10, Xte_e10 = wavelet_cache['energy_plus_top10']
    add_exp('wavelet_energy_plus_top10', 8, 'wavelet', 'wavelet_energy_plus_top10', Xtr_e10, Xte_e10, f'wavelet={args.wavelet}, energy + top10')

# Family 9 — Feature selection on raw waveform
for k in [5, 10, 20]:
    Xtr, Xte = sel_cache[f'fstat_k{k}']
    add_exp(f'select_k{k}', 9, 'feature_selection', f'select_k{k}', Xtr, Xte, f'F-statistic top {k}')
    Xtr, Xte = sel_cache[f'mi_k{k}']
    add_exp(f'mi_k{k}', 9, 'feature_selection', f'mi_k{k}', Xtr, Xte, f'mutual info top {k}')

Xtr_l, Xte_l = sel_cache['lasso']
add_exp('lasso_selected', 9, 'feature_selection', 'lasso_selected', Xtr_l, Xte_l, 'MultiTaskLassoCV selected')
Xtr_en2, Xte_en2 = sel_cache['elasticnet']
add_exp('elasticnet_selected', 9, 'feature_selection', 'elasticnet_selected', Xtr_en2, Xte_en2, 'ElasticNetCV union selected')

# Family 10 — Combined feature sets
for n_comp in [3, 5, 10]:
    pca_key = f'pca_{n_comp}'
    if pca_key in pca_cache:
        Xtr_pca, Xte_pca, _ = pca_cache[pca_key]
        add_exp(f'echem_plus_pca{n_comp}', 10, 'combined', f'echem_plus_pca{n_comp}',
                np.concatenate([X_echem_tr, Xtr_pca], axis=1),
                np.concatenate([X_echem_te, Xte_pca], axis=1), f'echem + PCA({n_comp})')

if 'pca_5' in pca_cache:
    Xtr_p5, Xte_p5, _ = pca_cache['pca_5']
    add_exp('echem_area_plus_pca5', 10, 'combined', 'echem_area_plus_pca5',
            np.concatenate([X_echem_tr, X_area_tr, Xtr_p5], axis=1),
            np.concatenate([X_echem_te, X_area_te, Xte_p5], axis=1), 'echem + area + PCA(5)')
    add_exp('all_handcrafted_plus_pca5', 10, 'combined', 'all_handcrafted_plus_pca5',
            np.concatenate([X_all_hc_tr, Xtr_p5], axis=1),
            np.concatenate([X_all_hc_te, Xte_p5], axis=1), 'all_handcrafted + PCA(5)')

if 'pca_10' in pca_cache:
    Xtr_p10, Xte_p10, _ = pca_cache['pca_10']
    add_exp('all_handcrafted_plus_pca10', 10, 'combined', 'all_handcrafted_plus_pca10',
            np.concatenate([X_all_hc_tr, Xtr_p10], axis=1),
            np.concatenate([X_all_hc_te, Xte_p10], axis=1), 'all_handcrafted + PCA(10)')

if HAS_PYWT and not args.skip_wavelets and 'energy' in wavelet_cache:
    Xtr_we2, Xte_we2 = wavelet_cache['energy']
    add_exp('all_handcrafted_plus_wavelet_energy', 10, 'combined', 'all_handcrafted_plus_wavelet_energy',
            np.concatenate([X_all_hc_tr, Xtr_we2], axis=1),
            np.concatenate([X_all_hc_te, Xte_we2], axis=1), 'all_handcrafted + wavelet energy')

Xtr_s10, Xte_s10 = spline_cache[10]
add_exp('all_handcrafted_plus_spline10', 10, 'combined', 'all_handcrafted_plus_spline10',
        np.concatenate([X_all_hc_tr, Xtr_s10], axis=1),
        np.concatenate([X_all_hc_te, Xte_s10], axis=1), 'all_handcrafted + spline(10)')

# Family 11 — Autoencoder
if not args.skip_autoencoder:
    for bottleneck, (Xtr_ae, Xte_ae) in ae_cache.items():
        add_exp(f'autoencoder_{bottleneck}', 11, 'autoencoder', f'autoencoder_{bottleneck}',
                Xtr_ae, Xte_ae, f'autoencoder bottleneck={bottleneck}')

# Family 12 — Preprocessing ablation
for abl_name, (Xtr_abl, Xte_abl) in ablation_cache.items():
    add_exp(f'ablation_{abl_name}', 12, 'ablation', f'ablation_{abl_name}',
            Xtr_abl, Xte_abl, f'preprocessing ablation: {abl_name}')

print(f"\nTotal experiments registered: {len(experiments)}")

# ── Define models ───────────────────────────────────────────────────────────────
models = {
    'ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
    'rf':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'gbr':   MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
}

models_to_run = list(models.keys())
if not args.skip_ann:
    models_to_run.append('ann')

# ── Main experiment loop ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Running {len(experiments)} experiments × {len(models_to_run)} models = {len(experiments) * len(models_to_run)} total fits")
print(f"{'='*70}\n")

all_results = []

for exp_idx, (exp_name, family_num, family_name, feat_name, X_tr, X_te, n_feats, meta) in enumerate(experiments):
    if not sanity_check(X_tr, X_te, exp_name, y_tr, y_te):
        continue

    print(f"  [{family_num}] {exp_name} ({n_feats} features)")

    for model_name in models_to_run:
        try:
            if model_name == 'ann':
                metrics = eval_ann(X_tr, X_te, y_tr, y_te, args.n_seeds, args.ann_epochs)
            else:
                metrics = eval_sklearn(copy.deepcopy(models[model_name]), X_tr, X_te, y_tr, y_te, args.cv_folds)

            cv_str  = f"{metrics['mean_cv_rmse']:.2f}" if not np.isnan(metrics['mean_cv_rmse']) else "nan"
            tst_str = f"{metrics['mean_test_rmse']:.2f}"
            print(f"    {model_name:<8} cv_rmse={cv_str}  test_rmse={tst_str} µM")

            row = {
                'experiment_name': f"{exp_name}_{model_name}",
                'family':           family_num,
                'feature_family':   family_name,
                'feature_set':      feat_name,
                'feature_dim':      n_feats,
                'model':            model_name,
                'preprocessing':    meta,
                **metrics,
            }
            all_results.append(row)
        except Exception as exc:
            print(f"    {model_name:<8} ERROR: {exc}")

# ── Save results ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Saving results...")

results_df = pd.DataFrame(all_results)
results_path = os.path.join(OUT_DIR, 'results.csv')
results_df.to_csv(results_path, index=False)
print(f"  Saved {results_path}")

# Ranked by mean_cv_rmse (NaN last), then mean_test_rmse
ranked_df = results_df.sort_values(
    ['mean_cv_rmse', 'mean_test_rmse'],
    ascending=[True, True],
    na_position='last'
)
ranked_path = os.path.join(OUT_DIR, 'ranked_results.csv')
ranked_df.to_csv(ranked_path, index=False)
print(f"  Saved {ranked_path}")

# Best model summary
summary_path = os.path.join(OUT_DIR, 'best_model_summary.txt')
with open(summary_path, 'w') as f:
    f.write("TOP 10 EXPERIMENTS BY MEAN CV RMSE\n")
    f.write("=" * 80 + "\n")
    top10 = ranked_df.head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        cv_str  = f"{row['mean_cv_rmse']:.3f}" if not np.isnan(row['mean_cv_rmse']) else "nan"
        tst_str = f"{row['mean_test_rmse']:.3f}"
        f.write(f"\n{rank:2}. {row['experiment_name']}\n")
        f.write(f"    Feature family : {row['feature_family']}\n")
        f.write(f"    Feature set    : {row['feature_set']}\n")
        f.write(f"    Feature dim    : {row['feature_dim']}\n")
        f.write(f"    Model          : {row['model']}\n")
        f.write(f"    Mean CV RMSE   : {cv_str} µM\n")
        f.write(f"    Mean Test RMSE : {tst_str} µM\n")
        f.write(f"    DA / AA / UA test RMSE: {row['DA_test_rmse']:.2f} / {row['AA_test_rmse']:.2f} / {row['UA_test_rmse']:.2f} µM\n")
        f.write(f"    Preprocessing  : {row['preprocessing']}\n")
print(f"  Saved {summary_path}")

# ── Plotting ────────────────────────────────────────────────────────────────────
if HAS_MPL and len(results_df) > 0:
    print("\nGenerating plots...")

    # Color palettes
    family_colors = {
        1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
        5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f',
        9: '#bcbd22', 10: '#17becf', 11: '#aec7e8', 12: '#ffbb78',
    }
    model_colors = {'ridge': '#1f77b4', 'rf': '#ff7f0e', 'gbr': '#2ca02c', 'ann': '#d62728'}

    # Plot 1: Top 20 experiments by mean_test_rmse (bar chart, one color per family)
    try:
        top20 = results_df.nsmallest(20, 'mean_test_rmse')
        fig, ax = plt.subplots(figsize=(14, 6))
        bar_colors = [family_colors.get(int(f), '#888888') for f in top20['family']]
        bars = ax.bar(range(len(top20)), top20['mean_test_rmse'], color=bar_colors)
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(top20['experiment_name'], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean Test RMSE (µM)')
        ax.set_title('Top 20 Experiments by Mean Test RMSE')
        # Legend for families
        from matplotlib.patches import Patch
        legend_handles = [Patch(color=c, label=f'Family {f}') for f, c in family_colors.items()
                          if f in top20['family'].values]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'top20_test_rmse.png'), dpi=150)
        plt.close()
        print("  Saved top20_test_rmse.png")
    except Exception as exc:
        print(f"  Plot 1 failed: {exc}")

    # Plot 2: Scatter mean_cv_rmse vs mean_test_rmse, colored by model type
    try:
        scatter_df = results_df.dropna(subset=['mean_cv_rmse', 'mean_test_rmse'])
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name, grp in scatter_df.groupby('model'):
            ax.scatter(grp['mean_cv_rmse'], grp['mean_test_rmse'],
                       c=model_colors.get(model_name, '#888888'),
                       label=model_name, alpha=0.6, s=30)
        ax.set_xlabel('Mean CV RMSE (µM)')
        ax.set_ylabel('Mean Test RMSE (µM)')
        ax.set_title('CV RMSE vs Test RMSE by Model Type')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'cv_vs_test_rmse_scatter.png'), dpi=150)
        plt.close()
        print("  Saved cv_vs_test_rmse_scatter.png")
    except Exception as exc:
        print(f"  Plot 2 failed: {exc}")

    # Plot 3: Feature dim vs mean_test_rmse for PCA experiments (one line per model)
    try:
        pca_df = results_df[results_df['feature_family'] == 'pca'].copy()
        if len(pca_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            for model_name, grp in pca_df.groupby('model'):
                grp_sorted = grp.sort_values('feature_dim')
                ax.plot(grp_sorted['feature_dim'], grp_sorted['mean_test_rmse'],
                        marker='o', label=model_name, color=model_colors.get(model_name, '#888888'))
            ax.set_xlabel('Number of PCA Components')
            ax.set_ylabel('Mean Test RMSE (µM)')
            ax.set_title('PCA: Feature Dimension vs Test RMSE')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, 'pca_dim_vs_test_rmse.png'), dpi=150)
            plt.close()
            print("  Saved pca_dim_vs_test_rmse.png")
    except Exception as exc:
        print(f"  Plot 3 failed: {exc}")

    # Plot 4: PCA explained variance (cumulative)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_comps_show = min(len(pca_full.explained_variance_ratio_), 30)
        ax.plot(range(1, n_comps_show + 1),
                np.cumsum(pca_full.explained_variance_ratio_[:n_comps_show]),
                marker='o', markersize=4)
        ax.axhline(0.90, color='r', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(0.95, color='g', linestyle='--', alpha=0.5, label='95%')
        ax.axhline(0.99, color='b', linestyle='--', alpha=0.5, label='99%')
        ax.set_xlabel('Number of PCA Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA Explained Variance (raw waveform)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'pca_explained_variance.png'), dpi=150)
        plt.close()
        print("  Saved pca_explained_variance.png")
    except Exception as exc:
        print(f"  Plot 4 failed: {exc}")

# ── Final recommendations ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL RECOMMENDATIONS")
print("=" * 70)

if len(results_df) == 0:
    print("No results to analyze.")
else:
    # Best overall by test RMSE
    best_overall = results_df.loc[results_df['mean_test_rmse'].idxmin()]
    print(f"Best overall by test RMSE      : {best_overall['experiment_name']} — {best_overall['mean_test_rmse']:.2f} µM")

    # Best under 5 features
    under5 = results_df[results_df['feature_dim'] <= 5]
    if len(under5) > 0:
        b5 = under5.loc[under5['mean_test_rmse'].idxmin()]
        print(f"Best under 5 features          : {b5['experiment_name']} — {b5['mean_test_rmse']:.2f} µM")
    else:
        print("Best under 5 features          : no experiments with ≤5 features")

    # Best under 10 features
    under10 = results_df[results_df['feature_dim'] <= 10]
    if len(under10) > 0:
        b10 = under10.loc[under10['mean_test_rmse'].idxmin()]
        print(f"Best under 10 features         : {b10['experiment_name']} — {b10['mean_test_rmse']:.2f} µM")
    else:
        print("Best under 10 features         : no experiments with ≤10 features")

    # Best interpretable (echem, area, deriv, stat)
    interp_families = ['electrochemical', 'area', 'derivative', 'statistical']
    interp_df = results_df[results_df['feature_family'].isin(interp_families)]
    if len(interp_df) > 0:
        bi = interp_df.loc[interp_df['mean_test_rmse'].idxmin()]
        print(f"Best interpretable             : {bi['experiment_name']} — {bi['mean_test_rmse']:.2f} µM")
    else:
        print("Best interpretable             : no interpretable experiments run")

    # Best non-ANN
    non_ann = results_df[results_df['model'] != 'ann']
    if len(non_ann) > 0:
        bna = non_ann.loc[non_ann['mean_test_rmse'].idxmin()]
        print(f"Best non-ANN                   : {bna['experiment_name']} — {bna['mean_test_rmse']:.2f} µM")

    # Raw waveform baseline
    raw_rows = results_df[results_df['feature_set'] == 'raw_100']
    if len(raw_rows) > 0:
        raw_rmse = float(raw_rows['mean_test_rmse'].min())
        print(f"Raw waveform test RMSE         : {raw_rmse:.2f} µM (baseline)")
    else:
        raw_rmse = None
        print("Raw waveform test RMSE         : not computed")

    # PCA + echem vs PCA alone
    pca_alone  = results_df[results_df['feature_family'] == 'pca']['mean_test_rmse'].min() if len(results_df[results_df['feature_family'] == 'pca']) > 0 else None
    pca_echem  = results_df[results_df['feature_set'].str.startswith('echem_plus_pca', na=False)]['mean_test_rmse'].min() if len(results_df[results_df['feature_set'].str.startswith('echem_plus_pca', na=False)]) > 0 else None
    if pca_alone is not None and pca_echem is not None:
        improved = pca_echem < pca_alone
        print(f"PCA + echem vs PCA alone       : {'improved' if improved else 'not improved'} ({pca_echem:.2f} vs {pca_alone:.2f} µM)")

    # All handcrafted vs raw
    hc_rows = results_df[results_df['feature_set'] == 'all_handcrafted']
    if len(hc_rows) > 0 and raw_rmse is not None:
        hc_rmse = float(hc_rows['mean_test_rmse'].min())
        diff = hc_rmse - raw_rmse
        competitive = abs(diff) < 5.0 or hc_rmse <= raw_rmse
        print(f"All handcrafted vs raw         : {'competitive' if competitive else 'not competitive'} ({diff:+.2f} µM)")

print(f"\nResults saved to: {os.path.abspath(OUT_DIR)}/")
print(f"  results.csv, ranked_results.csv, best_model_summary.txt")
if HAS_MPL:
    print(f"  Plots: top20_test_rmse.png, cv_vs_test_rmse_scatter.png,")
    print(f"         pca_dim_vs_test_rmse.png, pca_explained_variance.png")
