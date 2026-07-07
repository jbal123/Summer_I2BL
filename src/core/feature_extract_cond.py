"""
Full feature extraction for one or all condition directories.

Implements the 44-feature ANN input vector recommended in Feature_Engineering_Digest.md:
  (1)  SWV  : {Ip, Ep, FWHM, Area, k_pre, AI, Ip_fwd, Ip_rev, fwd_rev_ratio} × 3 analytes
  (2)  DPV  : {Ip, Ep, Area} × 3 analytes
  (3)  CV   : {Ipa, Ep_a, Area, FWHM, k_pre} per analyte + DA cathodic {Ipc, Ep_c, deltaEp, E12, Ipa_Ipc} + 2 baselines
  (4)  Ratios: {Ip_DA/Ip_AA, Ip_DA/Ip_UA, Ip_UA/Ip_AA, ΔEp_DA-AA, ΔEp_UA-DA} from SWV + DPV
  (5)  CA-GC: {I0, Iss, Iss/I0, cott_slope, cott_R2, Q_early, Q_late, Q_late/Q_early,
               CE(1s), CE(10s), CE_ss, AF, η(10s)}

Output: one row per (condition, electrode). Each electrode is its own independent
feature vector — no averaging across electrodes.
SWV uses rep 3 only (reps 1 and 2 discarded).

Usage:
    python feature_extract_cond.py                              # all conditions, all 3 electrodes
    python feature_extract_cond.py --electrodes 1 3             # all conditions, electrodes 1 & 3
    python feature_extract_cond.py outputs_dataset_full/1.0/    # single condition, all electrodes
    python feature_extract_cond.py outputs_dataset_full/1.0/ --electrodes 1 5
"""

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import sys
import argparse
import glob as _glob
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapezoid
from scipy.interpolate import splrep, interp1d
import warnings
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ── Parameters ────────────────────────────────────────────────────────────────
SMOOTH_W       = 21
SMOOTH_P       = 3
ASLS_LAM       = 1e7
ASLS_P         = 0.005
SCAN_RATE      = 0.02          # V/s for CV
ALL_ELECTRODES = ['i1', 'i3', 'i5']
GC_MAP         = {'i1': ('i1', 'i2'), 'i3': ('i3', 'i4'), 'i5': ('i5', 'i6')}

WINDOWS = {
    # AA: scan starts ~0.104 V so the original (-0.10, 0.12) window had <5 points.
    # Shifted right to sit within the scan range. Note: AA and DA peaks overlap in
    # this region — SWV AA features will be noisy/partially contaminated by DA signal.
    # DA and UA windows are unchanged.
    'AA': ( 0.10,  0.20),
    'DA': ( 0.12,  0.32),
    'UA': ( 0.30,  0.55),
}
V_BASE_LOW  = -0.05
V_BASE_HIGH =  0.70


# ── Preprocessing utilities ───────────────────────────────────────────────────
def asls(I, lam=ASLS_LAM, p=ASLS_P, n_iter=10):
    n = len(I)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        D = lam * D.T @ D
        w = np.ones(n)
        for _ in range(n_iter):
            W = diags(w, 0)
            bl = spsolve(W + D, w * I)
            w = np.where(I > bl, p, 1 - p)
    return I - bl, bl


def sg(arr):
    return savgol_filter(arr, SMOOTH_W, SMOOTH_P)


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def linear_baseline(V, I, v_lo, v_hi):
    mask = (V >= v_lo) & (V <= v_hi)
    Vw, Iw = V[mask], I[mask]
    if len(Vw) < 4:
        return Vw, Iw
    slope = (Iw[-1] - Iw[0]) / (Vw[-1] - Vw[0]) if (Vw[-1] - Vw[0]) != 0 else 0
    bl = Iw[0] + slope * (Vw - Vw[0])
    return Vw, Iw - bl


# ── File discovery helpers ─────────────────────────────────────────────────────
def _find_file(base, pattern):
    matches = _glob.glob(os.path.join(base, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {base}")
    return matches[0]


def parse_labels(base):
    """Extract DA/AA/UA concentrations from SWV filename."""
    swv1 = _find_file(base, 'SWV_*_1.txt')
    fname = os.path.basename(swv1)
    m = re.search(r'_DA(\d+)_AA(\d+)_UA(\d+)_', fname)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Could not parse labels from filename: {fname}")


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_cv_n(base):
    if _glob.glob(os.path.join(base, 'CV_Norm_Cond*.txt')):
        path = _find_file(base, 'CV_Norm_Cond*.txt')
    else:
        path = os.path.join(base, 'CV_N.txt')
    df = pd.read_csv(path, header=None, names=['V', 'i1', 'i3', 'i5', 'i7'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df['V'].values, {e: df[e].values for e in ALL_ELECTRODES}


def load_cv_gc(base):
    path = _find_file(base, 'CV_GC_Cond*.txt')
    df = pd.read_csv(path, header=None,
                     names=['V', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return (df['V'].values,
            {e: df[e].values for e in ['i1', 'i3', 'i5']},
            {e: df[e].values for e in ['i2', 'i4', 'i6']})


def load_ca_norm(base):
    if _glob.glob(os.path.join(base, 'CA_Norm_Cond*.txt')):
        path = _find_file(base, 'CA_Norm_Cond*.txt')
    else:
        path = os.path.join(base, 'CA_Norm.txt')
    df = pd.read_csv(path, header=None, names=['t', 'i1', 'i3', 'i5', 'i7'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df['t'].values, {e: df[e].values for e in ALL_ELECTRODES}


def load_ca_gc(base):
    path = _find_file(base, 'CA_GC_Cond*.txt')
    df = pd.read_csv(path, header=None,
                     names=['t', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return (df['t'].values,
            {e: df[e].values for e in ['i1', 'i3', 'i5']},
            {e: df[e].values for e in ['i2', 'i4', 'i6']})


def load_dpv(base):
    if _glob.glob(os.path.join(base, 'DPV_Cond*.txt')):
        path = _find_file(base, 'DPV_Cond*.txt')
    else:
        path = os.path.join(base, 'DPV.txt')
    df = pd.read_csv(path, header=None, names=['V', 'i1', 'i3', 'i5', 'i7'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df.sort_values('V').reset_index(drop=True)
    return df['V'].values, {e: df[e].values for e in ALL_ELECTRODES}


SWV_COLS = ['V',
            'i1d', 'i1f', 'i1r',
            'i3d', 'i3f', 'i3r',
            'i5d', 'i5f', 'i5r',
            'i7d', 'i7f', 'i7r']


def load_swv(base):
    """Load SWV replicate 3 only (reps 1 and 2 discarded)."""
    rep3 = _find_file(base, 'SWV_*_3.txt')
    d = pd.read_csv(rep3, header=None, names=SWV_COLS)
    d = d.apply(pd.to_numeric, errors='coerce').dropna()
    return d['V'].values, d


# ── Peak extraction helpers ───────────────────────────────────────────────────
def peak_in_window(V, I_corr, v_lo, v_hi, direction='max'):
    mask = (V >= v_lo) & (V <= v_hi)
    if mask.sum() < 5:
        return None, np.nan, np.nan
    Vw, Iw = linear_baseline(V, I_corr, v_lo, v_hi)
    if direction == 'min':
        Iw = -Iw
    # Require 10% relative prominence — no argmax fallback.
    # If no peak clears this bar the window has no real faradaic feature.
    min_prom = max(Iw.max() * 0.10, 1e-9)
    peaks, props = find_peaks(Iw, prominence=min_prom)
    if len(peaks) == 0:
        return None, np.nan, np.nan
    pidx_w = peaks[np.argmax(Iw[peaks])]
    full_indices = np.where(mask)[0]
    pidx_full = full_indices[pidx_w]
    sign = 1 if direction == 'max' else -1
    return pidx_full, V[pidx_full], sign * I_corr[pidx_full]


def peak_shape_features(V, I_corr, pidx, v_lo, v_hi):
    Vw, Iw = linear_baseline(V, I_corr, v_lo, v_hi)
    full_indices = np.where((V >= v_lo) & (V <= v_hi))[0]
    if pidx not in full_indices:
        return dict(FWHM=np.nan, Area=np.nan, k_pre=np.nan, AI=np.nan)

    local_idx = np.where(full_indices == pidx)[0][0]
    Ip = Iw[local_idx]
    if Ip <= 0:
        # Linear baseline over-corrects when peak is near a window boundary.
        # Fall back to the AsLS-corrected signal without the secondary linear subtract.
        Iw = I_corr[full_indices]
        Ip = float(Iw[local_idx])
        if Ip <= 0:
            return dict(FWHM=np.nan, Area=np.nan, k_pre=np.nan, AI=np.nan)

    half  = Ip / 2.0
    left  = local_idx
    while left > 0 and Iw[left] > half:
        left -= 1
    right = local_idx
    while right < len(Iw) - 1 and Iw[right] > half:
        right += 1

    FWHM   = float(Vw[right] - Vw[left])
    W_pre  = float(Vw[local_idx] - Vw[left])
    W_post = float(Vw[right] - Vw[local_idx])
    AI     = (W_pre / W_post) if W_post > 0 else np.nan
    Area   = float(trapezoid(np.clip(Iw[left:right + 1], 0, None), Vw[left:right + 1]))

    onset = local_idx
    while onset > 0 and Iw[onset] > 0.10 * Ip:
        onset -= 1
    k_pre = np.nan
    if local_idx - onset > 2:
        k_pre = float(np.polyfit(Vw[onset:local_idx], Iw[onset:local_idx], 1)[0])

    return dict(FWHM=FWHM, Area=Area, k_pre=k_pre, AI=AI)


# ── Feature extraction per technique ─────────────────────────────────────────
def extract_swv(V, swv_df, elec):
    I_diff = sg(swv_df[f'{elec}d'].values * 1e6)
    I_fwd  = swv_df[f'{elec}f'].values * 1e6
    I_rev  = swv_df[f'{elec}r'].values * 1e6
    I_corr, _ = asls(I_diff)

    feats = {}
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        if v_lo < V.min() and v_hi < V.min() + 0.05:
            feats.update({f'swv_{elec}_{analyte}_{k}': np.nan
                          for k in ['Ip', 'Ep', 'FWHM', 'Area', 'k_pre', 'AI',
                                    'Ip_fwd', 'Ip_rev', 'fwd_rev_ratio']})
            continue

        pidx, Ep, Ip = peak_in_window(V, I_corr, max(v_lo, V.min()), v_hi)
        if pidx is None or np.isnan(Ep):
            feats.update({f'swv_{elec}_{analyte}_{k}': np.nan
                          for k in ['Ip', 'Ep', 'FWHM', 'Area', 'k_pre', 'AI',
                                    'Ip_fwd', 'Ip_rev', 'fwd_rev_ratio']})
            continue

        shape = peak_shape_features(V, I_corr, pidx, max(v_lo, V.min()), v_hi)
        Ipf   = float(I_fwd[pidx])
        Ipr   = float(I_rev[pidx])
        ratio = float(np.log1p(abs(Ipf) / max(abs(Ipr), 1e-9)))

        feats[f'swv_{elec}_{analyte}_Ip']            = round(Ip,    6)
        feats[f'swv_{elec}_{analyte}_Ep']            = round(Ep,    4)
        feats[f'swv_{elec}_{analyte}_FWHM']          = round(shape['FWHM'],  4) if not np.isnan(shape['FWHM'])  else np.nan
        feats[f'swv_{elec}_{analyte}_Area']          = round(shape['Area'],  6) if not np.isnan(shape['Area'])  else np.nan
        feats[f'swv_{elec}_{analyte}_k_pre']         = round(shape['k_pre'], 4) if not np.isnan(shape['k_pre']) else np.nan
        feats[f'swv_{elec}_{analyte}_AI']            = round(shape['AI'],    4) if not np.isnan(shape['AI'])    else np.nan
        feats[f'swv_{elec}_{analyte}_Ip_fwd']        = round(Ipf,   6)
        feats[f'swv_{elec}_{analyte}_Ip_rev']        = round(Ipr,   6)
        feats[f'swv_{elec}_{analyte}_fwd_rev_ratio'] = round(ratio, 4)

    return feats


def extract_dpv(V, currents, elec):
    I_flip = -sg(currents[elec] * 1e6)
    I_corr, _ = asls(I_flip)

    feats = {}
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        if v_hi < V.min() or v_lo > V.max():
            feats.update({f'dpv_{elec}_{analyte}_{k}': np.nan
                          for k in ['Ip', 'Ep', 'Area']})
            continue

        pidx, Ep, Ip = peak_in_window(V, I_corr, v_lo, v_hi)
        if pidx is None:
            feats.update({f'dpv_{elec}_{analyte}_{k}': np.nan
                          for k in ['Ip', 'Ep', 'Area']})
            continue

        shape = peak_shape_features(V, I_corr, pidx, v_lo, v_hi)
        feats[f'dpv_{elec}_{analyte}_Ip']   = round(Ip, 6)
        feats[f'dpv_{elec}_{analyte}_Ep']   = round(Ep, 4)
        feats[f'dpv_{elec}_{analyte}_Area'] = round(shape['Area'], 6) if not np.isnan(shape['Area']) else np.nan

    return feats


def extract_cv(V, currents, elec):
    I_raw = sg(currents[elec] * 1e6)

    split  = int(np.argmax(V))
    V_fwd, I_fwd = V[:split + 1], I_raw[:split + 1]
    V_rev, I_rev = V[split:][::-1], I_raw[split:][::-1]

    trim = (V_fwd >= V_fwd[0] + 0.05) & (V_fwd <= V_fwd[-1] - 0.05)
    V_fwd, I_fwd = V_fwd[trim], I_fwd[trim]

    I_corr_fwd, _ = asls(I_fwd)
    I_corr_rev, _ = asls(I_rev)

    feats = {}

    idx_low  = nearest_idx(V_fwd, V_BASE_LOW)
    idx_high = nearest_idx(V_fwd, V_BASE_HIGH)
    feats[f'cv_{elec}_baseline_low']  = round(float(I_fwd[idx_low]),  6)
    feats[f'cv_{elec}_baseline_high'] = round(float(I_fwd[idx_high]), 6)

    da_Ep_a = np.nan
    da_Ipa  = np.nan
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        pidx, Ep, Ipa = peak_in_window(V_fwd, I_corr_fwd, v_lo, v_hi)
        if pidx is None:
            feats.update({f'cv_{elec}_{analyte}_{k}': np.nan
                          for k in ['Ipa', 'Ep_a', 'Area', 'FWHM', 'k_pre']})
            continue

        shape = peak_shape_features(V_fwd, I_corr_fwd, pidx, v_lo, v_hi)
        feats[f'cv_{elec}_{analyte}_Ipa']   = round(Ipa, 6)
        feats[f'cv_{elec}_{analyte}_Ep_a']  = round(Ep,  4)
        feats[f'cv_{elec}_{analyte}_Area']  = round(shape['Area'],  6) if not np.isnan(shape['Area'])  else np.nan
        feats[f'cv_{elec}_{analyte}_FWHM']  = round(shape['FWHM'],  4) if not np.isnan(shape['FWHM'])  else np.nan
        feats[f'cv_{elec}_{analyte}_k_pre'] = round(shape['k_pre'], 4) if not np.isnan(shape['k_pre']) else np.nan

        if analyte == 'DA':
            da_Ep_a = Ep
            da_Ipa  = Ipa

    if not np.isnan(da_Ep_a):
        I_cat_inv = -I_corr_rev
        cat_peaks, _ = find_peaks(I_cat_inv, prominence=0.01)
        best = None
        for cp in cat_peaks:
            if abs(V_rev[cp] - da_Ep_a) < 0.15:
                if best is None or abs(V_rev[cp] - da_Ep_a) < abs(V_rev[best] - da_Ep_a):
                    best = cp
        if best is not None:
            Ep_c = float(V_rev[best])
            Ipc  = float(I_corr_rev[best])
            feats[f'cv_{elec}_DA_Ipc']     = round(Ipc, 6)
            feats[f'cv_{elec}_DA_Ep_c']    = round(Ep_c, 4)
            feats[f'cv_{elec}_DA_deltaEp'] = round(da_Ep_a - Ep_c, 4)
            feats[f'cv_{elec}_DA_E12']     = round((da_Ep_a + Ep_c) / 2, 4)
            feats[f'cv_{elec}_DA_Ipa_Ipc'] = round(abs(da_Ipa / Ipc), 4) if Ipc != 0 else np.nan
        else:
            for k in ['Ipc', 'Ep_c', 'deltaEp', 'E12', 'Ipa_Ipc']:
                feats[f'cv_{elec}_DA_{k}'] = np.nan
    else:
        for k in ['Ipc', 'Ep_c', 'deltaEp', 'E12', 'Ipa_Ipc']:
            feats[f'cv_{elec}_DA_{k}'] = np.nan

    return feats


# ── CV raw-waveform sampler (Option 2 replacement for extract_cv) ─────────────
# TO REVERT: delete extract_cv_raw, and in extract_condition change
#   raw.update(extract_cv_raw(...))  →  raw.update(extract_cv(...))
# Also update TEST_PREFIXES in bayes_optimize.py if needed (cv_ prefix is the same).
#
# Samples the AsLS-corrected forward and reverse CV sweeps at CV_N_POINTS evenly
# spaced voltages across the full scan range. Avoids peak detection entirely so
# overlapping AA/DA peaks don't cause misassignment. The Bayesian optimizer then
# learns which voltage points are actually informative.
CV_N_POINTS = 50   # number of sample points per sweep direction; tune as needed

def extract_cv_raw(V, currents, elec):
    I_raw = sg(currents[elec] * 1e6)

    split  = int(np.argmax(V))
    V_fwd, I_fwd = V[:split + 1], I_raw[:split + 1]
    V_rev, I_rev = V[split:][::-1], I_raw[split:][::-1]

    trim = (V_fwd >= V_fwd[0] + 0.05) & (V_fwd <= V_fwd[-1] - 0.05)
    V_fwd, I_fwd = V_fwd[trim], I_fwd[trim]

    I_corr_fwd, _ = asls(I_fwd)
    I_corr_rev, _ = asls(I_rev)

    v_lo = max(V_fwd.min(), V_rev.min())
    v_hi = min(V_fwd.max(), V_rev.max())
    v_grid = np.linspace(v_lo, v_hi, CV_N_POINTS)

    feats = {}
    for i, v in enumerate(v_grid):
        idx_fwd = nearest_idx(V_fwd, v)
        idx_rev = nearest_idx(V_rev, v)
        feats[f'cv_{elec}_fwd_{i:02d}'] = round(float(I_corr_fwd[idx_fwd]), 6)
        feats[f'cv_{elec}_rev_{i:02d}'] = round(float(I_corr_rev[idx_rev]), 6)

    return feats
# ── end Option 2 ──────────────────────────────────────────────────────────────


def _spline_coeffs_fe(V, I, v_grid, n_knots):
    target_len = n_knots + 4
    try:
        f = interp1d(V, I, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_grid = f(v_grid)
        knots = np.linspace(v_grid[1], v_grid[-2], n_knots)
        tck = splrep(v_grid, I_grid, t=knots, k=3)
        c = np.array(tck[1], dtype=np.float64)
        if len(c) >= target_len:
            return c[:target_len]
        return np.pad(c, (0, target_len - len(c)), constant_values=0.0)
    except Exception:
        return np.zeros(target_len, dtype=np.float64)


CV_SPLINE_KNOTS = 20
CV_SPLINE_N_COEFFS = CV_SPLINE_KNOTS + 4  # 24


def extract_cv_spline(V, currents, elec):
    I_raw = sg(currents[elec] * 1e6)

    split  = int(np.argmax(V))
    V_fwd, I_fwd = V[:split + 1], I_raw[:split + 1]
    V_rev, I_rev = V[split:][::-1], I_raw[split:][::-1]

    trim = (V_fwd >= V_fwd[0] + 0.05) & (V_fwd <= V_fwd[-1] - 0.05)
    V_fwd, I_fwd = V_fwd[trim], I_fwd[trim]

    I_corr_fwd, _ = asls(I_fwd)
    I_corr_rev, _ = asls(I_rev)

    v_lo = max(V_fwd.min(), V_rev.min())
    v_hi = min(V_fwd.max(), V_rev.max())
    v_grid = np.linspace(v_lo, v_hi, 200)

    fwd_coeffs = _spline_coeffs_fe(V_fwd, I_corr_fwd, v_grid, CV_SPLINE_KNOTS)
    rev_coeffs = _spline_coeffs_fe(V_rev, I_corr_rev, v_grid, CV_SPLINE_KNOTS)

    feats = {}
    for i, val in enumerate(fwd_coeffs):
        feats[f'cv_{elec}_spline_fwd_{i:02d}'] = round(float(val), 6)
    for i, val in enumerate(rev_coeffs):
        feats[f'cv_{elec}_spline_rev_{i:02d}'] = round(float(val), 6)
    return feats


def extract_ca(t, currents, elec):
    I = currents[elec] * 1e6

    idx_I0     = nearest_idx(t, 0.10)
    idx_Iss_lo = nearest_idx(t, 18.0)
    idx_Iss_hi = nearest_idx(t, 20.0)

    I0  = float(I[idx_I0])
    Iss = float(np.mean(I[idx_Iss_lo:idx_Iss_hi + 1]))

    mask_cott = (t >= 0.5) & (t <= 5.0)
    t_inv  = t[mask_cott] ** (-0.5)
    I_cott = I[mask_cott]
    if len(t_inv) > 4:
        coeffs   = np.polyfit(t_inv, I_cott, 1)
        I_fitted = np.polyval(coeffs, t_inv)
        ss_res   = np.sum((I_cott - I_fitted) ** 2)
        ss_tot   = np.sum((I_cott - I_cott.mean()) ** 2)
        R2         = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        cott_slope = float(coeffs[0])
    else:
        cott_slope = R2 = np.nan

    mask_early = (t >= 0.10) & (t <= 5.0)
    mask_late  = (t >= 10.0) & (t <= 20.0)
    Q_early = float(trapezoid(I[mask_early], t[mask_early])) if mask_early.sum() > 1 else np.nan
    Q_late  = float(trapezoid(I[mask_late],  t[mask_late]))  if mask_late.sum()  > 1 else np.nan
    Q_ratio = float(Q_late / Q_early) if (Q_early and Q_early != 0) else np.nan

    return {
        f'ca_{elec}_I0':           round(I0,         6),
        f'ca_{elec}_Iss':          round(Iss,        6),
        f'ca_{elec}_Iss_I0':       round(Iss / I0,   4) if I0 != 0 else np.nan,
        f'ca_{elec}_cott_slope':   round(cott_slope, 6) if not np.isnan(cott_slope) else np.nan,
        f'ca_{elec}_cott_R2':      round(R2,         4) if not np.isnan(R2)         else np.nan,
        f'ca_{elec}_Q_early':      round(Q_early,    6),
        f'ca_{elec}_Q_late':       round(Q_late,     6),
        f'ca_{elec}_Q_late_early': round(Q_ratio,    4) if not np.isnan(Q_ratio)    else np.nan,
    }


def extract_cagc(t, gens, cols, ca_norm_Iss, gen_ch, col_ch):
    I_gen = gens[gen_ch] * 1e6
    I_col = cols[col_ch] * 1e6

    def _ce(t_val):
        idx = nearest_idx(t, t_val)
        g, c = I_gen[idx], I_col[idx]
        return float(-c / g) if g != 0 else np.nan

    idx_ss_lo = nearest_idx(t, 18.0)
    idx_ss_hi = nearest_idx(t, 20.0)
    I_gen_ss  = float(np.mean(I_gen[idx_ss_lo:idx_ss_hi + 1]))
    I_col_ss  = float(np.mean(I_col[idx_ss_lo:idx_ss_hi + 1]))

    CE_ss = float(-I_col_ss / I_gen_ss) if I_gen_ss != 0 else np.nan
    AF    = float((I_gen_ss + abs(I_col_ss)) / ca_norm_Iss) if ca_norm_Iss != 0 else np.nan

    idx10    = nearest_idx(t, 10.0)
    g10, c10 = I_gen[idx10], I_col[idx10]
    eta10    = float((g10 - abs(c10)) / (g10 + abs(c10))) if (g10 + abs(c10)) != 0 else np.nan

    return {
        f'cagc_{gen_ch}_CE_1s':  round(_ce(1.0),  4),
        f'cagc_{gen_ch}_CE_10s': round(_ce(10.0), 4),
        f'cagc_{gen_ch}_CE_ss':  round(CE_ss,     4) if not np.isnan(CE_ss) else np.nan,
        f'cagc_{gen_ch}_AF':     round(AF,        4) if not np.isnan(AF)    else np.nan,
        f'cagc_{gen_ch}_eta10':  round(eta10,     4) if not np.isnan(eta10) else np.nan,
    }


# ── Inter-analyte ratios (single electrode) ───────────────────────────────────
def inter_analyte_ratios_single(feats_stripped, technique):
    """
    Compute cross-analyte ratios from one electrode's already-stripped feature dict.
    Keys are expected without electrode label, e.g. 'swv_DA_Ip'.
    """
    Ip_DA = feats_stripped.get(f'{technique}_DA_Ip', np.nan)
    Ip_AA = feats_stripped.get(f'{technique}_AA_Ip', np.nan)
    Ip_UA = feats_stripped.get(f'{technique}_UA_Ip', np.nan)
    Ep_DA = feats_stripped.get(f'{technique}_DA_Ep', np.nan)
    Ep_AA = feats_stripped.get(f'{technique}_AA_Ep', np.nan)
    Ep_UA = feats_stripped.get(f'{technique}_UA_Ep', np.nan)

    def _safe_ratio(a, b):
        return round(a / b, 4) if (b and not np.isnan(b) and b != 0 and not np.isnan(a)) else np.nan

    def _safe_diff(a, b):
        return round(a - b, 4) if not (np.isnan(a) or np.isnan(b)) else np.nan

    return {
        f'{technique}_ratio_Ip_DA_AA': _safe_ratio(Ip_DA, Ip_AA),
        f'{technique}_ratio_Ip_DA_UA': _safe_ratio(Ip_DA, Ip_UA),
        f'{technique}_ratio_Ip_UA_AA': _safe_ratio(Ip_UA, Ip_AA),
        f'{technique}_dEp_DA_AA':      _safe_diff(Ep_DA, Ep_AA),
        f'{technique}_dEp_UA_DA':      _safe_diff(Ep_UA, Ep_DA),
    }


# ── Single-condition extraction ───────────────────────────────────────────────
def _strip_elec(d, elec):
    """Remove electrode label from all keys: swv_i1_DA_Ip → swv_DA_Ip."""
    return {k.replace(f'_{elec}_', '_'): v for k, v in d.items()}


def extract_condition(base_dir, electrodes=None, verbose=True):
    """
    Run feature extraction for every requested electrode in one condition directory.

    Returns a list of dicts — one per electrode — each containing:
        {'electrode': 'i1', 'DA_uM': ..., 'AA_uM': ..., 'UA_uM': ...,
         'swv_DA_Ip': ..., 'dpv_UA_Ep': ..., ...}

    electrodes: which electrodes to extract (default: all three ['i1','i3','i5']).
    """
    if electrodes is None:
        electrodes = ALL_ELECTRODES

    base = base_dir if base_dir.endswith(os.sep) else base_dir + os.sep
    da, aa, ua = parse_labels(base)
    labels = {'DA_uM': da, 'AA_uM': aa, 'UA_uM': ua}

    if verbose:
        print(f"  Labels: DA={da}µM  AA={aa}µM  UA={ua}µM  "
              f"[electrodes: {', '.join(electrodes)}]")

    # Load all files once; each loader returns data for all physical electrodes
    V_cv,  cv_I                  = load_cv_n(base)
    _,     cvgc_gens, cvgc_cols  = load_cv_gc(base)   # loaded but unused for features
    t_ca,  ca_I                  = load_ca_norm(base)
    t_cagc, cagc_gens, cagc_cols = load_ca_gc(base)
    V_dpv, dpv_I                 = load_dpv(base)
    V_swv, swv_df                = load_swv(base)

    rows = []
    for elec in electrodes:
        gen_ch, col_ch = GC_MAP[elec]

        # Collect raw per-electrode features (keys still contain electrode label)
        raw = {}
        raw.update(extract_swv(V_swv, swv_df, elec))
        raw.update(extract_dpv(V_dpv, dpv_I, elec))
        raw.update(extract_cv_raw(V_cv, cv_I, elec))
        raw.update(extract_cv_spline(V_cv, cv_I, elec))
        ca_feats = extract_ca(t_ca, ca_I, elec)
        raw.update(ca_feats)
        raw.update(extract_cagc(t_cagc, cagc_gens, cagc_cols,
                                ca_feats[f'ca_{elec}_Iss'], gen_ch, col_ch))

        # Strip electrode label so all electrodes share identical column names
        stripped = _strip_elec(raw, elec)

        # Inter-analyte ratios (computed on stripped keys)
        stripped.update(inter_analyte_ratios_single(stripped, 'swv'))
        stripped.update(inter_analyte_ratios_single(stripped, 'dpv'))

        row = {'electrode': elec, **labels, **stripped}
        rows.append(row)

        if verbose:
            print(f"    {elec}: {len(stripped)} features")

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from pathlib import Path as _Path
    ROOT = str(_Path(__file__).resolve().parents[2] / "data" / "outputs_dataset_full")

    parser = argparse.ArgumentParser(description='Extract electrochemical features.')
    parser.add_argument('cond_dir', nargs='?', default=None,
                        help='Single condition directory (omit to run all conditions).')
    parser.add_argument('--electrodes', nargs='+', default=['1', '3', '5'],
                        metavar='N',
                        help='Electrode numbers to include: 1, 3, 5 (default: all three). '
                             'E.g. --electrodes 1 3  to hold out electrode 5 for validation.')
    args = parser.parse_args()

    VALID = {'1': 'i1', '3': 'i3', '5': 'i5', 'i1': 'i1', 'i3': 'i3', 'i5': 'i5'}
    try:
        electrodes = [VALID[e] for e in args.electrodes]
    except KeyError as bad:
        parser.error(f"Unknown electrode {bad}. Valid choices: 1, 3, 5")

    elec_tag = ''.join(e.replace('i', '') for e in electrodes)   # e.g. "13", "135"

    single    = args.cond_dir is not None
    cond_dirs = [args.cond_dir] if single else sorted(
        _glob.glob(os.path.join(ROOT, '*/')),
        key=lambda x: float(os.path.basename(x.rstrip('/\\')))
    )

    print(f"Electrodes: {electrodes}   "
          f"Conditions: {'1 (single)' if single else len(cond_dirs)}")

    all_rows = []
    errors   = []

    for cond_dir in cond_dirs:
        cond_num = float(os.path.basename(cond_dir.rstrip('/\\')))
        print(f"\n=== Condition {int(cond_num)} ({cond_dir}) ===")
        try:
            rows = extract_condition(cond_dir, electrodes=electrodes, verbose=True)
            for row in rows:
                all_rows.append({'condition': int(cond_num), **row})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append((cond_dir, str(exc)))

    if not all_rows:
        print("\nNo data extracted successfully.")
        sys.exit(1)

    out_df = pd.DataFrame(all_rows)

    if single:
        cond_num = int(float(os.path.basename(cond_dirs[0].rstrip('/\\'))))
        out_path = f'features_ml_vector_cond{cond_num}_e{elec_tag}.csv'
    else:
        out_path = f'features_all_conditions_ml_e{elec_tag}.csv'

    out_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Shape: {out_df.shape}  "
          f"({len(cond_dirs)} conditions × {len(electrodes)} electrode(s) = {len(all_rows)} rows)")

    if errors:
        print(f"\nFailed conditions ({len(errors)}):")
        for path, msg in errors:
            print(f"  {path}: {msg}")
