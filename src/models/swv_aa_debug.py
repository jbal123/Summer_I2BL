"""
SWV AA peak detection diagnostic.

Loads SWV rep 3 from a condition directory, shows the actual voltage range,
then sweeps candidate AA windows and reports which ones find a valid peak.
Also saves a plot of the raw + corrected signal with all candidate windows marked.

Usage:
    python swv_aa_debug.py                          # uses condition 1, electrode i1
    python swv_aa_debug.py --cond outputs_dataset_full/5.0 --elec i3
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
_DEFAULT_COND = str(_REPO_ROOT / "data" / "outputs_dataset_full" / "1.0")
_DEBUG_OUT = str(_REPO_ROOT / "results" / "swv_aa_debug.png")
# --- end repo anchors ---


import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings

# ── Parameters (must match feature_extract_cond.py) ──────────────────────────
SMOOTH_W  = 21
SMOOTH_P  = 3
ASLS_LAM  = 1e7
ASLS_P    = 0.005
SWV_COLS  = ['V', 'i1d','i1f','i1r', 'i3d','i3f','i3r', 'i5d','i5f','i5r', 'i7d','i7f','i7r']

# ── Candidate AA windows to test ──────────────────────────────────────────────
# Add or adjust these to probe different voltage regions
CANDIDATE_WINDOWS = [
    (-0.10,  0.12),   # original (almost certainly out of range)
    ( 0.00,  0.15),
    ( 0.05,  0.18),
    ( 0.08,  0.18),
    ( 0.08,  0.20),
    ( 0.10,  0.20),
    ( 0.10,  0.22),
    ( 0.10,  0.25),
    ( 0.12,  0.22),
    ( 0.12,  0.25),
]

# ── Helpers (copied from feature_extract_cond.py) ────────────────────────────
def sg(arr):
    return savgol_filter(arr, SMOOTH_W, SMOOTH_P)

def asls(I, lam=ASLS_LAM, p=ASLS_P, n_iter=10):
    n = len(I)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        D  = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        D  = lam * D.T @ D
        w  = np.ones(n)
        for _ in range(n_iter):
            W  = diags(w, 0)
            bl = spsolve(W + D, w * I)
            w  = np.where(I > bl, p, 1 - p)
    return I - bl, bl

def linear_baseline(V, I, v_lo, v_hi):
    mask = (V >= v_lo) & (V <= v_hi)
    Vw, Iw = V[mask], I[mask]
    if len(Vw) < 4:
        return Vw, Iw
    slope = (Iw[-1] - Iw[0]) / (Vw[-1] - Vw[0]) if (Vw[-1] - Vw[0]) != 0 else 0
    bl    = Iw[0] + slope * (Vw - Vw[0])
    return Vw, Iw - bl

def peak_in_window(V, I_corr, v_lo, v_hi):
    mask = (V >= v_lo) & (V <= v_hi)
    if mask.sum() < 5:
        return None, np.nan, np.nan
    Vw, Iw = linear_baseline(V, I_corr, v_lo, v_hi)
    peaks, _ = find_peaks(Iw, prominence=max(Iw.max() * 0.05, 1e-9))
    pidx_w   = peaks[np.argmax(Iw[peaks])] if len(peaks) > 0 else int(np.argmax(Iw))
    full_idx  = np.where(mask)[0]
    pidx_full = full_idx[pidx_w]
    return pidx_full, V[pidx_full], I_corr[pidx_full]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--cond', default=_DEFAULT_COND,
                    help='Condition directory (default: data/outputs_dataset_full/1.0)')
parser.add_argument('--elec', default='i1', choices=['i1', 'i3', 'i5'],
                    help='Electrode to inspect (default: i1)')
args = parser.parse_args()

# ── Load SWV rep 3 ────────────────────────────────────────────────────────────
matches = glob.glob(os.path.join(args.cond, 'SWV_*_3.txt'))
if not matches:
    raise FileNotFoundError(f"No SWV_*_3.txt found in {args.cond}")
rep3 = matches[0]
print(f"Loading: {rep3}")

df = pd.read_csv(rep3, skiprows=57, header=None, names=SWV_COLS)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

V      = df['V'].values
elec   = args.elec
I_diff = sg(df[f'{elec}d'].values * 1e6)
I_corr, baseline = asls(I_diff)

print(f"\nElectrode: {elec}")
print(f"Voltage range: {V.min():.4f} V  →  {V.max():.4f} V  ({len(V)} points)")
print(f"Signal range (AsLS corrected): {I_corr.min():.4f}  →  {I_corr.max():.4f} µA")
print(f"\nOriginal AA window: -0.10 → 0.12 V")
print(f"  Points in window: {((V >= -0.10) & (V <= 0.12)).sum()}  "
      f"← {'PROBLEM: too few' if ((V >= -0.10) & (V <= 0.12)).sum() < 5 else 'OK'}")

# ── Test all candidate windows ────────────────────────────────────────────────
print(f"\n{'Window':<22}  {'Points':>7}  {'Peak found':>12}  {'Ep (V)':>8}  {'Ip (µA)':>10}")
print('-' * 65)
results = []
for v_lo, v_hi in CANDIDATE_WINDOWS:
    n_pts = ((V >= v_lo) & (V <= v_hi)).sum()
    pidx, Ep, Ip = peak_in_window(V, I_corr, v_lo, v_hi)
    found = pidx is not None and not np.isnan(Ep)
    tag   = f"{Ep:.4f}" if found else "  —"
    istr  = f"{Ip:.4f}" if found else "  —"
    flag  = " ✓" if found else " ✗"
    print(f"  {v_lo:.2f} → {v_hi:.2f}        {n_pts:>7}  {flag:>12}  {tag:>8}  {istr:>10}")
    results.append((v_lo, v_hi, n_pts, found, Ep, Ip))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axes[0].plot(V, I_diff, color='gray', lw=0.8, alpha=0.6, label='SG smoothed (raw)')
axes[0].plot(V, baseline, color='darkorange', lw=1.2, ls='--', label='AsLS baseline')
axes[0].set_ylabel('Current (µA)')
axes[0].set_title(f'SWV {elec} — {os.path.basename(rep3)}', fontweight='bold')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

colors = plt.cm.tab10(np.linspace(0, 1, len(CANDIDATE_WINDOWS)))
axes[1].plot(V, I_corr, color='steelblue', lw=1.5, label='AsLS corrected')
axes[1].axhline(0, color='k', lw=0.5, ls=':')

for (v_lo, v_hi, n_pts, found, Ep, Ip), c in zip(results, colors):
    axes[1].axvspan(v_lo, v_hi, alpha=0.08, color=c)
    axes[1].axvline(v_lo, color=c, lw=0.8, ls='--', alpha=0.6)
    axes[1].axvline(v_hi, color=c, lw=0.8, ls='--', alpha=0.6)
    label = f"{v_lo:.2f}→{v_hi:.2f} {'✓' if found else '✗'}"
    if found:
        axes[1].scatter([Ep], [Ip], color=c, s=60, zorder=5)
    axes[1].plot([], [], color=c, label=label)

# Mark original AA / DA / UA windows
for name, (lo, hi), col in [('AA orig', (-0.10, 0.12), 'red'),
                              ('DA',     ( 0.12, 0.32), 'green'),
                              ('UA',     ( 0.30, 0.55), 'purple')]:
    axes[1].axvline(lo, color=col, lw=1.5, ls='-', alpha=0.4)
    axes[1].axvline(hi, color=col, lw=1.5, ls='-', alpha=0.4)
    axes[1].text((lo + hi) / 2, axes[1].get_ylim()[1] if axes[1].get_ylim()[1] != 0 else 1,
                 name, ha='center', fontsize=7, color=col)

axes[1].set_xlabel('Potential (V)')
axes[1].set_ylabel('Current (µA)')
axes[1].set_title('AsLS-corrected signal with candidate AA windows')
axes[1].legend(fontsize=7, loc='upper left', ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = _DEBUG_OUT
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved → {out}")
