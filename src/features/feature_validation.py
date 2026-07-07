"""
feature_validation.py — Raw signal + extracted features overlaid, all 6 techniques.

One page per condition × electrode in the master PDF (3 pages per condition).
Each page: 2 × 3 panel grid —
    [SWV]    [CV Normal]  [DPV   ]
    [CA]     [CA-GC]      [CV-GC ]

Raw data shown faded behind the SG+AsLS-corrected signal (bold).
Extracted features overlaid: peaks, FWHM arrows, shaded peak areas, analytic labels.
Individual PNGs saved per condition × electrode.

Usage:
    python feature_validation.py
    python feature_validation.py --conditions 1 3 5
    python feature_validation.py --data-dir "Day 1-4 Outputs w. TXT/Outputs_Day_1"
    python feature_validation.py --output validation.pdf --output-dir validation_pngs
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
_DEF_DATA_DIR = str(_REPO_ROOT / "data" / "Day 1-4 Outputs w. TXT" / "Outputs_Day_1")
_DEF_OUTPUT = str(_REPO_ROOT / "results" / "feature_validation_pngs" / "feature_validation.pdf")
_DEF_OUTPUT_DIR = str(_REPO_ROOT / "results" / "feature_validation_pngs")
# --- end repo anchors ---

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

from feature_extract_cond import (
    ALL_ELECTRODES, GC_MAP, WINDOWS,
    load_cv_n, load_cv_gc, load_ca_norm, load_ca_gc, load_dpv, load_swv,
    parse_labels, sg, asls, nearest_idx, linear_baseline,
    extract_cv, extract_swv, extract_dpv, extract_ca, extract_cagc,
    _strip_elec,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',   default=_DEF_DATA_DIR)
parser.add_argument('--output',     default=_DEF_OUTPUT)
parser.add_argument('--output-dir', default=_DEF_OUTPUT_DIR)
parser.add_argument('--conditions', nargs='+', type=int, default=None)
parser.add_argument('--no-asls',    action='store_true', default=False,
                    help='Skip AsLS baseline correction — plot SG-smoothed signal only.')
args = parser.parse_args()
APPLY_ASLS = not args.no_asls

# ── Nature-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':               150,
    'savefig.dpi':              300,
    'figure.facecolor':         'white',
    'axes.facecolor':           'white',
    'font.family':              'DejaVu Sans',
    'font.size':                9,
    'axes.titlesize':           10,
    'axes.labelsize':           9,
    'axes.linewidth':           1.2,
    'xtick.labelsize':          8,
    'ytick.labelsize':          8,
    'xtick.direction':          'in',
    'ytick.direction':          'in',
    'xtick.major.size':         5,
    'ytick.major.size':         5,
    'xtick.minor.size':         3,
    'ytick.minor.size':         3,
    'xtick.major.width':        1.0,
    'ytick.major.width':        1.0,
    'legend.frameon':           False,
    'legend.fontsize':          7,
    'lines.linewidth':          1.8,
    'figure.max_open_warning':  200,
})

# ── Colors ─────────────────────────────────────────────────────────────────────
ANALYTE_COLORS = {'AA': '#0072B2', 'DA': '#D55E00', 'UA': '#009E73'}
ELEC_COLORS    = {'i1': '#000000', 'i3': '#332288', 'i5': '#AA4499'}
ELEC_LABELS    = {'i1': 'Electrode 1', 'i3': 'Electrode 3', 'i5': 'Electrode 5'}
WIN_ALPHA      = 0.07


# ── Helpers ────────────────────────────────────────────────────────────────────
def _nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _fmt(v, d=3):
    return 'NaN' if _nan(v) else f'{v:.{d}f}'


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=10, fontweight='semibold', pad=5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', direction='in', length=5, width=1.0)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.8)
    ax.minorticks_on()
    ax.grid(False)
    ax.margins(x=0.02, y=0.12)
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)


def _fill_peak_area(ax, V, I_corr, v_lo, v_hi, color, alpha=0.15):
    """Shade the baseline-subtracted peak area within the analyte window."""
    try:
        Vw, Iw = linear_baseline(V, I_corr, v_lo, v_hi)
        if len(Vw) > 1 and np.any(Iw > 0):
            ax.fill_between(Vw, 0, np.clip(Iw, 0, None),
                            alpha=alpha, color=color, zorder=1)
    except Exception:
        pass


def _peak_dot(ax, Ep, Ip, color):
    if not _nan(Ep) and not _nan(Ip):
        ax.plot(Ep, Ip, 'o', color=color, ms=6, zorder=6,
                markeredgecolor='white', markeredgewidth=0.8)


def _fwhm_arrow(ax, Ep, Ip, FWHM, color):
    if _nan(FWHM) or _nan(Ep) or _nan(Ip) or FWHM <= 0:
        return
    half = Ip / 2.0
    ax.annotate('', xy=(Ep + FWHM / 2, half), xytext=(Ep - FWHM / 2, half),
                arrowprops=dict(arrowstyle='<->', color=color, lw=0.9), zorder=5)


# Fixed y-fraction positions per analyte so labels are always in the top margin,
# never sitting on the curve. Each analyte gets its own row, well-separated.
# x stays in data coordinates via get_xaxis_transform().
_ANALYTE_YTOP = {'AA': 0.97, 'DA': 0.76, 'UA': 0.55}
_CATHODIC_YBOT = 0.12   # for negative (cathodic) peaks — anchored near the bottom


def _annotate(ax, Ep, Ip, lines, color, y_top=0.97):
    """
    Vertical dotted guide line from peak dot to a text box in the top margin.
    Text is placed at a fixed axes-fraction y so it never overlaps the curve.
    x is in data coordinates (aligns with Ep) via get_xaxis_transform().
    """
    if _nan(Ep) or _nan(Ip):
        return
    # Thin vertical guide line (data x, axes y) from peak up to label
    ax.axvline(Ep, color=color, lw=0.6, ls=':', alpha=0.40, zorder=3)
    # Label box at fixed top position — clear of the curve
    ax.text(Ep, y_top, '\n'.join(lines),
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=6.0, color=color,
            bbox=dict(fc='white', alpha=0.93, ec=color,
                      boxstyle='round,pad=0.20', lw=0.6),
            zorder=9)


def _annotate_bottom(ax, Ep, Ip, lines, color, y_bot=_CATHODIC_YBOT):
    """Same as _annotate but anchors to the bottom margin for negative peaks."""
    if _nan(Ep) or _nan(Ip):
        return
    ax.axvline(Ep, color=color, lw=0.6, ls=':', alpha=0.40, zorder=3)
    ax.text(Ep, y_bot, '\n'.join(lines),
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=6.0, color=color,
            bbox=dict(fc='white', alpha=0.93, ec=color,
                      boxstyle='round,pad=0.20', lw=0.6),
            zorder=9)


def _feat_box(ax, lines, loc='lower right'):
    """Summary feature table in a fixed corner — always outside the data region."""
    coords = {
        'upper right': (0.99, 0.99, 'right', 'top'),
        'upper left':  (0.01, 0.99, 'left',  'top'),
        'lower right': (0.99, 0.01, 'right', 'bottom'),
        'lower left':  (0.01, 0.01, 'left',  'bottom'),
    }
    x, y, ha, va = coords.get(loc, (0.99, 0.01, 'right', 'bottom'))
    ax.text(x, y, '\n'.join(lines), transform=ax.transAxes,
            ha=ha, va=va, fontsize=5.8, fontfamily='monospace',
            bbox=dict(fc='#f8f8f8', alpha=0.93, ec='#bbbbbb',
                      boxstyle='round,pad=0.28', lw=0.5),
            zorder=8)


# ── SWV panel ─────────────────────────────────────────────────────────────────
def draw_swv(ax, V, swv_df, elec, feats, apply_asls=True):
    color = ELEC_COLORS[elec]
    I_raw = swv_df[f'{elec}d'].values * 1e6
    I_sg  = sg(I_raw)

    if apply_asls:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            I_corr, bl = asls(I_sg)
        ax.plot(V, bl,     color='gray', lw=0.8, alpha=0.45, ls=':', zorder=2, label='AsLS bl')
        main_label = 'SG + AsLS'
    else:
        I_corr = I_sg
        main_label = 'SG only'

    ax.plot(V, I_raw,  color=color, lw=0.8, alpha=0.20, zorder=1, label='Raw')
    ax.plot(V, I_corr, color=color, lw=2.0, zorder=3,             label=main_label)
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.30)

    feat_lines = []
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        ac     = ANALYTE_COLORS[analyte]
        v_lo_e = max(v_lo, float(V.min()))
        v_hi_e = min(v_hi, float(V.max()))
        if v_lo_e >= v_hi_e:
            continue
        ax.axvspan(v_lo_e, v_hi_e, alpha=WIN_ALPHA, color=ac, zorder=0)

        Ip   = feats.get(f'swv_{analyte}_Ip',   np.nan)
        Ep   = feats.get(f'swv_{analyte}_Ep',   np.nan)
        FWHM = feats.get(f'swv_{analyte}_FWHM', np.nan)
        Area = feats.get(f'swv_{analyte}_Area', np.nan)

        if not _nan(Ep) and not _nan(Ip):
            _fill_peak_area(ax, V, I_corr, v_lo_e, v_hi_e, ac)
            _peak_dot(ax, Ep, Ip, ac)
            _fwhm_arrow(ax, Ep, Ip, FWHM, ac)
            ann = [analyte, f'Ep={_fmt(Ep,3)}V', f'Ip={_fmt(Ip,2)}µA']
            if not _nan(FWHM): ann.append(f'FWHM={_fmt(FWHM,3)}V')
            if not _nan(Area): ann.append(f'Area={_fmt(Area,3)}')
            _annotate(ax, Ep, Ip, ann, ac, y_top=_ANALYTE_YTOP[analyte])

        feat_lines.append(f'{analyte}: Ip={_fmt(Ip,2)}  Ep={_fmt(Ep,3)}')
        feat_lines.append(f'  FWHM={_fmt(FWHM,3)}  Area={_fmt(Area,3)}')

    _feat_box(ax, feat_lines, 'lower right')
    _style_ax(ax, 'Potential (V)', 'Diff. Current (µA)',
              f'SWV  |  {ELEC_LABELS[elec]}  [{"SG+AsLS" if apply_asls else "SG only"}]')
    ax.legend(fontsize=6.5, loc='lower left')


# ── CV panel ──────────────────────────────────────────────────────────────────
def draw_cv(ax, V_cv, cv_I, elec, feats, apply_asls=True):
    color = ELEC_COLORS[elec]
    I_raw = cv_I[elec] * 1e6
    I_sg  = sg(I_raw)

    split      = int(np.argmax(V_cv))
    V_fwd_full = V_cv[:split + 1]
    I_fwd_full = I_sg[:split + 1]
    trim = (V_fwd_full >= V_fwd_full[0] + 0.05) & (V_fwd_full <= V_fwd_full[-1] - 0.05)
    V_fwd, I_fwd = V_fwd_full[trim], I_fwd_full[trim]

    # Natural descending reverse — forms proper closed loop when plotted
    V_rev_nat = V_cv[split:]
    I_rev_nat = I_sg[split:]

    if apply_asls:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            I_corr_f, bl_f = asls(I_fwd)
            I_corr_r_asc, _ = asls(I_rev_nat[::-1])
            I_corr_r_nat = I_corr_r_asc[::-1]
        ax.plot(V_fwd, bl_f, color='gray', lw=0.8, ls=':', alpha=0.50,
                zorder=2, label='AsLS bl')
        fwd_label = 'Fwd (SG+AsLS)'
        rev_label = 'Rev (SG+AsLS)'
    else:
        I_corr_f     = I_fwd
        I_corr_r_nat = I_rev_nat
        fwd_label = 'Fwd (SG)'
        rev_label = 'Rev (SG)'

    # Full raw loop faded (mentor style — no processing)
    ax.plot(V_cv, I_sg,            color=color, lw=0.9, alpha=0.25, zorder=1, label='Raw loop')
    # Corrected forward (bold) + reverse (dashed) — natural directions → proper loop
    ax.plot(V_fwd,    I_corr_f,     color=color, lw=2.2, zorder=3,             label=fwd_label)
    ax.plot(V_rev_nat, I_corr_r_nat, color=color, lw=1.5, ls='--', alpha=0.55,
            zorder=2, label=rev_label)
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.30)

    v_fmin, v_fmax = float(V_fwd.min()), float(V_fwd.max())
    da_Ep_a, da_Ipa = np.nan, np.nan
    feat_lines = []

    for analyte, (v_lo, v_hi) in WINDOWS.items():
        ac = ANALYTE_COLORS[analyte]
        lo = max(v_lo, v_fmin)
        hi = min(v_hi, v_fmax)
        if lo >= hi:
            continue
        ax.axvspan(lo, hi, alpha=WIN_ALPHA, color=ac, zorder=0)

        Ipa  = feats.get(f'cv_{analyte}_Ipa',  np.nan)
        Ep_a = feats.get(f'cv_{analyte}_Ep_a', np.nan)
        FWHM = feats.get(f'cv_{analyte}_FWHM', np.nan)
        Area = feats.get(f'cv_{analyte}_Area', np.nan)

        if not _nan(Ep_a) and not _nan(Ipa):
            _fill_peak_area(ax, V_fwd, I_corr_f, lo, hi, ac)
            _peak_dot(ax, Ep_a, Ipa, ac)
            _fwhm_arrow(ax, Ep_a, Ipa, FWHM, ac)
            ann = [analyte, f'Ep_a={_fmt(Ep_a,3)}V', f'Ipa={_fmt(Ipa,2)}µA']
            if not _nan(FWHM): ann.append(f'FWHM={_fmt(FWHM,3)}V')
            if not _nan(Area): ann.append(f'Area={_fmt(Area,3)}')
            _annotate(ax, Ep_a, Ipa, ann, ac, y_top=_ANALYTE_YTOP[analyte])
            if analyte == 'DA':
                da_Ep_a, da_Ipa = Ep_a, Ipa

        feat_lines.append(f'{analyte}_a: Ipa={_fmt(Ipa,2)}  Ep={_fmt(Ep_a,3)}')

    # DA cathodic peak — annotate at bottom margin (peak is negative/low)
    Ep_c = feats.get('cv_DA_Ep_c',    np.nan)
    Ipc  = feats.get('cv_DA_Ipc',     np.nan)
    dEp  = feats.get('cv_DA_deltaEp', np.nan)
    E12  = feats.get('cv_DA_E12',     np.nan)
    IrIc = feats.get('cv_DA_Ipa_Ipc', np.nan)

    if not _nan(Ep_c) and not _nan(Ipc):
        ac = ANALYTE_COLORS['DA']
        ax.plot(Ep_c, Ipc, 's', color=ac, ms=6, zorder=6,
                markeredgecolor='white', markeredgewidth=0.8)
        ann = ['DA cat', f'Ep_c={_fmt(Ep_c,3)}V', f'Ipc={_fmt(Ipc,2)}µA']
        if not _nan(dEp):  ann.append(f'ΔEp={_fmt(dEp,3)}V')
        if not _nan(E12):  ann.append(f'E½={_fmt(E12,3)}V')
        if not _nan(IrIc): ann.append(f'Ipa/Ipc={_fmt(IrIc,3)}')
        _annotate_bottom(ax, Ep_c, Ipc, ann, ac)

        # ΔEp dotted bracket between anodic and cathodic peaks
        if not _nan(da_Ep_a) and not _nan(da_Ipa):
            ax.annotate('', xy=(da_Ep_a, da_Ipa), xytext=(Ep_c, Ipc),
                        arrowprops=dict(arrowstyle='<->', color='#999999',
                                        lw=0.9, linestyle='dotted'), zorder=4)
        feat_lines.append(f'DA_c: Ipc={_fmt(Ipc,2)}  ΔEp={_fmt(dEp,3)}')

    _feat_box(ax, feat_lines, 'lower right')
    _style_ax(ax, 'Potential (V)', 'Current (µA)',
              f'CV  |  {ELEC_LABELS[elec]}  [{"SG+AsLS" if apply_asls else "SG only"}]')
    ax.legend(fontsize=6.5, loc='lower left')


# ── CV-GC panel ───────────────────────────────────────────────────────────────
def draw_cvgc(ax, V_gc, cvgc_gens, cvgc_cols, elec):
    gen_ch, col_ch = GC_MAP[elec]
    color  = ELEC_COLORS[elec]
    I_gen  = sg(cvgc_gens[gen_ch] * 1e6)
    I_col  = sg(cvgc_cols[col_ch] * 1e6)

    ax.plot(V_gc, I_gen, color=color, lw=2.0,               label=f'Gen ({gen_ch})')
    ax.plot(V_gc, I_col, color=color, lw=1.5, ls='--', alpha=0.60, label=f'Col ({col_ch})')
    ax.axhline(0, color='k', lw=0.4, alpha=0.30)

    v_lo_all, v_hi_all = float(V_gc.min()), float(V_gc.max())
    feat_lines = []

    for analyte, (v_lo, v_hi) in WINDOWS.items():
        ac = ANALYTE_COLORS[analyte]
        lo = max(v_lo, v_lo_all)
        hi = min(v_hi, v_hi_all)
        if lo >= hi:
            continue
        ax.axvspan(lo, hi, alpha=WIN_ALPHA, color=ac, zorder=0)

        mask = (V_gc >= lo) & (V_gc <= hi)
        if mask.sum() > 0:
            peak_i  = int(np.argmax(np.abs(I_gen[mask])))
            full_i  = np.where(mask)[0][peak_i]
            g = float(I_gen[full_i])
            c = float(I_col[full_i])
            ce_str = f'{-c/g:.3f}' if g != 0 else 'N/A'
            ax.annotate(f'{analyte}\nCE≈{ce_str}',
                        xy=(V_gc[full_i], g),
                        xytext=(4, 6), textcoords='offset points',
                        fontsize=6.5, color=ac,
                        bbox=dict(fc='white', alpha=0.85, ec=ac,
                                  boxstyle='round,pad=0.18', lw=0.5),
                        zorder=7)
            feat_lines.append(f'{analyte}: CE={ce_str}')

    _feat_box(ax, feat_lines, 'upper left')
    _style_ax(ax, 'Potential (V)', 'Current (µA)', f'CV-GC  |  {ELEC_LABELS[elec]}')
    ax.legend(fontsize=6.5, loc='lower right')


# ── CA panel ──────────────────────────────────────────────────────────────────
def draw_ca(ax, t, ca_I, elec, feats):
    color = ELEC_COLORS[elec]
    I     = ca_I[elec] * 1e6

    ax.plot(t, I, color=color, lw=1.8, zorder=3)
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.30)

    I0  = feats.get('ca_I0',  np.nan)
    Iss = feats.get('ca_Iss', np.nan)

    if not _nan(I0):
        ax.axhline(I0,  color='#E69F00', lw=1.2, ls='--', alpha=0.85,
                   label=f'I₀={_fmt(I0,2)} µA')
        ax.axvline(0.10, color='#E69F00', lw=0.7, ls=':', alpha=0.50)

    if not _nan(Iss):
        ax.axhline(Iss, color='#0072B2', lw=1.2, ls='--', alpha=0.85,
                   label=f'Iss={_fmt(Iss,2)} µA')

    # Q_early shading
    mask_e  = (t >= 0.10) & (t <= 5.0)
    Q_early = feats.get('ca_Q_early', np.nan)
    if mask_e.sum() > 1:
        lbl = f'Q_early={_fmt(Q_early,2)}' if not _nan(Q_early) else 'Q_early'
        ax.fill_between(t[mask_e], 0, I[mask_e],
                        alpha=0.14, color='#009E73', zorder=1, label=lbl)

    # Q_late shading
    mask_l = (t >= 10.0) & (t <= 20.0)
    Q_late = feats.get('ca_Q_late', np.nan)
    if mask_l.sum() > 1:
        lbl = f'Q_late={_fmt(Q_late,2)}' if not _nan(Q_late) else 'Q_late'
        ax.fill_between(t[mask_l], 0, I[mask_l],
                        alpha=0.14, color='#D55E00', zorder=1, label=lbl)

    # Cottrell fit line
    cott_slope = feats.get('ca_cott_slope', np.nan)
    cott_R2    = feats.get('ca_cott_R2',    np.nan)
    mask_c     = (t >= 0.5) & (t <= 5.0)
    if not _nan(cott_slope) and mask_c.sum() > 3:
        t_c   = t[mask_c]
        t_inv = t_c ** (-0.5)
        I_c   = I[mask_c]
        try:
            coeffs = np.polyfit(t_inv, I_c, 1)
            I_fit  = np.polyval(coeffs, t_inv)
            lbl = f'Cottrell R²={_fmt(cott_R2,3)}' if not _nan(cott_R2) else 'Cottrell'
            ax.plot(t_c, I_fit, color='#CC79A7', lw=1.5, ls='--',
                    alpha=0.9, label=lbl, zorder=4)
        except Exception:
            pass

    feat_lines = [
        f'I0:         {_fmt(feats.get("ca_I0",np.nan),3)}',
        f'Iss:        {_fmt(feats.get("ca_Iss",np.nan),3)}',
        f'Iss/I0:     {_fmt(feats.get("ca_Iss_I0",np.nan),3)}',
        f'cott_slope: {_fmt(feats.get("ca_cott_slope",np.nan),3)}',
        f'cott_R2:    {_fmt(feats.get("ca_cott_R2",np.nan),4)}',
        f'Q_early:    {_fmt(feats.get("ca_Q_early",np.nan),3)}',
        f'Q_late:     {_fmt(feats.get("ca_Q_late",np.nan),3)}',
        f'Q_late/Q_e: {_fmt(feats.get("ca_Q_late_early",np.nan),3)}',
    ]
    _feat_box(ax, feat_lines, 'upper right')
    _style_ax(ax, 'Time (s)', 'Current (µA)', f'CA  |  {ELEC_LABELS[elec]}')
    ax.legend(fontsize=6.5, loc='lower right')


# ── CA-GC panel ───────────────────────────────────────────────────────────────
def draw_cagc(ax, t, gens, cols, elec, feats):
    gen_ch, col_ch = GC_MAP[elec]
    color  = ELEC_COLORS[elec]
    I_gen  = gens[gen_ch] * 1e6
    I_col  = cols[col_ch] * 1e6

    ax.plot(t, I_gen, color=color, lw=1.8, zorder=3, label=f'Gen ({gen_ch})')
    ax.plot(t, I_col, color=color, lw=1.5, ls='--', alpha=0.60, zorder=2, label=f'Col ({col_ch})')
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.30)

    # Vertical markers + CE annotations at t=1s and t=10s
    I_all   = np.concatenate([I_gen, I_col])
    y_range = float(np.ptp(I_all)) if len(I_all) > 0 else 1.0
    y_top   = float(np.nanmax(I_all)) if len(I_all) > 0 else 1.0

    for t_mark, feat_key, label_str in [
        (1.0,  'cagc_CE_1s',  'CE(1s)'),
        (10.0, 'cagc_CE_10s', 'CE(10s)'),
    ]:
        idx    = nearest_idx(t, t_mark)
        ax.axvline(t[idx], color='#E69F00', lw=0.8, ls=':', alpha=0.70)
        ce_val = feats.get(feat_key, np.nan)
        if not _nan(ce_val):
            ax.text(t[idx] + 0.30, y_top - 0.08 * y_range,
                    f'{label_str}={_fmt(ce_val,3)}',
                    fontsize=6.5, color='#E69F00', va='top')

    feat_lines = [
        f'CE(1s):  {_fmt(feats.get("cagc_CE_1s", np.nan),3)}',
        f'CE(10s): {_fmt(feats.get("cagc_CE_10s",np.nan),3)}',
        f'CE_ss:   {_fmt(feats.get("cagc_CE_ss",  np.nan),3)}',
        f'AF:      {_fmt(feats.get("cagc_AF",      np.nan),3)}',
        f'η(10s):  {_fmt(feats.get("cagc_eta10",  np.nan),3)}',
    ]
    _feat_box(ax, feat_lines, 'upper right')
    _style_ax(ax, 'Time (s)', 'Current (µA)', f'CA-GC  |  {ELEC_LABELS[elec]}')
    ax.legend(fontsize=6.5, loc='lower right')


# ── DPV panel ─────────────────────────────────────────────────────────────────
def draw_dpv(ax, V, dpv_I, elec, feats, apply_asls=True):
    color  = ELEC_COLORS[elec]
    I_raw  = dpv_I[elec] * 1e6
    I_flip = -sg(I_raw)   # flip so reduction peaks are positive

    if apply_asls:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            I_corr, bl = asls(I_flip)
        ax.plot(V, bl,     color='gray', lw=0.8, alpha=0.45, ls=':', zorder=2, label='AsLS bl')
        main_label = 'SG+AsLS (−)'
    else:
        I_corr = I_flip
        main_label = 'SG (−)'

    ax.plot(V, I_raw,  color=color, lw=0.8, alpha=0.20, zorder=1, label='Raw')
    ax.plot(V, I_corr, color=color, lw=2.0, zorder=3,             label=main_label)
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.30)

    feat_lines = []
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        ac     = ANALYTE_COLORS[analyte]
        v_lo_e = max(v_lo, float(V.min()))
        v_hi_e = min(v_hi, float(V.max()))
        if v_lo_e >= v_hi_e:
            continue
        ax.axvspan(v_lo_e, v_hi_e, alpha=WIN_ALPHA, color=ac, zorder=0)

        Ip   = feats.get(f'dpv_{analyte}_Ip',   np.nan)
        Ep   = feats.get(f'dpv_{analyte}_Ep',   np.nan)
        Area = feats.get(f'dpv_{analyte}_Area', np.nan)

        if not _nan(Ep) and not _nan(Ip) and Ip > 0:
            _fill_peak_area(ax, V, I_corr, v_lo_e, v_hi_e, ac)
            _peak_dot(ax, Ep, Ip, ac)
            ann = [analyte, f'Ep={_fmt(Ep,3)}V', f'Ip={_fmt(Ip,2)}µA']
            if not _nan(Area): ann.append(f'Area={_fmt(Area,3)}')
            _annotate(ax, Ep, Ip, ann, ac, y_top=_ANALYTE_YTOP[analyte])

        feat_lines.append(f'{analyte}: Ip={_fmt(Ip,2)}  Ep={_fmt(Ep,3)}')

    ax.text(0.50, 1.02, '⚠ signal negated for peak clarity',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=5.5, color='gray', style='italic')
    _feat_box(ax, feat_lines, 'lower right')
    _style_ax(ax, 'Potential (V)', 'Diff. Current (µA)',
              f'DPV  |  {ELEC_LABELS[elec]}  [{"SG+AsLS" if apply_asls else "SG only"}]')
    ax.legend(fontsize=6.5, loc='lower left')


# ── Condition number from path ─────────────────────────────────────────────────
def _cond_num(path):
    name = os.path.basename(path.rstrip('/\\'))
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 0


# ── Main ──────────────────────────────────────────────────────────────────────
cond_dirs = sorted(glob.glob(os.path.join(args.data_dir, '*/')), key=_cond_num)
if not cond_dirs:
    sys.exit(f"No condition directories found in {args.data_dir!r}")

if args.conditions:
    cond_dirs = [d for d in cond_dirs if _cond_num(d) in args.conditions]
    if not cond_dirs:
        sys.exit('No directories matched --conditions filter.')

os.makedirs(args.output_dir, exist_ok=True)
print(f"Conditions : {len(cond_dirs)}")
print(f"Master PDF : {os.path.abspath(args.output)}")
print(f"PNG folder : {os.path.abspath(args.output_dir)}\n")

with PdfPages(args.output) as pdf:
    for cond_dir in cond_dirs:
        cond_num = _cond_num(cond_dir)
        base     = cond_dir if cond_dir.endswith(os.sep) else cond_dir + os.sep

        try:
            da, aa, ua = parse_labels(base)
        except Exception as exc:
            print(f"  WARN cond {cond_num}: label parse — {exc}")
            continue

        cond_label = f"cond_{cond_num:02d}_DA{da}_AA{aa}_UA{ua}"
        cond_out   = os.path.join(args.output_dir, cond_label)
        os.makedirs(cond_out, exist_ok=True)

        # Load all technique files (each independently faulted)
        def _load(fn, *largs):
            try:
                return fn(*largs)
            except Exception as exc:
                return None

        cv_data   = _load(load_cv_n,    base)
        cvgc_data = _load(load_cv_gc,   base)
        ca_data   = _load(load_ca_norm, base)
        cagc_data = _load(load_ca_gc,   base)
        dpv_data  = _load(load_dpv,     base)
        swv_data  = _load(load_swv,     base)

        if cv_data is None:
            print(f"  SKIP cond {cond_num}: CV data missing")
            continue

        V_cv,  cv_I                  = cv_data
        V_gc,  cvgc_gens, cvgc_cols  = cvgc_data  if cvgc_data  else (None, None, None)
        t_ca,  ca_I                  = ca_data    if ca_data    else (None, None)
        t_cagc, cagc_gens, cagc_cols = cagc_data  if cagc_data  else (None, None, None)
        V_dpv, dpv_I                 = dpv_data   if dpv_data   else (None, None)
        V_swv, swv_df                = swv_data   if swv_data   else (None, None)

        for elec in ALL_ELECTRODES:
            gen_ch, col_ch = GC_MAP[elec]

            # ── Feature extraction ────────────────────────────────────────────
            feats = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                if V_swv is not None:
                    try:
                        feats.update(_strip_elec(extract_swv(V_swv, swv_df, elec), elec))
                    except Exception as exc:
                        print(f"  WARN cond {cond_num} {elec}: swv — {exc}")

                try:
                    feats.update(_strip_elec(extract_cv(V_cv, cv_I, elec), elec))
                except Exception as exc:
                    print(f"  WARN cond {cond_num} {elec}: cv — {exc}")

                if V_dpv is not None:
                    try:
                        feats.update(_strip_elec(extract_dpv(V_dpv, dpv_I, elec), elec))
                    except Exception as exc:
                        print(f"  WARN cond {cond_num} {elec}: dpv — {exc}")

                ca_feats_raw = {}
                if t_ca is not None:
                    try:
                        ca_feats_raw = extract_ca(t_ca, ca_I, elec)
                        feats.update(_strip_elec(ca_feats_raw, elec))
                    except Exception as exc:
                        print(f"  WARN cond {cond_num} {elec}: ca — {exc}")

                if t_cagc is not None and cagc_gens is not None:
                    try:
                        ca_Iss = ca_feats_raw.get(f'ca_{elec}_Iss', np.nan)
                        cagc_f = extract_cagc(t_cagc, cagc_gens, cagc_cols,
                                              ca_Iss, gen_ch, col_ch)
                        feats.update(_strip_elec(cagc_f, elec))
                    except Exception as exc:
                        print(f"  WARN cond {cond_num} {elec}: cagc — {exc}")

            # ── Build 2×3 figure ──────────────────────────────────────────────
            fig, axes = plt.subplots(2, 3, figsize=(21, 12))
            fig.patch.set_facecolor('white')
            fig.suptitle(
                f'Condition {cond_num}  ·  DA = {da} µM   AA = {aa} µM   UA = {ua} µM'
                f'   ·   {ELEC_LABELS[elec]}',
                fontsize=13, fontweight='bold', y=0.998,
            )

            ax_swv,  ax_cv,   ax_dpv  = axes[0]
            ax_ca,   ax_cagc, ax_cvgc = axes[1]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # SWV
                if V_swv is not None and swv_df is not None:
                    try:
                        draw_swv(ax_swv, V_swv, swv_df, elec, feats, apply_asls=APPLY_ASLS)
                    except Exception as exc:
                        ax_swv.text(0.5, 0.5, f'SWV render error:\n{exc}',
                                    transform=ax_swv.transAxes,
                                    ha='center', va='center', fontsize=7, color='red')
                        _style_ax(ax_swv, 'Potential (V)', 'Current (µA)',
                                  f'SWV  |  {ELEC_LABELS[elec]}')
                else:
                    ax_swv.text(0.5, 0.5, 'No SWV data',
                                transform=ax_swv.transAxes,
                                ha='center', va='center', fontsize=9, color='gray')
                    _style_ax(ax_swv, 'Potential (V)', 'Current (µA)',
                              f'SWV  |  {ELEC_LABELS[elec]}')

                # CV
                try:
                    draw_cv(ax_cv, V_cv, cv_I, elec, feats, apply_asls=APPLY_ASLS)
                except Exception as exc:
                    ax_cv.text(0.5, 0.5, f'CV render error:\n{exc}',
                               transform=ax_cv.transAxes,
                               ha='center', va='center', fontsize=7, color='red')
                    _style_ax(ax_cv, 'Potential (V)', 'Current (µA)',
                              f'CV  |  {ELEC_LABELS[elec]}')

                # DPV
                if V_dpv is not None and dpv_I is not None:
                    try:
                        draw_dpv(ax_dpv, V_dpv, dpv_I, elec, feats, apply_asls=APPLY_ASLS)
                    except Exception as exc:
                        ax_dpv.text(0.5, 0.5, f'DPV render error:\n{exc}',
                                    transform=ax_dpv.transAxes,
                                    ha='center', va='center', fontsize=7, color='red')
                        _style_ax(ax_dpv, 'Potential (V)', 'Current (µA)',
                                  f'DPV  |  {ELEC_LABELS[elec]}')
                else:
                    ax_dpv.text(0.5, 0.5, 'No DPV data',
                                transform=ax_dpv.transAxes,
                                ha='center', va='center', fontsize=9, color='gray')
                    _style_ax(ax_dpv, 'Potential (V)', 'Current (µA)',
                              f'DPV  |  {ELEC_LABELS[elec]}')

                # CA
                if t_ca is not None and ca_I is not None:
                    try:
                        draw_ca(ax_ca, t_ca, ca_I, elec, feats)
                    except Exception as exc:
                        ax_ca.text(0.5, 0.5, f'CA render error:\n{exc}',
                                   transform=ax_ca.transAxes,
                                   ha='center', va='center', fontsize=7, color='red')
                        _style_ax(ax_ca, 'Time (s)', 'Current (µA)',
                                  f'CA  |  {ELEC_LABELS[elec]}')
                else:
                    ax_ca.text(0.5, 0.5, 'No CA data',
                               transform=ax_ca.transAxes,
                               ha='center', va='center', fontsize=9, color='gray')
                    _style_ax(ax_ca, 'Time (s)', 'Current (µA)',
                              f'CA  |  {ELEC_LABELS[elec]}')

                # CA-GC
                if t_cagc is not None and cagc_gens is not None:
                    try:
                        draw_cagc(ax_cagc, t_cagc, cagc_gens, cagc_cols, elec, feats)
                    except Exception as exc:
                        ax_cagc.text(0.5, 0.5, f'CA-GC render error:\n{exc}',
                                     transform=ax_cagc.transAxes,
                                     ha='center', va='center', fontsize=7, color='red')
                        _style_ax(ax_cagc, 'Time (s)', 'Current (µA)',
                                  f'CA-GC  |  {ELEC_LABELS[elec]}')
                else:
                    ax_cagc.text(0.5, 0.5, 'No CA-GC data',
                                 transform=ax_cagc.transAxes,
                                 ha='center', va='center', fontsize=9, color='gray')
                    _style_ax(ax_cagc, 'Time (s)', 'Current (µA)',
                              f'CA-GC  |  {ELEC_LABELS[elec]}')

                # CV-GC
                if V_gc is not None and cvgc_gens is not None:
                    try:
                        draw_cvgc(ax_cvgc, V_gc, cvgc_gens, cvgc_cols, elec)
                    except Exception as exc:
                        ax_cvgc.text(0.5, 0.5, f'CV-GC render error:\n{exc}',
                                     transform=ax_cvgc.transAxes,
                                     ha='center', va='center', fontsize=7, color='red')
                        _style_ax(ax_cvgc, 'Potential (V)', 'Current (µA)',
                                  f'CV-GC  |  {ELEC_LABELS[elec]}')
                else:
                    ax_cvgc.text(0.5, 0.5, 'No CV-GC data',
                                 transform=ax_cvgc.transAxes,
                                 ha='center', va='center', fontsize=9, color='gray')
                    _style_ax(ax_cvgc, 'Potential (V)', 'Current (µA)',
                              f'CV-GC  |  {ELEC_LABELS[elec]}')

            fig.subplots_adjust(hspace=0.50, wspace=0.38, top=0.972, bottom=0.06, left=0.07, right=0.97)
            fig.tight_layout(rect=[0, 0, 1, 0.972])

            # Save individual PNG per electrode
            png_path = os.path.join(cond_out, f'{elec}.png')
            fig.savefig(png_path, dpi=200, bbox_inches='tight')

            # Append to master PDF
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

        print(f'  Cond {cond_num:>2}  ✓  → {cond_out}')

    meta = pdf.infodict()
    meta['Title']  = 'MicroNeedleArrayML — Feature Validation'
    meta['Author'] = 'feature_validation.py'

print(f'\nDone — {len(cond_dirs)} conditions × 3 electrodes = {len(cond_dirs)*3} pages')
print(f'PDF  → {os.path.abspath(args.output)}')
print(f'PNGs → {os.path.abspath(args.output_dir)}/')
