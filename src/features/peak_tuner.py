"""
peak_tuner.py — CV-focused interactive peak-detection tuner.

Scrolls through all conditions across all 4 days in one continuous slider.
Shows 4 CV panels per condition (CV Normal anodic/cathodic, CV GC anodic/cathodic)
with detected peaks annotated directly on the charts.
A feature panel on the right displays all CV features in plain English.

Usage:
    python peak_tuner.py
    python peak_tuner.py --day 2 --cond-num 5 --elec i3
"""

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---


import argparse, os, sys, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.signal import find_peaks

from feature_extract_cond import (
    load_cv_n, load_cv_gc,
    sg, asls, linear_baseline,
    WINDOWS, parse_labels,
    peak_shape_features,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ANALYTE_COLORS = {'AA': '#0072B2', 'DA': '#D55E00', 'UA': '#009E73'}
CAT_SHADE      = 0.55            # darken analyte colour for cathodic markers
GEN_COLOR      = '#222222'
COL_COLOR      = '#AA5500'
WIN_ALPHA      = 0.07
DEFAULT_PROM   = 0.10
ELECTRODES     = ['i1', 'i3', 'i5']
GC_COL_MAP     = {'i1': 'i2', 'i3': 'i4', 'i5': 'i6'}
BASE_ROOT      = 'Day 1-4 Outputs w. TXT'
DAY_UA         = {1: 0, 2: 100, 3: 200, 4: 400}   # UA concentration per day
DAY_DIR_NAMES  = {
    1: 'Outputs_Day_1',
    2: 'Outputs_Day_2_with_txt',
    3: 'Outputs_Day_3_with_txt',
    4: 'Outputs_Day_4_with_txt',
}

# ── Build flat condition list across all 4 days ────────────────────────────────
def _cond_num(name):
    m = re.match(r'Condition_(\d+)$', name)
    return int(m.group(1)) if m else 999

def build_condition_list():
    conds = []
    for day in range(1, 5):
        day_dir = os.path.join(BASE_ROOT, DAY_DIR_NAMES[day])
        if not os.path.isdir(day_dir):
            continue
        names = sorted(
            (d for d in os.listdir(day_dir)
             if re.fullmatch(r'Condition_\d+', d)
             and os.path.isdir(os.path.join(day_dir, d))),
            key=_cond_num,
        )
        for name in names:
            conds.append({
                'day':  day,
                'name': name,
                'num':  _cond_num(name),
                'path': os.path.join(day_dir, name),
            })
    return conds

ALL_CONDS = build_condition_list()
if not ALL_CONDS:
    sys.exit(f'No Condition_N folders found under {BASE_ROOT}/')

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--day',      type=int, default=1)
parser.add_argument('--cond-num', type=int, default=1)
parser.add_argument('--elec',     default='i1', choices=ELECTRODES)
args = parser.parse_args()

def _start_idx(day, num):
    for i, c in enumerate(ALL_CONDS):
        if c['day'] == day and c['num'] == num:
            return i
    return 0

START_IDX = _start_idx(args.day, args.cond_num)

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    'cond_idx':  START_IDX,
    'elec':      args.elec,
    'prom_frac': DEFAULT_PROM,
    'use_asls':  True,
}

# ── Data preparation ───────────────────────────────────────────────────────────
def _corr(I_sg, use_asls):
    if use_asls:
        c, _ = asls(I_sg)
        return c
    return I_sg.copy()

def prepare_cv(cond_path, elec, use_asls):
    """Return dict of {V, I_sg, I_corr} for each of the 4 CV views."""
    base     = cond_path + os.sep
    col_ch   = GC_COL_MAP[elec]

    # ── CV Normal ──────────────────────────────────────────────────────────────
    V_n, cv_I = load_cv_n(base)
    I_n_sg    = sg(cv_I[elec] * 1e6)
    split     = int(np.argmax(V_n))

    # Anodic (forward)
    V_na, I_na = V_n[:split + 1], I_n_sg[:split + 1]
    trim = (V_na >= V_na[0] + 0.05) & (V_na <= V_na[-1] - 0.05)
    V_na, I_na = V_na[trim], I_na[trim]
    I_na_c = _corr(I_na, use_asls)

    # Cathodic (reverse, kept reversed so V goes low→high same as anodic)
    V_nc = V_n[split:][::-1]
    I_nc = I_n_sg[split:][::-1]   # actual cathodic signal (negative peaks)
    # No AsLS for cathodic (AsLS tracks the negative baseline, not useful here)

    # ── CV GC ──────────────────────────────────────────────────────────────────
    V_g, gc_gens, gc_cols = load_cv_gc(base)
    I_gen_sg = sg(gc_gens[elec]  * 1e6)
    I_col_sg = sg(gc_cols[col_ch] * 1e6)
    split_g  = int(np.argmax(V_g))

    # GC Anodic
    V_ga  = V_g[:split_g + 1]
    I_gen = I_gen_sg[:split_g + 1]
    I_col = I_col_sg[:split_g + 1]
    trim_g = (V_ga >= V_ga[0] + 0.05) & (V_ga <= V_ga[-1] - 0.05)
    V_ga, I_gen, I_col = V_ga[trim_g], I_gen[trim_g], I_col[trim_g]
    I_gen_c = _corr(I_gen, use_asls)
    I_col_c = _corr(I_col, use_asls)

    # GC Cathodic
    V_gc   = V_g[split_g:][::-1]
    I_genc = I_gen_sg[split_g:][::-1]
    I_colc = I_col_sg[split_g:][::-1]

    return {
        'norm_a': dict(V=V_na,  I_sg=I_na,   I_corr=I_na_c),
        'norm_c': dict(V=V_nc,  I_sg=I_nc,   I_corr=I_nc),      # display = sg
        'gc_a':   dict(V=V_ga,  I_sg=I_gen,  I_corr=I_gen_c,
                       I_col_sg=I_col, I_col_corr=I_col_c),
        'gc_c':   dict(V=V_gc,  I_sg=I_genc, I_corr=I_genc,    # display = sg
                       I_col_sg=I_colc, I_col_corr=I_colc),
    }

# ── Peak detection ─────────────────────────────────────────────────────────────
def detect(V, I_corr, prom_frac, cathodic=False):
    """
    Detect peaks in all analyte windows.
    cathodic=True → negate signal first so cathodic dips become positive peaks.
    Returns: {analyte: None | {'best': {Ep, Ip, prom}, 'all': [...], 'shape': {...}}}
    Ip is always the ACTUAL (un-negated) current value.
    """
    I_work = -I_corr if cathodic else I_corr
    out    = {}
    for analyte, (v_lo, v_hi) in WINDOWS.items():
        mask = (V >= v_lo) & (V <= v_hi)
        if mask.sum() < 5:
            out[analyte] = None
            continue
        Vw, Iw = linear_baseline(V, I_work, v_lo, v_hi)
        if Iw.max() <= 0:
            out[analyte] = None
            continue
        min_prom = max(Iw.max() * prom_frac, 1e-9)
        peaks, props = find_peaks(Iw, prominence=min_prom)
        full = np.where(mask)[0]
        if len(peaks) == 0:
            out[analyte] = None
            continue

        all_pks = [
            {'Ep':   float(V[full[pi]]),
             'Ip':   float(I_corr[full[pi]]),   # actual (cathodic = negative)
             'prom': float(pr)}
            for pi, pr in zip(peaks, props['prominences'])
        ]
        best_i    = int(np.argmax(Iw[peaks]))
        best_fidx = full[peaks[best_i]]
        shape     = peak_shape_features(V, I_work, best_fidx, v_lo, v_hi)

        out[analyte] = {'best': all_pks[best_i], 'all': all_pks, 'shape': shape}
    return out

# ── Compute features (plain-English keys) ─────────────────────────────────────
def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —    '
    return f'{v:+8.4f}'

def compute_features(cv_data, prom_frac):
    feats = {}
    for section, key_a, key_c in (
        ('CV Normal', 'norm_a', 'norm_c'),
        ('CV GC Gen', 'gc_a',   'gc_c'),
    ):
        d_a = cv_data[key_a]
        d_c = cv_data[key_c]
        pk_a = detect(d_a['V'], d_a['I_corr'], prom_frac, cathodic=False)
        pk_c = detect(d_c['V'], d_c['I_corr'], prom_frac, cathodic=True)

        for analyte in ('AA', 'DA', 'UA'):
            r = pk_a.get(analyte)
            if r:
                bp = r['best'];  sh = r['shape']
                feats[f'{section}|{analyte}|Anodic Peak Current (µA)']  = bp['Ip']
                feats[f'{section}|{analyte}|Anodic Peak Potential (V)'] = bp['Ep']
                feats[f'{section}|{analyte}|Peak Width – FWHM (V)']     = sh.get('FWHM', np.nan)
                feats[f'{section}|{analyte}|Peak Area (µA·V)']          = sh.get('Area', np.nan)
                feats[f'{section}|{analyte}|Pre-Peak Slope (µA/V)']     = sh.get('k_pre', np.nan)
            else:
                for lbl in ('Anodic Peak Current (µA)', 'Anodic Peak Potential (V)',
                            'Peak Width – FWHM (V)', 'Peak Area (µA·V)',
                            'Pre-Peak Slope (µA/V)'):
                    feats[f'{section}|{analyte}|{lbl}'] = np.nan

        # DA cathodic
        rc = pk_c.get('DA');  ra = pk_a.get('DA')
        if rc:
            bpc = rc['best']
            feats[f'{section}|DA|Cathodic Peak Current (µA)']     = bpc['Ip']
            feats[f'{section}|DA|Cathodic Peak Potential (V)']    = bpc['Ep']
            if ra:
                bpa  = ra['best']
                dEp  = abs(bpa['Ep'] - bpc['Ep'])
                E12  = (bpa['Ep'] + bpc['Ep']) / 2.0
                rat  = abs(bpa['Ip'] / bpc['Ip']) if bpc['Ip'] != 0 else np.nan
                feats[f'{section}|DA|Peak-to-Peak Separation (V)']      = dEp
                feats[f'{section}|DA|Half-Wave Potential E½ (V)']       = E12
                feats[f'{section}|DA|Anodic / Cathodic Current Ratio']  = rat
            else:
                for lbl in ('Peak-to-Peak Separation (V)', 'Half-Wave Potential E½ (V)',
                            'Anodic / Cathodic Current Ratio'):
                    feats[f'{section}|DA|{lbl}'] = np.nan
        else:
            for lbl in ('Cathodic Peak Current (µA)', 'Cathodic Peak Potential (V)',
                        'Peak-to-Peak Separation (V)', 'Half-Wave Potential E½ (V)',
                        'Anodic / Cathodic Current Ratio'):
                feats[f'{section}|DA|{lbl}'] = np.nan

    return feats

# ── Feature panel text ─────────────────────────────────────────────────────────
ANODIC_LABELS   = ('Anodic Peak Current (µA)', 'Anodic Peak Potential (V)',
                   'Peak Width – FWHM (V)',     'Peak Area (µA·V)',
                   'Pre-Peak Slope (µA/V)')
CATHODIC_LABELS = ('Cathodic Peak Current (µA)', 'Cathodic Peak Potential (V)',
                   'Peak-to-Peak Separation (V)', 'Half-Wave Potential E½ (V)',
                   'Anodic / Cathodic Current Ratio')
ANALYTE_LONG    = {'AA': 'Ascorbic Acid', 'DA': 'Dopamine', 'UA': 'Uric Acid'}

def build_feature_text(feats, cond_info, elec):
    lines = [
        f"Day {cond_info['day']}  ·  {cond_info['name']}",
        f"DA = {cond_info['da']} µM   AA = {cond_info['aa']} µM   UA = {cond_info['ua']} µM",
        f"Electrode {elec}   Prominence {cond_info['prom']:.3f}   "
        f"{'AsLS ON' if cond_info['asls'] else 'No AsLS'}",
        '',
    ]
    for section in ('CV Normal', 'CV GC Gen'):
        lines.append(f'┌─ {section} {"─"*(30-len(section))}')
        lines.append('│ ANODIC SWEEP')
        for analyte in ('AA', 'DA', 'UA'):
            col_name = ANALYTE_LONG[analyte]
            lines.append(f'│   {col_name}')
            for lbl in ANODIC_LABELS:
                v = feats.get(f'{section}|{analyte}|{lbl}', np.nan)
                lines.append(f'│     {lbl:<38} {_fmt(v)}')
        lines.append('│')
        lines.append('│ CATHODIC SWEEP – Dopamine (quasi-reversible)')
        for lbl in CATHODIC_LABELS:
            v = feats.get(f'{section}|DA|{lbl}', np.nan)
            lines.append(f'│   {lbl:<40} {_fmt(v)}')
        lines.append('│')
    return '\n'.join(lines)

# ── Figure layout ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

fig = plt.figure(figsize=(22, 13))

# Plots + feature panel occupy y = 0.22 … 0.93.
# Widget strip lives in y = 0.00 … 0.20, safely below the plots.
gs_root = gridspec.GridSpec(
    1, 2, figure=fig,
    left=0.04, right=0.99,
    top=0.93,  bottom=0.22,
    wspace=0.03,
    width_ratios=[3.4, 1],
)
gs_plots = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs_root[0],
    hspace=0.48, wspace=0.26,
)

ax_na = fig.add_subplot(gs_plots[0, 0])   # CV Normal  – Anodic
ax_nc = fig.add_subplot(gs_plots[0, 1])   # CV Normal  – Cathodic
ax_ga = fig.add_subplot(gs_plots[1, 0])   # CV GC Gen  – Anodic
ax_gc = fig.add_subplot(gs_plots[1, 1])   # CV GC Gen  – Cathodic

ax_feat = fig.add_subplot(gs_root[1])
ax_feat.axis('off')

for ax in (ax_na, ax_nc, ax_ga, ax_gc):
    ax.set_xlabel('Potential (V)', fontsize=8)
    ax.set_ylabel('Current (µA)', fontsize=8)

# ── Control widgets ────────────────────────────────────────────────────────────
# Sliders span the left 65 % of the figure width.
# Radio + buttons live on the right, well clear of the sliders.
# Nothing exceeds y = 0.19, so nothing overlaps the plot area (bottom = 0.22).
#
#  y=0.14  ──────  Condition slider  ──────────────────────
#  y=0.09  ──────  Prominence slider ──────────────────────
#  y=0.03–0.19  [Radio]   y=0.12–0.19 [AsLS]  y=0.05–0.12 [Print]
n = len(ALL_CONDS)
ax_csl  = fig.add_axes([0.07, 0.140, 0.60, 0.025])   # Condition slider
ax_psl  = fig.add_axes([0.07, 0.085, 0.60, 0.025])   # Prominence slider
ax_rad  = fig.add_axes([0.72, 0.030, 0.07, 0.160])   # Electrode radio (right side)
ax_ab   = fig.add_axes([0.81, 0.120, 0.10, 0.055])   # AsLS toggle
ax_pb   = fig.add_axes([0.81, 0.050, 0.10, 0.055])   # Print info
ax_ttl  = fig.add_axes([0.04, 0.955, 0.92, 0.034])
ax_ttl.axis('off')

title_txt = ax_ttl.text(
    0.5, 0.5, '', transform=ax_ttl.transAxes,
    fontsize=10, ha='center', va='center',
    color='#222222', family='monospace', fontweight='bold',
)

cond_slider = Slider(ax_csl, 'Condition',
                     0, n - 1, valinit=START_IDX, valstep=1, color='#88AA55')
prom_slider = Slider(ax_psl, 'Prominence (fraction of window max)',
                     0.01, 0.60, valinit=DEFAULT_PROM, valstep=0.005, color='#5588CC')
radio    = RadioButtons(ax_rad, ELECTRODES,
                        active=ELECTRODES.index(args.elec), activecolor='#333333')
btn_asls = Button(ax_ab, 'AsLS: ON',   color='#AADDAA', hovercolor='#88CC88')
btn_prnt = Button(ax_pb, 'Print info', color='#EEEEEE', hovercolor='#CCDDFF')

# ── Per-axis artist lists ──────────────────────────────────────────────────────
_arts = {k: [] for k in ('na', 'nc', 'ga', 'gc', 'feat')}

def _clear(tag):
    for a in _arts[tag]:
        try:
            a.remove()
        except Exception:
            pass
    _arts[tag] = []

# ── Drawing helpers ────────────────────────────────────────────────────────────
def _shade_windows(ax, y_label_frac=0.98):
    for analyte, (vl, vh) in WINDOWS.items():
        ax.axvspan(vl, vh, color=ANALYTE_COLORS[analyte], alpha=WIN_ALPHA, zorder=0)
        ax.axvline(vl, color=ANALYTE_COLORS[analyte], lw=0.4, ls='--', alpha=0.35)
        ax.axvline(vh, color=ANALYTE_COLORS[analyte], lw=0.4, ls='--', alpha=0.35)
        ax.text((vl + vh) / 2, y_label_frac,
                analyte, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7,
                color=ANALYTE_COLORS[analyte], fontweight='bold')

def _rescale(ax, *arrays):
    vals = np.concatenate(arrays)
    lo, hi = vals.min(), vals.max()
    pad = max((hi - lo) * 0.15, 0.05)
    ax.set_ylim(lo - pad, hi + pad * 2.8)

def _cat_color(analyte):
    """Darken the analyte colour for cathodic markers."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(ANALYTE_COLORS[analyte]))
    return tuple(rgb * CAT_SHADE)

def _draw_panel(ax, tag, V, I_sg, I_corr, peaks, cathodic=False,
                I_col_sg=None, I_col_corr=None):
    """Draw one CV panel; return list of summary strings."""
    _clear(tag)
    A = _arts[tag]

    A.append(ax.plot(V, I_sg,   color='#CCCCCC', lw=0.8, alpha=0.5, zorder=1)[0])
    A.append(ax.plot(V, I_corr, color=GEN_COLOR,  lw=1.3, zorder=2)[0])

    if I_col_corr is not None:
        A.append(ax.plot(V, I_col_sg,   color='#FFCC88', lw=0.7, alpha=0.5, zorder=1)[0])
        A.append(ax.plot(V, I_col_corr, color=COL_COLOR,  lw=1.0, ls='--',
                          label='Collector', zorder=2)[0])
        A.append(ax.legend(loc='lower right', fontsize=7))

    A.append(ax.axhline(0, color='#AAAAAA', lw=0.5, ls=':'))
    _shade_windows(ax, y_label_frac=0.99 if not cathodic else 0.99)

    summary = []
    for analyte, result in peaks.items():
        col = ANALYTE_COLORS[analyte] if not cathodic else _cat_color(analyte)
        if result is None:
            summary.append(f'{analyte}: —')
            continue

        for pk in result['all']:
            sc = ax.scatter(pk['Ep'], pk['Ip'], s=22,
                             facecolors='none', edgecolors=col,
                             lw=0.8, zorder=5, alpha=0.65)
            A.append(sc)

        bp  = result['best']
        sc  = ax.scatter(bp['Ep'], bp['Ip'], s=110, marker='*', color=col, zorder=7)
        A.append(sc)
        vl  = ax.axvline(bp['Ep'], color=col, lw=0.7, ls=':', alpha=0.45, zorder=4)
        A.append(vl)

        # Annotation: top of axes for anodic; bottom third for cathodic
        y_frac = 0.97 if not cathodic else 0.35
        va_txt = 'top'
        txt = ax.text(
            bp['Ep'], y_frac,
            f"{ANALYTE_LONG[analyte]}\nEp = {bp['Ep']:.3f} V\nIp = {bp['Ip']:.3f} µA",
            transform=ax.get_xaxis_transform(),
            ha='center', va=va_txt, fontsize=6.3, color=col,
            bbox=dict(fc='white', alpha=0.88, ec=col,
                      boxstyle='round,pad=0.18', lw=0.6),
            zorder=9,
        )
        A.append(txt)
        sign_str = '' if not cathodic else '  ↓'
        summary.append(f'{analyte} Ep={bp["Ep"]:.3f}V  Ip={bp["Ip"]:.3f}µA{sign_str}')

    return summary

# ── Full redraw ────────────────────────────────────────────────────────────────
def redraw():
    cond  = ALL_CONDS[state['cond_idx']]
    elec  = state['elec']
    pf    = state['prom_frac']
    asls_ = state['use_asls']

    try:
        da, aa, ua = parse_labels(cond['path'] + os.sep)
    except Exception:
        da = aa = ua = '?'

    cond_info = {**cond, 'da': da, 'aa': aa, 'ua': ua,
                 'prom': pf, 'asls': asls_}

    # ── Title ─────────────────────────────────────────────────────────────────
    idx = state['cond_idx']
    ua_day = DAY_UA.get(cond['day'], '?')
    title_txt.set_text(
        f"[{idx + 1}/{len(ALL_CONDS)}]  Day {cond['day']} (UA={ua_day} µM)  ·  {cond['name']}  |  "
        f"DA = {da} µM   AA = {aa} µM   UA = {ua} µM  |  "
        f"Electrode {elec}  |  Prominence {pf:.2f}  |  "
        f"{'AsLS ON' if asls_ else 'No AsLS (SG only)'}"
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        cv = prepare_cv(cond['path'], elec, asls_)
    except Exception as e:
        print(f'[peak_tuner] Error loading {cond["path"]}: {e}')
        return

    # ── Detect peaks ───────────────────────────────────────────────────────────
    pk_na = detect(cv['norm_a']['V'], cv['norm_a']['I_corr'], pf, cathodic=False)
    pk_nc = detect(cv['norm_c']['V'], cv['norm_c']['I_corr'], pf, cathodic=True)
    pk_ga = detect(cv['gc_a']['V'],   cv['gc_a']['I_corr'],   pf, cathodic=False)
    pk_gc = detect(cv['gc_c']['V'],   cv['gc_c']['I_corr'],   pf, cathodic=True)

    # ── Rescale & set x-limits ─────────────────────────────────────────────────
    for ax, key, include_col in (
        (ax_na, 'norm_a', False),
        (ax_nc, 'norm_c', False),
        (ax_ga, 'gc_a',   True),
        (ax_gc, 'gc_c',   True),
    ):
        d = cv[key]
        arrays = [d['I_sg'], d['I_corr']]
        if include_col:
            arrays += [d['I_col_sg'], d['I_col_corr']]
        _rescale(ax, *arrays)
        ax.set_xlim(d['V'].min() - 0.01, d['V'].max() + 0.01)

    # ── Draw panels ───────────────────────────────────────────────────────────
    lbl = 'AsLS' if asls_ else 'SG'
    s_na = _draw_panel(ax_na, 'na', cv['norm_a']['V'],
                       cv['norm_a']['I_sg'], cv['norm_a']['I_corr'], pk_na)
    s_nc = _draw_panel(ax_nc, 'nc', cv['norm_c']['V'],
                       cv['norm_c']['I_sg'], cv['norm_c']['I_corr'], pk_nc, cathodic=True)
    s_ga = _draw_panel(ax_ga, 'ga', cv['gc_a']['V'],
                       cv['gc_a']['I_sg'], cv['gc_a']['I_corr'], pk_ga,
                       I_col_sg=cv['gc_a']['I_col_sg'], I_col_corr=cv['gc_a']['I_col_corr'])
    s_gc = _draw_panel(ax_gc, 'gc', cv['gc_c']['V'],
                       cv['gc_c']['I_sg'], cv['gc_c']['I_corr'], pk_gc, cathodic=True,
                       I_col_sg=cv['gc_c']['I_col_sg'], I_col_corr=cv['gc_c']['I_col_corr'])

    # ── Axis titles ────────────────────────────────────────────────────────────
    ax_na.set_title(f'CV Normal  —  Anodic Sweep  ({lbl})   electrode {elec}',   fontsize=8)
    ax_nc.set_title(f'CV Normal  —  Cathodic Sweep  (SG)    electrode {elec}',   fontsize=8)
    ax_ga.set_title(f'CV GC  —  Anodic Sweep  ({lbl})       electrode {elec}',   fontsize=8)
    ax_gc.set_title(f'CV GC  —  Cathodic Sweep  (SG)        electrode {elec}',   fontsize=8)

    # ── Feature panel ─────────────────────────────────────────────────────────
    feats = compute_features(cv, pf)
    _clear('feat')
    _arts['feat'].append(
        ax_feat.text(
            0.02, 0.99,
            build_feature_text(feats, cond_info, elec),
            transform=ax_feat.transAxes,
            fontsize=6.8, va='top', ha='left',
            family='monospace', color='#1a1a1a',
            linespacing=1.45,
        )
    )

    fig.canvas.draw_idle()

# ── Callbacks ──────────────────────────────────────────────────────────────────
def on_cond(val):
    state['cond_idx'] = max(0, min(int(round(val)), len(ALL_CONDS) - 1))
    redraw()

def on_prom(val):
    state['prom_frac'] = val
    redraw()

def on_elec(label):
    state['elec'] = label
    redraw()

def on_asls(event):
    state['use_asls'] = not state['use_asls']
    if state['use_asls']:
        btn_asls.label.set_text('AsLS: ON')
        btn_asls.ax.set_facecolor('#AADDAA')
    else:
        btn_asls.label.set_text('AsLS: OFF')
        btn_asls.ax.set_facecolor('#FFCCAA')
    redraw()

def on_print(event):
    c = ALL_CONDS[state['cond_idx']]
    print(f"\nDay {c['day']}  {c['name']}  ({c['path']})")
    print(f"Electrode: {state['elec']}   Prominence: {state['prom_frac']:.4f}   AsLS: {state['use_asls']}")

cond_slider.on_changed(on_cond)
prom_slider.on_changed(on_prom)
radio.on_clicked(on_elec)
btn_asls.on_clicked(on_asls)
btn_prnt.on_clicked(on_print)

# ── Initial draw ───────────────────────────────────────────────────────────────
redraw()
plt.show()
