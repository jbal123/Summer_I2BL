"""
generate_feature_reference.py — Generates feature_reference.pdf

A visual reference guide explaining every extracted electrochemical feature:
what it is, what it measures, its units, and why it matters for DA/AA/UA detection.

One page per technique + a cover page and an inter-analyte ratios page.

Usage:
    python generate_feature_reference.py
"""

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from pathlib import Path as _Path
OUT_PATH = str(_Path(__file__).resolve().parents[2] / "docs" / "reference" / "feature_reference.pdf")

# ── Color palette ──────────────────────────────────────────────────────────────
C_DA   = '#D55E00'
C_AA   = '#0072B2'
C_UA   = '#009E73'
C_GREY = '#555555'

TECH_COLORS = {
    'SWV':   '#8B0000',
    'CV':    '#00008B',
    'DPV':   '#006400',
    'CA':    '#4B0082',
    'CA-GC': '#8B4513',
    'CV-GC': '#2F4F4F',
}

ROW_EVEN = '#F5F7FA'
ROW_ODD  = '#FFFFFF'
HDR_BG   = '#2C3E50'
HDR_FG   = 'white'

PAGE_W, PAGE_H = 11, 8.5   # landscape letter


# ── Helpers ────────────────────────────────────────────────────────────────────
def new_page(pdf, title, subtitle, accent):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('white')

    # Top accent bar
    bar = mpatches.FancyBboxPatch((0, PAGE_H - 0.72), PAGE_W, 0.72,
                                  boxstyle='square,pad=0', transform=fig.transFigure,
                                  fc=accent, ec='none', zorder=0, clip_on=False)
    fig.add_artist(bar)

    fig.text(0.04, 1 - 0.36/PAGE_H, title,
             ha='left', va='center', fontsize=22, fontweight='bold',
             color='white', transform=fig.transFigure)
    fig.text(0.04, 1 - 0.60/PAGE_H, subtitle,
             ha='left', va='center', fontsize=11, color='#DDDDDD',
             transform=fig.transFigure, style='italic')

    # Footer
    fig.text(0.5, 0.012, 'MicroNeedleArray ML — Electrochemical Feature Reference',
             ha='center', va='bottom', fontsize=7.5, color='#888888',
             transform=fig.transFigure)

    return fig


def draw_table(fig, rows, col_widths, col_headers, top, left=0.03,
               row_height=0.062, accent='#2C3E50'):
    """
    Draw a table in figure-fraction coordinates.

    rows        : list of tuples — one tuple per data row
    col_widths  : list of column widths as figure fractions (must sum ≤ 1)
    col_headers : list of header strings
    top         : figure-fraction y for the top of the header row
    """
    n_cols = len(col_headers)
    xs = [left]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w)

    # Header row
    fig.add_artist(mpatches.FancyBboxPatch(
        (left, top - row_height), sum(col_widths), row_height,
        boxstyle='square,pad=0', transform=fig.transFigure,
        fc=accent, ec='none', clip_on=False))
    for j, (hdr, x, w) in enumerate(zip(col_headers, xs, col_widths)):
        fig.text(x + 0.008, top - row_height / 2, hdr,
                 ha='left', va='center', fontsize=8.5, fontweight='bold',
                 color='white', transform=fig.transFigure)

    # Data rows
    y = top - row_height
    for i, row in enumerate(rows):
        bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
        fig.add_artist(mpatches.FancyBboxPatch(
            (left, y - row_height), sum(col_widths), row_height,
            boxstyle='square,pad=0', transform=fig.transFigure,
            fc=bg, ec='#DDDDDD', lw=0.4, clip_on=False))

        for j, (cell, x, w) in enumerate(zip(row, xs, col_widths)):
            color = '#111111'
            weight = 'normal'
            # First column (feature name) in monospace bold
            if j == 0:
                weight = 'bold'
                family = 'monospace'
            else:
                family = 'DejaVu Sans'
            # Colour-code analyte column
            if j == 1:
                cell_str = str(cell)
                if 'DA' in cell_str:   color = C_DA
                elif 'AA' in cell_str: color = C_AA
                elif 'UA' in cell_str: color = C_UA
                else:                  color = C_GREY

            fig.text(x + 0.008, y - row_height / 2, str(cell),
                     ha='left', va='center', fontsize=7.5,
                     color=color, fontweight=weight,
                     fontfamily=family,
                     transform=fig.transFigure,
                     wrap=False)
        y -= row_height

    return y   # bottom of last row


def description_box(fig, text, top, left=0.03, width=0.94, accent='#2C3E50'):
    """Draw a light shaded description paragraph."""
    lines = text.split('\n')
    line_h = 0.032
    box_h  = len(lines) * line_h + 0.018
    fig.add_artist(mpatches.FancyBboxPatch(
        (left, top - box_h), width, box_h,
        boxstyle='round,pad=0.005', transform=fig.transFigure,
        fc='#EEF2F7', ec=accent, lw=0.8, clip_on=False))
    for k, line in enumerate(lines):
        fig.text(left + 0.012, top - 0.012 - k * line_h, line,
                 ha='left', va='top', fontsize=8.2, color='#222222',
                 transform=fig.transFigure)
    return top - box_h - 0.012


# ── PAGE 0: Cover ──────────────────────────────────────────────────────────────
def page_cover(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor('#1A252F')

    fig.text(0.5, 0.70, 'Electrochemical Feature\nReference Guide',
             ha='center', va='center', fontsize=38, fontweight='bold',
             color='white', transform=fig.transFigure, linespacing=1.3)

    fig.text(0.5, 0.55, 'MicroNeedleArray ML — Feature Definitions & Electrochemical Context',
             ha='center', va='center', fontsize=14, color='#AAAAAA',
             transform=fig.transFigure, style='italic')

    # Analyte legend chips
    for i, (analyte, color, full) in enumerate([
        ('DA', C_DA, 'Dopamine'),
        ('AA', C_AA, 'Ascorbic Acid'),
        ('UA', C_UA, 'Uric Acid'),
    ]):
        x = 0.30 + i * 0.14
        chip = mpatches.FancyBboxPatch((x - 0.04, 0.41), 0.12, 0.045,
                                        boxstyle='round,pad=0.005',
                                        transform=fig.transFigure,
                                        fc=color, ec='none', clip_on=False)
        fig.add_artist(chip)
        fig.text(x, 0.432, analyte,   ha='center', va='center',
                 fontsize=13, fontweight='bold', color='white',
                 transform=fig.transFigure)
        fig.text(x, 0.405, full,      ha='center', va='center',
                 fontsize=9,  color='#CCCCCC', transform=fig.transFigure)

    # Techniques list
    tech_desc = [
        ('SWV', 'Square Wave Voltammetry'),
        ('CV',  'Cyclic Voltammetry'),
        ('DPV', 'Differential Pulse Voltammetry'),
        ('CA',  'Chronoamperometry'),
        ('CA-GC', 'Chronoamperometry — Generator/Collector'),
        ('CV-GC', 'Cyclic Voltammetry — Generator/Collector'),
    ]
    for i, (abbr, full) in enumerate(tech_desc):
        x = 0.22 if i < 3 else 0.57
        y = 0.30 - (i % 3) * 0.055
        color = TECH_COLORS[abbr]
        chip = mpatches.FancyBboxPatch((x - 0.02, y - 0.012), 0.065, 0.030,
                                        boxstyle='round,pad=0.003',
                                        transform=fig.transFigure,
                                        fc=color, ec='none', clip_on=False)
        fig.add_artist(chip)
        fig.text(x + 0.013, y + 0.003, abbr, ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white',
                 transform=fig.transFigure)
        fig.text(x + 0.055, y + 0.003, full, ha='left', va='center',
                 fontsize=9, color='#CCCCCC', transform=fig.transFigure)

    fig.text(0.5, 0.04, 'Each feature is extracted per electrode (i1 / i3 / i5) independently.',
             ha='center', fontsize=9, color='#888888', transform=fig.transFigure)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 1: SWV ───────────────────────────────────────────────────────────────
def page_swv(pdf):
    accent = TECH_COLORS['SWV']
    fig = new_page(pdf,
                   'SWV — Square Wave Voltammetry',
                   'High-sensitivity pulse technique that separates faradaic (analyte) signal from capacitive (background) noise.',
                   accent)

    principle = (
        'The potential is stepped as a square wave superimposed on a staircase ramp. Current is sampled at the end of each forward and reverse pulse; '
        'the DIFFERENTIAL current (Δi = i_fwd − i_rev) greatly suppresses double-layer charging current.\n'
        'Result: clean, symmetric peaks whose height is directly proportional to analyte concentration. '
        'Only replicate 3 is used (reps 1–2 are conditioning sweeps). '
        'Signal is SG-smoothed → AsLS baseline-corrected before peak extraction.'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Feature', 'Analyte', 'Unit', 'Physical Meaning', 'Why it matters for detection']
    rows = [
        ('swv_{A}_Ip',           'AA / DA / UA', 'µA',   'Peak differential current (i_fwd − i_rev) at Ep.',
         'Primary concentration indicator — Ip scales linearly with [analyte] via Cottrell/SWV theory.'),
        ('swv_{A}_Ep',           'AA / DA / UA', 'V',    'Potential at which peak current occurs.',
         'Analyte fingerprint — AA ≈ 0.14V, DA ≈ 0.22V, UA ≈ 0.36V. Separates analytes in potential space.'),
        ('swv_{A}_FWHM',         'AA / DA / UA', 'V',    'Full Width at Half Maximum of the corrected peak.',
         'Peak sharpness. Broadening suggests kinetic limitations, adsorption, or overlapping signals.'),
        ('swv_{A}_Area',         'AA / DA / UA', 'µA·V', 'Integrated area under baseline-subtracted peak.',
         'Proportional to total charge transferred. More robust than Ip when peak shape varies.'),
        ('swv_{A}_k_pre',        'AA / DA / UA', 'µA/V', 'Slope of the rising edge before the peak maximum.',
         'Reflects electron-transfer rate; steeper slope = faster kinetics at that electrode.'),
        ('swv_{A}_AI',           'AA / DA / UA', '—',    'Asymmetry Index = W_left / W_right at half-height.',
         'AI = 1 for symmetric peaks. Deviations indicate adsorption effects or overlapping analytes.'),
        ('swv_{A}_Ip_fwd',       'AA / DA / UA', 'µA',   'Raw forward-pulse current at peak potential.',
         'Used in fwd/rev ratio; also a direct measure of forward oxidation current.'),
        ('swv_{A}_Ip_rev',       'AA / DA / UA', 'µA',   'Raw reverse-pulse current at peak potential.',
         'Negative for reversible systems. Magnitude relative to Ip_fwd indicates reversibility.'),
        ('swv_{A}_fwd_rev_ratio','AA / DA / UA', '—',    'log(|Ip_fwd| / |Ip_rev|). Log-ratio of pulse currents.',
         '≈ 0 for fully reversible redox. Increases with irreversibility (AA and UA are irreversible).'),
    ]
    col_widths = [0.165, 0.095, 0.055, 0.295, 0.375]
    draw_table(fig, rows, col_widths, headers, top=y, accent=HDR_BG)

    # Analyte note
    fig.text(0.03, 0.05,
             '{A} = analyte placeholder: AA (Ascorbic Acid, 0.10–0.20 V window)  ·  '
             'DA (Dopamine, 0.12–0.32 V window)  ·  UA (Uric Acid, 0.30–0.55 V window)',
             ha='left', va='bottom', fontsize=7.5, color='#555555',
             transform=fig.transFigure, style='italic')

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 2: CV ────────────────────────────────────────────────────────────────
def page_cv(pdf):
    accent = TECH_COLORS['CV']
    fig = new_page(pdf,
                   'CV — Cyclic Voltammetry',
                   'Potential swept linearly forward then reversed; records the full oxidation and reduction response of the analyte.',
                   accent)

    principle = (
        'The electrode potential is ramped from V_start → V_max (forward sweep, oxidation) then reversed V_max → V_start (reverse sweep, reduction).\n'
        'Features are extracted from the SG-smoothed, AsLS baseline-corrected forward sweep (anodic peaks) and reverse sweep (DA cathodic peak only).\n'
        'AA and UA show only anodic peaks (irreversible). DA shows both anodic and cathodic peaks (quasi-reversible redox couple).'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Feature', 'Analyte', 'Unit', 'Physical Meaning', 'Why it matters for detection']
    rows = [
        ('cv_{A}_Ipa',      'AA / DA / UA', 'µA',   'Anodic (oxidation) peak current on forward sweep.',
         'Scales with concentration. CV is slower than SWV/DPV so Ip is smaller but more stable.'),
        ('cv_{A}_Ep_a',     'AA / DA / UA', 'V',    'Potential at anodic peak maximum (forward sweep).',
         'Analyte ID. Shifts with pH and ionic strength — useful for detecting matrix changes.'),
        ('cv_{A}_FWHM',     'AA / DA / UA', 'V',    'Full Width at Half Maximum of anodic peak.',
         'Narrows for surface-confined species (adsorption), broadens for diffusion-limited or coupled reactions.'),
        ('cv_{A}_Area',     'AA / DA / UA', 'µA·V', 'Integrated area under baseline-subtracted anodic peak.',
         'Proportional to charge; combines peak height and width — useful when peak shape shifts.'),
        ('cv_{A}_k_pre',    'AA / DA / UA', 'µA/V', 'Pre-peak slope on forward sweep.',
         'Steeper slope = faster heterogeneous electron transfer rate constant (k°).'),
        ('cv_DA_Ipc',       'DA only',      'µA',   'Cathodic (reduction) peak current on reverse sweep.',
         'Confirms DA reversibility. Ipa/Ipc ratio characterises the redox mechanism.'),
        ('cv_DA_Ep_c',      'DA only',      'V',    'Cathodic peak potential on reverse sweep.',
         'Paired with Ep_a to compute ΔEp and E½.'),
        ('cv_DA_deltaEp',   'DA only',      'V',    'ΔEp = Ep_a − Ep_c. Peak-to-peak separation.',
         '59/n mV for fully reversible (n = electrons). Larger ΔEp indicates quasi-reversible / slow kinetics.'),
        ('cv_DA_E12',       'DA only',      'V',    "E½ = (Ep_a + Ep_c)/2 ≈ formal potential E°'.",
         'Thermodynamic quantity. Shifts with pH (−59 mV/pH for DA). Stable fingerprint.'),
        ('cv_DA_Ipa_Ipc',   'DA only',      '—',    '|Ipa / Ipc| ratio.',
         '= 1 for ideal reversible. > 1 suggests chemical follow-up reaction (EC mechanism).'),
        ('cv_{e}_baseline_low',  '—', 'µA',  'Current at V = −0.05 V (before any oxidation).',
         'Background / double-layer reference. Shifts with electrode fouling or matrix changes.'),
        ('cv_{e}_baseline_high', '—', 'µA',  'Current at V = 0.70 V (post-UA oxidation).',
         'Background reference at high potential; captures residual slope of faradaic tail.'),
    ]
    col_widths = [0.175, 0.095, 0.048, 0.28, 0.385]
    draw_table(fig, rows, col_widths, headers, top=y, row_height=0.057, accent=HDR_BG)

    fig.text(0.03, 0.05,
             '{A} = AA / DA / UA   ·   {e} = electrode label (i1, i3, i5)',
             ha='left', va='bottom', fontsize=7.5, color='#555555',
             transform=fig.transFigure, style='italic')

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 3: DPV ───────────────────────────────────────────────────────────────
def page_dpv(pdf):
    accent = TECH_COLORS['DPV']
    fig = new_page(pdf,
                   'DPV — Differential Pulse Voltammetry',
                   'Fixed-amplitude pulses superimposed on a staircase; current sampled before and after each pulse to cancel background.',
                   accent)

    principle = (
        'The potential staircase steps slowly; at each step a fixed-height pulse is applied. '
        'Current is sampled just before the pulse and just before the next step; ΔI = I_after − I_before.\n'
        'This differential measurement cancels slowly-varying capacitive current, giving better sensitivity than linear-sweep CV '
        'but lower than SWV. DPV peaks for reduction reactions are negative — the signal is negated before processing '
        'so extracted features (Ip, Area) are reported as positive numbers.'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Feature', 'Analyte', 'Unit', 'Physical Meaning', 'Why it matters for detection']
    rows = [
        ('dpv_{A}_Ip',   'AA / DA / UA', 'µA',   'Differential pulse peak current (negated, so positive).',
         'Intermediate sensitivity between CV and SWV. Complements SWV when peaks overlap.'),
        ('dpv_{A}_Ep',   'AA / DA / UA', 'V',    'Potential at peak maximum.',
         'Ep(DPV) ≈ E½ − ΔE/2 where ΔE is the pulse amplitude. Slightly different from SWV Ep.'),
        ('dpv_{A}_Area', 'AA / DA / UA', 'µA·V', 'Integrated area under baseline-corrected peak.',
         'Integral is robust to noise; captures total faradaic charge better than Ip alone.'),
    ]
    col_widths = [0.14, 0.095, 0.055, 0.32, 0.375]
    draw_table(fig, rows, col_widths, headers, top=y, accent=HDR_BG)

    # Inter-analyte ratios sub-section
    y2 = y - len(rows) * 0.062 - 0.075
    fig.text(0.03, y2 + 0.035, 'Inter-Analyte Ratios  (computed for both SWV and DPV)',
             ha='left', fontsize=11, fontweight='bold', color=accent,
             transform=fig.transFigure)

    ratio_desc = (
        'After per-analyte features are extracted, five cross-analyte ratios are computed for each technique '
        '(prefix = swv_ or dpv_).\n'
        'These ratios encode the relative sensitivity of the electrode to each analyte pair — crucial when '
        'concentrations co-vary and individual peaks shift together.'
    )
    y2 = description_box(fig, ratio_desc, top=y2 + 0.010, accent=accent)

    ratio_rows = [
        ('swv/dpv_ratio_Ip_DA_AA', 'DA vs AA', '—',    'Ip_DA / Ip_AA',
         'DA-to-AA sensitivity ratio. Helps the model decouple DA signal from AA background.'),
        ('swv/dpv_ratio_Ip_DA_UA', 'DA vs UA', '—',    'Ip_DA / Ip_UA',
         'DA-to-UA sensitivity ratio. UA is often the largest signal; ratio normalises it.'),
        ('swv/dpv_ratio_Ip_UA_AA', 'UA vs AA', '—',    'Ip_UA / Ip_AA',
         'Relative peak sizes between the two irreversible analytes.'),
        ('swv/dpv_dEp_DA_AA',      'DA − AA',  'V',    'Ep_DA − Ep_AA',
         'Potential separation. Larger = better electrochemical resolution between the two analytes.'),
        ('swv/dpv_dEp_UA_DA',      'UA − DA',  'V',    'Ep_UA − Ep_DA',
         'Potential separation between UA and DA peaks.'),
    ]
    col_widths = [0.215, 0.085, 0.045, 0.17, 0.465]
    draw_table(fig, ratio_rows, col_widths, headers, top=y2, accent=HDR_BG)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 4: CA ────────────────────────────────────────────────────────────────
def page_ca(pdf):
    accent = TECH_COLORS['CA']
    fig = new_page(pdf,
                   'CA — Chronoamperometry',
                   'A potential step is applied and the current decay over time is recorded — probes diffusion and electrode kinetics.',
                   accent)

    principle = (
        'The electrode is held at rest potential, then stepped to an oxidising potential. '
        'The initial current spike (I₀) reflects fast surface processes; the decay follows the Cottrell equation '
        'I(t) = nFAC√(D/πt) for planar diffusion. The steady-state current (Iss) at long times reflects the '
        'geometry-dependent diffusion limit of the microneedle array. '
        'No SG/AsLS filtering is applied — the raw current transient is used directly.'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Feature', 'Analyte', 'Unit', 'Physical Meaning', 'Why it matters for detection']
    rows = [
        ('ca_I0',          '(all)',  'µA',      'Current at t = 0.1 s (earliest reliable sample after step).',
         'Proportional to total electroactive species. Dominated by double-layer charging + fast faradaic.'),
        ('ca_Iss',         '(all)',  'µA',      'Mean current averaged over t = 18–20 s.',
         'Diffusion steady-state for microneedle geometry. Scales with analyte concentration and D.'),
        ('ca_Iss_I0',      '(all)',  '—',       'Iss / I0 ratio.',
         'Describes how quickly diffusion relaxes. Sensitive to electrode geometry and analyte mobility.'),
        ('ca_cott_slope',  '(all)',  'µA·s½',   'Slope of I vs t^−½ in the Cottrell window (0.5–5 s).',
         'From Cottrell: slope = nFAC√(D/π). Linear in C×√D — key concentration feature.'),
        ('ca_cott_R2',     '(all)',  '—',       'R² of Cottrell (linear) fit.',
         'Quality metric. Low R² suggests non-planar diffusion, surface fouling, or signal noise.'),
        ('ca_Q_early',     '(all)',  'µA·s',    'Integrated charge, t = 0.1–5 s (early region).',
         'Captures combined double-layer + faradaic charge. Sensitive to surface area and fast kinetics.'),
        ('ca_Q_late',      '(all)',  'µA·s',    'Integrated charge, t = 10–20 s (late region).',
         'Pure diffusion-dominated charge. More selective for analyte concentration than Q_early.'),
        ('ca_Q_late_early','(all)',  '—',       'Q_late / Q_early ratio.',
         'High ratio = diffusion-controlled. Low ratio = surface or adsorption processes dominate.'),
    ]
    col_widths = [0.155, 0.065, 0.068, 0.30, 0.395]
    draw_table(fig, rows, col_widths, headers, top=y, accent=HDR_BG)

    fig.text(0.03, 0.05,
             '(all) = feature is not analyte-specific; it integrates the response of all electroactive species '
             'present at the step potential.',
             ha='left', va='bottom', fontsize=7.5, color='#555555',
             transform=fig.transFigure, style='italic')

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 5: CA-GC ─────────────────────────────────────────────────────────────
def page_cagc(pdf):
    accent = TECH_COLORS['CA-GC']
    fig = new_page(pdf,
                   'CA-GC — Chronoamperometry, Generator/Collector',
                   'Dual-electrode mode: one electrode oxidises the analyte (generator); an adjacent one re-reduces it (collector).',
                   accent)

    principle = (
        'In the generator/collector (GC) configuration, the generator electrode is stepped to oxidise analytes as in normal CA. '
        'A neighbouring collector electrode is held at a reducing potential and captures (re-reduces) species diffusing from the generator.\n'
        'The collection efficiency CE = −I_col / I_gen quantifies how many oxidised molecules are caught before diffusing away. '
        'CE and related features are strong functions of electrode spacing, geometry, and analyte diffusion coefficient — '
        'independent of bulk concentration, making them complementary to the concentration-sensitive normal CA features.'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Feature', 'Pair', 'Unit', 'Physical Meaning', 'Why it matters for detection']
    rows = [
        ('cagc_CE_1s',  'i1→i2 / i3→i4 / i5→i6', '—',
         'Collection efficiency at t = 1 s:  −I_col(1s) / I_gen(1s).',
         'Early CE dominated by fast-diffusing species. Sensitive to analyte diffusion coefficient D.'),
        ('cagc_CE_10s', 'i1→i2 / i3→i4 / i5→i6', '—',
         'Collection efficiency at t = 10 s:  −I_col(10s) / I_gen(10s).',
         'Later CE reflects steady-state geometry. Less concentration-dependent — encodes D and geometry.'),
        ('cagc_CE_ss',  'i1→i2 / i3→i4 / i5→i6', '—',
         'Steady-state CE: −mean(I_col) / mean(I_gen) over t = 18–20 s.',
         'True geometric CE of the electrode pair. Changes with electrode fouling or blocked diffusion.'),
        ('cagc_AF',     'i1→i2 / i3→i4 / i5→i6', '—',
         'Amplification Factor: (|I_gen_ss| + |I_col_ss|) / Iss_normal.',
         'Total current output relative to single-electrode mode. Higher AF = more efficient redox cycling.'),
        ('cagc_eta10',  'i1→i2 / i3→i4 / i5→i6', '—',
         'Net feedback efficiency at t = 10 s: (I_gen − |I_col|) / (I_gen + |I_col|).',
         'Ranges −1 to +1. Near 0 = perfect collection. Near +1 = little collection (low CE).'),
    ]
    col_widths = [0.135, 0.205, 0.045, 0.265, 0.335]
    draw_table(fig, rows, col_widths, headers, top=y, accent=HDR_BG)

    # Electrode pair map
    y3 = y - len(rows) * 0.062 - 0.04
    fig.text(0.03, y3,
             'Electrode pair map:   i1 (gen) ↔ i2 (col)   ·   i3 (gen) ↔ i4 (col)   ·   i5 (gen) ↔ i6 (col)\n'
             'After electrode-label stripping, all three pairs share the same feature names (cagc_CE_1s, etc.).',
             ha='left', va='top', fontsize=8, color='#444444',
             transform=fig.transFigure)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 6: CV-GC ─────────────────────────────────────────────────────────────
def page_cvgc(pdf):
    accent = TECH_COLORS['CV-GC']
    fig = new_page(pdf,
                   'CV-GC — Cyclic Voltammetry, Generator/Collector',
                   'CV sweeping on the generator while the collector is held at a fixed reducing potential — maps CE across the full potential range.',
                   accent)

    principle = (
        'The generator electrode sweeps potential as in normal CV; the collector is potentiostated at a fixed reducing potential. '
        'The collector current versus generator potential curve shows when each analyte is oxidised at the generator and '
        'how efficiently it is collected.\n'
        'No features are extracted into the ML vector from CV-GC — it is used for visual validation only. '
        'The collection efficiency CE ≈ −I_col / I_gen is annotated at the peak of each analyte window '
        'and gives a qualitative check that the electrode spacing and geometry are performing as expected.'
    )
    y = description_box(fig, principle, top=0.845, accent=accent)

    headers = ['Quantity shown', 'Analyte window', 'Unit', 'Physical Meaning', 'Use in validation']
    rows = [
        ('I_gen vs V',      'All',          'µA vs V',
         'Generator current as a function of potential — same shape as normal CV.',
         'Should show three analyte oxidation peaks at expected potentials.'),
        ('I_col vs V',      'All',          'µA vs V',
         'Collector current as a function of generator potential.',
         'Should mirror I_gen (inverted) at each analyte peak if collection is efficient.'),
        ('CE (annotated)',  'AA / DA / UA', '—',
         'CE = −I_col / I_gen evaluated at the peak of each analyte window.',
         'Typical CE for a well-functioning GC pair: 0.1–0.5. Low CE flags poor geometry or fouling.'),
    ]
    col_widths = [0.165, 0.12, 0.07, 0.285, 0.345]
    draw_table(fig, rows, col_widths, headers, top=y, accent=HDR_BG)

    # Summary box
    y2 = y - len(rows) * 0.062 - 0.06
    fig.text(0.03, y2,
             'Note: CV-GC is a diagnostic / validation technique in this pipeline. '
             'Its features are NOT included in the ANN input vector.\n'
             'If CV-GC CE values differ greatly across conditions, it may indicate electrode-to-electrode '
             'variability that the model will need to compensate for.',
             ha='left', va='top', fontsize=8.5, color='#333333',
             transform=fig.transFigure)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── PAGE 7: Quick-reference card ──────────────────────────────────────────────
def page_quickref(pdf):
    accent = '#2C3E50'
    fig = new_page(pdf,
                   'Quick Reference — Symbol & Unit Summary',
                   'All features at a glance, grouped by technique.',
                   accent)

    categories = [
        ('SWV (per analyte per electrode)', [
            ('Ip',           'µA',   'Peak differential current — primary concentration proxy'),
            ('Ep',           'V',    'Peak potential — analyte identity fingerprint'),
            ('FWHM',         'V',    'Peak width at half height — kinetic / selectivity indicator'),
            ('Area',         'µA·V', 'Integrated peak area — charge proxy, robust to shape changes'),
            ('k_pre',        'µA/V', 'Pre-peak slope — heterogeneous electron transfer rate'),
            ('AI',           '—',    'Asymmetry index W_left/W_right — adsorption / overlap flag'),
            ('Ip_fwd',       'µA',   'Forward-pulse current at Ep'),
            ('Ip_rev',       'µA',   'Reverse-pulse current at Ep'),
            ('fwd_rev_ratio','—',    'log(|Ip_fwd/Ip_rev|) — reversibility measure'),
        ]),
        ('CV — Anodic peaks (per analyte per electrode)', [
            ('Ipa',     'µA',   'Anodic peak current (oxidation, forward sweep)'),
            ('Ep_a',    'V',    'Anodic peak potential'),
            ('FWHM',    'V',    'Peak width at half height'),
            ('Area',    'µA·V', 'Integrated anodic peak area'),
            ('k_pre',   'µA/V', 'Pre-peak slope'),
        ]),
        ('CV — DA cathodic peak (per electrode)', [
            ('Ipc',       'µA', 'Cathodic (reduction) peak current — DA only'),
            ('Ep_c',      'V',  'Cathodic peak potential'),
            ('deltaEp',   'V',  'Ep_a − Ep_c: peak-to-peak separation (59/n mV ideal)'),
            ('E12',       'V',  'E½ = (Ep_a+Ep_c)/2: formal potential'),
            ('Ipa_Ipc',   '—',  '|Ipa/Ipc|: reversibility ratio (= 1 ideal)'),
        ]),
        ('DPV (per analyte per electrode)', [
            ('Ip',   'µA',   'Differential pulse peak current (signal negated → positive)'),
            ('Ep',   'V',    'Peak potential'),
            ('Area', 'µA·V', 'Integrated area'),
        ]),
        ('CA (per electrode)', [
            ('I0',          'µA',   'Current at t = 0.1 s after step'),
            ('Iss',         'µA',   'Steady-state current (avg 18–20 s)'),
            ('Iss_I0',      '—',    'Iss/I0 ratio'),
            ('cott_slope',  'µA·s½','Cottrell slope (I vs t^−½ in 0.5–5 s window)'),
            ('cott_R2',     '—',    'R² of Cottrell linear fit'),
            ('Q_early',     'µA·s', 'Integrated charge 0.1–5 s'),
            ('Q_late',      'µA·s', 'Integrated charge 10–20 s'),
            ('Q_late_early','—',    'Q_late/Q_early ratio'),
        ]),
        ('CA-GC (per electrode pair)', [
            ('CE_1s',  '—', 'Collection efficiency at t = 1 s'),
            ('CE_10s', '—', 'Collection efficiency at t = 10 s'),
            ('CE_ss',  '—', 'Steady-state collection efficiency'),
            ('AF',     '—', 'Amplification factor vs normal CA Iss'),
            ('eta10',  '—', 'Feedback efficiency at t = 10 s'),
        ]),
        ('Inter-analyte ratios (SWV & DPV)', [
            ('ratio_Ip_DA_AA', '—', 'Ip_DA / Ip_AA'),
            ('ratio_Ip_DA_UA', '—', 'Ip_DA / Ip_UA'),
            ('ratio_Ip_UA_AA', '—', 'Ip_UA / Ip_AA'),
            ('dEp_DA_AA',      'V', 'Ep_DA − Ep_AA'),
            ('dEp_UA_DA',      'V', 'Ep_UA − Ep_DA'),
        ]),
    ]

    col_widths = [0.195, 0.065, 0.72]
    col_w_px   = [0.195, 0.065, 0.72]

    y = 0.83
    left = 0.03
    total_w = 0.94

    for grp_name, feats in categories:
        if y < 0.06:
            break
        # Group header
        fig.add_artist(mpatches.FancyBboxPatch(
            (left, y - 0.026), total_w, 0.026,
            boxstyle='square,pad=0', transform=fig.transFigure,
            fc=accent, ec='none', clip_on=False))
        fig.text(left + 0.008, y - 0.013, grp_name,
                 ha='left', va='center', fontsize=8.2, fontweight='bold',
                 color='white', transform=fig.transFigure)
        y -= 0.026

        for i, (sym, unit, desc) in enumerate(feats):
            if y < 0.06:
                break
            bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
            rh = 0.028
            fig.add_artist(mpatches.FancyBboxPatch(
                (left, y - rh), total_w, rh,
                boxstyle='square,pad=0', transform=fig.transFigure,
                fc=bg, ec='#DDDDDD', lw=0.3, clip_on=False))
            fig.text(left + 0.008, y - rh/2, sym,  ha='left', va='center',
                     fontsize=7.2, fontweight='bold', fontfamily='monospace',
                     color='#111111', transform=fig.transFigure)
            fig.text(left + 0.21,  y - rh/2, unit, ha='left', va='center',
                     fontsize=7.2, color=C_GREY, transform=fig.transFigure)
            fig.text(left + 0.28,  y - rh/2, desc, ha='left', va='center',
                     fontsize=7.2, color='#222222', transform=fig.transFigure)
            y -= rh

        y -= 0.010

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── Build PDF ──────────────────────────────────────────────────────────────────
print(f"Generating {OUT_PATH} ...")
with PdfPages(OUT_PATH) as pdf:
    page_cover(pdf)
    page_swv(pdf)
    page_cv(pdf)
    page_dpv(pdf)
    page_ca(pdf)
    page_cagc(pdf)
    page_cvgc(pdf)
    page_quickref(pdf)

    meta = pdf.infodict()
    meta['Title']   = 'MicroNeedleArray ML — Electrochemical Feature Reference Guide'
    meta['Subject'] = 'Feature definitions for SWV, CV, DPV, CA, CA-GC, CV-GC'
    meta['Author']  = 'generate_feature_reference.py'

print(f"Done → {OUT_PATH}  (8 pages)")
