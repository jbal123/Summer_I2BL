#!/usr/bin/env python3
"""Regenerate revised single-analyte PCR per-concentration reconstruction PDFs, CV-normal only.

Reuses the validated repo pipeline (selected PCR setup -> fit_pcr_models -> per-condition
overlay/reconstruction pages) but emits ONLY cv_normal (anodic + cathodic) pages; cv_gc is
skipped per request. One PDF per analyte: intro + concentration-colored overlays + one
per-concentration reconstruction page-set (actual vs PCR-predicted vs residual, per electrode).
"""

from __future__ import annotations

# --- path bootstrap ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from fit_revised_single_analyte_pcr import (
    ANALYTE_ORDER,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    TECHNIQUE_CONFIGS,
    chunks,
    configure_plot_style,
    fit_pcr_models,
    load_trace_entries,
    params_from_summary,
    plot_single_condition_grid,
)
from plot_revised_single_analyte_pcr_predictions import (
    DEFAULT_SUMMARY,
    plot_intro_page,
    plot_overlay_page,
    selected_setup_rows,
)


def write_cv_only_pdf(output_path, analyte, selected_row, models, manifest_rows):
    cv_normal_electrodes = list(TECHNIQUE_CONFIGS["cv_normal"]["electrode_columns"].keys())
    conditions = sorted({int(c) for m in models for c in np.asarray(m["conditions"], dtype=int)})
    with PdfPages(output_path) as pdf:
        plot_intro_page(pdf, analyte, selected_row, manifest_rows, len(models))
        # concentration-colored overlays, cv_normal only
        for group in chunks(cv_normal_electrodes, 2):
            plot_overlay_page(pdf, analyte, models, "cv_normal", group, "-".join(group))
        # per-concentration reconstruction pages, cv_normal only
        for condition in conditions:
            for group in chunks(cv_normal_electrodes, 2):
                plot_single_condition_grid(pdf, analyte, condition, models, "cv_normal", group, "-".join(group))
    return output_path


def run(args):
    configure_plot_style()
    data_root = args.data_root.resolve()
    selected = selected_setup_rows(args.summary_csv.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for analyte in ANALYTE_ORDER:
        entries, manifest_rows, _ = load_trace_entries(data_root, analyte)
        params = params_from_summary(selected[analyte])
        models = fit_pcr_models(entries, params, scopes={"electrode"})
        out_path = out_dir / f"revised_{analyte}_pcr_reconstruction_cv_only.pdf"
        write_cv_only_pdf(out_path, analyte, selected[analyte], models, manifest_rows)
        outputs.append(out_path)
        print(f"{analyte}: wrote {out_path}")
    for o in outputs:
        print("PDF:", o)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=Path.cwd())
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
