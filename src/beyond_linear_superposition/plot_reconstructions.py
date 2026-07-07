#!/usr/bin/env python3
"""Reconstruction showcase plots for the working CV-normal panels.

For every mixture condition this draws, per sweep (anodic / cathodic):

  col 1-3 : the separate per-analyte PCR reconstructions (DA, AA, UA) at the
            condition's concentrations -- i.e. the isolated pieces we predict;
  col 4   : the actual measured mixture overlaid with
              * the linear superposition (sum of the three pieces), and
              * the residual-corrected reconstruction (superposition + learned
                residual), which is the "replica" of the known mixture data.

This mirrors the layout of the earlier linear-superposition PDF in
All_Analyte_Superposition_Analysis*, adding the corrected reconstruction.

The corrected curve is an *out-of-fold* prediction (leave-one-UA-level-out), so
each condition is reconstructed by a residual model that never saw its UA level.

Run:  python plot_reconstructions.py --residual-model gpr
"""

from __future__ import annotations

# --- path bootstrap: ensure this package dir is importable regardless of launcher ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# --- end path bootstrap ---

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

_CACHE = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(_CACHE / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cross_validation import fit_residual_model, leave_one_ua_out_folds
from data_loading import (
    ANALYTE_ORDER,
    CURRENT_UNIT,
    ROOT,
    RESULTS_ROOT,
    load_panel,
    read_isolate_conditions,
    read_mixture_conditions,
)
from pcr_model import fit_analyte_models
from pipeline import FittedPanelPipeline
from residual_model import GPRResidualModel
from run_pipeline import preprocess_panel
from superposition import compute_residuals, superposition_prediction

ANALYTE_COLORS = {"DA": "#D55E00", "AA": "#0072B2", "UA": "#009E73"}
ANALYTE_LABELS = {"DA": "Dopamine", "AA": "Ascorbic Acid", "UA": "Uric Acid"}
WORKING_PANELS = [("cv_normal", "anodic"), ("cv_normal", "cathodic")]

DEFAULT_OUTPUT = RESULTS_ROOT / "beyond_linear_superposition"


def style(ax):
    ax.set_xlabel("Potential (V)", fontsize=7)
    ax.set_ylabel(f"Current ({CURRENT_UNIT})", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def r2(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - np.sum((actual - pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def build_panel_assets(panel, config, args):
    """Return everything needed to draw a panel: grid, PCR models, per-condition
    out-of-fold pipelines, and the preprocessed mixtures."""
    reference, iso_curves, mixtures = preprocess_panel(panel, config)
    pcr_models = fit_analyte_models(
        iso_curves, n_components=args.pcr_components, score_degree=args.pcr_degree
    )
    conditions = np.vstack([cond.concentration_vector for cond, _ in mixtures])
    real_curves = np.vstack([curve for _, curve in mixtures])
    _, residuals = compute_residuals(conditions, real_curves, pcr_models)

    # Map each condition index to a residual model trained without its UA level.
    index_to_model: dict[int, object] = {}
    for fold in leave_one_ua_out_folds(conditions):
        model = fit_residual_model(
            conditions[fold.train_index],
            residuals[fold.train_index],
            model_type=args.residual_model,
            poly_degree=args.poly_degree,
            poly_alpha=args.poly_alpha,
            gpr_components=args.gpr_components,
        )
        for idx in fold.test_index:
            index_to_model[int(idx)] = model

    return {
        "grid": panel.grid,
        "reference": reference,
        "pcr_models": pcr_models,
        "mixtures": mixtures,
        "index_to_model": index_to_model,
        "config": config,
    }


def draw_condition_page(pdf, cond_key, panel_assets_by_sweep, residual_model_type):
    """One page per mixture condition; rows = sweeps, cols = DA/AA/UA/combined."""
    sweeps = list(panel_assets_by_sweep.keys())
    fig, axes = plt.subplots(len(sweeps), 4, figsize=(16, 3.4 * len(sweeps)), squeeze=False)
    cond = None
    for row, sweep in enumerate(sweeps):
        assets, idx = panel_assets_by_sweep[sweep]
        cond, actual = assets["mixtures"][idx]
        grid = assets["grid"]
        models = assets["pcr_models"]
        concs = cond.concentrations

        # Columns 1-3: separate per-analyte PCR reconstructions.
        for col, analyte in enumerate(ANALYTE_ORDER):
            ax = axes[row][col]
            style(ax)
            piece = models[analyte].predict_curve(concs[analyte]) if analyte in models else np.zeros_like(grid)
            ax.plot(grid, piece, color=ANALYTE_COLORS[analyte], lw=1.4)
            lo, hi = np.percentile(piece, 1), np.percentile(piece, 99)
            if hi > lo:
                pad = 0.15 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
            if row == 0:
                ax.set_title(f"{ANALYTE_LABELS[analyte]} PCR piece", fontsize=9)
            ax.text(0.02, 0.96, f"{analyte} = {concs[analyte]:g} µM", transform=ax.transAxes,
                    fontsize=7, va="top", color=ANALYTE_COLORS[analyte])
            if col == 0:
                ax.text(-0.30, 0.5, f"CV normal\n{sweep}", transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=8)

        # Column 4: actual vs superposition vs corrected reconstruction.
        ax = axes[row][3]
        style(ax)
        baseline = superposition_prediction(concs["DA"], concs["AA"], concs["UA"], models)
        residual_model = assets["index_to_model"].get(idx)
        fitted = FittedPanelPipeline(
            technique="cv_normal", sweep=sweep, grid=grid, cow_reference=assets["reference"],
            preprocess_config=assets["config"], pcr_models=models,
            residual_model=residual_model, residual_model_type=residual_model_type,
        )
        corrected, std = fitted.predict_with_uncertainty(concs["DA"], concs["AA"], concs["UA"])
        ax.plot(grid, actual, color="#000000", lw=1.4, label="actual mixture")
        ax.plot(grid, baseline, color="#999999", lw=1.2, ls="--", label="linear superposition")
        ax.plot(grid, corrected, color="#CC0000", lw=1.6, label="corrected replica")
        if isinstance(residual_model, GPRResidualModel) and np.any(std > 0):
            ax.fill_between(grid, corrected - std, corrected + std, color="#CC0000", alpha=0.15, lw=0)
        # Focus the y-axis on the actual+corrected signal so the diagnostic peak
        # region is legible; the superposition tail is allowed to clip off-axis.
        focus = np.concatenate([actual, corrected])
        lo, hi = np.percentile(focus, 1), np.percentile(focus, 99)
        pad = 0.15 * (hi - lo) if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        if row == 0:
            ax.set_title("Actual vs reconstruction", fontsize=9)
        ax.text(
            0.02, 0.96,
            f"superpos: R²={r2(actual, baseline):.2f}\n"
            f"corrected: R²={r2(actual, corrected):.2f}\n"
            f"RMSE {rmse(actual, baseline):.2g}→{rmse(actual, corrected):.2g} {CURRENT_UNIT}",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.legend(loc="lower right", fontsize=6)

    fig.suptitle(
        f"Mixture reconstruction | Day {cond.day} Condition {cond.condition} | "
        f"DA {cond.da_uM:g} µM, AA {cond.aa_uM:g} µM, UA {cond.ua_uM:g} µM | "
        f"residual model: {residual_model_type} (out-of-fold)",
        fontsize=11, fontweight="semibold",
    )
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def run(args):
    output_root = (args.output_root or DEFAULT_OUTPUT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "als_lam": args.als_lam, "als_p": args.als_p, "als_iter": args.als_iter,
        "cow_segment_length": args.cow_segment_length, "cow_slack": args.cow_slack,
        "apply_als": not args.no_als, "apply_cow": not args.no_cow,
    }

    isolates = read_isolate_conditions()
    mixtures = read_mixture_conditions()
    if args.limit_conditions is not None:
        mixtures = mixtures[: args.limit_conditions]

    # Build assets for each working panel, keyed by sweep.
    assets_by_sweep = {}
    for technique, sweep in WORKING_PANELS:
        panel = load_panel(technique, sweep, isolates, mixtures, args.grid_points)
        if panel is None or not panel.mixture_curves:
            print(f"Skipping {technique}/{sweep}: no curves")
            continue
        assets_by_sweep[sweep] = build_panel_assets(panel, config, args)

    if not assets_by_sweep:
        print("No working panels available.")
        return

    # Index conditions by (day, condition) so anodic/cathodic line up on a page.
    def cond_index(assets):
        return {(c.day, c.condition): i for i, (c, _) in enumerate(assets["mixtures"])}

    sweep_index = {sweep: cond_index(assets) for sweep, assets in assets_by_sweep.items()}
    all_keys = sorted(set().union(*[set(idx) for idx in sweep_index.values()]))

    pdf_path = output_root / f"reconstruction_showcase_cv_normal_{args.residual_model}.pdf"
    with PdfPages(pdf_path) as pdf:
        for key in all_keys:
            page = {}
            for sweep, assets in assets_by_sweep.items():
                idx = sweep_index[sweep].get(key)
                if idx is not None:
                    page[sweep] = (assets, idx)
            if page:
                draw_condition_page(pdf, key, page, args.residual_model)
    print(f"Wrote reconstruction showcase: {pdf_path} ({len(all_keys)} conditions)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--residual-model", choices=["polynomial", "gpr"], default="gpr")
    p.add_argument("--grid-points", type=int, default=400)
    p.add_argument("--limit-conditions", type=int, default=None)
    p.add_argument("--no-als", action="store_true")
    p.add_argument("--no-cow", action="store_true")
    p.add_argument("--als-lam", type=float, default=1e5)
    p.add_argument("--als-p", type=float, default=0.01)
    p.add_argument("--als-iter", type=int, default=10)
    p.add_argument("--cow-segment-length", type=int, default=20)
    p.add_argument("--cow-slack", type=int, default=3)
    p.add_argument("--pcr-components", type=int, default=5)
    p.add_argument("--pcr-degree", type=int, default=2)
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--poly-alpha", type=float, default=1e-3)
    p.add_argument("--gpr-components", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
