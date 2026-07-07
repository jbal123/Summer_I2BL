#!/usr/bin/env python3
"""End-to-end driver for the Beyond-Linear-Superposition residual pipeline.

Ties together Steps 1-8:
  1. Load isolate + mixture CV curves (standalone parser).
  2. Preprocess (AsLSSR baseline + COW alignment to a fixed DA-20uM reference).
  3. Fit per-analyte PCR models on preprocessed isolates.
  4. Build the superposition baseline and training residuals for every mixture.
  5. Fit a residual model (polynomial or GPR) and evaluate it with
     leave-one-UA-level-out cross-validation, reporting baseline-vs-corrected
     metrics.
  6. Serialize the fitted per-panel pipelines and write metrics + plots.

Run:  python run_pipeline.py --residual-model polynomial
"""

from __future__ import annotations

# --- path bootstrap: ensure this package dir is importable regardless of launcher ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# --- end path bootstrap ---

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

# Headless matplotlib cache, consistent with the rest of the repo.
_CACHE = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(_CACHE / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "mplconfig"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cross_validation import (
    fit_residual_model,
    leave_one_ua_out_folds,
)
from data_loading import (
    ANALYTE_ORDER,
    CURRENT_UNIT,
    PANEL_ORDER,
    ROOT,
    RESULTS_ROOT,
    LoadedPanel,
    load_panel,
    read_isolate_conditions,
    read_mixture_conditions,
)
from evaluation import (
    DEFAULT_PEAK_WINDOWS,
    aggregate_metrics,
    evaluate_curve_reconstruction,
    improvement_summary,
    plot_condition,
)
from pcr_model import fit_analyte_models
from pipeline import FittedPanelPipeline
from preprocessing import build_cow_reference, correct_baseline, preprocess_curve
from superposition import compute_residuals


DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "beyond_linear_superposition"
COW_REFERENCE_CONC = 20.0  # mid-range DA isolate concentration (uM)


def build_reference(panel: LoadedPanel, config: dict) -> np.ndarray:
    """Fixed COW reference: mean of the AsLSSR-corrected DA isolate curves at
    the concentration closest to 20 uM."""
    da_curves = panel.isolate_curves.get("DA", [])
    if not da_curves:
        # Fall back to the mean of whatever isolates exist.
        all_curves = [c for samples in panel.isolate_curves.values() for _, c in samples]
        corrected = [correct_baseline(c, config["als_lam"], config["als_p"], config["als_iter"]) for c in all_curves]
        return build_cow_reference(np.vstack(corrected))
    concs = np.array([c for c, _ in da_curves])
    target = concs[np.argmin(np.abs(concs - COW_REFERENCE_CONC))]
    chosen = [curve for conc, curve in da_curves if np.isclose(conc, target)]
    corrected = [correct_baseline(c, config["als_lam"], config["als_p"], config["als_iter"]) for c in chosen]
    return build_cow_reference(np.vstack(corrected))


def preprocess_panel(panel: LoadedPanel, config: dict):
    """Return (reference, preprocessed isolate curves, preprocessed mixtures)."""
    reference = build_reference(panel, config)
    kw = dict(
        reference=reference,
        als_lam=config["als_lam"],
        als_p=config["als_p"],
        als_iter=config["als_iter"],
        cow_segment_length=config["cow_segment_length"],
        cow_slack=config["cow_slack"],
        apply_als=config["apply_als"],
        apply_cow=config["apply_cow"],
    )
    iso = {
        analyte: [(conc, preprocess_curve(curve, **kw)) for conc, curve in samples]
        for analyte, samples in panel.isolate_curves.items()
    }
    mixtures = [(cond, preprocess_curve(curve, **kw)) for cond, curve in panel.mixture_curves]
    return reference, iso, mixtures


def run_panel(panel: LoadedPanel, config: dict, args) -> tuple[FittedPanelPipeline, list[dict], list[dict], list]:
    grid = panel.grid
    reference, iso_curves, mixtures = preprocess_panel(panel, config)

    pcr_models = fit_analyte_models(
        iso_curves, n_components=args.pcr_components, score_degree=args.pcr_degree
    )

    conditions = np.vstack([cond.concentration_vector for cond, _ in mixtures])
    real_curves = np.vstack([curve for _, curve in mixtures])
    _, residuals_all = compute_residuals(conditions, real_curves, pcr_models)

    peak_windows = DEFAULT_PEAK_WINDOWS

    # ---- Leave-one-UA-level-out cross-validation -------------------------- #
    metric_rows: list[dict] = []
    plot_records: list = []  # (cond, actual, baseline, corrected, std)
    folds = leave_one_ua_out_folds(conditions)
    for fold in folds:
        train_conditions = conditions[fold.train_index]
        train_residuals = residuals_all[fold.train_index]
        residual_model = fit_residual_model(
            train_conditions,
            train_residuals,
            model_type=args.residual_model,
            poly_degree=args.poly_degree,
            poly_alpha=args.poly_alpha,
            gpr_components=args.gpr_components,
        )
        fold_pipeline = FittedPanelPipeline(
            technique=panel.technique,
            sweep=panel.sweep,
            grid=grid,
            cow_reference=reference,
            preprocess_config=config,
            pcr_models=pcr_models,
            residual_model=residual_model,
            residual_model_type=args.residual_model,
        )
        for idx in fold.test_index:
            cond, actual = mixtures[idx]
            c = cond.concentrations
            baseline = fold_pipeline.predict_baseline(c["DA"], c["AA"], c["UA"])
            corrected, std = fold_pipeline.predict_with_uncertainty(c["DA"], c["AA"], c["UA"])
            base_m = evaluate_curve_reconstruction(baseline, actual, grid, peak_windows)
            corr_m = evaluate_curve_reconstruction(corrected, actual, grid, peak_windows)
            row = {
                "technique": panel.technique,
                "sweep": panel.sweep,
                "held_out_ua": fold.held_out_ua,
                "day": cond.day,
                "condition": cond.condition,
                "da_uM": cond.da_uM,
                "aa_uM": cond.aa_uM,
                "ua_uM": cond.ua_uM,
            }
            for key, value in base_m.items():
                row[f"baseline_{key}"] = value
            for key, value in corr_m.items():
                row[f"corrected_{key}"] = value
            metric_rows.append(row)
            plot_records.append((cond, actual, baseline, corrected, std))

    # ---- Final residual model fit on ALL mixtures (for the saved pipeline) - #
    final_residual_model = fit_residual_model(
        conditions,
        residuals_all,
        model_type=args.residual_model,
        poly_degree=args.poly_degree,
        poly_alpha=args.poly_alpha,
        gpr_components=args.gpr_components,
    )
    fitted = FittedPanelPipeline(
        technique=panel.technique,
        sweep=panel.sweep,
        grid=grid,
        cow_reference=reference,
        preprocess_config=config,
        pcr_models=pcr_models,
        residual_model=final_residual_model,
        residual_model_type=args.residual_model,
    )

    # ---- Per-UA-level residual structure summary (Step 4 inspection) ------ #
    residual_rows = []
    for level in sorted({float(c[2]) for c in conditions}):
        mask = np.isclose(conditions[:, 2], level)
        block = residuals_all[mask]
        residual_rows.append(
            {
                "technique": panel.technique,
                "sweep": panel.sweep,
                "ua_uM": level,
                "n_mixtures": int(mask.sum()),
                "mean_abs_residual_uA": float(np.mean(np.abs(block))),
                "max_abs_residual_uA": float(np.max(np.abs(block))),
                "residual_std_uA": float(np.std(block)),
            }
        )

    return fitted, metric_rows, residual_rows, plot_records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.10g}" if isinstance(v, (float, np.floating)) and np.isfinite(v)
                     else ("" if isinstance(v, (float, np.floating)) else v))
                 for k, v in row.items()}
            )


def print_summary(metric_rows: list[dict], args) -> list[dict]:
    """Print and return per-panel baseline-vs-corrected aggregate metrics."""
    summary_rows = []
    panels = sorted({(r["technique"], r["sweep"]) for r in metric_rows})
    metric_keys = [k[len("baseline_"):] for k in metric_rows[0] if k.startswith("baseline_")]
    print("\n=== Cross-validated metrics (mean over held-out mixtures) ===")
    for technique, sweep in panels:
        rows = [r for r in metric_rows if r["technique"] == technique and r["sweep"] == sweep]
        base = aggregate_metrics([{k: r[f"baseline_{k}"] for k in metric_keys} for r in rows])
        corr = aggregate_metrics([{k: r[f"corrected_{k}"] for k in metric_keys} for r in rows])
        impr = improvement_summary(base, corr)
        # Median is robust to dead/failed electrodes that produce extreme R2.
        def _median(metric_prefix, key):
            vals = [r[f"{metric_prefix}_{key}"] for r in rows if np.isfinite(r.get(f"{metric_prefix}_{key}", np.nan))]
            return float(np.median(vals)) if vals else float("nan")
        print(f"\n[{technique} / {sweep}]  ({len(rows)} held-out mixtures, model={args.residual_model})")
        print(f"  RMSE global (mean)  : baseline {base['rmse_global']:.4g} -> corrected {corr['rmse_global']:.4g} "
              f"({100*impr['rmse_global_reduction_frac']:+.1f}%)")
        print(f"  R2 global   (mean)  : baseline {base['r2_global']:.4f} -> corrected {corr['r2_global']:.4f}")
        print(f"  R2 global   (median): baseline {_median('baseline','r2_global'):.4f} -> "
              f"corrected {_median('corrected','r2_global'):.4f}")
        summary_extra = {
            "baseline_r2_global_median": _median("baseline", "r2_global"),
            "corrected_r2_global_median": _median("corrected", "r2_global"),
            "baseline_rmse_global_median": _median("baseline", "rmse_global"),
            "corrected_rmse_global_median": _median("corrected", "rmse_global"),
        }
        summary_row = {"technique": technique, "sweep": sweep, "n_held_out": len(rows)}
        for k in metric_keys:
            summary_row[f"baseline_{k}"] = base[k]
            summary_row[f"corrected_{k}"] = corr[k]
        summary_row.update(impr)
        summary_row.update(summary_extra)
        summary_rows.append(summary_row)
    return summary_rows


def make_plots(pdf_path: Path, all_plot_records: dict, grid_by_panel: dict) -> None:
    with PdfPages(pdf_path) as pdf:
        for (technique, sweep), records in all_plot_records.items():
            grid = grid_by_panel[(technique, sweep)]
            # Group by held-out UA level for readability: one page per UA level.
            by_ua: dict[float, list] = {}
            for cond, actual, baseline, corrected, std in records:
                by_ua.setdefault(float(cond.ua_uM), []).append((cond, actual, baseline, corrected, std))
            for ua_level, items in sorted(by_ua.items()):
                n = len(items)
                ncol = 3
                nrow = int(np.ceil(n / ncol))
                fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
                fig.suptitle(
                    f"{technique} / {sweep} — held-out UA = {ua_level:g} uM "
                    f"(actual vs superposition vs corrected)",
                    fontsize=11,
                )
                for ax, (cond, actual, baseline, corrected, std) in zip(axes.ravel(), items):
                    plot_condition(
                        ax, grid, actual, baseline, corrected,
                        title=f"D{cond.day} C{cond.condition} | DA{cond.da_uM:g} AA{cond.aa_uM:g} UA{cond.ua_uM:g}",
                        residual_std=std, unit=CURRENT_UNIT,
                    )
                for ax in axes.ravel()[n:]:
                    ax.axis("off")
                axes.ravel()[0].legend(loc="best", fontsize=6)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig)
                plt.close(fig)


def run(args) -> None:
    output_root = (args.output_root or DEFAULT_OUTPUT_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "als_lam": args.als_lam,
        "als_p": args.als_p,
        "als_iter": args.als_iter,
        "cow_segment_length": args.cow_segment_length,
        "cow_slack": args.cow_slack,
        "apply_als": not args.no_als,
        "apply_cow": not args.no_cow,
    }

    isolates = read_isolate_conditions()
    mixtures = read_mixture_conditions()
    if args.limit_conditions is not None:
        mixtures = mixtures[: args.limit_conditions]
    print(f"Isolates: " + ", ".join(f"{a}={len(isolates[a])}" for a in ANALYTE_ORDER))
    print(f"Mixtures: {len(mixtures)}")

    panels = PANEL_ORDER if args.panel is None else [tuple(args.panel.split("/"))]

    fitted_pipelines = {}
    all_metric_rows = []
    all_residual_rows = []
    all_plot_records = {}
    grid_by_panel = {}

    for technique, sweep in panels:
        print(f"\n--- Panel {technique}/{sweep} ---")
        panel = load_panel(technique, sweep, isolates, mixtures, args.grid_points, verbose=args.verbose)
        if panel is None or not panel.mixture_curves:
            print("  No usable curves; skipping.")
            continue
        fitted, metric_rows, residual_rows, plot_records = run_panel(panel, config, args)
        fitted_pipelines[f"{technique}/{sweep}"] = fitted
        all_metric_rows.extend(metric_rows)
        all_residual_rows.extend(residual_rows)
        all_plot_records[(technique, sweep)] = plot_records
        grid_by_panel[(technique, sweep)] = panel.grid

    if not all_metric_rows:
        print("\nNo metrics produced — check data paths.")
        return

    tag = args.residual_model
    write_csv(output_root / f"cv_metrics_per_condition_{tag}.csv", all_metric_rows)
    write_csv(output_root / f"residual_structure_by_ua_{tag}.csv", all_residual_rows)
    summary_rows = print_summary(all_metric_rows, args)
    write_csv(output_root / f"cv_metrics_summary_{tag}.csv", summary_rows)

    if not args.no_plots:
        pdf_path = output_root / f"beyond_superposition_{args.residual_model}.pdf"
        make_plots(pdf_path, all_plot_records, grid_by_panel)
        print(f"\nWrote plots: {pdf_path}")

    model_path = output_root / f"fitted_pipelines_{args.residual_model}.joblib"
    joblib.dump({"pipelines": fitted_pipelines, "config": config, "args": vars(args)}, model_path)
    print(f"Wrote fitted pipelines: {model_path}")
    print(f"Wrote metrics: {output_root / f'cv_metrics_per_condition_{tag}.csv'}")
    print(f"Wrote summary: {output_root / f'cv_metrics_summary_{tag}.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--residual-model", choices=["polynomial", "gpr"], default="polynomial")
    p.add_argument("--panel", default=None, help="Restrict to one panel, e.g. cv_gc/anodic")
    p.add_argument("--grid-points", type=int, default=400)
    p.add_argument("--limit-conditions", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    # Preprocessing
    p.add_argument("--no-als", action="store_true", help="Disable AsLSSR baseline correction")
    p.add_argument("--no-cow", action="store_true", help="Disable COW alignment")
    p.add_argument("--als-lam", type=float, default=1e5)
    p.add_argument("--als-p", type=float, default=0.01)
    p.add_argument("--als-iter", type=int, default=10)
    p.add_argument("--cow-segment-length", type=int, default=20)
    p.add_argument("--cow-slack", type=int, default=3)
    # PCR
    p.add_argument("--pcr-components", type=int, default=5)
    p.add_argument("--pcr-degree", type=int, default=2)
    # Residual model
    p.add_argument("--poly-degree", type=int, default=2)
    p.add_argument("--poly-alpha", type=float, default=1e-3)
    p.add_argument("--gpr-components", type=int, default=5)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
