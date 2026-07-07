# Beyond Linear Superposition

Standalone implementation of the residual-correction pipeline specified in
`../beyond_linear_superposition.md`. It replaces naive superposition of the
per-analyte PCR predictions with a learned residual correction trained on the
real mixture data.

Everything here is **self-contained** — no imports from the rest of the repo.
It reads only the two data manifests (and the raw CV `.txt` files they point to):

- `../All_Analyte_Isolated_PCR/isolated_analyte_conditions.csv` — DA/AA/UA isolates
- `../All_Analyte_Superposition_Analysis/multi_analyte_conditions.csv` — mixtures

## Module map (one spec step per file)

| File | Spec step | Contents |
|---|---|---|
| `preprocessing.py` | Step 1 | `als_baseline` / `correct_baseline` (AsLSSR) and `cow_align` (COW potential-shift alignment) |
| `data_loading.py`  | — | Standalone CV `.txt` parser, sweep splitting, electrode averaging, manifest readers, common-grid resampling |
| `pcr_model.py`     | Step 2 | `AnalytePCRModel` (PCA + polynomial concentration→score), `fit_analyte_models` |
| `superposition.py` | Steps 3–4 | `superposition_prediction`, `compute_residuals` |
| `residual_model.py`| Step 5 | `fit_polynomial_residual_model` (5a), `fit_gpr_residual_model` (5b, with per-voltage uncertainty) |
| `pipeline.py`      | Step 6 | `predict_mixture_curve`, `FittedPanelPipeline` (serializable bundle) |
| `cross_validation.py` | Step 7 | Leave-one-UA-level-out folds |
| `evaluation.py`    | Step 8 | `evaluate_curve_reconstruction` (global + per-peak), aggregation, plotting |
| `run_pipeline.py`  | all | End-to-end driver: load → preprocess → fit PCR → residual CV → metrics/plots/serialize |
| `plot_reconstructions.py` | — | Reconstruction-showcase plots for the working CV-normal panels (separate per-analyte pieces + replica vs actual) |

## Usage

```bash
cd beyond_linear_superposition
../.venv/bin/python run_pipeline.py --residual-model polynomial   # Step 5a
../.venv/bin/python run_pipeline.py --residual-model gpr          # Step 5b
```

Useful flags: `--panel cv_normal/anodic` (one panel only), `--no-cow` /
`--no-als` (ablate preprocessing), `--grid-points`, `--limit-conditions`
(fast smoke test), `--no-plots`, plus AsLSSR/COW/PCR/residual hyperparameters
(`--als-lam`, `--cow-slack`, `--pcr-components`, `--poly-alpha`,
`--gpr-components`, …).

Outputs land in `results/` (suffixed by model):
`cv_metrics_per_condition_<model>.csv`, `cv_metrics_summary_<model>.csv`,
`residual_structure_by_ua_<model>.csv`, `beyond_superposition_<model>.pdf`
(actual vs superposition vs corrected, ±1σ band for GPR), and
`fitted_pipelines_<model>.joblib` (PCR models + COW reference + residual model
serialized together, per the spec's note).

The pipeline runs all four panels in `PANEL_ORDER`
(`cv_normal`/`cv_gc` × `anodic`/`cathodic`); the residual model is fit
independently per panel.

### Reconstruction showcase (working CV-normal panels)

```bash
../.venv/bin/python plot_reconstructions.py --residual-model gpr        # or polynomial
```

Writes `results/reconstruction_showcase_cv_normal_<model>.pdf`, one page per
mixture condition. Each page (rows = anodic/cathodic) shows the three **separate
per-analyte PCR reconstructions** (DA, AA, UA at that condition's
concentrations) and a fourth column overlaying the **actual measured mixture**,
the **linear superposition** (sum of the three pieces — the old method), and the
**corrected replica** (superposition + learned residual, with a ±1σ band for
GPR). The corrected curve is an *out-of-fold* prediction (its UA level was held
out), so the match is a genuine reconstruction of unseen mixture data. The
comparison axis is focused on the actual signal range so the diagnostic peaks
are legible; the shared water-oxidation tail is allowed to clip off-axis.

Typical CV-normal results: corrected R² ≈ 0.7–0.9 with RMSE ≈ 0.2–0.3 µA,
versus strongly negative R² for linear superposition.

## Results on the current dataset (leave-one-UA-out CV)

The residual correction is a clear win on the **CV-normal** channel and largely
ineffective on the **CV-GC** channel:

| Panel | Baseline RMSE (µA) | Polynomial | GPR |
|---|---|---|---|
| cv_normal / anodic   | 1.04 | −39.5% | **−73.2%** |
| cv_normal / cathodic | 1.11 | −47.5% | **−81.2%** |
| cv_gc / anodic       | 3.15 | +236% (worse) | −6% |
| cv_gc / cathodic     | 2.79 | +184% (worse) | +8% |

(negative % = RMSE reduction = improvement.) Per the spec's decision table, the
CV-normal result (GPR > polynomial, both >40% reduction) is "strong evidence of
learnable non-additivity → deploy GPR, report uncertainty bands."

## Important data caveat

`R²` values are strongly negative even after correction. This is **not a
pipeline bug** — it reflects two real properties of the dataset, confirmed by
direct inspection:

1. **Run-to-run / cross-day amplitude inconsistency.** Isolate amplitudes are
   not monotonic in concentration (e.g. CV-GC anodic DA isolates: 0 µM→3.3,
   20 µM→**36**, 100 µM→**2.8** µA), so the PCR concentration→score polynomial
   has little to fit on the CV-GC channel.
2. **Failed/near-dead electrodes.** Some mixtures (mostly Days 2–4, CV-GC) have
   actual amplitudes near 0.8 µA while the prediction is ~24 µA, giving R²
   around −1900 and dominating the *mean*. Median R² and RMSE are reported
   alongside the mean for this reason.

AsLSSR baseline correction is essential: on a clean Day-1 binary it lifts
additivity from R² ≈ −148 (raw sum) to R² ≈ +0.85. The remaining gap on CV-GC
is a measurement-scale problem (cross-day/electrode normalization) that lies
*outside* the spec's pipeline; the CV-normal channel, which is far more stable,
is where the residual correction demonstrably works.
