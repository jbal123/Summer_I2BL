
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
_RESULTS_RUNS_DIR = str(_REPO_ROOT / "results" / "bayes_runs")
# --- end repo anchors ---

import argparse
import glob
import json
import multiprocessing as mp
import os
import matplotlib
matplotlib.use('agg')  # non-interactive backend — required for worker processes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler

from feature_extract_cond import extract_condition, ALL_ELECTRODES
from model import make_model
from plot_training import plot_training

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Bayesian optimization over electrochemical test/feature combinations.')
parser.add_argument('--data-dir', default=_DATA_DIR)
parser.add_argument('--test-conditions', nargs='+', type=int,
                    default=[3, 5, 11, 14, 20], metavar='N',
                    help='Condition numbers held out for test/validation. '
                         'Default: 3 5 11 14 20 (Latin-hypercube selection across all analyte levels).')
parser.add_argument('--train-conditions', nargs='+', type=int, default=None, metavar='N',
                    help='Condition numbers used for training. '
                         'Default: all conditions not in --test-conditions.')
parser.add_argument('--n-trials', type=int, default=200)
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--top-n', type=int, default=10)
parser.add_argument('--n-startup-trials', type=int, default=25,
                    help='Random exploration trials before TPE starts guiding. Default: 25.')
parser.add_argument('--patience', type=int, default=0,
                    help='Val-check intervals with no improvement to allow past --epochs before '
                         'stopping. 0 = disabled (stop exactly at --epochs). Default: 0.')
parser.add_argument('--max-epochs', type=int, default=None,
                    help='Hard ceiling on total epochs per trial when --patience is active. '
                         'Default: no limit.')
parser.add_argument('--jobs', type=int, default=10,
                    help='Number of parallel workers (processes) for this run. '
                         'Uses Optuna constant-liar strategy so workers stay diverse. '
                         'Default: 1 (single-threaded).')
parser.add_argument('--asymmetric-loss', action='store_true', default=False,
                    help='Use asymmetric MSE loss that penalises underprediction by --alpha.')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='Underprediction penalty multiplier for asymmetric loss. Default: 2.0.')
parser.add_argument('--require-tests', nargs='+', default=None,
                    metavar='TEST', choices=['swv', 'dpv', 'cv', 'ca', 'cagc'],
                    help='Test types that are always active in every trial (never toggled off). '
                         'E.g. --require-tests cv')
parser.add_argument('--cv-pca-variance', type=float, default=0.95, metavar='F',
                    help='Minimum cumulative variance explained by PCA of raw CV waveform columns. '
                         'Default: 0.95.')
parser.add_argument('--resume', default=None, metavar='RUN_DIR',
                    help='Path to an existing run directory to continue from where it left off.')
args = parser.parse_args()

TEST_PREFIXES = {
    'swv':  lambda c: c.startswith('swv_'),
    'dpv':  lambda c: c.startswith('dpv_'),
    'cv':   lambda c: c.startswith('cv_'),
    'ca':   lambda c: c.startswith('ca_') and not c.startswith('cagc_'),
    'cagc': lambda c: c.startswith('cagc_'),
}

# ── Resume or new run ─────────────────────────────────────────────────────────
if args.resume:
    run_dir = args.resume.rstrip('/\\')
    if not os.path.isdir(run_dir):
        parser.error(f"Resume directory not found: {run_dir!r}")

    config_path = os.path.join(run_dir, 'run_config.json')
    if not os.path.exists(config_path):
        parser.error(f"No run_config.json found in {run_dir!r} — cannot resume.")

    with open(config_path) as fh:
        cfg = json.load(fh)

    # Support both new condition-based configs and legacy electrode-based configs
    if 'train_conds' in cfg:
        train_conds = cfg['train_conds']
        test_conds  = cfg['test_conds']
    else:
        print("WARNING: legacy run used electrode-based splitting — "
              "resuming with default condition split (test: 3 5 11 14 20).")
        test_conds  = args.test_conditions
        train_conds = None  # resolved after feature extraction

    n_trials         = cfg['n_trials']
    epochs           = cfg['epochs']
    seed             = cfg['seed']
    top_n            = cfg.get('top_n', args.top_n)
    data_dir         = cfg.get('data_dir', args.data_dir)
    n_startup_trials = cfg.get('n_startup_trials', args.n_startup_trials)
    patience         = cfg.get('patience', args.patience)
    max_epochs       = cfg.get('max_epochs', args.max_epochs)
    require_tests    = cfg.get('require_tests', args.require_tests or [])
    asymmetric_loss  = cfg.get('asymmetric_loss', args.asymmetric_loss)
    alpha            = cfg.get('alpha', args.alpha)
    cv_pca_variance  = cfg.get('cv_pca_variance', args.cv_pca_variance)
    jobs             = args.jobs  # always take from CLI — can scale up/down on resume

    print(f"Resuming run: {run_dir}")
    print(f"Config loaded — n_trials={n_trials}, epochs={epochs}, "
          f"train_conds={train_conds}, test_conds={test_conds}")

    feat_csv = os.path.join(run_dir, 'features.csv')
    print(f"Loading features from {feat_csv}…")
    df = pd.read_csv(feat_csv)

else:
    test_conds  = args.test_conditions
    train_conds = args.train_conditions  # None means "derive after extraction"
    n_trials         = args.n_trials
    epochs           = args.epochs
    seed             = args.seed
    top_n            = args.top_n
    data_dir         = args.data_dir
    n_startup_trials = args.n_startup_trials
    patience         = args.patience
    max_epochs       = args.max_epochs
    require_tests    = args.require_tests or []
    asymmetric_loss  = args.asymmetric_loss
    alpha            = args.alpha
    cv_pca_variance  = args.cv_pca_variance
    jobs             = args.jobs

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tc_tag    = '-'.join(str(c) for c in sorted(test_conds))
    run_name  = f"run_{timestamp}_testconds{tc_tag}_n{n_trials}_e{epochs}"
    run_dir   = os.path.join(_RESULTS_RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config so resume can reconstruct everything
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as fh:
        json.dump({
            'data_dir':         data_dir,
            'train_conds':      train_conds,
            'test_conds':       test_conds,
            'n_trials':         n_trials,
            'epochs':           epochs,
            'seed':             seed,
            'top_n':            top_n,
            'n_startup_trials': n_startup_trials,
            'patience':         patience,
            'max_epochs':       max_epochs,
            'require_tests':    require_tests,
            'asymmetric_loss':  asymmetric_loss,
            'alpha':            alpha,
            'cv_pca_variance':  cv_pca_variance,
        }, fh, indent=2)

    # Extract and save features
    cond_dirs = sorted(
        glob.glob(os.path.join(data_dir, '*/')),
        key=lambda x: float(os.path.basename(x.rstrip('/\\')))
    )
    if not cond_dirs:
        parser.error(f"No condition subdirectories found in {data_dir!r}")

    print(f"Extracting features from {len(cond_dirs)} conditions…", flush=True)
    all_rows = []
    for cond_dir in cond_dirs:
        cond_num = int(float(os.path.basename(cond_dir.rstrip('/\\'))))
        try:
            rows = extract_condition(cond_dir, electrodes=ALL_ELECTRODES, verbose=False)
            for row in rows:
                all_rows.append({'condition': cond_num, **row})
        except Exception as exc:
            print(f"  WARNING: skipped {cond_dir}: {exc}")

    if not all_rows:
        parser.error("Feature extraction produced no rows.")

    df = pd.DataFrame(all_rows)
    feat_csv = os.path.join(run_dir, 'features.csv')
    df.to_csv(feat_csv, index=False)
    print(f"Features saved → {feat_csv}  ({len(df)} rows)")

# ── Shared setup ──────────────────────────────────────────────────────────────
META_COLS  = ['condition', 'electrode']
LABEL_COLS = ['DA_uM', 'AA_uM', 'UA_uM']

all_conds = sorted(df['condition'].unique().tolist())
if train_conds is None:
    train_conds = [c for c in all_conds if c not in test_conds]

# ── CV PCA: compress raw waveform cols into PCA scores (fit on train only) ─────
_raw_cv_cols = [c for c in df.columns
                if c.startswith('cv_') and ('_fwd_' in c or '_rev_' in c)
                and '_spline_' not in c]
if _raw_cv_cols:
    _train_mask = df['condition'].isin(train_conds)
    _pca_scaler = StandardScaler()
    _X_tr_raw   = _pca_scaler.fit_transform(df.loc[_train_mask, _raw_cv_cols].fillna(0).values)
    _pca_full   = PCA().fit(_X_tr_raw)
    _cum_var    = np.cumsum(_pca_full.explained_variance_ratio_)
    _n_comp     = min(int(np.searchsorted(_cum_var, cv_pca_variance)) + 1,
                      len(_raw_cv_cols), int(_train_mask.sum()))
    _pca        = PCA(n_components=_n_comp).fit(_X_tr_raw)
    _X_all_pca  = _pca.transform(_pca_scaler.transform(df[_raw_cv_cols].fillna(0).values))
    _pca_cols   = [f'cv_pca_{i:02d}' for i in range(_n_comp)]
    df          = df.drop(columns=_raw_cv_cols)
    for i, col in enumerate(_pca_cols):
        df[col] = _X_all_pca[:, i].astype(np.float32)
    print(f"CV PCA: {len(_raw_cv_cols)} raw cols → {_n_comp} components "
          f"({_cum_var[_n_comp - 1]:.3f} variance explained, threshold={cv_pca_variance})")
else:
    print("CV PCA: no raw waveform columns found in feature set — skipped.")
# ── end CV PCA ─────────────────────────────────────────────────────────────────

ALL_FEAT_COLS = [c for c in df.columns if c not in META_COLS + LABEL_COLS]

TEST_FEAT_MAP = {
    t: [c for c in ALL_FEAT_COLS if TEST_PREFIXES[t](c)]
    for t in TEST_PREFIXES
}

train_df = df[df['condition'].isin(train_conds)]
test_df  = df[df['condition'].isin(test_conds)] if test_conds else None

print(f"Train rows: {len(train_df)} ({len(train_conds)} conditions)  "
      f"Test rows: {len(test_df) if test_df is not None else 0} ({len(test_conds)} conditions)")
print(f"Total features available: {len(ALL_FEAT_COLS)}")
for t, cols in TEST_FEAT_MAP.items():
    print(f"  {t}: {len(cols)} features")

# ── Optuna study (persisted to SQLite) ────────────────────────────────────────
db_path  = os.path.join(os.path.abspath(run_dir), 'study.db')
storage  = f"sqlite:///{db_path}"
sampler  = TPESampler(seed=seed, multivariate=True, warn_independent_sampling=False,
                      n_startup_trials=n_startup_trials,
                      constant_liar=True)  # keeps parallel workers diverse; safe with jobs=1 too
study    = optuna.create_study(
    study_name='bo_study',
    direction='minimize',
    sampler=sampler,
    storage=storage,
    load_if_exists=True,
)

already_done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
remaining    = n_trials - already_done

if remaining <= 0:
    print(f"\nAll {n_trials} trials already complete — nothing to run.")
else:
    print(f"\n{already_done} trials already complete. "
          f"Running {remaining} more (target: {n_trials})…\n")

# ── Objective ─────────────────────────────────────────────────────────────────
def objective(trial):
    active_tests = []
    for t in TEST_PREFIXES:
        if t in require_tests:
            trial.suggest_categorical(f'use_{t}', [True])  # fixed on; still registers param
            active_tests.append(t)
        elif trial.suggest_categorical(f'use_{t}', [True, False]):
            active_tests.append(t)
    if not active_tests:
        raise optuna.exceptions.TrialPruned()

    # Always suggest every feature parameter (static space so multivariate TPE
    # can model correlations). Only include the value if its test is active.
    #
    # CV PCA scores and spline coefficients are each treated as atomic blocks —
    # a single True/False toggle per block rather than per-column.
    CV_PCA_COLS    = {col for col in TEST_FEAT_MAP.get('cv', [])
                      if col.startswith('cv_pca_')}
    CV_SPLINE_COLS = {col for col in TEST_FEAT_MAP.get('cv', [])
                      if '_spline_' in col}

    use_cv_pca    = trial.suggest_categorical('use_cv_pca',    [True, False])
    use_cv_spline = trial.suggest_categorical('use_cv_spline', [True, False])

    selected_cols = []
    for t in TEST_PREFIXES:
        for col in TEST_FEAT_MAP[t]:
            if col in CV_PCA_COLS:
                if t in active_tests and use_cv_pca:
                    selected_cols.append(col)
            elif col in CV_SPLINE_COLS:
                if t in active_tests and use_cv_spline:
                    selected_cols.append(col)
            else:
                include = trial.suggest_categorical(f'feat_{col}', [True, False])
                if t in active_tests and include:
                    selected_cols.append(col)
    if not selected_cols:
        raise optuna.exceptions.TrialPruned()

    X_tr = train_df[selected_cols].fillna(0).values.astype(np.float32)
    y_tr = train_df[LABEL_COLS].values.astype(np.float32)

    if test_df is not None:
        X_te = test_df[selected_cols].fillna(0).values.astype(np.float32)
        y_te = test_df[LABEL_COLS].values.astype(np.float32)
    else:
        X_te, y_te = X_tr, y_tr

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr_sc  = scaler_X.fit_transform(X_tr).astype(np.float32)
    y_tr_sc  = scaler_y.fit_transform(y_tr).astype(np.float32)
    X_te_sc  = scaler_X.transform(X_te).astype(np.float32)
    y_te_sc  = scaler_y.transform(y_te).astype(np.float32)

    np.random.seed(seed + trial.number)
    model   = make_model(X_tr_sc.shape[1])
    val_every = max(1, epochs // 500)
    history = model.train(X_tr_sc, y_tr_sc, epochs=epochs, lr=0.001, verbose=False,
                          X_val=X_te_sc, y_val=y_te_sc, val_every=val_every,
                          patience=patience, max_epochs=max_epochs,
                          asymmetric=asymmetric_loss, alpha=alpha)

    y_pred    = scaler_y.inverse_transform(model.predict(X_te_sc))
    rmse_per  = np.sqrt(np.mean((y_te - y_pred) ** 2, axis=0))
    mean_rmse = float(np.mean(rmse_per))

    trial.set_user_attr('rmse_DA',        float(rmse_per[0]))
    trial.set_user_attr('rmse_AA',        float(rmse_per[1]))
    trial.set_user_attr('rmse_UA',        float(rmse_per[2]))
    trial.set_user_attr('n_features',     len(selected_cols))
    trial.set_user_attr('selected_tests', active_tests)
    trial.set_user_attr('selected_feats', selected_cols)

    # Per-trial folder
    trial_dir = os.path.join(run_dir, f'trial_{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)

    plot_training(
        history,
        save_path=os.path.join(trial_dir, 'training_curve.png'),
        title=(f"Trial {trial.number} — tests: {'+'.join(active_tests)}, "
               f"{len(selected_cols)} features, mean RMSE={mean_rmse:.2f} µM"),
        show=False,
    )

    actual_epochs = history['epochs'][-1] if history['epochs'] else epochs
    with open(os.path.join(trial_dir, 'summary.txt'), 'w') as fh:
        fh.write(f"Trial:          {trial.number}\n")
        fh.write(f"Mean RMSE:      {mean_rmse:.4f} µM\n")
        fh.write(f"RMSE DA:        {rmse_per[0]:.4f} µM\n")
        fh.write(f"RMSE AA:        {rmse_per[1]:.4f} µM\n")
        fh.write(f"RMSE UA:        {rmse_per[2]:.4f} µM\n")
        fh.write(f"Selected tests: {', '.join(active_tests)}\n")
        fh.write(f"N features:     {len(selected_cols)}\n")
        fh.write(f"Train conds:    {sorted(train_conds)}\n")
        fh.write(f"Test conds:     {sorted(test_conds)}\n")
        fh.write(f"Epoch floor:    {epochs}\n")
        fh.write(f"Actual epochs:  {actual_epochs}"
                 f"{' (extended)' if actual_epochs > epochs else ''}\n")
        fh.write(f"Patience:       {patience} val-checks"
                 f"{f'  |  max_epochs: {max_epochs}' if max_epochs else ''}\n")
        fh.write(f"\nFeatures:\n")
        for col in selected_cols:
            fh.write(f"  {col}\n")
        fh.write(f"\nPredictions vs targets (test set):\n")
        fh.write(f"{'Sample':<8}{'DA_true':<12}{'DA_pred':<12}"
                 f"{'AA_true':<12}{'AA_pred':<12}"
                 f"{'UA_true':<12}{'UA_pred':<12}\n")
        fh.write('-' * 80 + '\n')
        for i, (yt, yp) in enumerate(zip(y_te, y_pred)):
            fh.write(f"{i+1:<8}"
                     f"{yt[0]:<12.1f}{yp[0]:<12.1f}"
                     f"{yt[1]:<12.1f}{yp[1]:<12.1f}"
                     f"{yt[2]:<12.1f}{yp[2]:<12.1f}\n")

    return mean_rmse


def trial_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        best    = study.best_value
        rmse_da = trial.user_attrs.get('rmse_DA', float('nan'))
        rmse_aa = trial.user_attrs.get('rmse_AA', float('nan'))
        rmse_ua = trial.user_attrs.get('rmse_UA', float('nan'))
        nf      = trial.user_attrs.get('n_features', '?')
        tests   = '+'.join(trial.user_attrs.get('selected_tests', []))
        done    = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"  [{done:>4}/{n_trials}] Trial {trial.number:>4}  "
              f"mean={trial.value:.2f}  DA={rmse_da:.1f}  "
              f"AA={rmse_aa:.1f}  UA={rmse_ua:.1f}  "
              f"n_feat={nf}  tests=[{tests}]  best={best:.2f}")


# ── Run (with pause support) ──────────────────────────────────────────────────
def generate_reports(study):
    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No completed trials to report.")
        return

    n_done        = len(completed_trials)
    top_k         = min(top_n, n_done)
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:top_k]

    print(f"\n{'='*70}")
    print(f"Results — {n_done}/{n_trials} trials complete")
    print(f"{'='*70}\n")

    header = (f"{'Rank':<6}{'Trial':<8}{'MeanRMSE':<12}"
              f"{'DA':<10}{'AA':<10}{'UA':<10}{'nFeat':<8}{'Tests'}")
    print(f"Top {top_k} trials:")
    print(header)
    print('-' * len(header))
    for rank, t in enumerate(sorted_trials, 1):
        da  = t.user_attrs.get('rmse_DA', float('nan'))
        aa  = t.user_attrs.get('rmse_AA', float('nan'))
        ua  = t.user_attrs.get('rmse_UA', float('nan'))
        nf  = t.user_attrs.get('n_features', '?')
        tst = '+'.join(t.user_attrs.get('selected_tests', []))
        print(f"{rank:<6}{t.number:<8}{t.value:<12.2f}"
              f"{da:<10.1f}{aa:<10.1f}{ua:<10.1f}{nf:<8}{tst}")

    best       = study.best_trial
    best_feats = best.user_attrs.get('selected_feats', [])
    print(f"\nBest trial #{best.number} — mean RMSE {best.value:.2f} µM")
    print(f"Selected features ({len(best_feats)}):")
    for f in best_feats:
        print(f"  {f}")

    feat_counts = {col: 0 for col in ALL_FEAT_COLS}
    for t in sorted_trials:
        for col in t.user_attrs.get('selected_feats', []):
            if col in feat_counts:
                feat_counts[col] += 1
    nonzero       = {k: v for k, v in feat_counts.items() if v > 0}
    sorted_counts = sorted(nonzero.items(), key=lambda x: -x[1])

    # JSON
    json_path = os.path.join(run_dir, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump({
            'run_dir':        run_dir,
            'trials_done':    n_done,
            'n_trials_total': n_trials,
            'train_conds':    sorted(train_conds),
            'test_conds':     sorted(test_conds),
            'best_trial':     best.number,
            'best_mean_rmse': best.value,
            'best_features':  best_feats,
            'best_tests':     best.user_attrs.get('selected_tests', []),
            'top_trials': [
                {
                    'rank':           i + 1,
                    'trial':          t.number,
                    'mean_rmse':      t.value,
                    'rmse_DA':        t.user_attrs.get('rmse_DA'),
                    'rmse_AA':        t.user_attrs.get('rmse_AA'),
                    'rmse_UA':        t.user_attrs.get('rmse_UA'),
                    'n_features':     t.user_attrs.get('n_features'),
                    'selected_tests': t.user_attrs.get('selected_tests', []),
                    'features':       t.user_attrs.get('selected_feats', []),
                }
                for i, t in enumerate(sorted_trials)
            ],
            'feature_frequency': {k: v for k, v in sorted_counts},
        }, fh, indent=2)
    print(f"\nResults JSON saved → {json_path}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trial_nums  = [t.number for t in completed_trials]
    trial_vals  = [t.value  for t in completed_trials]
    best_so_far = np.minimum.accumulate(trial_vals)
    axes[0].plot(trial_nums, trial_vals,  '.', color='steelblue', alpha=0.5, ms=4, label='Trial RMSE')
    axes[0].plot(trial_nums, best_so_far, '-', color='darkred',   lw=1.8,         label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Mean RMSE (µM)')
    axes[0].set_title(f'BO Convergence ({n_done}/{n_trials} trials)', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    top20 = sorted_counts[:20]
    if top20:
        fnames, fcounts = zip(*top20)
        y_pos = range(len(fnames))
        axes[1].barh(y_pos, fcounts, color='steelblue', alpha=0.8)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(fnames, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlabel(f'Frequency in top-{top_k} trials')
        axes[1].set_title('Feature Importance (top-20 by frequency)', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'bo_convergence.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"BO convergence plot saved → {plot_path}")


# ── Parallel worker (fork-based — inherits all globals from parent) ───────────
def _worker(worker_cfg):
    """Runs in a separate forked process. Connects to the shared SQLite study
    and optimises its assigned slice of trials independently."""
    w_sampler = TPESampler(seed=worker_cfg['worker_seed'], multivariate=True,
                           warn_independent_sampling=False,
                           n_startup_trials=n_startup_trials, constant_liar=True)
    db_path = os.path.join(os.path.abspath(run_dir), 'study.db')
    w_study  = optuna.create_study(
        study_name='bo_study', direction='minimize',
        sampler=w_sampler,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )
    w_study.optimize(objective, n_trials=worker_cfg['n_trials'], callbacks=[trial_callback])
# ── end parallel worker ───────────────────────────────────────────────────────


if remaining > 0:
    try:
        if jobs > 1:
            n_workers      = min(jobs, remaining)
            base, leftover = divmod(remaining, n_workers)
            worker_cfgs    = [
                {'n_trials': base + (1 if i < leftover else 0), 'worker_seed': seed + i}
                for i in range(n_workers)
            ]
            print(f"Parallelising across {n_workers} workers (fork + constant-liar TPE).\n")
            ctx = mp.get_context('fork')
            with ctx.Pool(n_workers) as pool:
                pool.map(_worker, worker_cfgs)
        else:
            study.optimize(objective, n_trials=remaining, callbacks=[trial_callback])
        generate_reports(study)
    except KeyboardInterrupt:
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\n\nPaused after {done}/{n_trials} trials.")
        print(f"All progress saved. Resume with:")
        print(f"  python bayes_optimize.py --resume \"{run_dir}\"\n")
        generate_reports(study)
else:
    generate_reports(study)
