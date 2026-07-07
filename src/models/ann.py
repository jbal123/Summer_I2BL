
# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---

# --- repo data anchor (auto-added during repo reorg) ---
from pathlib import Path as _Path
_DATA_DIR = str(_Path(__file__).resolve().parents[2] / "data" / "outputs_dataset_full")
# --- end repo anchor ---

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from feature_extract_cond import extract_condition, ALL_ELECTRODES
from plot_training import plot_training
from model import ANN, make_model

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Train ANN on extracted electrochemical features.')
parser.add_argument('--data-dir', default=_DATA_DIR,
                    help='Root directory of per-condition subdirectories (e.g. data/outputs_dataset_full).')
parser.add_argument('--test-conditions', nargs='+', type=int,
                    default=[3, 5, 11, 14, 20], metavar='N',
                    help='Condition numbers held out for final evaluation. Default: 3 5 11 14 20.')
parser.add_argument('--train-conditions', nargs='+', type=int, default=None, metavar='N',
                    help='Condition numbers used for training. Default: all not in --test-conditions.')
parser.add_argument('--val-conditions', nargs='+', type=int, default=None, metavar='N',
                    help='Condition numbers monitored every --val-every epochs during training.')
parser.add_argument('--val-every', type=int, default=100, metavar='N',
                    help='Print train/val loss every N epochs. Default: 100.')
parser.add_argument('--tests', nargs='+', default=None,
                    choices=['swv', 'dpv', 'cv', 'ca', 'cagc'], metavar='TEST',
                    help='Electrochemical test types to include as features. '
                         'Choices: swv dpv cv ca cagc. Default: all.')
args = parser.parse_args()

# Maps each test name to a column-inclusion predicate.
# 'ca' explicitly excludes 'cagc_' so they can be selected independently.
TEST_PREFIXES = {
    'swv':  lambda c: c.startswith('swv_'),
    'dpv':  lambda c: c.startswith('dpv_'),
    'cv':   lambda c: c.startswith('cv_'),
    'ca':   lambda c: c.startswith('ca_') and not c.startswith('cagc_'),
    'cagc': lambda c: c.startswith('cagc_'),
}

# ── Feature extraction ────────────────────────────────────────────────────────
cond_dirs = sorted(
    glob.glob(os.path.join(args.data_dir, '*/')),
    key=lambda x: float(os.path.basename(x.rstrip('/\\')))
)
if not cond_dirs:
    parser.error(f"No condition subdirectories found in {args.data_dir!r}")

print(f"Extracting features from {len(cond_dirs)} conditions in {args.data_dir!r}…", flush=True)
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
    parser.error("Feature extraction produced no rows — check that data_dir contains valid condition folders.")

df = pd.DataFrame(all_rows)

META_COLS  = ['condition', 'electrode']
LABEL_COLS = ['DA_uM', 'AA_uM', 'UA_uM']
feat_cols  = [c for c in df.columns if c not in META_COLS + LABEL_COLS]

# Filter feature columns by selected test types
if args.tests is not None:
    selected_tests = set(args.tests)
    feat_cols = [c for c in feat_cols
                 if any(TEST_PREFIXES[t](c) for t in selected_tests)]
    if not feat_cols:
        parser.error(f"No feature columns matched the selected tests: {args.tests}")

# Partition rows by role: train / val (epoch monitor) / test (final eval)
all_conds   = sorted(df['condition'].unique().tolist())
test_conds  = args.test_conditions
val_conds   = args.val_conditions
train_conds = args.train_conditions if args.train_conditions else [c for c in all_conds if c not in test_conds]
train_df = df[df['condition'].isin(train_conds)]
val_df   = df[df['condition'].isin(val_conds)]  if val_conds  else None
test_df  = df[df['condition'].isin(test_conds)] if test_conds else None

X = train_df[feat_cols].fillna(0).values.astype(np.float32)
y = train_df[LABEL_COLS].values.astype(np.float32)

print(f"Data dir: {args.data_dir}  ({len(df)} rows, {len(cond_dirs)} conditions)")
print(f"Tests: {', '.join(sorted(args.tests)) if args.tests else 'all'}")
print(f"Training conditions: {sorted(train_conds)}")
print(f"Training rows: {X.shape[0]}  ({X.shape[1]} features)")
if val_df is not None:
    print(f"Val conditions (epoch monitor, every {args.val_every}): {sorted(val_conds)}  ({len(val_df)} rows)")
if test_df is not None:
    print(f"Test conditions (final eval): {sorted(test_conds)}  ({len(test_df)} rows)")
print(f"Analyte ranges — DA: {y[:,0].min():.0f}–{y[:,0].max():.0f} µM  "
      f"AA: {y[:,1].min():.0f}–{y[:,1].max():.0f} µM  "
      f"UA: {y[:,2].min():.0f}–{y[:,2].max():.0f} µM")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X).astype(np.float32)
y_scaled = scaler_y.fit_transform(y).astype(np.float32)


def print_results(y_true, y_pred, header="Predictions vs Targets (µM)"):
    analytes = LABEL_COLS
    print(f"\n=== {header} ===")
    hdr = f"{'Sample':<8}" + "".join(
        f"{'Target_'+a:<14}{'Pred_'+a:<14}{'Err_'+a:<12}" for a in analytes
    )
    print(hdr)
    print("-" * len(hdr))
    for i, (yi, yp) in enumerate(zip(y_true, y_pred)):
        row = f"{i+1:<8}"
        for j in range(3):
            row += f"{yi[j]:<14.1f}{yp[j]:<14.1f}{abs(yi[j]-yp[j]):<12.1f}"
        print(row)
    print(f"\n{'Analyte':<12}{'RMSE (µM)':<14}{'MAE (µM)'}")
    print("-" * 36)
    for j, name in enumerate(analytes):
        rmse = np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2))
        mae  = np.mean(np.abs(y_true[:, j] - y_pred[:, j]))
        print(f"  {name:<10}{rmse:<14.2f}{mae:.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n_features = X_scaled.shape[1]

    # ── Prepare val data for epoch-level monitoring ────────────────────────────
    X_val_mon = y_val_mon = None
    if val_df is not None:
        _Xv       = val_df[feat_cols].fillna(0).values.astype(np.float32)
        _yv       = val_df[LABEL_COLS].values.astype(np.float32)
        X_val_mon = scaler_X.transform(_Xv).astype(np.float32)
        y_val_mon = scaler_y.transform(_yv).astype(np.float32)

    # ── Full-dataset training ──────────────────────────────────────────────────
    print(f"\nTraining ANN [{n_features} → 16 → 8 → 3] on {X_scaled.shape[0]} rows…")
    model   = make_model(n_features)
    history = model.train(X_scaled, y_scaled, epochs=500000, lr=0.001, verbose=True,
                          X_val=X_val_mon, y_val=y_val_mon, val_every=args.val_every)

    # ── Convergence plot ───────────────────────────────────────────────────────
    cond_tag  = '-'.join(str(c) for c in sorted(train_conds))
    test_tag  = '_'.join(sorted(args.tests)) if args.tests else 'all'
    plot_path = f'training_curves_conds{cond_tag}_{test_tag}.png'
    plot_title = (f"Training Convergence — conditions {cond_tag}, "
                  f"tests: {test_tag}, {X_scaled.shape[0]} rows")
    plot_training(history, save_path=plot_path, title=plot_title)

    y_pred_scaled = model.predict(X_scaled)
    y_pred        = scaler_y.inverse_transform(y_pred_scaled)
    print_results(y, y_pred, header="Full-Dataset Fit (in-sample)")

    # ── Held-out test evaluation ───────────────────────────────────────────────
    if test_df is not None:
        X_te      = test_df[feat_cols].fillna(0).values.astype(np.float32)
        y_te      = test_df[LABEL_COLS].values.astype(np.float32)
        X_te_sc   = scaler_X.transform(X_te).astype(np.float32)
        y_te_pred = scaler_y.inverse_transform(model.predict(X_te_sc))
        print_results(y_te, y_te_pred,
                      header=f"Held-Out Test (conditions: {sorted(test_conds)})")

    # ── Leave-One-Out cross-validation ────────────────────────────────────────
    print(f"\nRunning Leave-One-Out CV ({X_scaled.shape[0]} folds)…")
    loo       = LeaveOneOut()
    loo_preds = np.zeros_like(y, dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_scaled), 1):
        X_tr, X_te_loo = X_scaled[train_idx], X_scaled[test_idx]
        y_tr           = y_scaled[train_idx]

        loo_model = make_model(n_features)
        loo_model.train(X_tr, y_tr, epochs=1000, lr=0.001, verbose=False)

        pred_scaled         = loo_model.predict(X_te_loo)
        loo_preds[test_idx] = scaler_y.inverse_transform(pred_scaled)
        print(f"  Fold {fold:>2}/{X_scaled.shape[0]} done", end='\r')

    print()
    print_results(y, loo_preds, header="Leave-One-Out CV")
