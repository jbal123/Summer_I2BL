#!/usr/bin/env python3
"""Standalone CV data loading for the Beyond-Linear-Superposition pipeline.

Self-contained (no imports from the rest of the repo). It re-implements the
minimal CV ``.txt`` parsing needed to read mean cyclic-voltammetry curves and
drives off the two existing data manifests:

  * All_Analyte_Isolated_PCR/isolated_analyte_conditions.csv   (DA/AA/UA isolates)
  * All_Analyte_Superposition_Analysis/multi_analyte_conditions.csv (mixtures)

Curves are averaged across the four working electrodes, split into anodic /
cathodic sweeps, and resampled onto a shared voltage grid so that every curve
(isolate or mixture, of a given technique+sweep) is directly comparable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Repo root is two levels above this file: src/beyond_linear_superposition/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Raw condition data lives under data/; the manifest CSVs store condition_folder
# paths relative to this data root.
DATA_ROOT = REPO_ROOT / "data"
RESULTS_ROOT = REPO_ROOT / "results"
# ROOT is retained as the anchor for resolving condition_folder entries (data side).
ROOT = DATA_ROOT

ISOLATE_MANIFEST = RESULTS_ROOT / "All_Analyte_Isolated_PCR" / "isolated_analyte_conditions.csv"
MIXTURE_MANIFEST = RESULTS_ROOT / "All_Analyte_Superposition_Analysis" / "multi_analyte_conditions.csv"

CURRENT_SCALE = 1e6  # amps -> microamps
CURRENT_UNIT = "uA"

ANALYTE_ORDER = ("DA", "AA", "UA")

# Technique -> filename token + 1-based current column indices for the four electrodes.
TECHNIQUES = {
    "cv_normal": {"label": "CV normal", "pattern": "cv_norm", "columns": [1, 2, 3, 4]},
    "cv_gc": {"label": "CV GC", "pattern": "cv_gc", "columns": [1, 3, 5, 7]},
}
SWEEPS = ("anodic", "cathodic")

# (technique, sweep) panels processed by the pipeline, in display order.
PANEL_ORDER = [
    ("cv_normal", "anodic"),
    ("cv_normal", "cathodic"),
    ("cv_gc", "anodic"),
    ("cv_gc", "cathodic"),
]

_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- #
# Raw file parsing
# --------------------------------------------------------------------------- #
def find_technique_file(condition_folder: Path, pattern: str) -> Path | None:
    matches = sorted(
        path for path in condition_folder.glob("*.txt") if pattern in path.name.lower()
    )
    return matches[0] if matches else None


def load_numeric_rows(file_path: Path) -> list[list[float]]:
    rows = []
    with file_path.open("r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not _NUMBER_PATTERN.match(stripped):
                continue
            values = [float(value) for value in _NUMBER_PATTERN.findall(stripped)]
            if len(values) >= 2:
                rows.append(values)
    return rows


def split_cv_sweeps(rows: list[list[float]]) -> dict[str, list[list[float]]]:
    """Split a CV record into anodic (increasing-V) and cathodic (decreasing-V)
    halves at the first turning point of the potential sweep."""
    x_values = [row[0] for row in rows]
    deltas = [x_values[i + 1] - x_values[i] for i in range(len(x_values) - 1)]
    nonzero = [(i, d) for i, d in enumerate(deltas) if abs(d) > 1e-12]
    if not nonzero:
        return {"anodic": rows, "cathodic": []}

    first_sign = 1 if nonzero[0][1] > 0 else -1
    split_index = None
    for i, delta in nonzero[1:]:
        if (1 if delta > 0 else -1) != first_sign:
            split_index = i
            break

    if split_index is None:
        forward, reverse = rows, []
    else:
        forward, reverse = rows[: split_index + 1], rows[split_index:]

    if first_sign > 0:
        return {"anodic": forward, "cathodic": reverse}
    return {"anodic": reverse, "cathodic": forward}


def mean_electrode_trace(
    sweep_rows: list[list[float]],
    columns: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Average the electrode current columns into a single mean trace (in uA),
    sorted by ascending potential."""
    per_electrode = []
    x_ref = None
    for col in columns:
        xs, ys = [], []
        for row in sweep_rows:
            if col >= len(row):
                continue
            xs.append(row[0])
            ys.append(row[col] * CURRENT_SCALE)
        if len(xs) <= 10:
            continue
        order = np.argsort(xs)
        xs = np.asarray(xs, dtype=float)[order]
        ys = np.asarray(ys, dtype=float)[order]
        if x_ref is None:
            x_ref = xs
            per_electrode.append(ys)
        elif len(xs) == len(x_ref) and np.allclose(xs, x_ref):
            per_electrode.append(ys)
        else:
            per_electrode.append(np.interp(x_ref, xs, ys))
    if not per_electrode or x_ref is None:
        return None
    return x_ref, np.mean(np.vstack(per_electrode), axis=0)


def load_condition_curve(
    condition_folder: Path,
    technique: str,
    sweep: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the mean CV curve for one (technique, sweep) of a condition folder."""
    config = TECHNIQUES[technique]
    source = find_technique_file(condition_folder, config["pattern"])
    if source is None:
        return None
    rows = load_numeric_rows(source)
    if not rows:
        return None
    sweeps = split_cv_sweeps(rows)
    sweep_rows = sweeps.get(sweep)
    if not sweep_rows:
        return None
    return mean_electrode_trace(sweep_rows, config["columns"])


# --------------------------------------------------------------------------- #
# Manifest readers
# --------------------------------------------------------------------------- #
def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [
            {k.strip(): (v.strip() if v is not None else "") for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]


@dataclass
class IsolateCondition:
    analyte: str
    concentration_uM: float
    folder: Path


@dataclass
class MixtureCondition:
    day: int
    condition: int
    condition_code: int
    da_uM: float
    aa_uM: float
    ua_uM: float
    folder: Path

    @property
    def concentrations(self) -> dict[str, float]:
        return {"DA": self.da_uM, "AA": self.aa_uM, "UA": self.ua_uM}

    @property
    def concentration_vector(self) -> np.ndarray:
        return np.array([self.da_uM, self.aa_uM, self.ua_uM], dtype=float)


def read_isolate_conditions(manifest: Path = ISOLATE_MANIFEST) -> dict[str, list[IsolateCondition]]:
    grouped: dict[str, list[IsolateCondition]] = {a: [] for a in ANALYTE_ORDER}
    for row in _read_csv_dicts(manifest):
        analyte = row["analyte"].strip()
        if analyte not in grouped:
            continue
        grouped[analyte].append(
            IsolateCondition(
                analyte=analyte,
                concentration_uM=float(row["calibration_concentration_uM"]),
                folder=ROOT / row["condition_folder"],
            )
        )
    for analyte in grouped:
        grouped[analyte].sort(key=lambda c: c.concentration_uM)
    return grouped


def read_mixture_conditions(manifest: Path = MIXTURE_MANIFEST) -> list[MixtureCondition]:
    conditions = []
    for row in _read_csv_dicts(manifest):
        conditions.append(
            MixtureCondition(
                day=int(float(row["day"])),
                condition=int(float(row["condition"])),
                condition_code=int(float(row["condition_code"])),
                da_uM=float(row["dopamine_uM"]),
                aa_uM=float(row["ascorbic_acid_uM"]),
                ua_uM=float(row["uric_acid_uM"]),
                folder=ROOT / row["condition_folder"],
            )
        )
    return conditions


# --------------------------------------------------------------------------- #
# Common-grid assembly
# --------------------------------------------------------------------------- #
def common_voltage_grid(curves: list[tuple[np.ndarray, np.ndarray]], points: int) -> np.ndarray:
    """Voltage grid spanning the overlap (intersection) of all supplied curves."""
    lo = max(float(np.min(x)) for x, _ in curves)
    hi = min(float(np.max(x)) for x, _ in curves)
    if hi <= lo:
        raise ValueError("Curves do not share an overlapping voltage range")
    return np.linspace(lo, hi, int(points))


def resample(curve: tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    x, y = curve
    return np.interp(grid, np.asarray(x, dtype=float), np.asarray(y, dtype=float))


@dataclass
class LoadedPanel:
    """All raw curves for one (technique, sweep), resampled to a common grid."""

    technique: str
    sweep: str
    grid: np.ndarray
    isolate_curves: dict[str, list[tuple[float, np.ndarray]]] = field(default_factory=dict)
    mixture_curves: list[tuple["MixtureCondition", np.ndarray]] = field(default_factory=list)


def load_panel(
    technique: str,
    sweep: str,
    isolates: dict[str, list[IsolateCondition]],
    mixtures: list[MixtureCondition],
    grid_points: int,
    verbose: bool = False,
) -> LoadedPanel | None:
    """Load + resample every isolate and mixture curve for one panel."""
    raw_isolates: dict[str, list[tuple[float, tuple[np.ndarray, np.ndarray]]]] = {}
    raw_mixtures: list[tuple[MixtureCondition, tuple[np.ndarray, np.ndarray]]] = []
    all_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for analyte in ANALYTE_ORDER:
        raw_isolates[analyte] = []
        for cond in isolates.get(analyte, []):
            curve = load_condition_curve(cond.folder, technique, sweep)
            if curve is None:
                if verbose:
                    print(f"  [skip] {analyte} {cond.concentration_uM} uM: no {technique}/{sweep}")
                continue
            raw_isolates[analyte].append((cond.concentration_uM, curve))
            all_curves.append(curve)

    for cond in mixtures:
        curve = load_condition_curve(cond.folder, technique, sweep)
        if curve is None:
            if verbose:
                print(f"  [skip] mixture D{cond.day} C{cond.condition}: no {technique}/{sweep}")
            continue
        raw_mixtures.append((cond, curve))
        all_curves.append(curve)

    if not all_curves:
        return None

    grid = common_voltage_grid(all_curves, grid_points)
    panel = LoadedPanel(technique=technique, sweep=sweep, grid=grid)
    for analyte in ANALYTE_ORDER:
        panel.isolate_curves[analyte] = [
            (conc, resample(curve, grid)) for conc, curve in raw_isolates[analyte]
        ]
    panel.mixture_curves = [(cond, resample(curve, grid)) for cond, curve in raw_mixtures]
    return panel


__all__ = [
    "ROOT",
    "ISOLATE_MANIFEST",
    "MIXTURE_MANIFEST",
    "CURRENT_UNIT",
    "ANALYTE_ORDER",
    "TECHNIQUES",
    "SWEEPS",
    "PANEL_ORDER",
    "IsolateCondition",
    "MixtureCondition",
    "LoadedPanel",
    "read_isolate_conditions",
    "read_mixture_conditions",
    "load_condition_curve",
    "common_voltage_grid",
    "resample",
    "load_panel",
]
