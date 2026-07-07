#!/usr/bin/env python3
"""Plot Day 1 DA-only CV and CV GC curves into one PDF.

The DA-only subset is expected to live in:

    Day1_DA_Only_Conditions/

Each page in the output PDF contains one technique. Each condition gets two
panels: anodic and cathodic sweeps. Within each panel, all four electrodes are
overlaid.
"""

from __future__ import annotations

# --- path bootstrap (auto-added during repo reorg): make sibling modules importable ---
import os as _os, sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "features", "day1_da_only", "single_analyte", "multi_analyte", "models"):
    _p = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---


import argparse
import csv
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CACHE_DIR / "xdg_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_SUBSET_DIR = ROOT / "Day1_DA_Only_Conditions"
DEFAULT_MANIFEST = DEFAULT_SUBSET_DIR / "day1_da_only_conditions.csv"
DEFAULT_OUTPUT = DEFAULT_SUBSET_DIR / "day1_da_only_cv_curves.pdf"

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)

ELECTRODE_COLORS = {
    "E1": "#000000",
    "E3": "#D55E00",
    "E5": "#0072B2",
    "E7": "#009E73",
}

TECHNIQUES = {
    "CV normal": {
        "filename_pattern": "cv_norm",
        # Matches Data Analysis.ipynb.
        "electrode_columns": {"E1": 1, "E3": 2, "E5": 3, "E7": 4},
    },
    "CV GC": {
        "filename_pattern": "cv_gc",
        # Matches Data Analysis.ipynb.
        "electrode_columns": {"E1": 1, "E3": 3, "E5": 5, "E7": 7},
    },
}


def configure_plot_style() -> None:
    """Use the same clean plotting style as Data Analysis.ipynb."""
    nature_colors = [
        "#000000",
        "#D55E00",
        "#0072B2",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#999999",
    ]

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "axes.prop_cycle": cycler(color=nature_colors),
            "figure.max_open_warning": 200,
        }
    )


def clean_value(value: str) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value_float.is_integer():
        return str(int(value_float))
    return str(value_float)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw_row in reader:
            rows.append({key.strip(): value.strip() for key, value in raw_row.items()})

    required_columns = {"Condition", "Dopamine (uM)", "Subset Folder"}
    missing = required_columns.difference(rows[0].keys() if rows else [])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Manifest is missing required columns: {missing_text}")

    return sorted(rows, key=lambda row: float(row["Dopamine (uM)"]))


def find_technique_file(condition_folder: Path, filename_pattern: str) -> Path | None:
    matches = sorted(
        file_path
        for file_path in condition_folder.glob("*.txt")
        if filename_pattern in file_path.name.lower()
    )
    return matches[0] if matches else None


def load_numeric_rows(file_path: Path) -> list[list[float]]:
    rows = []

    with file_path.open("r", errors="ignore") as handle:
        for line in handle:
            if not NUMBER_PATTERN.match(line.strip()):
                continue

            values = [float(value) for value in NUMBER_PATTERN.findall(line)]
            if len(values) >= 2:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows found in {file_path}")

    return rows


def split_cv_sweeps(rows: list[list[float]]) -> dict[str, list[list[float]]]:
    """Split a CV loop into anodic and cathodic sweeps by x-direction change."""
    x_values = [row[0] for row in rows]
    deltas = [x_values[index + 1] - x_values[index] for index in range(len(x_values) - 1)]
    nonzero_deltas = [(index, delta) for index, delta in enumerate(deltas) if abs(delta) > 1e-12]

    if not nonzero_deltas:
        return {"Anodic": rows, "Cathodic": []}

    first_sign = 1 if nonzero_deltas[0][1] > 0 else -1
    split_index = None

    for index, delta in nonzero_deltas[1:]:
        current_sign = 1 if delta > 0 else -1
        if current_sign != first_sign:
            split_index = index
            break

    if split_index is None:
        forward_rows = rows
        reverse_rows = []
    else:
        forward_rows = rows[: split_index + 1]
        reverse_rows = rows[split_index:]

    if first_sign > 0:
        return {"Anodic": forward_rows, "Cathodic": reverse_rows}

    return {"Anodic": reverse_rows, "Cathodic": forward_rows}


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current (A)")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.7)
    ax.minorticks_on()
    ax.grid(False)
    ax.margins(x=0.02, y=0.08)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)


def plot_sweep(
    ax: plt.Axes,
    sweep_rows: list[list[float]],
    electrode_columns: dict[str, int],
) -> None:
    if not sweep_rows:
        ax.text(0.5, 0.5, "No sweep data", ha="center", va="center", transform=ax.transAxes)
        return

    for electrode, column_index in electrode_columns.items():
        x_values = []
        y_values = []

        for row in sweep_rows:
            if column_index >= len(row):
                continue
            x_values.append(row[0])
            y_values.append(row[column_index])

        if len(x_values) > 2:
            ax.plot(
                x_values,
                y_values,
                label=electrode,
                color=ELECTRODE_COLORS[electrode],
                linewidth=1.0,
                alpha=0.9,
            )


def add_page_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.4, label=electrode)
        for electrode, color in ELECTRODE_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )


def plot_technique_page(
    pdf: PdfPages,
    conditions: list[dict[str, str]],
    technique_name: str,
    technique_config: dict[str, object],
    subset_dir: Path,
) -> None:
    condition_count = len(conditions)
    fig_height = max(12.0, condition_count * 2.35)
    fig, axes = plt.subplots(
        condition_count,
        2,
        figsize=(13.5, fig_height),
        squeeze=False,
    )

    fig.suptitle(
        f"Day 1 DA-only {technique_name} curves",
        fontsize=14,
        fontweight="semibold",
        y=0.985,
    )

    filename_pattern = str(technique_config["filename_pattern"])
    electrode_columns = technique_config["electrode_columns"]

    for row_index, condition in enumerate(conditions):
        condition_id = clean_value(condition["Condition"])
        dopamine = clean_value(condition["Dopamine (uM)"])
        condition_folder = ROOT / condition["Subset Folder"]

        if not condition_folder.is_dir():
            condition_folder = subset_dir / f"Condition_{condition_id}"

        source_file = find_technique_file(condition_folder, filename_pattern)

        for column_index, sweep_name in enumerate(("Anodic", "Cathodic")):
            ax = axes[row_index][column_index]
            ax.set_title(f"Condition {condition_id} | DA {dopamine} uM | {sweep_name}")
            style_axis(ax)

            if source_file is None:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing {technique_name} file",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            rows = load_numeric_rows(source_file)
            sweeps = split_cv_sweeps(rows)
            plot_sweep(ax, sweeps[sweep_name], electrode_columns)

    add_page_legend(fig)
    fig.tight_layout(rect=(0.04, 0.045, 0.985, 0.965), h_pad=1.15, w_pad=1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf(manifest_path: Path, subset_dir: Path, output_path: Path) -> None:
    configure_plot_style()
    conditions = read_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for technique_name, technique_config in TECHNIQUES.items():
            plot_technique_page(
                pdf=pdf,
                conditions=conditions,
                technique_name=technique_name,
                technique_config=technique_config,
                subset_dir=subset_dir,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DA-only Day 1 CV and CV GC curves into a single PDF."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Subset manifest CSV. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=DEFAULT_SUBSET_DIR,
        help=f"Folder containing copied condition folders. Default: {DEFAULT_SUBSET_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_pdf(
        manifest_path=args.manifest.resolve(),
        subset_dir=args.subset_dir.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"Saved PDF: {args.output.resolve()}")


if __name__ == "__main__":
    main()
