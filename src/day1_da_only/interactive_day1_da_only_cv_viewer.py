#!/usr/bin/env python3
"""Interactive viewer for Day 1 DA-only CV and CV GC curves.

Shows one condition at a time with four panels:

    CV normal anodic
    CV normal cathodic
    CV GC anodic
    CV GC cathodic

Each panel overlays the four electrodes. The condition slider selects the
condition. The x-width slider changes the physical axis width while preserving
the same voltage range. The y-zoom slider tightens or loosens the y limits.
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

CACHE_ROOT = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(CACHE_ROOT / "mplconfig").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider


ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_SUBSET_DIR = ROOT / "Day1_DA_Only_Conditions"
DEFAULT_MANIFEST = DEFAULT_SUBSET_DIR / "day1_da_only_conditions.csv"

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)

ELECTRODE_COLORS = {
    "E1": "#000000",
    "E3": "#D55E00",
    "E5": "#0072B2",
    "E7": "#009E73",
}

CURRENT_SCALE = 1e6
CURRENT_UNIT = "uA"
Y_FOCUS_FLOOR = -1.0

TECHNIQUES = {
    "CV normal": {
        "filename_pattern": "cv_norm",
        "electrode_columns": {"E1": 1, "E3": 2, "E5": 3, "E7": 4},
    },
    "CV GC": {
        "filename_pattern": "cv_gc",
        "electrode_columns": {"E1": 1, "E3": 3, "E5": 5, "E7": 7},
    },
}


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
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]

    required_columns = {"Condition", "Dopamine (uM)", "Ascorbic Acid (uM)", "Uric Acid (uM)"}
    required_columns.add("Subset Folder")
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
            stripped = line.strip()
            if not NUMBER_PATTERN.match(stripped):
                continue

            values = [float(value) for value in NUMBER_PATTERN.findall(stripped)]
            if len(values) >= 2:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows found in {file_path}")

    return rows


def split_cv_sweeps(rows: list[list[float]]) -> dict[str, list[list[float]]]:
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


def extract_electrode_traces(
    sweep_rows: list[list[float]],
    electrode_columns: dict[str, int],
) -> dict[str, tuple[list[float], list[float]]]:
    traces = {}

    for electrode, column_index in electrode_columns.items():
        x_values = []
        y_values = []

        for row in sweep_rows:
            if column_index >= len(row):
                continue
            x_values.append(row[0])
            y_values.append(row[column_index] * CURRENT_SCALE)

        if len(x_values) > 2:
            traces[electrode] = (x_values, y_values)

    return traces


def load_condition_data(condition: dict[str, str], subset_dir: Path) -> dict[str, object]:
    condition_id = clean_value(condition["Condition"])
    condition_folder = ROOT / condition["Subset Folder"]

    if not condition_folder.is_dir():
        condition_folder = subset_dir / f"Condition_{condition_id}"

    panels = {}
    files = {}

    for technique_name, technique_config in TECHNIQUES.items():
        source_file = find_technique_file(
            condition_folder,
            str(technique_config["filename_pattern"]),
        )
        files[technique_name] = source_file

        if source_file is None:
            for sweep_name in ("Anodic", "Cathodic"):
                panels[(technique_name, sweep_name)] = {}
            continue

        rows = load_numeric_rows(source_file)
        sweeps = split_cv_sweeps(rows)
        electrode_columns = technique_config["electrode_columns"]

        for sweep_name in ("Anodic", "Cathodic"):
            panels[(technique_name, sweep_name)] = extract_electrode_traces(
                sweeps[sweep_name],
                electrode_columns,
            )

    return {"panels": panels, "files": files, "condition_folder": condition_folder}


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(f"Current ({CURRENT_UNIT})")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.7)
    ax.minorticks_on()
    ax.grid(False)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)


def set_axis_limits(ax: plt.Axes, traces: dict[str, tuple[list[float], list[float]]], y_zoom: float) -> None:
    x_values = []
    y_values = []

    for x_trace, y_trace in traces.values():
        x_values.extend(x_trace)
        y_values.extend(y_trace)

    if not x_values or not y_values:
        return

    ax.set_xlim(min(x_values), max(x_values))

    y_min = min(y_values)
    y_max = max(y_values)
    y_span = max(y_max - y_min, 1e-9)
    padding = max(y_span * 0.08, 0.05)

    lower = max(y_min - padding, Y_FOCUS_FLOOR)
    lower = min(lower, -0.03)

    upper = max(y_max + padding, 0.05)
    focused_span = max(upper - lower, 0.1) / max(y_zoom, 0.05)
    upper = max(lower + focused_span, 0.05)

    ax.set_ylim(lower, upper)


def apply_x_width(axes: list[plt.Axes], x_width: float) -> None:
    # Smaller x_width -> larger box aspect -> physically narrower x axis.
    box_aspect = 0.56 / max(x_width, 0.05)
    for ax in axes:
        ax.set_box_aspect(box_aspect)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
        }
    )


def nearest_condition(condition_numbers: list[int], requested: int) -> int:
    return min(condition_numbers, key=lambda condition_number: abs(condition_number - requested))


def build_viewer(
    conditions: list[dict[str, str]],
    subset_dir: Path,
    start_condition: int,
    no_show: bool,
) -> None:
    configure_plot_style()

    condition_numbers = [int(float(row["Condition"])) for row in conditions]
    condition_by_number = dict(zip(condition_numbers, conditions))
    current_condition = nearest_condition(condition_numbers, start_condition)

    state = {
        "condition": current_condition,
        "x_width": 0.72,
        "y_zoom": 1.25,
    }

    fig, axes_grid = plt.subplots(2, 2, figsize=(12.0, 8.7))
    axes = [axes_grid[0][0], axes_grid[0][1], axes_grid[1][0], axes_grid[1][1]]
    plt.subplots_adjust(left=0.075, right=0.985, top=0.875, bottom=0.245, hspace=0.42, wspace=0.18)

    panel_order = [
        ("CV normal", "Anodic"),
        ("CV normal", "Cathodic"),
        ("CV GC", "Anodic"),
        ("CV GC", "Cathodic"),
    ]

    legend_handles = [
        Line2D([0], [0], color=color, lw=1.6, label=electrode)
        for electrode, color in ELECTRODE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=4,
        bbox_to_anchor=(0.975, 0.018),
        frameon=False,
    )

    ax_condition = fig.add_axes([0.12, 0.165, 0.58, 0.030])
    ax_x_width = fig.add_axes([0.12, 0.108, 0.58, 0.030])
    ax_y_zoom = fig.add_axes([0.12, 0.051, 0.58, 0.030])
    ax_reset = fig.add_axes([0.775, 0.112, 0.11, 0.052])
    ax_print = fig.add_axes([0.775, 0.045, 0.11, 0.052])

    condition_slider = Slider(
        ax_condition,
        "Condition",
        min(condition_numbers),
        max(condition_numbers),
        valinit=current_condition,
        valstep=condition_numbers,
        valfmt="%0.0f",
        color="#88AA55",
    )
    x_width_slider = Slider(
        ax_x_width,
        "X width",
        0.45,
        1.25,
        valinit=state["x_width"],
        valstep=0.01,
        color="#CC8855",
    )
    y_zoom_slider = Slider(
        ax_y_zoom,
        f"Y zoom >= {Y_FOCUS_FLOOR:g} {CURRENT_UNIT}",
        0.50,
        8.00,
        valinit=state["y_zoom"],
        valstep=0.05,
        color="#5588CC",
    )
    reset_button = Button(ax_reset, "Reset scale", color="#EEEEEE", hovercolor="#DDDDDD")
    print_button = Button(ax_print, "Print files", color="#EEEEEE", hovercolor="#CCDDFF")

    def redraw() -> None:
        condition = condition_by_number[int(state["condition"])]
        condition_id = clean_value(condition["Condition"])
        dopamine = clean_value(condition["Dopamine (uM)"])
        aa = clean_value(condition["Ascorbic Acid (uM)"])
        ua = clean_value(condition["Uric Acid (uM)"])

        loaded = load_condition_data(condition, subset_dir)
        panels = loaded["panels"]

        fig.suptitle(
            f"Day 1 DA-only CV viewer | Condition {condition_id} | "
            f"DA {dopamine} uM, AA {aa} uM, UA {ua} uM",
            fontsize=12,
            fontweight="semibold",
        )

        for ax, panel_key in zip(axes, panel_order):
            ax.cla()
            style_axis(ax)
            technique_name, sweep_name = panel_key
            traces = panels[panel_key]

            ax.set_title(f"{technique_name} - {sweep_name}")
            if not traces:
                ax.text(
                    0.5,
                    0.5,
                    "No data found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            for electrode, (x_values, y_values) in traces.items():
                ax.plot(
                    x_values,
                    y_values,
                    color=ELECTRODE_COLORS[electrode],
                    label=electrode,
                    linewidth=1.15,
                    alpha=0.92,
                )

            set_axis_limits(ax, traces, state["y_zoom"])

        apply_x_width(axes, state["x_width"])
        fig.canvas.draw_idle()

    def on_condition(value: float) -> None:
        state["condition"] = int(round(value))
        redraw()

    def on_x_width(value: float) -> None:
        state["x_width"] = float(value)
        apply_x_width(axes, state["x_width"])
        fig.canvas.draw_idle()

    def on_y_zoom(value: float) -> None:
        state["y_zoom"] = float(value)
        redraw()

    def on_reset(_event) -> None:
        x_width_slider.set_val(0.72)
        y_zoom_slider.set_val(1.25)

    def on_print(_event) -> None:
        condition = condition_by_number[int(state["condition"])]
        loaded = load_condition_data(condition, subset_dir)
        print(f"\nCondition {clean_value(condition['Condition'])}")
        print(f"Folder: {loaded['condition_folder']}")
        for technique_name, file_path in loaded["files"].items():
            print(f"{technique_name}: {file_path}")

    condition_slider.on_changed(on_condition)
    x_width_slider.on_changed(on_x_width)
    y_zoom_slider.on_changed(on_y_zoom)
    reset_button.on_clicked(on_reset)
    print_button.on_clicked(on_print)

    redraw()

    if no_show:
        plt.close(fig)
        print("Loaded viewer data successfully.")
        return

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Day 1 DA-only CV/CV GC curve viewer."
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
        "--condition",
        type=int,
        default=5,
        help="Starting condition number. Default: 5",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Load and render once without opening a GUI window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = read_manifest(args.manifest.resolve())
    build_viewer(
        conditions=conditions,
        subset_dir=args.subset_dir.resolve(),
        start_condition=args.condition,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
