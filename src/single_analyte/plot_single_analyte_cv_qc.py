#!/usr/bin/env python3
"""Create CV/CV-GC electrode QC plots for the revised single-analyte dataset."""

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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CACHE = Path(tempfile.gettempdir()) / "microneedlearrayml_matplotlib"
(_CACHE / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATA_ROOT = ROOT / "Single_Analyte_Data_revised"
DEFAULT_OUTPUT_PDF = DEFAULT_DATA_ROOT / "single_analyte_cv_cvgc_electrode_qc.pdf"
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_ROOT / "single_analyte_cv_cvgc_electrode_qc_summary.csv"

CURRENT_SCALE = 1e6
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
COND_RE = re.compile(r"Cond(\d+)")
DA_RE = re.compile(r"DA(\d+(?:p\d+)?)")
AA_RE = re.compile(r"AA(\d+(?:p\d+)?)")
UA_RE = re.compile(r"UA(\d+(?:p\d+)?)")

ANALYTE_ORDER = ("DA", "AA", "UA")
TECHNIQUE_ORDER = ("CV_Norm", "CV_GC")

CV_NORM_LABELS = [f"Ch{i}" for i in range(1, 9)]
CV_GC_LABELS = [
    "E1 generator",
    "E1 collector",
    "E3 generator",
    "E3 collector",
    "E5 generator",
    "E5 collector",
    "E7 generator",
    "E7 collector",
]


@dataclass(frozen=True)
class CVFile:
    analyte: str
    folder: str
    condition: int
    da_uM: float | None
    aa_uM: float | None
    ua_uM: float | None
    technique: str
    path: Path


@dataclass(frozen=True)
class SweepSegment:
    start: int
    end: int
    sign: int


@dataclass(frozen=True)
class CycleSelection:
    matrix: np.ndarray
    start_row: int
    end_row: int
    selected_cycle: int
    total_complete_cycles: int


def parse_value(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))


def parse_condition(path: Path) -> int | None:
    match = COND_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def technique_for(path: Path) -> str | None:
    name = path.name.lower()
    if not name.endswith(".txt"):
        return None
    if "cv_norm" in name:
        return "CV_Norm"
    if "cv_gc" in name:
        return "CV_GC"
    return None


def is_numeric_folder(name: str) -> bool:
    try:
        float(name)
    except ValueError:
        return False
    return True


def is_target_analyte_file(analyte: str, da: float | None, aa: float | None, ua: float | None) -> bool:
    da_v = 0.0 if da is None else da
    aa_v = 0.0 if aa is None else aa
    ua_v = 0.0 if ua is None else ua
    if analyte == "DA":
        return da_v >= 0.0 and aa_v == 0.0 and ua_v == 0.0 and (da is not None or (aa_v == 0.0 and ua_v == 0.0))
    if analyte == "AA":
        return da_v == 0.0 and aa_v > 0.0 and ua_v == 0.0
    if analyte == "UA":
        return da_v == 0.0 and aa_v == 0.0 and ua_v >= 0.0 and ua is not None
    return False


def discover_cv_files(data_root: Path) -> tuple[list[CVFile], list[Path]]:
    selected: list[CVFile] = []
    skipped: list[Path] = []
    for analyte in ANALYTE_ORDER:
        analyte_root = data_root / analyte
        if not analyte_root.is_dir():
            continue
        for folder in sorted(analyte_root.iterdir(), key=lambda p: (not is_numeric_folder(p.name), float(p.name) if is_numeric_folder(p.name) else 1e9, p.name)):
            if not folder.is_dir() or not is_numeric_folder(folder.name):
                continue
            for path in sorted(folder.glob("CV*.txt")):
                technique = technique_for(path)
                if technique is None:
                    skipped.append(path)
                    continue
                condition = parse_condition(path)
                da = parse_value(DA_RE, path.name)
                aa = parse_value(AA_RE, path.name)
                ua = parse_value(UA_RE, path.name)
                if condition is None or not is_target_analyte_file(analyte, da, aa, ua):
                    skipped.append(path)
                    continue
                selected.append(
                    CVFile(
                        analyte=analyte,
                        folder=folder.name,
                        condition=condition,
                        da_uM=da,
                        aa_uM=aa,
                        ua_uM=ua,
                        technique=technique,
                        path=path,
                    )
                )
    selected.sort(
        key=lambda item: (
            ANALYTE_ORDER.index(item.analyte),
            float(item.folder),
            TECHNIQUE_ORDER.index(item.technique),
            item.path.name,
        )
    )
    return selected, skipped


def load_numeric_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            values = [float(value) for value in NUMBER_RE.findall(line)]
            if len(values) >= 2:
                rows.append(values)
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    width = max(len(row) for row in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        matrix[index, : len(row)] = row
    return matrix


def sweep_segments(potential: np.ndarray) -> list[SweepSegment]:
    deltas = np.diff(potential)
    nonzero = np.flatnonzero(np.abs(deltas) > 1e-12)
    if len(nonzero) == 0:
        return []

    segments: list[SweepSegment] = []
    start = 0
    previous_sign = 1 if deltas[nonzero[0]] > 0 else -1
    for idx in nonzero[1:]:
        sign = 1 if deltas[idx] > 0 else -1
        if sign != previous_sign:
            segments.append(SweepSegment(start=start, end=idx + 1, sign=previous_sign))
            start = idx
            previous_sign = sign
    segments.append(SweepSegment(start=start, end=len(potential), sign=previous_sign))
    return segments


def select_last_complete_cycle(matrix: np.ndarray) -> CycleSelection:
    segments = sweep_segments(matrix[:, 0])
    complete_cycles = len(segments) // 2
    if complete_cycles == 0:
        return CycleSelection(
            matrix=matrix,
            start_row=0,
            end_row=len(matrix),
            selected_cycle=1,
            total_complete_cycles=1,
        )

    first_segment = 2 * (complete_cycles - 1)
    start = segments[first_segment].start
    end = segments[first_segment + 1].end
    return CycleSelection(
        matrix=matrix[start:end],
        start_row=start,
        end_row=end,
        selected_cycle=complete_cycles,
        total_complete_cycles=complete_cycles,
    )


def split_sweep_indices(potential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    segments = sweep_segments(potential)
    if not segments:
        indices = np.arange(len(potential))
        return indices, np.array([], dtype=int)
    if len(segments) == 1:
        indices = np.arange(segments[0].start, segments[0].end)
        if segments[0].sign > 0:
            return indices, np.array([], dtype=int)
        return np.array([], dtype=int), indices

    first = np.arange(segments[0].start, segments[0].end)
    second = np.arange(segments[1].start, segments[1].end)
    if segments[0].sign > 0:
        return first, second
    return second, first


def cycle_label(selection: CycleSelection) -> str:
    return (
        f"last complete run {selection.selected_cycle}/{selection.total_complete_cycles}; "
        f"source rows {selection.start_row + 1}-{selection.end_row}"
    )


def channel_flags(values_uA: np.ndarray) -> list[str]:
    finite = values_uA[np.isfinite(values_uA)]
    if len(finite) == 0:
        return ["no numeric data"]
    flags: list[str] = []
    span = float(np.nanmax(finite) - np.nanmin(finite))
    max_abs = float(np.nanmax(np.abs(finite)))
    near_limit_fraction = float(np.mean(np.abs(finite) >= 95.0))
    if max_abs >= 95.0:
        flags.append("near +/-100 uA limit")
    if near_limit_fraction >= 0.05:
        flags.append("saturation plateau")
    if span <= 0.05:
        flags.append("flat/low span")
    if np.nanstd(finite) <= 0.01:
        flags.append("near-constant")
    return flags


def concentration_label(item: CVFile) -> str:
    def fmt(value: float | None) -> str:
        if value is None:
            return "?"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    return f"DA {fmt(item.da_uM)} uM | AA {fmt(item.aa_uM)} uM | UA {fmt(item.ua_uM)} uM"


def current_polarity_multiplier(item: CVFile) -> float:
    if item.analyte == "AA":
        return -1.0
    return 1.0


def current_polarity_label(item: CVFile) -> str:
    if current_polarity_multiplier(item) < 0:
        return "displayed current = -raw current (AA polarity normalized to legacy CV/CV-GC convention)"
    return "displayed current = raw current"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", labelsize=7)
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)


def plot_page(pdf: PdfPages, item: CVFile, selection: CycleSelection, csv_rows: list[dict[str, object]]) -> None:
    matrix = selection.matrix
    potential = matrix[:, 0]
    currents = matrix[:, 1:] * current_polarity_multiplier(item)
    labels = CV_GC_LABELS if item.technique == "CV_GC" else CV_NORM_LABELS
    labels = labels[: currents.shape[1]]
    forward_idx, reverse_idx = split_sweep_indices(potential)

    fig, axes = plt.subplots(4, 2, figsize=(11.0, 14.0), sharex=True)
    axes_flat = axes.ravel()
    all_current_uA = currents * CURRENT_SCALE
    finite = all_current_uA[np.isfinite(all_current_uA)]
    if len(finite):
        y_min = float(np.nanpercentile(finite, 1))
        y_max = float(np.nanpercentile(finite, 99))
        y_span = max(y_max - y_min, 1.0)
        y_limits = (y_min - 0.15 * y_span, y_max + 0.15 * y_span)
    else:
        y_limits = None

    page_flags: list[str] = []
    for channel_index, ax in enumerate(axes_flat):
        if channel_index >= currents.shape[1]:
            ax.axis("off")
            continue
        y_uA = currents[:, channel_index] * CURRENT_SCALE
        flags = channel_flags(y_uA)
        if flags:
            page_flags.append(f"{labels[channel_index]}: {', '.join(flags)}")
        color = "#0072B2" if item.technique == "CV_Norm" else ("#D55E00" if channel_index % 2 == 0 else "#009E73")
        if len(forward_idx):
            ax.plot(potential[forward_idx], y_uA[forward_idx], color=color, lw=0.9, label="increasing V")
        if len(reverse_idx):
            ax.plot(potential[reverse_idx], y_uA[reverse_idx], color=color, lw=0.9, ls="--", alpha=0.85, label="decreasing V")
        ax.axhline(0, color="#888888", lw=0.5)
        ax.set_title(labels[channel_index], fontsize=9)
        ax.set_ylabel("Current (uA)", fontsize=8)
        if y_limits is not None:
            channel_max = float(np.nanmax(np.abs(y_uA[np.isfinite(y_uA)]))) if np.isfinite(y_uA).any() else 0.0
            if channel_max < 95.0:
                ax.set_ylim(*y_limits)
        style_axis(ax)
        if channel_index in {0, 1}:
            ax.legend(loc="best", fontsize=6, frameon=False)

        finite_y = y_uA[np.isfinite(y_uA)]
        csv_rows.append(
            {
                "analyte": item.analyte,
                "folder": item.folder,
                "condition": item.condition,
                "technique": item.technique,
                "source_file": str(item.path.relative_to(ROOT)),
                "concentration_label": concentration_label(item),
                "cycle_selection": cycle_label(selection),
                "current_polarity": current_polarity_label(item),
                "channel": labels[channel_index],
                "channel_index": channel_index + 1,
                "n_points": int(len(finite_y)),
                "min_uA": float(np.nanmin(finite_y)) if len(finite_y) else "",
                "max_uA": float(np.nanmax(finite_y)) if len(finite_y) else "",
                "span_uA": float(np.nanmax(finite_y) - np.nanmin(finite_y)) if len(finite_y) else "",
                "std_uA": float(np.nanstd(finite_y)) if len(finite_y) else "",
                "max_abs_uA": float(np.nanmax(np.abs(finite_y))) if len(finite_y) else "",
                "flags": "; ".join(flags),
            }
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Potential (V)", fontsize=8)

    review_flags = [flag for flag in page_flags if "near +/-100" in flag or "saturation" in flag or "flat" in flag or "near-constant" in flag]
    note = "Review flags: " + (" | ".join(review_flags[:8]) if review_flags else "none by simple heuristic")
    if len(review_flags) > 8:
        note += f" | +{len(review_flags) - 8} more"
    fig.suptitle(
        f"{item.analyte} condition {item.condition} ({item.folder}) - {item.technique}\n"
        f"{concentration_label(item)}\n{item.path.relative_to(ROOT)}",
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(0.03, 0.032, f"{cycle_label(selection)} | {current_polarity_label(item)}", ha="left", va="bottom", fontsize=7)
    fig.text(0.03, 0.018, note, ha="left", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def write_intro_page(pdf: PdfPages, items: list[CVFile], skipped: list[Path]) -> None:
    counts = Counter((item.analyte, item.technique) for item in items)
    lines = [
        "Single Analyte CV/CV-GC Electrode QC",
        "",
        "Scope:",
        "- Current folders only: DA, AA, UA numeric condition folders.",
        "- Techniques: CV_Norm and CV_GC text files.",
        "- Pages show all eight current channels. CV-GC labels odd columns as generator and even columns as collector pairs.",
        "- If a source file contains multiple CV cycles, only the last complete cycle is plotted and summarized.",
        "- Plot legends use sweep direction labels, not inferred oxidation/reduction labels.",
        "- AA CV/CV-GC source files are opposite polarity from the legacy AA plots, so AA is displayed and summarized as -raw current.",
        "- UA pages filter to filenames with UA concentration labels and skip AA-labeled duplicate files found inside UA folders.",
        "",
        "Review flag heuristics:",
        "- near +/-100 uA limit: absolute current reaches at least 95 uA.",
        "- saturation plateau: at least 5% of points are at/near +/-95 uA.",
        "- flat/low span: channel span is <= 0.05 uA.",
        "- near-constant: channel standard deviation is <= 0.01 uA.",
        "",
        "These flags are screening hints only. Use the plots for the final electrode keep/drop decision.",
        "",
        "Included pages:",
    ]
    for analyte in ANALYTE_ORDER:
        total = sum(counts[(analyte, tech)] for tech in TECHNIQUE_ORDER)
        lines.append(f"- {analyte}: {total} pages ({counts[(analyte, 'CV_Norm')]} CV, {counts[(analyte, 'CV_GC')]} CV-GC)")
    lines.extend(
        [
            "",
            f"Skipped CV-like files after folder/analyte filtering: {len(skipped)}",
            f"Companion CSV: {DEFAULT_OUTPUT_CSV.name}",
        ]
    )
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.07, 0.93, lines[0], fontsize=18, fontweight="semibold", va="top")
    fig.text(0.07, 0.86, "\n".join(lines[2:]), fontsize=10, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def write_summary_pages(pdf: PdfPages, csv_rows: list[dict[str, object]]) -> None:
    flagged = [row for row in csv_rows if row["flags"]]
    high_priority = [
        row for row in flagged
        if "near +/-100" in row["flags"] or "saturation" in row["flags"]
    ]
    lines = [
        "Flag Summary",
        "",
        f"Total channel traces reviewed: {len(csv_rows)}",
        f"Channel traces with any flag: {len(flagged)}",
        f"Channel traces with saturation/current-limit flags: {len(high_priority)}",
        "",
        "Saturation/current-limit flags by analyte/condition/technique:",
    ]
    grouped = Counter((row["analyte"], row["folder"], row["condition"], row["technique"]) for row in high_priority)
    for key, count in sorted(grouped.items(), key=lambda x: (ANALYTE_ORDER.index(x[0][0]), float(x[0][1]), x[0][3])):
        analyte, folder, condition, technique = key
        lines.append(f"- {analyte} {folder} Cond{condition} {technique}: {count} channel(s)")
    if not grouped:
        lines.append("- none")

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.07, 0.93, lines[0], fontsize=18, fontweight="semibold", va="top")
    fig.text(0.07, 0.86, "\n".join(lines[2:70]), fontsize=9, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output_pdf = args.output_pdf.resolve()
    output_csv = args.output_csv.resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    items, skipped = discover_cv_files(data_root)
    if args.limit_pages is not None:
        items = items[: args.limit_pages]
    if not items:
        raise RuntimeError(f"No CV/CV-GC files found under {data_root}")

    csv_rows: list[dict[str, object]] = []
    errors: list[str] = []
    with PdfPages(output_pdf) as pdf:
        write_intro_page(pdf, items, skipped)
        for index, item in enumerate(items, 1):
            print(f"[{index}/{len(items)}] {item.path.relative_to(ROOT)}")
            try:
                matrix = load_numeric_matrix(item.path)
                selection = select_last_complete_cycle(matrix)
                plot_page(pdf, item, selection, csv_rows)
            except Exception as exc:  # keep the large report moving and document failures.
                errors.append(f"{item.path.relative_to(ROOT)}: {exc}")
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.08, 0.9, "Plotting Error", fontsize=18, fontweight="semibold")
                fig.text(0.08, 0.82, f"{item.path.relative_to(ROOT)}\n\n{exc}", fontsize=10, va="top")
                pdf.savefig(fig)
                plt.close(fig)
        write_summary_pages(pdf, csv_rows)
        if errors:
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.07, 0.93, "Errors", fontsize=18, fontweight="semibold", va="top")
            fig.text(0.07, 0.86, "\n".join(errors[:60]), fontsize=8, va="top", family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

    write_csv(output_csv, csv_rows)
    print(f"Wrote PDF: {output_pdf}")
    print(f"Wrote CSV: {output_csv}")
    print(f"Pages plotted: {len(items)} data pages + documentation/summary")
    print(f"Skipped filtered CV files: {len(skipped)}")
    if errors:
        print(f"Plotting errors: {len(errors)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--limit-pages", type=int, default=None, help="Debug only: stop after N data pages.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
