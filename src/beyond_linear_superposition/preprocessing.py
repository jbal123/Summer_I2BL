#!/usr/bin/env python3
"""Step 1 preprocessing: AsLSSR baseline correction + COW potential alignment.

Standalone implementation of the "Beyond Linear Superposition" spec, Step 1.
These corrections must be applied identically to isolate (training) and mixture
(prediction) curves. The baseline is fit per-curve; the COW reference is fixed
once (mean of the DA isolate curves at the mid-range concentration) and reused
for every curve and every cross-validation fold.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import correlate


# --------------------------------------------------------------------------- #
# 1a. Asymmetric least-squares baseline correction (AsLSSR)
# --------------------------------------------------------------------------- #
def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline estimate.

    lam: smoothness parameter (1e4-1e7 typical for CV).
    p:   asymmetry parameter (0.001-0.1; smaller = baseline hugs minima harder).
    n_iter: number of reweighting iterations.

    Returns the estimated baseline (same shape as ``y``).
    """
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    H = lam * (D.T @ D)
    w = np.ones(L)
    Z = y.copy()
    for _ in range(n_iter):
        W = sparse.diags(w)
        Z = spsolve((W + H).tocsc(), w * y)
        w = np.where(y > Z, p, 1.0 - p)
    return Z


def correct_baseline(current: np.ndarray, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Return ``current`` with its asymmetric least-squares baseline removed."""
    baseline = als_baseline(current, lam=lam, p=p, n_iter=n_iter)
    return np.asarray(current, dtype=float) - baseline


# --------------------------------------------------------------------------- #
# 1b. Correlation optimized warping (COW) potential-shift alignment
# --------------------------------------------------------------------------- #
def cow_align(
    reference: np.ndarray,
    signal: np.ndarray,
    segment_length: int = 20,
    slack: int = 3,
) -> np.ndarray:
    """Simplified COW alignment of ``signal`` onto ``reference``.

    Divides the signal into segments, finds the optimal integer shift per
    segment by cross-correlation within ``slack``, then gathers the shifted
    samples to realign. ``reference`` and ``signal`` must lie on the same grid.

    segment_length: points per segment (~1/5 of a peak width).
    slack: maximum shift in index units allowed per segment.
    """
    reference = np.asarray(reference, dtype=float)
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == 0 or segment_length < 2 or n < segment_length:
        return signal.copy()

    n_segs = n // segment_length
    warped = signal.copy()
    for i in range(n_segs):
        sl = slice(i * segment_length, (i + 1) * segment_length)
        ref_seg = reference[sl]
        lo = max(0, i * segment_length - slack)
        hi = min(n, (i + 1) * segment_length + slack)
        sig_window = signal[lo:hi]
        if len(sig_window) < len(ref_seg) or len(ref_seg) == 0:
            continue
        corr = correlate(sig_window, ref_seg, mode="valid")
        # Offset of the best-matching window relative to the segment start.
        best_shift = int(np.argmax(corr)) - (i * segment_length - lo)
        src_idx = np.arange(i * segment_length, (i + 1) * segment_length) + best_shift
        src_idx = np.clip(src_idx, 0, n - 1)
        warped[sl] = signal[src_idx]
    return warped


def build_cow_reference(curves: np.ndarray) -> np.ndarray:
    """Build a COW reference as the mean of the supplied (already baseline-
    corrected) curves. The spec fixes this to the DA isolate curves at the
    mid-range concentration (20 uM)."""
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    return curves.mean(axis=0)


def preprocess_curve(
    current: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    als_lam: float = 1e5,
    als_p: float = 0.01,
    als_iter: int = 10,
    cow_segment_length: int = 20,
    cow_slack: int = 3,
    apply_als: bool = True,
    apply_cow: bool = True,
) -> np.ndarray:
    """Apply AsLSSR baseline correction then COW alignment to one curve.

    ``reference`` is required when ``apply_cow`` is True. Pass a baseline-
    corrected reference so the two curves are compared on the same footing.
    """
    out = np.asarray(current, dtype=float)
    if apply_als:
        out = correct_baseline(out, lam=als_lam, p=als_p, n_iter=als_iter)
    if apply_cow:
        if reference is None:
            raise ValueError("COW alignment requires a reference curve")
        out = cow_align(reference, out, segment_length=cow_segment_length, slack=cow_slack)
    return out


__all__ = [
    "als_baseline",
    "correct_baseline",
    "cow_align",
    "build_cow_reference",
    "preprocess_curve",
]
