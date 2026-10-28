"""Segmented Gauss-Legendre quadrature, plain and in log-space.

Log-space evaluation is mandatory for the heavy-tail integrands: near tail
index 2 the integrands span hundreds of orders of magnitude, and any linear-
space evaluation silently under/overflows.  Every integral returns per-segment
contributions so callers can hard-assert that the terminal segment is
negligible (tail-remainder check) instead of trusting a truncation.
"""

from functools import lru_cache

import numpy as np
from scipy import special


@lru_cache(maxsize=32)
def _gl_raw(n):
    return special.roots_legendre(n)


def gl_nodes(a, b, n):
    """Nodes and weights of n-point Gauss-Legendre on [a, b]."""
    xs, ws = _gl_raw(int(n))
    return (a + b) / 2 + (b - a) / 2 * xs, (b - a) / 2 * ws


def gl_integrate(f, pts, n=2000):
    """int f over [pts[0], pts[-1]] split at the interior points; returns
    (total, per_segment) with one n-point GL rule per segment."""
    pts = sorted(float(p) for p in pts)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            segs.append(0.0)
            continue
        xs, ws = gl_nodes(a, b, n)
        segs.append(float(np.sum(ws * f(xs))))
    return float(np.sum(segs)), np.asarray(segs)


def gl_integrate_log(logf, pts, n=2000):
    """int exp(logf) over segmented [pts[0], pts[-1]], evaluated stably.

    Per segment, the max of logf on the nodes is subtracted before
    exponentiating (log-sum-exp).  logf may return -inf.  Returns
    (total, per_segment).
    """
    pts = sorted(float(p) for p in pts)
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            segs.append(0.0)
            continue
        xs, ws = gl_nodes(a, b, n)
        lv = np.asarray(logf(xs), dtype=float)
        m = np.max(lv)
        if not np.isfinite(m):
            segs.append(0.0)
            continue
        segs.append(float(np.exp(m) * np.sum(ws * np.exp(lv - m))))
    return float(np.sum(segs)), np.asarray(segs)


def assert_tail_negligible(segs, rel=1e-10, what="integral"):
    """Raise if the terminal segment carries more than `rel` of the total —
    the truncation point was too aggressive."""
    total = float(np.sum(segs))
    if total <= 0:
        return
    frac = abs(float(segs[-1])) / total
    if frac > rel:
        raise AssertionError(
            f"tail remainder of {what}: last segment holds {frac:.2e} "
            f"of the total (> {rel:.0e}); enlarge the integration range")
