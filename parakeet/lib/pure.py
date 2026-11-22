"""Library functions written in the Python subset Parakeet compiles.

These are ordinary Python: they are inlined and optimised together with the
code that calls them.
"""

import numpy as np

from . import adverbs
from .core import source

__parakeet_lib__ = True

imap = adverbs.imap


@source(np.linspace)
def linspace(start, stop, num=50, endpoint=True):
    n = int(num)
    if n < 0:
        raise ValueError("Number of samples must be non-negative")
    div = n - 1 if endpoint else n
    lo = float(start)
    delta = float(stop) - lo
    if div > 0:
        step = delta / div
        y = np.arange(0, n) * step + lo
    else:
        y = np.arange(0, n) * delta + lo
    if endpoint and n > 1:
        y[n - 1] = stop
    return y


@source(np.sort)
def sort(a):
    """Bottom-up merge sort of a 1-d array (stable; NaNs sort last)."""
    x = np.ravel(a).copy()
    n = len(x)
    tmp = np.empty_like(x)
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i = lo
            j = mid
            for k in range(lo, hi):
                if i < mid and (j >= hi or x[i] <= x[j] or x[j] != x[j]):
                    tmp[k] = x[i]
                    i += 1
                else:
                    tmp[k] = x[j]
                    j += 1
        swap = x
        x = tmp
        tmp = swap
        width *= 2
    return x


@source(np.argsort)
def argsort(a):
    """Indices that sort a 1-d array (stable merge sort; NaNs sort last)."""
    v = np.ravel(a)
    n = len(v)
    x = np.arange(n)
    tmp = np.empty_like(x)
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i = lo
            j = mid
            for k in range(lo, hi):
                if i < mid and (j >= hi or v[x[i]] <= v[x[j]] or v[x[j]] != v[x[j]]):
                    tmp[k] = x[i]
                    i += 1
                else:
                    tmp[k] = x[j]
                    j += 1
        swap = x
        x = tmp
        tmp = swap
        width *= 2
    return x


@source(np.median)
def median(a):
    s = np.sort(a)
    n = len(s)
    if n == 0:
        return np.nan
    if n % 2 == 1:
        return s[n // 2] * 1.0
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


@source(np.searchsorted)
def searchsorted(a, v):
    """Index of the first element of sorted 1-d ``a`` that is >= scalar ``v``."""
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return np.int64(lo)


@source(np.cross)
def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@source(np.convolve)
def convolve(a, v):
    """Full discrete convolution of two 1-d arrays."""
    n = len(a)
    m = len(v)

    def at(k):
        lo = max(0, k - m + 1)
        hi = min(k + 1, n)
        total = a[0] * v[0] * 0
        for i in range(lo, hi):
            total += a[i] * v[k - i]
        return total

    return imap(at, n + m - 1)


@source(np.histogram)
def histogram(a, bins=10, range=None):  # noqa: A002
    """Counts of a 1-d array in equal-width bins; returns ``(counts, edges)``."""
    x = np.ravel(a)
    if range is None:
        lo = float(np.min(x)) if len(x) > 0 else 0.0
        hi = float(np.max(x)) if len(x) > 0 else 1.0
    else:
        lo = float(range[0])
        hi = float(range[1])
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    counts = np.zeros(bins, dtype=np.int64)
    scale = bins / (hi - lo)
    for i in builtins_range(len(x)):
        xi = x[i]
        if xi >= lo and xi <= hi:
            k = int((xi - lo) * scale)
            if k == bins:
                k = bins - 1
            counts[k] += 1
    edges = np.linspace(lo, hi, bins + 1)
    return counts, edges


builtins_range = __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range


def pmap1(f, x, w=3):
    """Apply ``f`` to a sliding window of width ``w`` around every element of a
    1-d array (windows are truncated at the borders)."""
    n = x.shape[0]
    half = w // 2

    def local_apply(i):
        lower = max(i - half, 0)
        upper = min(i + half + 1, n)
        return f(x[lower:upper])

    return imap(local_apply, n)


def pmap2(f, x, width=(3, 3)):
    """Apply ``f`` to the ``width``-sized neighbourhood of every element of a
    2-d array (neighbourhoods are truncated at the borders)."""
    width_x, width_y = width
    n_rows, n_cols = x.shape
    hx = width_x // 2
    hy = width_y // 2

    def local_apply(idx):
        i, j = idx
        lx = max(i - hx, 0)
        ux = min(i + hx + 1, n_rows)
        ly = max(j - hy, 0)
        uy = min(j + hy + 1, n_cols)
        return f(x[lx:ux, ly:uy])

    return imap(local_apply, x.shape)


def pmap2_trim(f, x, width=(3, 3), step=(1, 1)):
    """Like :func:`pmap2` but only over interior windows (no border)."""
    width_x, width_y = width
    step_x, step_y = step
    n_rows, n_cols = x.shape
    hx = width_x // 2
    hy = width_y // 2
    return [
        [f(x[i - hx : i + hx + 1, j - hy : j + hy + 1]) for j in builtins_range(hy, n_cols - hy, step_y)]
        for i in builtins_range(hx, n_rows - hx, step_x)
    ]
