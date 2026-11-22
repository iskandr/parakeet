"""Time the benchmark suite: NumPy versus Parakeet on each available backend.

    python benchmarks/run.py                 # all benchmarks, CPU backends
    python benchmarks/run.py --gpu           # also Metal / CUDA (float32 data)
    python benchmarks/run.py -k julia --scale 0.25

Times are best-of-N wall-clock milliseconds, excluding compilation.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import parakeet  # noqa: E402
from benchmarks import suite  # noqa: E402
from parakeet import jit  # noqa: E402


def copy_args(args):
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)


def best_time(fn, args, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        a = copy_args(args)
        start = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-k", "--filter", default="", help="only run benchmarks whose name contains this")
    p.add_argument("--scale", type=float, default=1.0, help="problem size multiplier")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--gpu", action="store_true", help="include GPU backends and use float32 data")
    p.add_argument("--backends", default="", help="comma separated list (default: c, multicore[, gpu])")
    opts = p.parse_args(argv)

    available = parakeet.available_backends()
    if opts.backends:
        backends = opts.backends.split(",")
    else:
        backends = ["c", "multicore"]
        if opts.gpu:
            backends += [b for b in ("metal", "cuda") if available.get(b) == "ok"]
    dtype = np.float32 if opts.gpu else np.float64
    print(
        f"# {platform.platform()} / {platform.processor() or platform.machine()} / "
        f"Python {platform.python_version()} / NumPy {np.__version__} / Parakeet {parakeet.__version__}"
    )
    print(
        f"# {parakeet.config.num_threads} threads, dtype {np.dtype(dtype).name}, scale {opts.scale}, best of {opts.repeat}"
    )
    header = f"{'benchmark':24s} {'numpy':>10s} " + " ".join(f"{b:>12s}" for b in backends)
    print(header)
    print("-" * len(header))
    rows = []
    for name, (fn, numpy_fn, args) in suite.make(dtype, opts.scale).items():
        if opts.filter not in name:
            continue
        row = {"name": name}
        if numpy_fn == "python":
            base = None
        else:
            base = best_time(numpy_fn or fn, args, max(2, opts.repeat // 2))
        row["numpy"] = base
        cells = [f"{base:8.2f}ms" if base is not None else "       n/a"]
        expected = None
        for backend in backends:
            compiled = jit(fn, backend=backend)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", parakeet.FallbackWarning)
                result = compiled(*copy_args(args))  # compile + warm up
                if expected is None:
                    expected = result
                elif not np.allclose(result, expected, rtol=1e-3, atol=1e-3):
                    raise AssertionError(f"{name}: backend {backend} disagrees")
                t = best_time(compiled, args, opts.repeat)
            row[backend] = t
            speed = f"{base / t:5.1f}x" if base else "     "
            cells.append(
                f"{t:7.2f}ms{speed}"[:14].rjust(12) if not base else f"{t:6.2f}ms {speed.strip():>5s}".rjust(12)
            )
        rows.append(row)
        print(f"{name:24s} {cells[0]} " + " ".join(f"{c:>12s}" for c in cells[1:]))
    return rows


if __name__ == "__main__":
    main()
