"""Regenerate docs/supported.md from the library registry.

python tools/gen_supported.py
"""

from __future__ import annotations

import builtins
import math
from pathlib import Path

import numpy as np

from parakeet import lib, scalars


def names_in(module, prefix: str) -> list[str]:
    out = set()
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        try:
            known = lib.lookup(obj) is not None
        except TypeError:
            known = False
        if known:
            out.add(prefix + name)
    return sorted(out)


def table(names: list[str], cols: int = 4) -> str:
    rows = [names[i : i + cols] for i in range(0, len(names), cols)]
    lines = ["| " + " | ".join([""] * cols) + " |", "|" + "---|" * cols]
    for r in rows:
        r = r + [""] * (cols - len(r))
        lines.append("| " + " | ".join(f"`{n}`" if n else "" for n in r) + " |")
    return "\n".join(lines)


def main() -> None:
    ufuncs = sorted({f"np.{u.__name__}" for u in scalars.BY_NUMPY})
    np_funcs = [n for n in names_in(np, "np.") if n not in ufuncs]
    sections = [
        "# Supported functions",
        "",
        "This page is generated from Parakeet's library registry by `tools/gen_supported.py`, so it always",
        "matches the installed version.  Anything not listed here raises an `UnsupportedError` that names the",
        "function (and suggests similarly named supported ones).",
        "",
        "## NumPy element-wise functions (ufuncs)",
        "",
        "All follow NumPy's broadcasting and type promotion rules exactly.",
        "",
        table(ufuncs),
        "",
        "## NumPy functions",
        "",
        table(np_funcs),
        "",
        "## `np.linalg` and `np.random`",
        "",
        table(names_in(np.linalg, "np.linalg.") + names_in(np.random, "np.random.")),
        "",
        "## Array attributes and methods",
        "",
        table(sorted(f".{n}" for n in lib.PROPERTIES) + sorted(f".{n}()" for n in lib.METHODS)),
        "",
        "## Python builtins",
        "",
        table(names_in(builtins, "") + ["bool", "float", "int"]),
        "",
        "## `math`",
        "",
        table(names_in(math, "math.") + ["math.e", "math.inf", "math.nan", "math.pi", "math.tau"]),
        "",
        "## Parakeet operators",
        "",
        table(
            sorted(
                f"parakeet.{n}"
                for n in (
                    "map each allpairs outer_map reduce scan imap ireduce parfor filter pmap1 pmap2 pmap2_trim".split()
                )
            )
        ),
        "",
    ]
    out = Path(__file__).resolve().parent.parent / "docs" / "supported.md"
    out.write_text("\n".join(sections))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
