"""Generate crisp, 680px architectural & algorithmic SVG diagrams for docs/.

Usage:
  python tools/gen_doc_diagrams.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

SANS = "'Source Sans 3', Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'IBM Plex Mono', 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# Cohesive palette matching docs/stylesheets/extra.css and docs/assets/bench-*.svg
INK = "#0f172a"
SUB = "#475569"
MUTED = "#64748b"
BORDER = "#cbd5e1"
LINE = "#94a3b8"
BG_SOFT = "#f8fafc"
BG_LIGHT = "#f1f5f9"

GREEN_DARK = "#245744"
GREEN = "#0d9488"
GREEN_BG = "#edf4ef"
GREEN_BD = "#86c5a8"

BLUE_DARK = "#176083"
BLUE = "#2563eb"
BLUE_BG = "#eff6ff"
BLUE_BD = "#93c5fd"

AMBER_DARK = "#92400e"
AMBER_BG = "#fef3c7"
AMBER_BD = "#f59e0b"

RED_DARK = "#991b1b"
RED = "#dc2626"
RED_BG = "#fef2f2"
RED_BD = "#fca5a5"


def _svg_open(width: int, height: int, aria_label: str, extra_defs: str = "") -> str:
  return (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
    f'width="{width}" height="{height}" role="img" aria-label="{escape(aria_label)}">\n'
    f"<defs>\n"
    f'  <marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
    f'    <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="{MUTED}"/>\n'
    f"  </marker>\n"
    f'  <marker id="arr-green" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
    f'    <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="{GREEN_DARK}"/>\n'
    f"  </marker>\n"
    f'  <marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
    f'    <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="{BLUE_DARK}"/>\n'
    f"  </marker>\n"
    f'  <marker id="arr-red" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">\n'
    f'    <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="{RED}"/>\n'
    f"  </marker>\n"
    f'  <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">\n'
    f'    <line x1="0" y1="0" x2="0" y2="7" stroke="#cbd5e1" stroke-width="1.5"/>\n'
    f"  </pattern>\n"
    f"{extra_defs}"
    f"</defs>\n"
    f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" fill="#ffffff" stroke="{BORDER}"/>\n'
  )


def _title(title: str, subtitle: str) -> str:
  return (
    f'<text x="20" y="28" font-family="{SANS}" font-size="15.5" font-weight="700" fill="{INK}">{escape(title)}</text>\n'
    f'<text x="20" y="46" font-family="{SANS}" font-size="12" fill="{SUB}">{escape(subtitle)}</text>\n'
  )


def _box(
  x: float,
  y: float,
  w: float,
  h: float,
  *,
  fill: str = BG_SOFT,
  stroke: str = BORDER,
  rx: float = 7,
  stroke_width: float = 1.2,
  dash: str = "",
) -> str:
  d = f' stroke-dasharray="{dash}"' if dash else ""
  return (
    f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
    f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{d}/>\n'
  )


def _text(
  x: float,
  y: float,
  text: str,
  *,
  size: float = 12,
  weight: str = "500",
  fill: str = INK,
  anchor: str = "start",
  mono: bool = False,
) -> str:
  fam = MONO if mono else SANS
  return (
    f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{fam}" '
    f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>\n'
  )


def _pill(
  x: float,
  y: float,
  w: float,
  h: float,
  label: str,
  *,
  fill: str = GREEN_BG,
  stroke: str = GREEN_BD,
  ink: str = GREEN_DARK,
  mono: bool = False,
  size: float = 10.5,
) -> str:
  return _box(x, y, w, h, fill=fill, stroke=stroke, rx=h / 2, stroke_width=1.0) + _text(
    x + w / 2, y + h / 2 + size * 0.34, label, size=size, weight="600", fill=ink, anchor="middle", mono=mono
  )


def _arrow(
  x1: float, y1: float, x2: float, y2: float, *, color: str = MUTED, marker: str = "arr", width: float = 1.6
) -> str:
  return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>\n'


# ---------------------------------------------------------------------------
# 1. Compiler pipeline (docs/internals.md)
# ---------------------------------------------------------------------------


def gen_pipeline() -> str:
  w, h = 680, 316
  out = [_svg_open(w, h, "Parakeet compiler pipeline from Python to CPU, Metal, and CUDA")]
  out.append(
    _title(
      "Parakeet Compilation Pipeline",
      "Staged elaboration, algebraic operator optimization, loop lowering, and multi-target codegen.",
    )
  )

  # Top row: 4 main stages
  stages = [
    (
      20,
      66,
      134,
      88,
      BG_SOFT,
      BORDER,
      INK,
      "1. Python Source",
      ["@par.jit function", "NumPy, loops, par.*", "+ argument type key"],
    ),
    (
      176,
      66,
      146,
      88,
      BLUE_BG,
      BLUE_BD,
      BLUE_DARK,
      "2. Elaboration",
      ["Staged interpreter", "Delayed arrays", "AD tape (grad / jvp)"],
    ),
    (
      344,
      66,
      156,
      88,
      GREEN_BG,
      GREEN_BD,
      GREEN_DARK,
      "3. Operator Algebra",
      ["7 index primitives", "12 rewrite passes", "Fusion & Laws 1–6"],
    ),
    (
      522,
      66,
      138,
      88,
      BG_SOFT,
      BORDER,
      INK,
      "4. Lowered Loop IR",
      ["ParLoop & ForRange", "Flat buffer offsets", "Verified static SSA"],
    ),
  ]
  for x, y, bw, bh, fill, stroke, hdr, title, lines in stages:
    out.append(_box(x, y, bw, bh, fill=fill, stroke=stroke))
    out.append(_text(x + bw / 2, y + 22, title, size=12, weight="700", fill=hdr, anchor="middle"))
    out.append(
      f'<line x1="{x + 10:.1f}" y1="{y + 30:.1f}" x2="{x + bw - 10:.1f}" y2="{y + 30:.1f}" stroke="{stroke}" stroke-opacity="0.7" stroke-width="1"/>\n'
    )
    for idx, line in enumerate(lines):
      out.append(_text(x + bw / 2, y + 48 + idx * 15, line, size=10.8, weight="400", fill=SUB, anchor="middle"))

  out.append(_arrow(154, 110, 174, 110))
  out.append(_arrow(322, 110, 342, 110))
  out.append(_arrow(500, 110, 520, 110))

  # Inspection hooks strip
  out.append(_text(20, 178, "Inspect:", size=11, weight="600", fill=MUTED))
  out.append(
    _pill(205, 164, 108, 22, "spec.operators", fill=BLUE_BG, stroke=BLUE_BD, ink=BLUE_DARK, mono=True, size=10)
  )
  out.append(
    _pill(348, 164, 148, 22, "par.show_rewrite()", fill=GREEN_BG, stroke=GREEN_BD, ink=GREEN_DARK, mono=True, size=10)
  )
  out.append(_pill(550, 164, 82, 22, "spec.ir", fill=BG_LIGHT, stroke=BORDER, ink=INK, mono=True, size=10))

  # Dashed connectors down from stages to inspect pills
  for px in (259, 422, 591):
    out.append(
      f'<line x1="{px}" y1="154" x2="{px}" y2="164" stroke="{LINE}" stroke-width="1" stroke-dasharray="2,2"/>\n'
    )

  # Branching arrows from stage 4 down to 3 codegen backends
  out.append(
    f'<path d="M 591 186 L 591 208 L 122 208 L 122 222" fill="none" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#arr)"/>\n'
  )
  out.append(
    f'<path d="M 340 208 L 340 222" fill="none" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#arr)"/>\n'
  )
  out.append(
    f'<path d="M 558 208 L 558 222" fill="none" stroke="{MUTED}" stroke-width="1.5" marker-end="url(#arr)"/>\n'
  )
  out.append(_text(20, 200, "Codegen & runtime:", size=11, weight="600", fill=MUTED))

  # Bottom row: 3 targets
  backends = [
    (
      20,
      224,
      204,
      74,
      BG_SOFT,
      BORDER,
      INK,
      "Host C (CPU & Orchestration)",
      "Thread pool · OpenMP · GCD · BLAS",
      "spec.source  →  cached .so / .dylib",
    ),
    (
      238,
      224,
      204,
      74,
      GREEN_BG,
      GREEN_BD,
      GREEN_DARK,
      "NVIDIA CUDA",
      "NVRTC · Shared SRAM · TF32 WMMA",
      "spec.kernels  →  device-resident",
    ),
    (
      456,
      224,
      204,
      74,
      BLUE_BG,
      BLUE_BD,
      BLUE_DARK,
      "Apple Metal",
      "MSL · Threadgroup · simdgroup_matrix",
      "spec.kernels  →  unified memory",
    ),
  ]
  for x, y, bw, bh, fill, stroke, hdr, title, sub1, sub2 in backends:
    out.append(_box(x, y, bw, bh, fill=fill, stroke=stroke))
    out.append(_text(x + bw / 2, y + 22, title, size=11.8, weight="700", fill=hdr, anchor="middle"))
    out.append(_text(x + bw / 2, y + 41, sub1, size=10.5, weight="400", fill=SUB, anchor="middle"))
    out.append(_text(x + bw / 2, y + 59, sub2, size=9.8, weight="500", fill=MUTED, anchor="middle", mono=True))

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 1b. Indexification: array-driven Python -> coordinate-driven Index* IR
# ---------------------------------------------------------------------------


def gen_indexification() -> str:
  w, h = 680, 316
  out = [
    _svg_open(
      w,
      h,
      "Indexification lowers diverse array-driven Python and NumPy operations onto uniform coordinate-driven Index* primitives",
    )
  ]
  out.append(
    _title(
      "Why the 'Index' Prefix? Array-Driven Python → Coordinate-Driven IR",
      "Views, broadcasts, transposes, and par.* operators disappear into integer index expressions.",
    )
  )

  # Column headers
  out.append(_text(20, 72, "SURFACE PYTHON / NUMPY / PAR.*", size=10.2, weight="700", fill=SUB))
  out.append(_text(324, 72, "INDEX-SPACE PRIMITIVE", size=10.2, weight="700", fill=GREEN_DARK))

  # Top band: Map family -> IndexMap
  y1 = 82
  out.append(_box(20, y1, 260, 108, fill=BG_SOFT, stroke=BORDER))
  map_inputs = [
    ("x[:, None] - y[None, :]", "broadcast"),
    ("x.T * 2.0", "transpose + ufunc"),
    ("par.allpairs(f, x, y)", "pairwise map"),
    ("[f(xi) for xi in x]", "comprehension"),
  ]
  for idx, (code, tag) in enumerate(map_inputs):
    ry = y1 + 10 + idx * 23
    out.append(_box(28, ry, 244, 19, fill="#ffffff", stroke=BORDER, rx=4, stroke_width=1.0))
    out.append(_text(35, ry + 13, code, size=9.6, weight="600", fill=INK, mono=True))
    out.append(_text(265, ry + 13, tag, size=9.2, fill=MUTED, anchor="end"))

  out.append(_arrow(284, y1 + 54, 316, y1 + 54, color=GREEN_DARK, marker="arr-green", width=2.0))

  out.append(_box(324, y1, 336, 108, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(_text(338, y1 + 21, "IndexMap", size=12.5, weight="700", fill=GREEN_DARK, mono=True))
  out.append(
    _pill(422, y1 + 7, 134, 20, "coordinate box (M, N)", fill="#ffffff", stroke=GREEN_BD, ink=GREEN_DARK, size=9.5)
  )
  out.append(_box(338, y1 + 34, 308, 42, fill="#ffffff", stroke=GREEN_BD, rx=5, stroke_width=1.0))
  out.append(_text(348, y1 + 51, "for (i, j) in (0..M-1, 0..N-1):", size=9.8, weight="600", fill=INK, mono=True))
  out.append(_text(348, y1 + 67, "  out[i, j] = body(x[i], y[j])", size=9.8, fill=SUB, mono=True))
  out.append(
    _text(
      338,
      y1 + 95,
      "No input arrays on the node — broadcasts & views are just index math",
      size=9.8,
      weight="500",
      fill=GREEN_DARK,
    )
  )

  # Bottom band: Reduce family -> IndexReduce
  y2 = 202
  out.append(_box(20, y2, 260, 96, fill=BG_SOFT, stroke=BORDER))
  red_inputs = [
    ("np.sum(x * x, axis=1)", "reduction"),
    ("np.argmin(d2, axis=1)", "value + index"),
    ("par.fold(lift, merge, x)", "ordered carry"),
    ("par.ireduce(f, (K,))", "index reduction"),
  ]
  for idx, (code, tag) in enumerate(red_inputs):
    ry = y2 + 7 + idx * 21
    out.append(_box(28, ry, 244, 18, fill="#ffffff", stroke=BORDER, rx=4, stroke_width=1.0))
    out.append(_text(35, ry + 12.5, code, size=9.4, weight="600", fill=INK, mono=True))
    out.append(_text(265, ry + 12.5, tag, size=9.0, fill=MUTED, anchor="end"))

  out.append(_arrow(284, y2 + 48, 316, y2 + 48, color=BLUE_DARK, marker="arr-blue", width=2.0))

  out.append(_box(324, y2, 336, 96, fill=BLUE_BG, stroke=BLUE_BD))
  out.append(_text(338, y2 + 21, "IndexReduce", size=12.5, weight="700", fill=BLUE_DARK, mono=True))
  out.append(
    _pill(444, y2 + 7, 134, 20, "reduction axis (0..K-1)", fill="#ffffff", stroke=BLUE_BD, ink=BLUE_DARK, size=9.5)
  )
  out.append(_box(338, y2 + 32, 308, 38, fill="#ffffff", stroke=BLUE_BD, rx=5, stroke_width=1.0))
  out.append(_text(348, y2 + 48, "fold (⊕, carries) over k in 0..K-1:", size=9.8, weight="600", fill=INK, mono=True))
  out.append(_text(348, y2 + 63, "  acc = combine(acc, body(k))", size=9.8, fill=SUB, mono=True))
  out.append(
    _text(
      338,
      y2 + 86,
      "One fusion & tiling pass now handles sum, argmin, @, einsum & fold",
      size=9.8,
      weight="500",
      fill=BLUE_DARK,
    )
  )

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 2. Algebraic fusion patterns (docs/primitives.md, docs/rewrites.md)
# ---------------------------------------------------------------------------


def gen_fusion() -> str:
  w, h = 680, 294
  out = [_svg_open(w, h, "Vertical, filter-reduce, and sibling fusion eliminate intermediate arrays")]
  out.append(
    _title(
      "Algebraic Fusion: Collapsing Memory Passes",
      "Producer-consumer chains, boolean filters, and sibling passes merge before loop lowering.",
    )
  )

  # Column headers
  out.append(_text(20, 72, "NAIVE NUMPY (ALLOCATES INTERMEDIATES)", size=10.2, weight="700", fill=RED_DARK))
  out.append(_text(404, 72, "PARAKEET FUSED (SINGLE PASS)", size=10.2, weight="700", fill=GREEN_DARK))

  # Row 1: Vertical fusion
  r1 = 82
  out.append(_box(20, r1, 344, 58, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(30, r1 + 16, "Vertical:", size=10.5, weight="700", fill=SUB))
  out.append(_text(82, r1 + 16, "np.tanh(x * a) + b", size=10.5, weight="600", fill=INK, mono=True))
  out.append(_pill(30, r1 + 26, 42, 22, "x", fill="#ffffff", stroke=BORDER, ink=INK, mono=True))
  out.append(_arrow(72, r1 + 37, 88, r1 + 37, width=1.3))
  out.append(_pill(90, r1 + 26, 74, 22, "tmp1 = x*a", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=9.5))
  out.append(_arrow(164, r1 + 37, 180, r1 + 37, width=1.3))
  out.append(_pill(182, r1 + 26, 86, 22, "tmp2 = tanh", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=9.5))
  out.append(_arrow(268, r1 + 37, 284, r1 + 37, width=1.3))
  out.append(_pill(286, r1 + 26, 68, 22, "y = + b", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=9.5))

  out.append(_arrow(368, r1 + 29, 398, r1 + 29, color=GREEN_DARK, marker="arr-green", width=2.0))

  out.append(_box(404, r1, 256, 58, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(
    _pill(
      416,
      r1 + 10,
      174,
      22,
      "y[i] = tanh(x[i]*a) + b",
      fill="#ffffff",
      stroke=GREEN_BD,
      ink=GREEN_DARK,
      mono=True,
      size=9.6,
    )
  )
  out.append(_pill(596, r1 + 10, 54, 22, "1 pass", fill=GREEN_DARK, stroke=GREEN_DARK, ink="#ffffff", size=9.8))
  out.append(_text(416, r1 + 48, "0 temporaries · 3 memory passes → 1", size=10.5, weight="500", fill=GREEN_DARK))

  # Row 2: Filter + Reduce fusion
  r2 = 152
  out.append(_box(20, r2, 344, 58, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(30, r2 + 16, "Filter-Reduce:", size=10.5, weight="700", fill=SUB))
  out.append(_text(112, r2 + 16, "np.sum(x[x > 0])", size=10.5, weight="600", fill=INK, mono=True))
  out.append(_pill(30, r2 + 26, 42, 22, "x", fill="#ffffff", stroke=BORDER, ink=INK, mono=True))
  out.append(_arrow(72, r2 + 37, 90, r2 + 37, width=1.3))
  out.append(
    _pill(92, r2 + 26, 134, 22, "kept = compact(x>0)", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=9.5)
  )
  out.append(_arrow(226, r2 + 37, 244, r2 + 37, width=1.3))
  out.append(_pill(246, r2 + 26, 108, 22, "sum(kept)", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=9.5))

  out.append(_arrow(368, r2 + 29, 398, r2 + 29, color=GREEN_DARK, marker="arr-green", width=2.0))

  out.append(_box(404, r2, 256, 58, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(
    _pill(
      416, r2 + 10, 174, 22, "sum(x[i] if x[i]>0)", fill="#ffffff", stroke=GREEN_BD, ink=GREEN_DARK, mono=True, size=9.6
    )
  )
  out.append(_pill(596, r2 + 10, 54, 22, "0 alloc", fill=GREEN_DARK, stroke=GREEN_DARK, ink="#ffffff", size=9.8))
  out.append(_text(416, r2 + 48, "No index array · no gather bounds check", size=10.5, weight="500", fill=GREEN_DARK))

  # Row 3: Sibling fusion
  r3 = 222
  out.append(_box(20, r3, 344, 58, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(30, r3 + 16, "Sibling:", size=10.5, weight="700", fill=SUB))
  out.append(_text(80, r3 + 16, "x.min(), x.max()", size=10.5, weight="600", fill=INK, mono=True))
  out.append(
    _pill(30, r3 + 26, 148, 22, "Pass 1: read x → lo", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=9.5)
  )
  out.append(
    _pill(190, r3 + 26, 162, 22, "Pass 2: read x → hi", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=9.5)
  )

  out.append(_arrow(368, r3 + 29, 398, r3 + 29, color=GREEN_DARK, marker="arr-green", width=2.0))

  out.append(_box(404, r3, 256, 58, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(
    _pill(
      416, r3 + 10, 174, 22, "lo, hi = fold2(x)", fill="#ffffff", stroke=GREEN_BD, ink=GREEN_DARK, mono=True, size=9.6
    )
  )
  out.append(_pill(596, r3 + 10, 54, 22, "1 read", fill=GREEN_DARK, stroke=GREEN_DARK, ink="#ffffff", size=9.8))
  out.append(
    _text(416, r3 + 48, "Also fuses sibling maps & same-key scatters", size=10.5, weight="500", fill=GREEN_DARK)
  )

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 3. Causal attention: Naive O(T^2) vs Monoidal Tiling + Law 5 Mask Pruning
# ---------------------------------------------------------------------------


def gen_attention_tiling() -> str:
  w, h = 680, 338
  out = [
    _svg_open(w, h, "Causal self-attention compiled from naive O(T^2) NumPy to O(T) tiled monoid with block skipping")
  ]
  out.append(
    _title(
      "Causal Attention: O(T²) Buffers → Single Tiled Kernel",
      "Law 1 (rescaled monoid) eliminates 3 quadratic buffers; Law 5 (tile intervals) skips upper-triangle blocks.",
    )
  )

  # Left panel: Naive NumPy
  out.append(_box(20, 62, 232, 258, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(34, 83, "Naive NumPy Execution", size=12.5, weight="700", fill=RED_DARK))
  out.append(_text(34, 99, "4 passes · 3 × O(T²) buffers in RAM", size=10.8, fill=SUB))

  out.append(
    _pill(36, 112, 200, 24, "Inputs: Q, K, V  (T × D)", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=10)
  )
  out.append(_arrow(136, 136, 136, 150, color=RED, marker="arr-red", width=1.4))
  out.append(
    _pill(36, 152, 200, 26, "1. scores  (T × T)  alloc", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=10)
  )
  out.append(_arrow(136, 178, 136, 192, color=RED, marker="arr-red", width=1.4))
  out.append(
    _pill(36, 194, 200, 26, "2. exp_s   (T × T)  alloc", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=10)
  )
  out.append(_arrow(136, 220, 136, 234, color=RED, marker="arr-red", width=1.4))
  out.append(
    _pill(36, 236, 200, 26, "3. weights (T × T)  alloc", fill=RED_BG, stroke=RED_BD, ink=RED_DARK, mono=True, size=10)
  )
  out.append(_arrow(136, 262, 136, 276, color=MUTED, marker="arr", width=1.4))
  out.append(
    _pill(36, 278, 200, 26, "4. Output O  (T × D)", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=10)
  )

  # Transition arrow
  out.append(_arrow(258, 190, 286, 190, color=GREEN_DARK, marker="arr-green", width=2.2))

  # Right panel: Parakeet Tiled + Law 5
  out.append(_box(294, 62, 366, 258, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(_text(310, 83, "Parakeet Tiled Kernel (Law 1 + Law 5)", size=12.5, weight="700", fill=GREEN_DARK))
  out.append(_text(310, 99, "1 pass · 0 quadratic buffers · only O(T · D) output", size=10.8, fill=GREEN_DARK))

  # 4x4 tile grid showing causal block structure
  gx, gy, cell, gap = 312, 124, 34, 4
  out.append(_text(gx + 74, gy - 8, "Key / Value tiles  j₀  →", size=10, weight="600", fill=SUB, anchor="middle"))
  out.append(
    f'<text x="{gx - 8}" y="{gy + 74}" text-anchor="middle" font-family="{SANS}" '
    f'font-size="10" font-weight="600" fill="{SUB}" transform="rotate(-90 {gx - 8} {gy + 74})">Query tiles  i₀  →</text>\n'
  )

  for r in range(4):
    for c in range(4):
      cx = gx + c * (cell + gap)
      cy = gy + r * (cell + gap)
      if c > r:
        # Upper triangle: skipped by Law 5
        out.append(_box(cx, cy, cell, cell, fill="url(#hatch)", stroke="#cbd5e1", rx=4, stroke_width=1))
        out.append(_text(cx + cell / 2, cy + cell / 2 + 3.5, "skip", size=9, weight="600", fill=MUTED, anchor="middle"))
      elif c == r:
        # Diagonal: masked tile
        out.append(_box(cx, cy, cell, cell, fill=AMBER_BG, stroke=AMBER_BD, rx=4, stroke_width=1.2))
        out.append(
          _text(cx + cell / 2, cy + cell / 2 + 3.5, "mask", size=9, weight="600", fill=AMBER_DARK, anchor="middle")
        )
      else:
        # Interior: branchless tile
        out.append(_box(cx, cy, cell, cell, fill="#d1fae5", stroke=GREEN, rx=4, stroke_width=1.2))
        out.append(
          _text(cx + cell / 2, cy + cell / 2 + 3.5, "full", size=9.2, weight="700", fill=GREEN_DARK, anchor="middle")
        )

  # On-chip SRAM state box to the right of the grid
  sx, sy = 478, 124
  out.append(_box(sx, sy, 168, 148, fill="#ffffff", stroke=GREEN_BD, rx=7))
  out.append(_text(sx + 84, sy + 20, "On-Chip SRAM / Cache", size=11, weight="700", fill=GREEN_DARK, anchor="middle"))
  out.append(_text(sx + 12, sy + 40, "Stage Q_tile  (BM × D)", size=10, fill=SUB, mono=True))
  out.append(_text(sx + 12, sy + 56, "Stream K, V   (BN × D)", size=10, fill=SUB, mono=True))
  out.append(
    f'<line x1="{sx + 10}" y1="{sy + 68}" x2="{sx + 158}" y2="{sy + 68}" stroke="{BORDER}" stroke-width="1"/>\n'
  )
  out.append(_text(sx + 12, sy + 85, "Running Monoid State:", size=10.2, weight="700", fill=INK))
  out.append(_text(sx + 12, sy + 103, "m : row max", size=10, fill=SUB, mono=True))
  out.append(_text(sx + 12, sy + 118, "l : exp normalizer", size=10, fill=SUB, mono=True))
  out.append(_text(sx + 12, sy + 133, "o : weighted output", size=10, fill=SUB, mono=True))

  # Legend at bottom of right panel
  ly = 294
  out.append(_box(310, ly, 12, 12, fill="#d1fae5", stroke=GREEN, rx=2.5))
  out.append(_text(327, ly + 10, "Branchless (j₀+BN ≤ i₀)", size=9.8, fill=INK))
  out.append(_box(448, ly, 12, 12, fill=AMBER_BG, stroke=AMBER_BD, rx=2.5))
  out.append(_text(465, ly + 10, "Masked diag", size=9.8, fill=INK))
  out.append(_box(544, ly, 12, 12, fill="url(#hatch)", stroke="#cbd5e1", rx=2.5))
  out.append(_text(561, ly + 10, "Skipped (not loaded)", size=9.8, fill=MUTED))

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 4. Exact 1-NN pairwise fold tiling (docs/primitives.md, docs/algebra.md)
# ---------------------------------------------------------------------------


def gen_pairwise_1nn() -> str:
  w, h = 680, 276
  out = [_svg_open(w, h, "Exact 1-NN tiled pairwise fold avoids O(Q*R) distance matrix while keeping tie order")]
  out.append(
    _title(
      "Exact 1-NN: Tiled Ordered Pairwise Fold (Law 6)",
      "Streams 16×128 score tiles in L1/SRAM left-to-right across references, preserving NumPy tie order.",
    )
  )

  # Left: Conceptual Q x R matrix never allocated
  out.append(_box(20, 64, 272, 194, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(34, 84, "Logical Q × R Distance Grid (never allocated)", size=11.5, weight="700", fill=INK))
  out.append(_text(34, 100, "At Q=1,024, R=16,384: 64 MiB eliminated", size=10.5, fill=RED_DARK))

  # Draw tiled strips across R
  gx, gy = 36, 114
  out.append(_box(gx, gy, 240, 108, fill="#ffffff", stroke=BORDER, dash="4,3", rx=5))
  # Active query strip (row band)
  out.append(_box(gx + 2, gy + 24, 236, 30, fill=GREEN_BG, stroke=GREEN_BD, rx=3))
  # Active tile inside strip
  out.append(_box(gx + 60, gy + 26, 56, 26, fill="#d1fae5", stroke=GREEN, rx=3, stroke_width=1.5))
  out.append(_text(gx + 88, gy + 43, "16×128", size=9.5, weight="700", fill=GREEN_DARK, anchor="middle", mono=True))
  # Completed tile to the left
  out.append(_box(gx + 4, gy + 26, 52, 26, fill=BG_LIGHT, stroke=BORDER, rx=3))
  out.append(_text(gx + 30, gy + 43, "done", size=9, fill=MUTED, anchor="middle"))
  # Arrow moving right across reference blocks
  out.append(_arrow(gx + 120, gy + 39, gx + 174, gy + 39, color=GREEN_DARK, marker="arr-green", width=1.6))
  out.append(_text(gx + 206, gy + 42, "r₀ += 128", size=9.2, weight="600", fill=GREEN_DARK, anchor="middle", mono=True))

  out.append(
    _text(gx + 120, gy + 74, "One query strip (16 rows) per thread / block", size=10, fill=SUB, anchor="middle")
  )
  out.append(
    _text(gx + 120, gy + 92, "Left-to-right reference order keeps first-index ties", size=10, fill=SUB, anchor="middle")
  )

  out.append(_text(34, 244, "Scratch: 8 KiB L1 (CPU) · 16 KiB SRAM (GPU)", size=10.3, weight="600", fill=GREEN_DARK))

  # Arrow to right panel
  out.append(_arrow(298, 160, 326, 160, color=GREEN_DARK, marker="arr-green", width=2.0))

  # Right: Ordered Carry Merge
  out.append(_box(334, 64, 326, 194, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(_text(350, 84, "Ordered Carry Reduction", size=12, weight="700", fill=GREEN_DARK))
  out.append(_text(350, 100, "Each tile produces local winner (d_tile, r₀ + arg);", size=10.5, fill=SUB))
  out.append(_text(350, 115, "merged strictly left-to-right in global index order.", size=10.5, fill=SUB))

  # 3 tile summary pills merging into final winner
  tiles = [("Tile 0", "(d₀, idx₀)"), ("Tile 1", "(d₁, idx₁)"), ("Tile 2", "(d₂, idx₂)")]
  for i, (lbl, val) in enumerate(tiles):
    tx = 350 + i * 102
    out.append(_box(tx, 128, 92, 46, fill="#ffffff", stroke=GREEN_BD, rx=6))
    out.append(_text(tx + 46, 146, lbl, size=10, weight="700", fill=GREEN_DARK, anchor="middle"))
    out.append(_text(tx + 46, 163, val, size=9.8, fill=SUB, anchor="middle", mono=True))
    out.append(_arrow(tx + 46, 174, 497, 200, color=GREEN_DARK, marker="arr-green", width=1.3))

  out.append(_box(363, 202, 268, 42, fill="#ffffff", stroke=GREEN, rx=6, stroke_width=1.5))
  out.append(
    _text(
      497, 220, "merge(a, b):  a if a.d <= b.d else b", size=10.2, weight="700", fill=INK, anchor="middle", mono=True
    )
  )
  out.append(_text(497, 236, "Exact NumPy argmin tie & NaN semantics", size=10, fill=GREEN_DARK, anchor="middle"))

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 5. Law 4: Chunkwise state-space duality (docs/algebra.md)
# ---------------------------------------------------------------------------


def gen_chunkwise_scan() -> str:
  w, h = 680, 290
  out = [
    _svg_open(w, h, "Law 4 chunkwise state-space duality turns an O(T) serial recurrence into parallel 4-step chunks")
  ]
  out.append(
    _title(
      "Law 4: Chunkwise State-Space Duality",
      "Rewrites a serial O(T) recurrence into 64× shorter inter-chunk scan + parallel matmuls (C = 64).",
    )
  )

  # Top strip: Serial recurrence
  out.append(_box(20, 62, 640, 62, fill=RED_BG, stroke=RED_BD))
  out.append(_text(34, 80, "Before: Serial for-loop  (O(T) depth · 1 core)", size=11.5, weight="700", fill=RED_DARK))
  out.append(
    _text(644, 80, "S = a[t]·S + k[t] ⊗ v[t]", size=10.2, weight="600", fill=RED_DARK, mono=True, anchor="end")
  )
  serial_labels = ["t = 0", "t = 1", "t = 2", "t = 3", "…", "t = T−1"]
  for i, lbl in enumerate(serial_labels):
    bx = 34 + i * 106
    out.append(_pill(bx, 90, 76, 24, lbl, fill="#ffffff", stroke=RED_BD, ink=RED_DARK, mono=True, size=9.8))
    if i < len(serial_labels) - 1:
      out.append(_arrow(bx + 76, 102, bx + 104, 102, color=RED, marker="arr-red", width=1.4))

  # Down arrow
  out.append(_arrow(340, 124, 340, 144, color=GREEN_DARK, marker="arr-green", width=2.0))

  # Bottom panel: 4 parallel chunkwise steps
  out.append(_box(20, 148, 640, 126, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(
    _text(
      34,
      168,
      "After: Parallel 4-Step Chunkwise Scan  (C = 64 · division-free · Tensor Core ready)",
      size=11.5,
      weight="700",
      fill=GREEN_DARK,
    )
  )

  steps = [
    (
      34,
      180,
      142,
      80,
      "#ffffff",
      GREEN_BD,
      GREEN_DARK,
      "Step 1 · Parallel",
      "Intra-chunk matmul",
      "Y_intra = (Q@Kᵀ·L)@V",
      "All T/C chunks at once",
    ),
    (
      190,
      180,
      142,
      80,
      BLUE_BG,
      BLUE_BD,
      BLUE_DARK,
      "Step 2 · Parallel",
      "Chunk summaries",
      "B[c] = K_s[c]ᵀ @ V[c]",
      "One matmul per chunk",
    ),
    (
      346,
      180,
      142,
      80,
      AMBER_BG,
      AMBER_BD,
      AMBER_DARK,
      "Step 3 · 64× Shorter",
      "Inter-chunk scan",
      "S[c+1] = A·S[c] + B[c]",
      "Tiny T/64 recurrence",
    ),
    (
      502,
      180,
      144,
      80,
      "#ffffff",
      GREEN_BD,
      GREEN_DARK,
      "Step 4 · Parallel",
      "Off-diag readout",
      "Y = Y_intra + Q@S[c]",
      "Flows into WMMA/SIMD",
    ),
  ]
  for x, y, bw, bh, fill, stroke, hdr, k, t, eq, note in steps:
    out.append(_box(x, y, bw, bh, fill=fill, stroke=stroke, rx=6))
    out.append(_text(x + bw / 2, y + 17, k, size=9.5, weight="700", fill=hdr, anchor="middle"))
    out.append(_text(x + bw / 2, y + 33, t, size=10.8, weight="700", fill=INK, anchor="middle"))
    out.append(_text(x + bw / 2, y + 51, eq, size=9.2, fill=SUB, anchor="middle", mono=True))
    out.append(_text(x + bw / 2, y + 69, note, size=9.5, fill=MUTED, anchor="middle"))

  out.append(_arrow(176, 220, 188, 220, color=GREEN_DARK, marker="arr-green", width=1.4))
  out.append(_arrow(332, 220, 344, 220, color=GREEN_DARK, marker="arr-green", width=1.4))
  out.append(_arrow(488, 220, 500, 220, color=GREEN_DARK, marker="arr-green", width=1.4))

  out.append("</svg>\n")
  return "".join(out)


# ---------------------------------------------------------------------------
# 6. Automatic differentiation: Forward vs Reverse & Law 2 Rematerialization
# ---------------------------------------------------------------------------


def gen_autodiff_modes() -> str:
  w, h = 680, 236
  out = [_svg_open(w, h, "Forward-mode JVP vs reverse-mode VJP data flow")]
  out.append(
    _title(
      "Forward Mode (JVP) vs. Reverse Mode (VJP)",
      "Choose forward mode when outputs m >> inputs n, and reverse mode when inputs n >> outputs m.",
    )
  )

  # Forward mode row
  out.append(_box(20, 62, 640, 70, fill=BG_SOFT, stroke=BORDER))
  out.append(_text(36, 84, "Forward Mode  (par.jvp / par.jacfwd)", size=12, weight="700", fill=BLUE_DARK))
  out.append(_text(36, 100, "Cost scales with inputs n · O(1) memory, no backward tape", size=10.5, fill=SUB))
  out.append(_pill(340, 82, 82, 28, "(x, v)", fill=BLUE_BG, stroke=BLUE_BD, ink=BLUE_DARK, mono=True, size=10.5))
  out.append(_arrow(422, 96, 450, 96, color=BLUE_DARK, marker="arr-blue", width=1.6))
  out.append(_pill(452, 82, 80, 28, "f  +  df", fill="#ffffff", stroke=BLUE_BD, ink=INK, mono=True, size=10.5))
  out.append(_arrow(532, 96, 560, 96, color=BLUE_DARK, marker="arr-blue", width=1.6))
  out.append(_pill(562, 82, 84, 28, "(y, J·v)", fill=BLUE_BG, stroke=BLUE_BD, ink=BLUE_DARK, mono=True, size=10.5))

  # Reverse mode row
  out.append(_box(20, 144, 640, 78, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(_text(36, 166, "Reverse Mode  (par.grad / par.vjp / par.jacrev)", size=12, weight="700", fill=GREEN_DARK))
  out.append(_text(36, 182, "Cost scales with outputs m · ideal for scalar training losses", size=10.5, fill=SUB))
  out.append(_pill(340, 154, 82, 24, "x", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=10))
  out.append(_arrow(422, 166, 450, 166, color=MUTED, marker="arr", width=1.4))
  out.append(_pill(452, 154, 80, 24, "forward f", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=9.8))
  out.append(_arrow(532, 166, 560, 166, color=MUTED, marker="arr", width=1.4))
  out.append(_pill(562, 154, 84, 24, "y (+ stats)", fill="#ffffff", stroke=BORDER, ink=INK, mono=True, size=9.5))

  out.append(_pill(340, 188, 82, 24, "uᵀ · J", fill="#ffffff", stroke=GREEN, ink=GREEN_DARK, mono=True, size=10))
  out.append(_arrow(450, 200, 424, 200, color=GREEN_DARK, marker="arr-green", width=1.5))
  out.append(_pill(452, 188, 80, 24, "pullback", fill="#ffffff", stroke=GREEN, ink=GREEN_DARK, mono=True, size=9.8))
  out.append(_arrow(560, 200, 534, 200, color=GREEN_DARK, marker="arr-green", width=1.5))
  out.append(_pill(562, 188, 84, 24, "seed u", fill="#ffffff", stroke=GREEN, ink=GREEN_DARK, mono=True, size=9.8))

  out.append("</svg>\n")
  return "".join(out)


def gen_rematerialization() -> str:
  w, h = 680, 204
  out = [
    _svg_open(w, h, "Law 2 algebraic rematerialization replaces O(T^2) backward tapes with on-chip tile recomputation")
  ]
  out.append(
    _title(
      "Law 2: Algebraic Rematerialization in Reverse-Mode AD",
      "Substitutes sufficient-statistic adjoints before lowering so backward never allocates O(T²) or O(M·V) tapes.",
    )
  )

  # Left box: Naive tape
  out.append(_box(20, 64, 294, 122, fill=RED_BG, stroke=RED_BD))
  out.append(_text(36, 86, "Naive Reverse-Mode Tape", size=12, weight="700", fill=RED_DARK))
  out.append(_text(36, 108, "Saves full score & probability matrices:", size=10.5, fill=SUB))
  out.append(
    _pill(
      36,
      118,
      260,
      26,
      "S (T×T), P (T×T)  or  logits (M×V)",
      fill="#ffffff",
      stroke=RED_BD,
      ink=RED_DARK,
      mono=True,
      size=9.8,
    )
  )
  out.append(_text(36, 168, "O(T²) / O(M·V) DRAM traffic & memory blow-up", size=10.2, weight="600", fill=RED_DARK))

  out.append(_arrow(318, 125, 358, 125, color=GREEN_DARK, marker="arr-green", width=2.2))

  # Right box: Algebraic VJP
  out.append(_box(364, 64, 296, 122, fill=GREEN_BG, stroke=GREEN_BD))
  out.append(_text(380, 86, "Parakeet Algebraic VJP / HVP", size=12, weight="700", fill=GREEN_DARK))
  out.append(_text(380, 108, "Saves only compact O(T·D) sufficient stats:", size=10.5, fill=SUB))
  out.append(
    _pill(
      380,
      118,
      264,
      26,
      "(Q, K, V, O, lse)  or  (H, W, y, lse)",
      fill="#ffffff",
      stroke=GREEN,
      ink=INK,
      mono=True,
      size=9.8,
    )
  )
  out.append(
    _text(380, 168, "Recomputes P = exp(S − lse) tile-by-tile on-chip", size=10.2, weight="600", fill=GREEN_DARK)
  )

  out.append("</svg>\n")
  return "".join(out)


def main() -> None:
  ASSETS.mkdir(parents=True, exist_ok=True)
  diagrams = {
    "diag-pipeline.svg": gen_pipeline(),
    "diag-indexification.svg": gen_indexification(),
    "diag-fusion.svg": gen_fusion(),
    "diag-attention-tiling.svg": gen_attention_tiling(),
    "diag-pairwise-1nn.svg": gen_pairwise_1nn(),
    "diag-chunkwise-scan.svg": gen_chunkwise_scan(),
    "diag-autodiff-modes.svg": gen_autodiff_modes(),
    "diag-rematerialization.svg": gen_rematerialization(),
  }
  for name, svg in diagrams.items():
    path = ASSETS / name
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path} ({len(svg)} bytes)")


if __name__ == "__main__":
  main()
