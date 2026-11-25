"""Generate reproducible SVG benchmark figures with per-replicate scatter, IQR, and median.

Reads raw timing samples from ``docs/assets/colab-l4-0dcb7d5.json`` (no external
plotting libraries required) and writes publication-ready SVGs into ``docs/assets/``.

Usage::

    python tools/gen_benchmark_figures.py
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "docs" / "assets" / "colab-l4-0dcb7d5.json"
DEFAULT_OUT = ROOT / "docs" / "assets"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# High-contrast, colorblind-safe palette on a clean light card (legible in both light & dark themes).
BG = "#ffffff"
CARD_BORDER = "#d8dee9"
GRID = "#e5e9f0"
GRID_SUB = "#f1f5f9"
TEXT = "#0f172a"
MUTED = "#475569"
SUBTLE = "#64748b"

PK_MAIN = "#0d9488"  # Parakeet primary (teal)
PK_ALT = "#0f766e"  # Parakeet secondary (deep teal)
PK_LIGHT = "#99f6e4"
BASE_BLUE = "#3b82f6"  # Specialist / compiled baselines
BASE_SLATE = "#64748b"  # Eager / general baselines
BASE_WARN = "#d97706"  # 1.0.0 baseline / reference


def _quantile(vals: list[float], q: float) -> float:
  s = sorted(vals)
  if not s:
    return 0.0
  if len(s) == 1:
    return s[0]
  pos = q * (len(s) - 1)
  lo = int(math.floor(pos))
  hi = int(math.ceil(pos))
  frac = pos - lo
  return s[lo] * (1.0 - frac) + s[hi] * frac


def _stats(samples: list[float]) -> tuple[float, float, float, float, float]:
  s = sorted(samples)
  return (
    s[0],
    _quantile(s, 0.25),
    median(s),
    _quantile(s, 0.75),
    s[-1],
  )


def _jitter(key: str, idx: int, amplitude: float = 4.2) -> float:
  h = hashlib.blake2b(f"{key}:{idx}".encode(), digest_size=4).digest()
  u = int.from_bytes(h, "little") / 0xFFFFFFFF
  return (u * 2.0 - 1.0) * amplitude


def _fmt_ms(v: float) -> str:
  if v >= 100.0:
    return f"{v:.0f}"
  if v >= 10.0:
    return f"{v:.1f}"
  if v >= 1.0:
    return f"{v:.2f}"
  return f"{v:.3f}"


def _esc(s: str) -> str:
  return html.escape(s, quote=True)


def _find_rows(data: dict) -> dict[tuple[str, str, str, str], dict]:
  """Index provider rows by (provider_group, task, shape_tag, method)."""
  idx: dict[tuple[str, str, str, str], dict] = {}
  for prov_key, pval in data["providers"].items():
    device = "cpu" if prov_key == "cpu" else "gpu"
    for row in pval["rows"]:
      shapes = row["shapes"]
      if row["task"] in ("nn", "rbf"):
        size = "small" if shapes[0][0] == 256 else "large"
      else:
        size = "default"
      idx[(device, row["task"], size, row["method"])] = row
  return idx


def _log_ticks(vmin: float, vmax: float) -> list[float]:
  lo_exp = int(math.floor(math.log10(vmin)))
  hi_exp = int(math.ceil(math.log10(vmax)))
  ticks: list[float] = []
  for e in range(lo_exp, hi_exp + 1):
    for mult in (1, 2, 5):
      val = mult * (10.0**e)
      if vmin * 0.999 <= val <= vmax * 1.001:
        ticks.append(val)
  return ticks


def _render_dot_panel(
  out: list[str],
  *,
  x0: float,
  y0: float,
  width: float,
  height: float,
  title: str,
  subtitle: str,
  arms: list[tuple[str, list[float], str, bool]],
  vmin: float,
  vmax: float,
  x_label: str = "Latency (ms, log scale, lower is better)",
) -> None:
  """Render one horizontal log-scale scatter + IQR + median panel."""
  label_w = 148.0
  val_w = 98.0
  plot_x0 = x0 + label_w
  plot_w = width - label_w - val_w
  plot_y0 = y0 + 38.0
  plot_h = height - 66.0

  log_min = math.log10(vmin)
  log_max = math.log10(vmax)

  def sx(v: float) -> float:
    clamped = max(vmin, min(vmax, v))
    return plot_x0 + (math.log10(clamped) - log_min) / (log_max - log_min) * plot_w

  # Panel title & subtitle
  out.append(
    f'<text x="{x0:.1f}" y="{y0 + 14:.1f}" font-family="{FONT}" font-size="13" '
    f'font-weight="700" fill="{TEXT}">{_esc(title)}</text>'
  )
  out.append(
    f'<text x="{x0:.1f}" y="{y0 + 29:.1f}" font-family="{FONT}" font-size="10.5" fill="{MUTED}">{_esc(subtitle)}</text>'
  )

  # Plot frame background
  out.append(
    f'<rect x="{plot_x0:.1f}" y="{plot_y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
    f'rx="4" fill="#fafbfc" stroke="{GRID}" stroke-width="1"/>'
  )

  # Gridlines & x-axis ticks
  for t in _log_ticks(vmin, vmax):
    tx = sx(t)
    is_decade = abs(math.log10(t) - round(math.log10(t))) < 1e-6
    stroke = GRID if is_decade else GRID_SUB
    out.append(
      f'<line x1="{tx:.1f}" y1="{plot_y0:.1f}" x2="{tx:.1f}" y2="{plot_y0 + plot_h:.1f}" '
      f'stroke="{stroke}" stroke-width="1"/>'
    )
    t_str = f"{t:g}"
    out.append(
      f'<text x="{tx:.1f}" y="{plot_y0 + plot_h + 13:.1f}" text-anchor="middle" '
      f'font-family="{MONO}" font-size="9.5" fill="{SUBTLE}">{t_str}</text>'
    )

  out.append(
    f'<text x="{plot_x0 + plot_w / 2:.1f}" y="{plot_y0 + plot_h + 26:.1f}" text-anchor="middle" '
    f'font-family="{FONT}" font-size="9.5" fill="{SUBTLE}">{_esc(x_label)}</text>'
  )

  n_arms = len(arms)
  row_h = plot_h / max(1, n_arms)

  for i, (name, samples, color, is_pk) in enumerate(arms):
    cy = plot_y0 + (i + 0.5) * row_h
    lo, q1, med, q3, hi = _stats(samples)
    iqr = q3 - q1

    if is_pk:
      out.append(
        f'<rect x="{plot_x0 + 1:.1f}" y="{cy - row_h * 0.44:.1f}" '
        f'width="{plot_w - 2:.1f}" height="{row_h * 0.88:.1f}" '
        f'rx="2" fill="#f0fdfa"/>'
      )

    # Row separator
    if i > 0:
      sy_sep = plot_y0 + i * row_h
      out.append(
        f'<line x1="{plot_x0:.1f}" y1="{sy_sep:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{sy_sep:.1f}" '
        f'stroke="{GRID_SUB}" stroke-width="1"/>'
      )

    # Label
    weight = "700" if is_pk else "500"
    label_fill = PK_ALT if is_pk else TEXT
    out.append(
      f'<text x="{plot_x0 - 8:.1f}" y="{cy + 3.5:.1f}" text-anchor="end" '
      f'font-family="{FONT}" font-size="11" font-weight="{weight}" fill="{label_fill}">{_esc(name)}</text>'
    )

    # Min-max whisker
    x_lo, x_q1, x_med, x_q3, x_hi = sx(lo), sx(q1), sx(med), sx(q3), sx(hi)
    out.append(
      f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" '
      f'stroke="{color}" stroke-opacity="0.45" stroke-width="1.5"/>'
    )

    # Individual replicate dots (deterministic vertical jitter)
    amp = min(4.5, row_h * 0.24)
    for j, sv in enumerate(samples):
      dx = sx(sv)
      dy = cy + _jitter(f"{title}:{name}", j, amplitude=amp)
      out.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="2.0" fill="{color}" fill-opacity="0.32"/>')

    # IQR box (25th-75th percentile)
    box_w = max(2.5, x_q3 - x_q1)
    box_x = x_q1 if (x_q3 - x_q1) >= 2.5 else x_med - 1.25
    out.append(
      f'<rect x="{box_x:.1f}" y="{cy - 4.5:.1f}" width="{box_w:.1f}" height="9.0" '
      f'rx="2" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="1.3"/>'
    )

    # Median marker (crisp vertical tick + solid dot)
    out.append(f'<circle cx="{x_med:.1f}" cy="{cy:.1f}" r="3.6" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')

    # Right-side median (IQR) annotation
    med_txt = f"{_fmt_ms(med)} ms"
    iqr_txt = f"±{_fmt_ms(iqr / 2)}" if iqr >= 0.002 else ""
    out.append(
      f'<text x="{plot_x0 + plot_w + 6:.1f}" y="{cy + 3.5:.1f}" '
      f'font-family="{MONO}" font-size="9.8" font-weight="{weight}" fill="{label_fill}">'
      f"{_esc(med_txt)}"
      f'<tspan dx="3" font-weight="400" font-size="8.6" fill="{SUBTLE}">{iqr_txt}</tspan></text>'
    )


def make_pairwise_svg(data: dict) -> str:
  rows = _find_rows(data)
  w, h = 980, 560
  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    'aria-label="Pairwise exact 1-NN and Gaussian kernel regression latencies on NVIDIA L4 GPU and 4-thread CPU">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="24" y="30" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">'
    "Pairwise Search &amp; Kernel Regression (1,024 × 4,096 × 32, float32)</text>",
    f'<text x="24" y="48" font-family="{FONT}" font-size="11.5" fill="{MUTED}">'
    "Warmed latency on NVIDIA L4 (n=21) and 4-thread CPU (n=11) — dots: all replicates, box: IQR (25th–75th), marker: median. Lower is faster.</text>",
  ]

  # Legend top-right
  lx, ly = 656, 22
  out.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="{PK_MAIN}"/>')
  out.append(
    f'<text x="{lx + 9}" y="{ly + 4}" font-family="{FONT}" font-size="11" font-weight="600" fill="{PK_ALT}">Parakeet</text>'
  )
  out.append(f'<circle cx="{lx + 78}" cy="{ly}" r="4" fill="{BASE_BLUE}"/>')
  out.append(
    f'<text x="{lx + 87}" y="{ly + 4}" font-family="{FONT}" font-size="11" fill="{TEXT}">Compiled / Specialist</text>'
  )
  out.append(f'<circle cx="{lx + 216}" cy="{ly}" r="4" fill="{BASE_SLATE}"/>')
  out.append(f'<text x="{lx + 225}" y="{ly + 4}" font-family="{FONT}" font-size="11" fill="{TEXT}">NumPy / CuPy</text>')

  gpu_nn = [
    ("Faiss GpuFlatL2", rows[("gpu", "nn", "large", "faiss-flat")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (NumPy)", rows[("gpu", "nn", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("Parakeet (nested map)", rows[("gpu", "nn", "large", "pk-dsl")]["samples_ms"], PK_ALT, True),
    ("PyTorch cdist", rows[("gpu", "nn", "large", "torch-cdist")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile (cdist)", rows[("gpu", "nn", "large", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "nn", "large", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile (bcast)", rows[("gpu", "nn", "large", "torch-compile-broadcast")]["samples_ms"], BASE_BLUE, False),
    ("PyKeOps Genred", rows[("gpu", "nn", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("CuPy broadcast", rows[("gpu", "nn", "large", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  cpu_nn = [
    ("scikit-learn brute", rows[("cpu", "nn", "large", "sklearn-brute")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (nested map)", rows[("cpu", "nn", "large", "pk-dsl")]["samples_ms"], PK_ALT, True),
    ("Parakeet (NumPy)", rows[("cpu", "nn", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("Faiss IndexFlatL2", rows[("cpu", "nn", "large", "faiss-flat")]["samples_ms"], BASE_BLUE, False),
    ("PyKeOps Genred", rows[("cpu", "nn", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("SciPy cKDTree", rows[("cpu", "nn", "large", "scipy-tree")]["samples_ms"], BASE_BLUE, False),
    ("NumPy broadcast", rows[("cpu", "nn", "large", "numpy")]["samples_ms"], BASE_SLATE, False),
  ]

  gpu_rbf = [
    ("torch.compile (cdist)", rows[("gpu", "rbf", "large", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("PyTorch cdist", rows[("gpu", "rbf", "large", "torch-cdist")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile (bcast)", rows[("gpu", "rbf", "large", "torch-compile-broadcast")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "rbf", "large", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (NumPy)", rows[("gpu", "rbf", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("PyKeOps Genred", rows[("gpu", "rbf", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("CuPy broadcast", rows[("gpu", "rbf", "large", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  cpu_rbf = [
    ("Parakeet (NumPy)", rows[("cpu", "rbf", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("PyKeOps Genred", rows[("cpu", "rbf", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("NumPy broadcast", rows[("cpu", "rbf", "large", "numpy")]["samples_ms"], BASE_SLATE, False),
  ]

  _render_dot_panel(
    out,
    x0=24,
    y0=66,
    width=456,
    height=252,
    title="A. GPU Exact 1-NN (NVIDIA L4)",
    subtitle="GPU-resident inputs, 0 host transfers, exact tie/NaN semantics",
    arms=gpu_nn,
    vmin=0.1,
    vmax=25.0,
  )
  _render_dot_panel(
    out,
    x0=500,
    y0=66,
    width=456,
    height=252,
    title="B. CPU Exact 1-NN (4 threads)",
    subtitle="Tiled pairwise fold in 8 KiB L1 scratch vs libraries",
    arms=cpu_nn,
    vmin=4.0,
    vmax=700.0,
  )
  _render_dot_panel(
    out,
    x0=24,
    y0=332,
    width=456,
    height=205,
    title="C. GPU Gaussian Kernel Regression (NVIDIA L4)",
    subtitle="4 normalized weighted sums/query, single tiled launch",
    arms=gpu_rbf,
    vmin=0.2,
    vmax=25.0,
  )
  _render_dot_panel(
    out,
    x0=500,
    y0=332,
    width=456,
    height=205,
    title="D. CPU Gaussian Kernel Regression (4 threads)",
    subtitle="Single-pass Law 1 monoidal reduction (11.8× faster than NumPy)",
    arms=cpu_rbf,
    vmin=20.0,
    vmax=700.0,
  )

  out.append(
    f'<text x="24" y="{h - 10}" font-family="{FONT}" font-size="9.5" fill="{SUBTLE}">'
    "Source: docs/assets/colab-l4-0dcb7d5.json (rev 0dcb7d5) · ± half-IQR shown when ≥ 0.001 ms · Generated by tools/gen_benchmark_figures.py</text>"
  )
  out.append("</svg>")
  return "\n".join(out) + "\n"


def make_scaling_svg(data: dict) -> str:
  w, h = 980, 355
  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    'aria-label="GPU exact 1-NN latency and scratch memory scaling with reference count on NVIDIA L4">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="24" y="30" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">'
    "Exact 1-NN Scaling on NVIDIA L4: Full Score Matrix (1.0.0) vs. Tiled Pairwise Fold</text>",
    f'<text x="24" y="48" font-family="{FONT}" font-size="11.5" fill="{MUTED}">'
    "1,024 queries × R references × 32 float32 features, GPU-resident, n=21 warmed replicates per point.</text>",
  ]

  base_rows = {
    r["references"]: r for r in data["baseline_scaling"]["rows"] if r["source"] == "numpy" and "samples_ms" in r
  }
  cur_np = {
    r["references"]: r
    for r in data["pairwise_scaling"]["rows"]
    if r["source"] == "numpy" and r["enabled"] and "samples_ms" in r
  }
  cur_dsl = {
    r["references"]: r
    for r in data["pairwise_scaling"]["rows"]
    if r["source"] == "dsl" and r["enabled"] and "samples_ms" in r
  }
  refs = [1024, 4096, 16384]

  # --- Left panel: Latency vs R ---
  lx0, ly0, lw, lh = 68.0, 92.0, 440.0, 206.0
  out.append(
    f'<text x="{lx0:.1f}" y="76" font-family="{FONT}" font-size="13" font-weight="700" fill="{TEXT}">'
    "A. Median Latency &amp; Replicate Spread</text>"
  )
  out.append(
    f'<rect x="{lx0:.1f}" y="{ly0:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="4" fill="#fafbfc" stroke="{GRID}"/>'
  )

  # X positions evenly spaced on log2(R)
  def sx(r: int) -> float:
    idx = math.log2(r / 1024.0) / 4.0  # 0, 0.5, 1.0
    return lx0 + 44.0 + idx * (lw - 88.0)

  ymax = 2.95

  def sy(v: float) -> float:
    return ly0 + lh - (v / ymax) * (lh - 14.0) - 4.0

  for ytick in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    yy = sy(ytick)
    out.append(f'<line x1="{lx0:.1f}" y1="{yy:.1f}" x2="{lx0 + lw:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{lx0 - 8:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" font-family="{MONO}" '
      f'font-size="10" fill="{SUBTLE}">{ytick:.1f}</text>'
    )

  for r in refs:
    xx = sx(r)
    out.append(f'<line x1="{xx:.1f}" y1="{ly0:.1f}" x2="{xx:.1f}" y2="{ly0 + lh:.1f}" stroke="{GRID_SUB}"/>')
    out.append(
      f'<text x="{xx:.1f}" y="{ly0 + lh + 16:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="10.5" fill="{TEXT}">{r:,}</text>'
    )
  out.append(
    f'<text x="{lx0 + lw / 2:.1f}" y="{ly0 + lh + 32:.1f}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="10.5" fill="{SUBTLE}">Number of reference rows R (log scale)</text>'
  )
  out.append(
    f'<text x="20" y="{ly0 + lh / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {ly0 + lh / 2:.1f})" '
    f'font-family="{FONT}" font-size="10.5" fill="{SUBTLE}">Latency (ms)</text>'
  )

  series = [
    ("Parakeet 1.0.0 (full Q×R matrix)", base_rows, BASE_WARN, "0", -4.0),
    ("Parakeet tiled fold (NumPy)", cur_np, PK_MAIN, "0", 0.0),
    ("Parakeet tiled fold (explicit map)", cur_dsl, PK_ALT, "4,3", 4.0),
  ]

  for label, rdict, color, dash, xoff in series:
    pts = []
    for r in refs:
      s = rdict[r]["samples_ms"]
      _, q1, med, q3, _ = _stats(s)
      xx = sx(r) + xoff
      pts.append((xx, sy(med), med, r))
      # IQR bar
      out.append(
        f'<line x1="{xx:.1f}" y1="{sy(q1):.1f}" x2="{xx:.1f}" y2="{sy(q3):.1f}" '
        f'stroke="{color}" stroke-width="4" stroke-opacity="0.35" stroke-linecap="round"/>'
      )
      for j, sv in enumerate(s):
        jx = xx + _jitter(f"{label}:{r}", j, amplitude=2.8)
        out.append(f'<circle cx="{jx:.1f}" cy="{sy(sv):.1f}" r="1.8" fill="{color}" fill-opacity="0.35"/>')
    d_attr = " ".join(f"{'M' if k == 0 else 'L'}{p[0]:.1f},{p[1]:.1f}" for k, p in enumerate(pts))
    dash_str = f' stroke-dasharray="{dash}"' if dash != "0" else ""
    out.append(f'<path d="{d_attr}" fill="none" stroke="{color}" stroke-width="2.3"{dash_str}/>')
    for xx, yy, _med, _ in pts:
      out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.0" fill="{color}" stroke="#ffffff" stroke-width="1.3"/>')

  # Callout at R = 16,384
  x16 = sx(16384)
  y_old = sy(base_rows[16384]["median_ms"])
  y_new = sy(cur_np[16384]["median_ms"])
  out.append(
    f'<text x="{x16 - 10:.1f}" y="{y_old - 8:.1f}" text-anchor="end" font-family="{MONO}" font-size="10.5" '
    f'font-weight="700" fill="{BASE_WARN}">2.60 ms (1.0.0)</text>'
  )
  out.append(
    f'<text x="{x16 - 10:.1f}" y="{y_new - 10:.1f}" text-anchor="end" font-family="{MONO}" font-size="10.5" '
    f'font-weight="700" fill="{PK_ALT}">0.54 ms (4.8× faster)</text>'
  )

  # Compact legend inside left panel top-left
  out.append(
    f'<rect x="{lx0 + 10:.1f}" y="{ly0 + 10:.1f}" width="224" height="58" rx="5" fill="#ffffff" fill-opacity="0.92" stroke="{GRID}"/>'
  )
  for k, (label, _, color, dash, _) in enumerate(series):
    yy = ly0 + 24 + k * 17
    dash_str = f' stroke-dasharray="{dash}"' if dash != "0" else ""
    out.append(
      f'<line x1="{lx0 + 18:.1f}" y1="{yy:.1f}" x2="{lx0 + 36:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="2.4"{dash_str}/>'
    )
    out.append(
      f'<text x="{lx0 + 42:.1f}" y="{yy + 3.5:.1f}" font-family="{FONT}" font-size="10.5" fill="{TEXT}">{_esc(label)}</text>'
    )

  # --- Right panel: Global scratch memory (MiB) ---
  rx0, ry0, rw, rh = 576.0, 92.0, 376.0, 206.0
  out.append(
    f'<text x="{rx0:.1f}" y="76" font-family="{FONT}" font-size="13" font-weight="700" fill="{TEXT}">'
    "B. Global Device Scratch Memory (MiB)</text>"
  )
  out.append(
    f'<rect x="{rx0:.1f}" y="{ry0:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="4" fill="#fafbfc" stroke="{GRID}"/>'
  )

  mem_old = [4.0, 16.0, 64.0]
  mem_new = [0.1875, 0.75, 3.0]
  mmax = 72.0

  def smy(m: float) -> float:
    return ry0 + rh - (m / mmax) * (rh - 16.0) - 2.0

  for mt in (0, 16, 32, 48, 64):
    yy = smy(mt)
    out.append(f'<line x1="{rx0:.1f}" y1="{yy:.1f}" x2="{rx0 + rw:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{rx0 - 8:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" font-family="{MONO}" '
      f'font-size="10" fill="{SUBTLE}">{mt}</text>'
    )

  bar_w = 34.0
  for i, r in enumerate(refs):
    cx = rx0 + (i + 0.5) * (rw / 3.0)
    # 1.0.0 bar
    y1 = smy(mem_old[i])
    out.append(
      f'<rect x="{cx - bar_w - 3:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{smy(0) - y1:.1f}" '
      f'rx="3" fill="{BASE_WARN}" fill-opacity="0.82"/>'
    )
    out.append(
      f'<text x="{cx - bar_w / 2 - 3:.1f}" y="{y1 - 5:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="9.5" font-weight="600" fill="{BASE_WARN}">{mem_old[i]:.0f}M</text>'
    )
    # Tiled bar
    y2 = smy(mem_new[i])
    h2 = max(2.0, smy(0) - y2)
    out.append(
      f'<rect x="{cx + 3:.1f}" y="{smy(0) - h2:.1f}" width="{bar_w:.1f}" height="{h2:.1f}" rx="3" fill="{PK_MAIN}"/>'
    )
    lbl_new = "192K" if r == 1024 else ("768K" if r == 4096 else "3.0M")
    out.append(
      f'<text x="{cx + bar_w / 2 + 3:.1f}" y="{smy(0) - h2 - 5:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="9.5" font-weight="700" fill="{PK_ALT}">{lbl_new}</text>'
    )
    out.append(
      f'<text x="{cx:.1f}" y="{ry0 + rh + 16:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="10.5" fill="{TEXT}">R = {r:,}</text>'
    )

  out.append(
    f'<text x="{rx0 + rw / 2:.1f}" y="{ry0 + rh + 32:.1f}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="10.5" fill="{SUBTLE}">21× smaller global scratch (+ 16 KiB shared tile/block)</text>'
  )
  out.append(
    f'<rect x="{rx0 + 10:.1f}" y="{ry0 + 10:.1f}" width="186" height="42" rx="5" fill="#ffffff" fill-opacity="0.92" stroke="{GRID}"/>'
  )
  out.append(
    f'<rect x="{rx0 + 18:.1f}" y="{ry0 + 19:.1f}" width="12" height="10" rx="2" fill="{BASE_WARN}" fill-opacity="0.82"/>'
  )
  out.append(
    f'<text x="{rx0 + 36:.1f}" y="{ry0 + 28:.1f}" font-family="{FONT}" font-size="10.5" fill="{TEXT}">1.0.0 full score matrix</text>'
  )
  out.append(f'<rect x="{rx0 + 18:.1f}" y="{ry0 + 35:.1f}" width="12" height="10" rx="2" fill="{PK_MAIN}"/>')
  out.append(
    f'<text x="{rx0 + 36:.1f}" y="{ry0 + 44:.1f}" font-family="{FONT}" font-size="10.5" fill="{TEXT}">Tiled partial winners</text>'
  )

  out.append(
    f'<text x="24" y="{h - 10}" font-family="{FONT}" font-size="9.5" fill="{SUBTLE}">'
    "Source: docs/assets/colab-l4-0dcb7d5.json (rev 0dcb7d5 vs 1.0.0 baseline 7ddd90a) · Generated by tools/gen_benchmark_figures.py</text>"
  )
  out.append("</svg>")
  return "\n".join(out) + "\n"


def make_workloads_svg(data: dict) -> str:
  rows = _find_rows(data)
  w, h = 980, 520
  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    'aria-label="GPU attention, cross-entropy, GBDT split finding, and nanoGPT training step timings on NVIDIA L4">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="24" y="30" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">'
    "Forward Kernels &amp; End-to-End nanoGPT Training (NVIDIA L4)</text>",
    f'<text x="24" y="48" font-family="{FONT}" font-size="11.5" fill="{MUTED}">'
    "Top: forward kernel latency (n=21 replicates, IQR, median). Bottom: full forward + backward + Adam step (median of 20 warmed steps).</text>",
  ]

  gbdt = [
    ("torch.compile", rows[("gpu", "split", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "split", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("PyTorch scatter", rows[("gpu", "split", "default", "torch-scatter")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet", rows[("gpu", "split", "default", "pk-dsl")]["samples_ms"], PK_MAIN, True),
    ("CuPy", rows[("gpu", "split", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  attn = [
    ("PyTorch SDPA", rows[("gpu", "causal", "default", "torch-sdpa")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "causal", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile", rows[("gpu", "causal", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("CuPy", rows[("gpu", "causal", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
    ("Parakeet", rows[("gpu", "causal", "default", "pk-numpy")]["samples_ms"], PK_MAIN, True),
  ]

  cce = [
    ("PyTorch CE", rows[("gpu", "cce", "default", "torch-cross-entropy")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile", rows[("gpu", "cce", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "cce", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("CuPy", rows[("gpu", "cce", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
    ("Parakeet", rows[("gpu", "cce", "default", "pk-numpy")]["samples_ms"], PK_MAIN, True),
  ]

  # Top 3 panels (slightly narrower label_w handled nicely by 2-col or 3-col — use 2 top + 2 bottom for readability!)
  _render_dot_panel(
    out,
    x0=24,
    y0=66,
    width=456,
    height=196,
    title="A. GBDT Histogram & Best Split",
    subtitle="262,144 rows × 256 bins (2 fused histograms + prefix scans + argmax)",
    arms=gbdt,
    vmin=0.15,
    vmax=1.0,
  )
  _render_dot_panel(
    out,
    x0=500,
    y0=66,
    width=456,
    height=196,
    title="B. Causal Attention (4 heads × 512 × 64)",
    subtitle="Single tiled Law 1 + Law 5 kernel vs cuDNN SDPA / XLA",
    arms=attn,
    vmin=0.07,
    vmax=0.9,
  )
  _render_dot_panel(
    out,
    x0=24,
    y0=276,
    width=456,
    height=205,
    title="C. Linear Cross-Entropy (512 × 128 × 2,048)",
    subtitle="Zero logit matrix allocation vs cuBLAS + fused CE",
    arms=cce,
    vmin=0.05,
    vmax=2.5,
  )

  # --- Panel D: nanoGPT training step bars ---
  dx0, dy0, dw, dh = 500.0, 276.0, 456.0, 205.0
  out.append(
    f'<text x="{dx0:.1f}" y="{dy0 + 14:.1f}" font-family="{FONT}" font-size="13" '
    f'font-weight="700" fill="{TEXT}">D. nanoGPT Training Step (fwd + bwd + Adam)</text>'
  )
  out.append(
    f'<text x="{dx0:.1f}" y="{dy0 + 29:.1f}" font-family="{FONT}" font-size="10.5" '
    f'fill="{MUTED}">Median of 20 warmed steps on L4 (identical loss curves; lower is faster)</text>'
  )

  cfgs = data["training"]["configs"]
  small = cfgs["small_B16_T64_C128_L2"]
  med_cfg = cfgs["med_B32_T128_C256_L4"]

  items = [
    ("JAX jit", small["jax_jit"]["median_ms"], med_cfg["jax_jit"]["median_ms"], BASE_BLUE, False),
    ("torch.compile", small["torch_compile"]["median_ms"], med_cfg["torch_compile"]["median_ms"], BASE_BLUE, False),
    ("PyTorch eager", small["torch_eager"]["median_ms"], med_cfg["torch_eager"]["median_ms"], BASE_SLATE, False),
    (
      "Parakeet TF32",
      small["parakeet_tf32_wmma"]["median_ms"],
      med_cfg["parakeet_tf32_wmma"]["median_ms"],
      PK_MAIN,
      True,
    ),
    ("Parakeet FP32", small["parakeet_fp32"]["median_ms"], med_cfg["parakeet_fp32"]["median_ms"], PK_ALT, True),
  ]

  lbl_w = 104.0
  px0 = dx0 + lbl_w
  pw = dw - lbl_w - 76.0
  py0 = dy0 + 40.0
  ph = dh - 68.0
  vmax_t = 50.0

  out.append(
    f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{pw:.1f}" height="{ph:.1f}" rx="4" fill="#fafbfc" stroke="{GRID}"/>'
  )
  for t in (0, 10, 20, 30, 40, 50):
    tx = px0 + (t / vmax_t) * pw
    out.append(f'<line x1="{tx:.1f}" y1="{py0:.1f}" x2="{tx:.1f}" y2="{py0 + ph:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{tx:.1f}" y="{py0 + ph + 13:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="9.5" fill="{SUBTLE}">{t}</text>'
    )
  out.append(
    f'<text x="{px0 + pw / 2:.1f}" y="{py0 + ph + 26:.1f}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="9.5" fill="{SUBTLE}">Step time (ms) — light bar: 16×64×128×2 · dark bar: 32×128×256×4</text>'
  )

  row_h = ph / len(items)
  for i, (name, v_small, v_med, color, is_pk) in enumerate(items):
    y_top = py0 + i * row_h
    cy = y_top + row_h * 0.5
    if is_pk:
      out.append(
        f'<rect x="{px0 + 1:.1f}" y="{y_top + 1:.1f}" width="{pw - 2:.1f}" height="{row_h - 2:.1f}" rx="2" fill="#f0fdfa"/>'
      )
    weight = "700" if is_pk else "500"
    lfill = PK_ALT if is_pk else TEXT
    out.append(
      f'<text x="{px0 - 8:.1f}" y="{cy + 3.5:.1f}" text-anchor="end" font-family="{FONT}" '
      f'font-size="10.5" font-weight="{weight}" fill="{lfill}">{_esc(name)}</text>'
    )
    w_s = (v_small / vmax_t) * pw
    w_m = (v_med / vmax_t) * pw
    bh = max(5.5, row_h * 0.31)
    # Small config bar (upper, lighter)
    out.append(
      f'<rect x="{px0:.1f}" y="{cy - bh - 1:.1f}" width="{w_s:.1f}" height="{bh:.1f}" '
      f'rx="2" fill="{color}" fill-opacity="0.45"/>'
    )
    # Medium config bar (lower, solid)
    out.append(f'<rect x="{px0:.1f}" y="{cy + 1:.1f}" width="{w_m:.1f}" height="{bh:.1f}" rx="2" fill="{color}"/>')
    out.append(
      f'<text x="{px0 + pw + 5:.1f}" y="{cy - 1:.1f}" font-family="{MONO}" font-size="9" '
      f'fill="{SUBTLE}">{v_small:.1f} / <tspan font-weight="{weight}" fill="{lfill}">{v_med:.1f}</tspan></text>'
    )

  out.append(
    f'<text x="24" y="{h - 10}" font-family="{FONT}" font-size="9.5" fill="{SUBTLE}">'
    "Source: docs/assets/colab-l4-0dcb7d5.json (rev 0dcb7d5) · Parakeet TF32 beats PyTorch eager on 16×64×128×2 (6.8 vs 9.2 ms) · Generated by tools/gen_benchmark_figures.py</text>"
  )
  out.append("</svg>")
  return "\n".join(out) + "\n"


def generate_all(data_path: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT) -> list[Path]:
  data = json.loads(data_path.read_text())
  out_dir.mkdir(parents=True, exist_ok=True)
  outputs = [
    (out_dir / "bench-pairwise-l4.svg", make_pairwise_svg(data)),
    (out_dir / "bench-scaling-1nn.svg", make_scaling_svg(data)),
    (out_dir / "bench-workloads-l4.svg", make_workloads_svg(data)),
  ]
  written: list[Path] = []
  for path, content in outputs:
    path.write_text(content, encoding="utf-8")
    written.append(path)
  return written


if __name__ == "__main__":
  for p in generate_all():
    print(f"wrote {p}")
