"""Generate reproducible, single-panel SVG benchmark figures at native content width.

Reads raw timing samples from ``docs/assets/colab-l4-0dcb7d5.json``,
``docs/assets/colab-l4-nanogpt.json``, and ``docs/assets/bio-benchmarks-m1pro.json``
(no external plotting libraries required) and writes crisp, high-legibility
individual SVG figures (``680px`` native width so MkDocs renders them at >= 1:1
scale without shrinking fonts) into ``docs/assets/``.

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
DEFAULT_NANOGPT_DATA = ROOT / "docs" / "assets" / "colab-l4-nanogpt.json"
DEFAULT_BIO_DATA = ROOT / "docs" / "assets" / "bio-benchmarks-m1pro.json"
DEFAULT_OUT = ROOT / "docs" / "assets"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# High-contrast, colorblind-safe palette on a clean light card (legible in both light & dark themes).
BG = "#ffffff"
CARD_BORDER = "#cbd5e1"
GRID = "#e2e8f0"
GRID_SUB = "#f1f5f9"
TEXT = "#0f172a"
MUTED = "#475569"
SUBTLE = "#64748b"

PK_MAIN = "#0d9488"  # Parakeet primary (teal)
PK_ALT = "#0f766e"  # Parakeet secondary (deep teal)
PK_LEGACY = "#d97706"  # Earlier Parakeet baseline (amber)
BASE_BLUE = "#2563eb"  # Compiled / specialist baselines
BASE_PURPLE = "#7c3aed"  # Bio C/Cython baselines
BASE_SLATE = "#64748b"  # Eager / NumPy / CuPy baselines

CARD_W = 680


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
  return (s[0], _quantile(s, 0.25), median(s), _quantile(s, 0.75), s[-1])


def _jitter(key: str, idx: int, amplitude: float = 6.0) -> float:
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


def _make_single_dot_svg(
  *,
  title: str,
  subtitle: str,
  footer: str,
  arms: list[tuple[str, list[float], str, bool]],
  vmin: float,
  vmax: float,
  x_label: str = "Latency (ms, log scale, lower is better)",
  show_legacy_legend: bool = False,
) -> str:
  """Render one 680px single-panel log-scale scatter + IQR + median chart."""
  w = CARD_W
  row_h = 38.0
  header_h = 84.0
  footer_h = 60.0
  plot_h = len(arms) * row_h
  h = int(header_h + plot_h + footer_h)

  label_w = 188.0
  val_w = 128.0
  plot_x0 = 18.0 + label_w
  plot_w = w - 36.0 - label_w - val_w
  plot_y0 = header_h

  log_min = math.log10(vmin)
  log_max = math.log10(vmax)

  def sx(v: float) -> float:
    clamped = max(vmin, min(vmax, v))
    return plot_x0 + (math.log10(clamped) - log_min) / (log_max - log_min) * plot_w

  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    f'aria-label="{_esc(title)}">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="20" y="28" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">{_esc(title)}</text>',
    f'<text x="20" y="48" font-family="{FONT}" font-size="12.5" fill="{MUTED}">{_esc(subtitle)}</text>',
  ]

  # Clean legend row underneath subtitle (never collides with title)
  legend_items = [(PK_MAIN, "Parakeet")]
  if show_legacy_legend:
    legend_items.append((PK_LEGACY, "Parakeet 1.0.0"))
  legend_items.extend([(BASE_BLUE, "Compiled / Specialist"), (BASE_SLATE, "NumPy / CuPy")])
  lx = 20.0
  for color, lbl in legend_items:
    out.append(f'<circle cx="{lx + 5:.1f}" cy="66" r="4.5" fill="{color}"/>')
    out.append(
      f'<text x="{lx + 14:.1f}" y="70" font-family="{FONT}" font-size="12" '
      f'font-weight="{"600" if color == PK_MAIN else "500"}" fill="{PK_ALT if color == PK_MAIN else TEXT}">{_esc(lbl)}</text>'
    )
    lx += len(lbl) * 6.8 + 30.0

  # Plot frame
  out.append(
    f'<rect x="{plot_x0:.1f}" y="{plot_y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
    f'rx="5" fill="#fafbfc" stroke="{GRID}" stroke-width="1"/>'
  )

  # Gridlines & ticks
  for t in _log_ticks(vmin, vmax):
    tx = sx(t)
    is_decade = abs(math.log10(t) - round(math.log10(t))) < 1e-6
    stroke = GRID if is_decade else GRID_SUB
    out.append(
      f'<line x1="{tx:.1f}" y1="{plot_y0:.1f}" x2="{tx:.1f}" y2="{plot_y0 + plot_h:.1f}" '
      f'stroke="{stroke}" stroke-width="1"/>'
    )
    out.append(
      f'<text x="{tx:.1f}" y="{plot_y0 + plot_h + 18:.1f}" text-anchor="middle" '
      f'font-family="{MONO}" font-size="12" fill="{SUBTLE}">{t:g}</text>'
    )

  out.append(
    f'<text x="{plot_x0 + plot_w / 2:.1f}" y="{plot_y0 + plot_h + 36:.1f}" text-anchor="middle" '
    f'font-family="{FONT}" font-size="12" fill="{SUBTLE}">{_esc(x_label)}</text>'
  )

  for i, (name, samples, color, is_pk) in enumerate(arms):
    cy = plot_y0 + (i + 0.5) * row_h
    lo, q1, med, q3, hi = _stats(samples)
    iqr = q3 - q1

    if is_pk:
      bg_fill = "#fffbeb" if color == PK_LEGACY else "#f0fdfa"
      out.append(
        f'<rect x="{plot_x0 + 1:.1f}" y="{cy - row_h * 0.44:.1f}" '
        f'width="{plot_w - 2:.1f}" height="{row_h * 0.88:.1f}" rx="3" fill="{bg_fill}"/>'
      )

    if i > 0:
      sy_sep = plot_y0 + i * row_h
      out.append(
        f'<line x1="{plot_x0:.1f}" y1="{sy_sep:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{sy_sep:.1f}" '
        f'stroke="{GRID_SUB}" stroke-width="1"/>'
      )

    weight = "700" if is_pk else "500"
    label_fill = PK_LEGACY if color == PK_LEGACY else (PK_ALT if is_pk else TEXT)
    out.append(
      f'<text x="{plot_x0 - 10:.1f}" y="{cy + 4.8:.1f}" text-anchor="end" '
      f'font-family="{FONT}" font-size="13.5" font-weight="{weight}" fill="{label_fill}">{_esc(name)}</text>'
    )

    x_lo, x_q1, x_med, x_q3, x_hi = sx(lo), sx(q1), sx(med), sx(q3), sx(hi)
    out.append(
      f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" '
      f'stroke="{color}" stroke-opacity="0.55" stroke-width="2.0"/>'
    )

    for j, sv in enumerate(samples):
      dx = sx(sv)
      dy = cy + _jitter(f"{title}:{name}", j, amplitude=7.0)
      out.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="2.8" fill="{color}" fill-opacity="0.34"/>')

    box_w = max(4.0, x_q3 - x_q1)
    box_x = x_q1 if (x_q3 - x_q1) >= 4.0 else x_med - 2.0
    out.append(
      f'<rect x="{box_x:.1f}" y="{cy - 7.0:.1f}" width="{box_w:.1f}" height="14.0" '
      f'rx="3" fill="{color}" fill-opacity="0.30" stroke="{color}" stroke-width="1.6"/>'
    )
    out.append(f'<circle cx="{x_med:.1f}" cy="{cy:.1f}" r="5.0" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')

    med_txt = f"{_fmt_ms(med)} ms"
    iqr_txt = f"±{_fmt_ms(iqr / 2)}" if iqr >= 0.002 else ""
    out.append(
      f'<text x="{plot_x0 + plot_w + 10:.1f}" y="{cy + 4.8:.1f}" '
      f'font-family="{MONO}" font-size="12.5" font-weight="{weight}" fill="{label_fill}">'
      f"{_esc(med_txt)}"
      f'<tspan dx="3" font-weight="400" font-size="11" fill="{SUBTLE}">{iqr_txt}</tspan></text>'
    )

  out.append(f'<text x="20" y="{h - 10}" font-family="{FONT}" font-size="11" fill="{SUBTLE}">{_esc(footer)}</text>')
  out.append("</svg>")
  return "\n".join(out) + "\n"


def make_scaling_latency_svg(data: dict) -> str:
  w, h = CARD_W, 360
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

  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    'aria-label="GPU exact 1-NN latency scaling from 1,024 to 16,384 references on NVIDIA L4">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="20" y="28" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">'
    "Exact 1-NN Latency Scaling on NVIDIA L4 (1.0.0 vs. Tiled Fold)</text>",
    f'<text x="20" y="48" font-family="{FONT}" font-size="12.5" fill="{MUTED}">'
    "1,024 queries × R references × 32 float32 features, GPU-resident, n=21 replicates.</text>",
  ]

  lx0, ly0, lw, lh = 66.0, 68.0, w - 92.0, 218.0
  out.append(
    f'<rect x="{lx0:.1f}" y="{ly0:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="5" fill="#fafbfc" stroke="{GRID}"/>'
  )

  def sx(r: int) -> float:
    idx = math.log2(r / 1024.0) / 4.0
    return lx0 + 58.0 + idx * (lw - 116.0)

  ymax = 2.95

  def sy(v: float) -> float:
    return ly0 + lh - (v / ymax) * (lh - 16.0) - 4.0

  for ytick in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
    yy = sy(ytick)
    out.append(f'<line x1="{lx0:.1f}" y1="{yy:.1f}" x2="{lx0 + lw:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{lx0 - 10:.1f}" y="{yy + 4.5:.1f}" text-anchor="end" font-family="{MONO}" '
      f'font-size="12" fill="{SUBTLE}">{ytick:.1f}</text>'
    )

  for r in refs:
    xx = sx(r)
    out.append(f'<line x1="{xx:.1f}" y1="{ly0:.1f}" x2="{xx:.1f}" y2="{ly0 + lh:.1f}" stroke="{GRID_SUB}"/>')
    out.append(
      f'<text x="{xx:.1f}" y="{ly0 + lh + 20:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="12.5" fill="{TEXT}">R = {r:,}</text>'
    )

  out.append(
    f'<text x="{lx0 + lw / 2:.1f}" y="{ly0 + lh + 40:.1f}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="12.5" fill="{SUBTLE}">Number of reference rows R (log2 spacing)</text>'
  )
  out.append(
    f'<text x="20" y="{ly0 + lh / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {ly0 + lh / 2:.1f})" '
    f'font-family="{FONT}" font-size="12.5" fill="{SUBTLE}">Latency (ms)</text>'
  )

  series = [
    ("Parakeet 1.0.0 (full Q×R score matrix)", base_rows, PK_LEGACY, "0", -6.0),
    ("Parakeet tiled fold (broadcasting NumPy)", cur_np, PK_MAIN, "0", 0.0),
    ("Parakeet tiled fold (nested par.map)", cur_dsl, PK_ALT, "5,3", 6.0),
  ]

  for label, rdict, color, dash, xoff in series:
    pts = []
    for r in refs:
      s = rdict[r]["samples_ms"]
      _, q1, med, q3, _ = _stats(s)
      xx = sx(r) + xoff
      pts.append((xx, sy(med), med, r))
      out.append(
        f'<line x1="{xx:.1f}" y1="{sy(q1):.1f}" x2="{xx:.1f}" y2="{sy(q3):.1f}" '
        f'stroke="{color}" stroke-width="5.5" stroke-opacity="0.35" stroke-linecap="round"/>'
      )
      for j, sv in enumerate(s):
        jx = xx + _jitter(f"{label}:{r}", j, amplitude=3.8)
        out.append(f'<circle cx="{jx:.1f}" cy="{sy(sv):.1f}" r="2.4" fill="{color}" fill-opacity="0.35"/>')
    d_attr = " ".join(f"{'M' if k == 0 else 'L'}{p[0]:.1f},{p[1]:.1f}" for k, p in enumerate(pts))
    dash_str = f' stroke-dasharray="{dash}"' if dash != "0" else ""
    out.append(f'<path d="{d_attr}" fill="none" stroke="{color}" stroke-width="2.8"{dash_str}/>')
    for xx, yy, _med, _ in pts:
      out.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5.0" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')

  x16 = sx(16384)
  out.append(
    f'<text x="{x16 - 14:.1f}" y="{sy(base_rows[16384]["median_ms"]) - 10:.1f}" text-anchor="end" '
    f'font-family="{MONO}" font-size="12.5" font-weight="700" fill="{PK_LEGACY}">2.60 ms (1.0.0)</text>'
  )
  out.append(
    f'<text x="{x16 - 14:.1f}" y="{sy(cur_np[16384]["median_ms"]) - 12:.1f}" text-anchor="end" '
    f'font-family="{MONO}" font-size="12.5" font-weight="700" fill="{PK_ALT}">0.54 ms (4.8× faster)</text>'
  )

  # Legend box top-left
  out.append(
    f'<rect x="{lx0 + 12:.1f}" y="{ly0 + 12:.1f}" width="298" height="76" rx="6" fill="#ffffff" fill-opacity="0.95" stroke="{GRID}"/>'
  )
  for k, (label, _, color, dash, _) in enumerate(series):
    yy = ly0 + 30 + k * 22
    dash_str = f' stroke-dasharray="{dash}"' if dash != "0" else ""
    out.append(
      f'<line x1="{lx0 + 22:.1f}" y1="{yy:.1f}" x2="{lx0 + 46:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="3"{dash_str}/>'
    )
    out.append(
      f'<text x="{lx0 + 54:.1f}" y="{yy + 4.5:.1f}" font-family="{FONT}" font-size="12.5" fill="{TEXT}">{_esc(label)}</text>'
    )

  out.append(
    f'<text x="20" y="{h - 10}" font-family="{FONT}" font-size="11" fill="{SUBTLE}">'
    "Source: docs/assets/colab-l4-0dcb7d5.json (rev 0dcb7d5 vs 1.0.0 baseline 7ddd90a)</text>"
  )
  out.append("</svg>")
  return "\n".join(out) + "\n"


def make_scaling_memory_svg() -> str:
  w, h = CARD_W, 330
  refs = [1024, 4096, 16384]
  mem_old = [4.0, 16.0, 64.0]
  mem_new = [0.1875, 0.75, 3.0]
  mmax = 74.0

  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    'aria-label="Global device scratch memory scaling for exact 1-NN on NVIDIA L4">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="20" y="28" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">'
    "Exact 1-NN Global Device Scratch Memory (21× Reduction)</text>",
    f'<text x="20" y="48" font-family="{FONT}" font-size="12.5" fill="{MUTED}">'
    "Full Q × R score matrix (1.0.0) vs. tiled partial winners (+ 16 KiB shared tile per block).</text>",
  ]

  rx0, ry0, rw, rh = 66.0, 68.0, w - 92.0, 192.0
  out.append(
    f'<rect x="{rx0:.1f}" y="{ry0:.1f}" width="{rw:.1f}" height="{rh:.1f}" rx="5" fill="#fafbfc" stroke="{GRID}"/>'
  )

  def smy(m: float) -> float:
    return ry0 + rh - (m / mmax) * (rh - 22.0) - 2.0

  for mt in (0, 16, 32, 48, 64):
    yy = smy(mt)
    out.append(f'<line x1="{rx0:.1f}" y1="{yy:.1f}" x2="{rx0 + rw:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{rx0 - 10:.1f}" y="{yy + 4.5:.1f}" text-anchor="end" font-family="{MONO}" '
      f'font-size="12" fill="{SUBTLE}">{mt}</text>'
    )
  out.append(
    f'<text x="20" y="{ry0 + rh / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {ry0 + rh / 2:.1f})" '
    f'font-family="{FONT}" font-size="12.5" fill="{SUBTLE}">Global scratch (MiB)</text>'
  )

  bar_w = 50.0
  for i, r in enumerate(refs):
    cx = rx0 + (i + 0.5) * (rw / 3.0)
    y1 = smy(mem_old[i])
    out.append(
      f'<rect x="{cx - bar_w - 5:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{smy(0) - y1:.1f}" '
      f'rx="4" fill="{PK_LEGACY}" fill-opacity="0.88"/>'
    )
    out.append(
      f'<text x="{cx - bar_w / 2 - 5:.1f}" y="{y1 - 6:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="12" font-weight="700" fill="{PK_LEGACY}">{mem_old[i]:.0f} MiB</text>'
    )
    y2 = smy(mem_new[i])
    h2 = max(4.0, smy(0) - y2)
    out.append(
      f'<rect x="{cx + 5:.1f}" y="{smy(0) - h2:.1f}" width="{bar_w:.1f}" height="{h2:.1f}" rx="4" fill="{PK_MAIN}"/>'
    )
    lbl_new = "192 KiB" if r == 1024 else ("768 KiB" if r == 4096 else "3.0 MiB")
    out.append(
      f'<text x="{cx + bar_w / 2 + 5:.1f}" y="{smy(0) - h2 - 6:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="12" font-weight="700" fill="{PK_ALT}">{lbl_new}</text>'
    )
    out.append(
      f'<text x="{cx:.1f}" y="{ry0 + rh + 22:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="12.5" fill="{TEXT}">R = {r:,}</text>'
    )

  out.append(
    f'<rect x="{rx0 + 12:.1f}" y="{ry0 + 12:.1f}" width="266" height="54" rx="6" fill="#ffffff" fill-opacity="0.95" stroke="{GRID}"/>'
  )
  out.append(f'<rect x="{rx0 + 22:.1f}" y="{ry0 + 23:.1f}" width="14" height="12" rx="2" fill="{PK_LEGACY}"/>')
  out.append(
    f'<text x="{rx0 + 44:.1f}" y="{ry0 + 33:.1f}" font-family="{FONT}" font-size="12.5" fill="{TEXT}">Parakeet 1.0.0 (full Q×R matrix)</text>'
  )
  out.append(f'<rect x="{rx0 + 22:.1f}" y="{ry0 + 44:.1f}" width="14" height="12" rx="2" fill="{PK_MAIN}"/>')
  out.append(
    f'<text x="{rx0 + 44:.1f}" y="{ry0 + 54:.1f}" font-family="{FONT}" font-size="12.5" fill="{TEXT}">Parakeet tiled partial winners</text>'
  )

  out.append(
    f'<text x="20" y="{h - 10}" font-family="{FONT}" font-size="11" fill="{SUBTLE}">'
    "Source: docs/assets/colab-l4-0dcb7d5.json · Exact scratch sizes from CUDA allocator</text>"
  )
  out.append("</svg>")
  return "\n".join(out) + "\n"


def _make_nanogpt_single_svg(
  *,
  title: str,
  subtitle: str,
  cfg_d: dict,
  vmax: float,
  ticks: tuple[int, ...],
  footer: str,
) -> str:
  """Render one crisp 680px single-panel horizontal bar chart for a nanoGPT config."""
  order = [
    ("JAX jit (XLA + cuBLAS)", "jax_jit", BASE_BLUE, False),
    ("torch.compile (cuBLAS)", "torch_compile", BASE_BLUE, False),
    ("Parakeet TF32 (WMMA)", "parakeet_tf32_wmma", PK_MAIN, True),
    ("Parakeet FP32", "parakeet_fp32", PK_ALT, True),
    ("Parakeet 0dcb7d5 TF32", "parakeet_prev_tf32", PK_LEGACY, True),
    ("PyTorch eager", "torch_eager", BASE_SLATE, False),
  ]
  rows = [(lbl, float(cfg_d[key]["median_ms"]), col, is_pk) for lbl, key, col, is_pk in order if key in cfg_d]
  rows.sort(key=lambda r: r[1])

  w = CARD_W
  row_h = 38.0
  header_h = 64.0
  footer_h = 58.0
  plot_h = len(rows) * row_h
  h = int(header_h + plot_h + footer_h)

  lbl_w = 186.0
  val_w = 88.0
  px0 = 18.0 + lbl_w
  pw = w - 36.0 - lbl_w - val_w
  py0 = header_h

  out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
    f'aria-label="{_esc(title)}">',
    f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
    f'<text x="20" y="28" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">{_esc(title)}</text>',
    f'<text x="20" y="48" font-family="{FONT}" font-size="12.5" fill="{MUTED}">{_esc(subtitle)}</text>',
    f'<rect x="{px0:.1f}" y="{py0:.1f}" width="{pw:.1f}" height="{plot_h:.1f}" rx="5" fill="#fafbfc" stroke="{GRID}"/>',
  ]

  for t in ticks:
    tx = px0 + (t / vmax) * pw
    out.append(f'<line x1="{tx:.1f}" y1="{py0:.1f}" x2="{tx:.1f}" y2="{py0 + plot_h:.1f}" stroke="{GRID}"/>')
    out.append(
      f'<text x="{tx:.1f}" y="{py0 + plot_h + 18:.1f}" text-anchor="middle" font-family="{MONO}" '
      f'font-size="12" fill="{SUBTLE}">{t}</text>'
    )
  out.append(
    f'<text x="{px0 + pw / 2:.1f}" y="{py0 + plot_h + 36:.1f}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="12" fill="{SUBTLE}">Step time (ms, lower is better)</text>'
  )

  for i, (label, val, color, is_pk) in enumerate(rows):
    cy = py0 + (i + 0.5) * row_h
    if is_pk:
      bg_fill = "#fffbeb" if color == PK_LEGACY else "#f0fdfa"
      out.append(
        f'<rect x="{px0 + 1:.1f}" y="{cy - row_h * 0.44:.1f}" width="{pw - 2:.1f}" '
        f'height="{row_h * 0.88:.1f}" rx="3" fill="{bg_fill}"/>'
      )
    if i > 0:
      out.append(
        f'<line x1="{px0:.1f}" y1="{py0 + i * row_h:.1f}" x2="{px0 + pw:.1f}" y2="{py0 + i * row_h:.1f}" stroke="{GRID_SUB}"/>'
      )
    weight = "700" if is_pk else "500"
    lfill = PK_LEGACY if color == PK_LEGACY else (PK_ALT if is_pk else TEXT)
    out.append(
      f'<text x="{px0 - 10:.1f}" y="{cy + 4.8:.1f}" text-anchor="end" font-family="{FONT}" '
      f'font-size="13.5" font-weight="{weight}" fill="{lfill}">{_esc(label)}</text>'
    )
    bw = max(4.0, (val / vmax) * pw)
    bh = 22.0
    out.append(
      f'<rect x="{px0:.1f}" y="{cy - bh / 2:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
      f'rx="3.5" fill="{color}" fill-opacity="0.88"/>'
    )
    out.append(
      f'<text x="{px0 + pw + 10:.1f}" y="{cy + 4.8:.1f}" font-family="{MONO}" '
      f'font-size="13" font-weight="{weight}" fill="{lfill}">{val:.2f} ms</text>'
    )

  out.append(f'<text x="20" y="{h - 10}" font-family="{FONT}" font-size="11" fill="{SUBTLE}">{_esc(footer)}</text>')
  out.append("</svg>")
  return "\n".join(out) + "\n"


def make_bio_individual_svgs(bio_data: dict) -> list[tuple[str, str]]:
  """Render one dedicated 680px SVG per bioinformatics workload (always 2 Parakeet variants)."""
  workloads = bio_data["workloads"]
  specs = [
    (
      "bench-bio-myers-m1pro.svg",
      "edit_distance_1000x256",
      "Myers 64-Bit Edit Distance (1,000 × 256 bp)",
      "Exact bit-parallel Levenshtein distance vs. C++ edlib.",
    ),
    (
      "bench-bio-gotoh-m1pro.svg",
      "gotoh_align_200x128",
      "Affine-Gap Gotoh Alignment + Traceback (200 × 128 bp)",
      "Full 3-state traceback + RLE CIGAR vs. Biopython (score-only) & Biotite.",
    ),
    (
      "bench-bio-rna-m1pro.svg",
      "patro_longread_rna",
      "SSHash + Delta-Map Long-Read RNA (3.01 Mbp)",
      "80 human Ensembl isoforms, 9.70× unitig collapse, 92.2% hash-probe skip.",
    ),
    (
      "bench-bio-tanimoto-m1pro.svg",
      "tanimoto_256x8192",
      "2048-Bit Tanimoto 1-NN Search (256 × 8,192)",
      "Tiled POPCNT bit-vector search vs. C++ RDKit BulkTanimoto.",
    ),
    (
      "bench-bio-kabsch-m1pro.svg",
      "kabsch_rmsd_2000x128",
      "Batched Kabsch 3D Protein RMSD (2,000 × 128 CA)",
      "Compiled 3×3 SVD + determinant vs. Biotite (f32) & NumPy batched SVD (f64).",
    ),
    (
      "bench-bio-pileup-m1pro.svg",
      "bam_pileup_2m",
      "6-Channel BAM Pileup (2,000,000 Alignments)",
      "Single-pass fused multi-channel scatter vs. 6× NumPy bincount.",
    ),
  ]

  def color_for(name: str) -> tuple[str, bool]:
    if name.startswith("Parakeet (multicore)"):
      return PK_MAIN, True
    if name.startswith("Parakeet"):
      return PK_ALT, True
    if "edlib" in name or "RDKit" in name:
      return BASE_BLUE, False
    if "Biopython" in name or "Biotite" in name:
      return BASE_PURPLE, False
    return BASE_SLATE, False

  outputs: list[tuple[str, str]] = []
  for fname, key, title, sub in specs:
    wdata = workloads[key]
    arms = list(wdata["arms"].items())
    w = CARD_W
    row_h = 40.0
    header_h = 64.0
    footer_h = 42.0
    plot_h = len(arms) * row_h
    h = int(header_h + plot_h + footer_h)

    max_med = max(a["median_ms"] for _, a in arms)
    all_s = [s for _, a in arms for s in a["samples_ms"]]
    max_val = max(max_med * 1.15, float(sorted(all_s)[int(0.92 * (len(all_s) - 1))]), 1e-3)

    bar_left = 224.0
    val_w = 152.0
    bar_max_w = w - bar_left - val_w - 16.0
    py0 = header_h

    lines = [
      f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
      f'aria-label="{_esc(title)}">',
      f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{BG}" stroke="{CARD_BORDER}"/>',
      f'<text x="20" y="28" font-family="{FONT}" font-size="16" font-weight="700" fill="{TEXT}">{_esc(title)}</text>',
      f'<text x="20" y="48" font-family="{FONT}" font-size="12.5" fill="{MUTED}">{_esc(sub)}</text>',
      f'<rect x="{bar_left:.1f}" y="{py0:.1f}" width="{bar_max_w:.1f}" height="{plot_h:.1f}" rx="5" fill="#fafbfc" stroke="{GRID}"/>',
    ]

    for i, (arm_name, st) in enumerate(arms):
      cy = py0 + (i + 0.5) * row_h
      med = st["median_ms"]
      iqr = st["iqr_ms"]
      bw = max(4.0, min(bar_max_w, (med / max_val) * bar_max_w))
      col_hex, is_pk = color_for(arm_name)
      weight = "700" if is_pk else "500"
      lfill = PK_ALT if is_pk else TEXT

      if is_pk:
        lines.append(
          f'<rect x="{bar_left + 1:.1f}" y="{cy - row_h * 0.44:.1f}" width="{bar_max_w - 2:.1f}" '
          f'height="{row_h * 0.88:.1f}" rx="3" fill="#f0fdfa"/>'
        )
      if i > 0:
        lines.append(
          f'<line x1="{bar_left:.1f}" y1="{py0 + i * row_h:.1f}" x2="{bar_left + bar_max_w:.1f}" '
          f'y2="{py0 + i * row_h:.1f}" stroke="{GRID_SUB}"/>'
        )

      lines.append(
        f'<text x="{bar_left - 10:.1f}" y="{cy + 4.8:.1f}" text-anchor="end" font-family="{FONT}" '
        f'font-size="13.5" font-weight="{weight}" fill="{lfill}">{_esc(arm_name)}</text>'
      )
      lines.append(
        f'<rect x="{bar_left:.1f}" y="{cy - 10.5:.1f}" width="{bw:.1f}" height="21" '
        f'rx="3.5" fill="{col_hex}" fill-opacity="0.88"/>'
      )
      for s in st["samples_ms"]:
        sx_dot = bar_left + max(4.0, min(bar_max_w - 6.0, (s / max_val) * bar_max_w))
        lines.append(f'<circle cx="{sx_dot:.1f}" cy="{cy:.1f}" r="2.6" fill="{TEXT}" fill-opacity="0.44"/>')

      val_txt = f"{med:.2f} ms" if med >= 1.0 else f"{med:.3f} ms"
      if "mbp_per_s" in st:
        val_txt += f" ({st['mbp_per_s']:.0f} Mbp/s)"
      elif iqr > 0:
        val_txt += f" (±{iqr:.2f})"
      lines.append(
        f'<text x="{bar_left + bar_max_w + 10:.1f}" y="{cy + 4.8:.1f}" font-family="{MONO}" '
        f'font-size="12.5" font-weight="{weight}" fill="{lfill}">{_esc(val_txt)}</text>'
      )

    lines.append(
      f'<text x="20" y="{h - 12}" font-family="{FONT}" font-size="11" fill="{SUBTLE}">'
      f"{_esc(bio_data['machine'])} · n={bio_data['repeats']} replicates (median bar + replicate dots; lower is faster)</text>"
    )
    lines.append("</svg>")
    outputs.append((fname, "\n".join(lines) + "\n"))
  return outputs


def generate_all(
  data_path: Path = DEFAULT_DATA,
  nanogpt_path: Path = DEFAULT_NANOGPT_DATA,
  bio_path: Path = DEFAULT_BIO_DATA,
  out_dir: Path = DEFAULT_OUT,
) -> list[Path]:
  data = json.loads(data_path.read_text())
  rows = _find_rows(data)
  base_4096 = next(
    r["samples_ms"] for r in data["baseline_scaling"]["rows"] if r["source"] == "numpy" and r["references"] == 4096
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  src_footer = "Source: docs/assets/colab-l4-0dcb7d5.json (rev 0dcb7d5) · ± half-IQR shown when ≥ 0.001 ms"

  gpu_nn = [
    ("Faiss GpuIndexFlatL2", rows[("gpu", "nn", "large", "faiss-flat")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (NumPy, tiled)", rows[("gpu", "nn", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("Parakeet (nested par.map)", rows[("gpu", "nn", "large", "pk-dsl")]["samples_ms"], PK_ALT, True),
    ("PyTorch cdist", rows[("gpu", "nn", "large", "torch-cdist")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile (cdist)", rows[("gpu", "nn", "large", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "nn", "large", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    (
      "torch.compile (broadcast)",
      rows[("gpu", "nn", "large", "torch-compile-broadcast")]["samples_ms"],
      BASE_BLUE,
      False,
    ),
    ("Parakeet 1.0.0 (un-tiled)", base_4096, PK_LEGACY, True),
    ("PyKeOps Genred", rows[("gpu", "nn", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("CuPy broadcast", rows[("gpu", "nn", "large", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  cpu_nn = [
    ("scikit-learn brute", rows[("cpu", "nn", "large", "sklearn-brute")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (nested par.map)", rows[("cpu", "nn", "large", "pk-dsl")]["samples_ms"], PK_ALT, True),
    ("Parakeet (NumPy)", rows[("cpu", "nn", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("Faiss IndexFlatL2", rows[("cpu", "nn", "large", "faiss-flat")]["samples_ms"], BASE_BLUE, False),
    ("PyKeOps Genred", rows[("cpu", "nn", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("SciPy cKDTree", rows[("cpu", "nn", "large", "scipy-tree")]["samples_ms"], BASE_BLUE, False),
    ("NumPy broadcast", rows[("cpu", "nn", "large", "numpy")]["samples_ms"], BASE_SLATE, False),
  ]

  gpu_rbf = [
    ("torch.compile (cdist)", rows[("gpu", "rbf", "large", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("PyTorch cdist", rows[("gpu", "rbf", "large", "torch-cdist")]["samples_ms"], BASE_BLUE, False),
    (
      "torch.compile (broadcast)",
      rows[("gpu", "rbf", "large", "torch-compile-broadcast")]["samples_ms"],
      BASE_BLUE,
      False,
    ),
    ("JAX jit", rows[("gpu", "rbf", "large", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (NumPy, online)", rows[("gpu", "rbf", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("PyKeOps Genred", rows[("gpu", "rbf", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("CuPy broadcast", rows[("gpu", "rbf", "large", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  cpu_rbf = [
    ("Parakeet (NumPy, 4T)", rows[("cpu", "rbf", "large", "pk-numpy")]["samples_ms"], PK_MAIN, True),
    ("PyKeOps Genred (4T)", rows[("cpu", "rbf", "large", "keops")]["samples_ms"], BASE_BLUE, False),
    ("NumPy broadcast", rows[("cpu", "rbf", "large", "numpy")]["samples_ms"], BASE_SLATE, False),
  ]

  gbdt = [
    ("torch.compile", rows[("gpu", "split", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "split", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("PyTorch scatter", rows[("gpu", "split", "default", "torch-scatter")]["samples_ms"], BASE_BLUE, False),
    ("Parakeet (fused scatter)", rows[("gpu", "split", "default", "pk-dsl")]["samples_ms"], PK_MAIN, True),
    ("CuPy", rows[("gpu", "split", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
  ]

  attn = [
    ("PyTorch SDPA (cuDNN)", rows[("gpu", "causal", "default", "torch-sdpa")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "causal", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile", rows[("gpu", "causal", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("CuPy", rows[("gpu", "causal", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
    ("Parakeet (monoidal tile)", rows[("gpu", "causal", "default", "pk-numpy")]["samples_ms"], PK_MAIN, True),
  ]

  cce = [
    ("PyTorch cross_entropy", rows[("gpu", "cce", "default", "torch-cross-entropy")]["samples_ms"], BASE_BLUE, False),
    ("torch.compile", rows[("gpu", "cce", "default", "torch-compile")]["samples_ms"], BASE_BLUE, False),
    ("JAX jit", rows[("gpu", "cce", "default", "jax-jit")]["samples_ms"], BASE_BLUE, False),
    ("CuPy", rows[("gpu", "cce", "default", "cupy")]["samples_ms"], BASE_SLATE, False),
    ("Parakeet (0-logit tile)", rows[("gpu", "cce", "default", "pk-numpy")]["samples_ms"], PK_MAIN, True),
  ]

  outputs: list[tuple[Path, str]] = [
    (
      out_dir / "bench-1nn-gpu-l4.svg",
      _make_single_dot_svg(
        title="GPU Exact 1-NN Search on NVIDIA L4 (1,024 × 4,096 × 32)",
        subtitle="GPU-resident float32, 0 host transfers, exact tie/NaN semantics (n=21).",
        footer=src_footer,
        arms=gpu_nn,
        vmin=0.1,
        vmax=25.0,
        show_legacy_legend=True,
      ),
    ),
    (
      out_dir / "bench-1nn-cpu.svg",
      _make_single_dot_svg(
        title="CPU Exact 1-NN Search on 4 Threads (1,024 × 4,096 × 32)",
        subtitle="Tiled pairwise fold in 8 KiB L1 scratch vs. specialist libraries (n=11).",
        footer=src_footer,
        arms=cpu_nn,
        vmin=4.0,
        vmax=700.0,
      ),
    ),
    (
      out_dir / "bench-rbf-gpu-l4.svg",
      _make_single_dot_svg(
        title="GPU Gaussian Kernel Regression on NVIDIA L4 (1,024 × 4,096 × 32)",
        subtitle="4 normalized weighted sums per query in one tiled launch (n=21).",
        footer=src_footer,
        arms=gpu_rbf,
        vmin=0.2,
        vmax=25.0,
      ),
    ),
    (
      out_dir / "bench-rbf-cpu.svg",
      _make_single_dot_svg(
        title="CPU Gaussian Kernel Regression on 4 Threads (1,024 × 4,096 × 32)",
        subtitle="Single-pass online monoidal reduction (11.8× faster than NumPy, n=11).",
        footer=src_footer,
        arms=cpu_rbf,
        vmin=20.0,
        vmax=700.0,
      ),
    ),
    (out_dir / "bench-scaling-1nn-latency.svg", make_scaling_latency_svg(data)),
    (out_dir / "bench-scaling-1nn-memory.svg", make_scaling_memory_svg()),
    (
      out_dir / "bench-gbdt-l4.svg",
      _make_single_dot_svg(
        title="GPU GBDT Histogram & Best Split on NVIDIA L4 (262,144 × 256)",
        subtitle="2 fused sibling histograms + prefix scans + gain argmax (n=21).",
        footer=src_footer,
        arms=gbdt,
        vmin=0.15,
        vmax=1.0,
      ),
    ),
    (
      out_dir / "bench-attn-l4.svg",
      _make_single_dot_svg(
        title="GPU Causal Self-Attention on NVIDIA L4 (4 × 512 × 64)",
        subtitle="Single tiled monoidal kernel with Law 5 causal pruning (n=21).",
        footer=src_footer,
        arms=attn,
        vmin=0.07,
        vmax=0.9,
      ),
    ),
    (
      out_dir / "bench-cce-l4.svg",
      _make_single_dot_svg(
        title="GPU Linear Cross-Entropy on NVIDIA L4 (512 × 128 × 2,048)",
        subtitle="Zero M × V logit allocation vs. cuBLAS + fused cross_entropy (n=21).",
        footer=src_footer,
        arms=cce,
        vmin=0.05,
        vmax=2.5,
      ),
    ),
  ]

  ng_data = json.loads(nanogpt_path.read_text()) if nanogpt_path.exists() else data["training"]
  ng_cfgs = ng_data["configs"]
  ng_footer = "Source: docs/assets/colab-l4-nanogpt.json · NVIDIA L4, median of 20 warmed steps (0 cuBLAS in Parakeet)"
  outputs.extend(
    [
      (
        out_dir / "bench-nanogpt-small-l4.svg",
        _make_nanogpt_single_svg(
          title="nanoGPT Small Training Step on NVIDIA L4 (16 × 64 × 128 × 2L)",
          subtitle="Forward + reverse-mode AD + Adam, 4 heads. Parakeet 2.0× faster than eager PyTorch.",
          cfg_d=ng_cfgs["small_B16_T64_C128_L2"],
          vmax=10.5,
          ticks=(0, 2, 4, 6, 8, 10),
          footer=ng_footer,
        ),
      ),
      (
        out_dir / "bench-nanogpt-med-l4.svg",
        _make_nanogpt_single_svg(
          title="nanoGPT Medium Training Step on NVIDIA L4 (32 × 128 × 256 × 4L)",
          subtitle="Forward + reverse-mode AD + Adam, 4 heads. Double-buffered cp.async WMMA cuts step 1.65×.",
          cfg_d=ng_cfgs["med_B32_T128_C256_L4"],
          vmax=42.0,
          ticks=(0, 10, 20, 30, 40),
          footer=ng_footer,
        ),
      ),
    ]
  )

  if bio_path.exists():
    bio_data = json.loads(bio_path.read_text())
    for fname, content in make_bio_individual_svgs(bio_data):
      outputs.append((out_dir / fname, content))

  written: list[Path] = []
  for path, content in outputs:
    path.write_text(content, encoding="utf-8")
    written.append(path)
  return written


if __name__ == "__main__":
  for p in generate_all():
    print(f"wrote {p}")
