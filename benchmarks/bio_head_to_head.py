"""Reproducible head-to-head bioinformatics benchmarks: Parakeet vs C/C++/Cython bio libraries.

Compares Parakeet (serial ``c`` and ``multicore``) against installed specialist
bioinformatics libraries (``edlib``, ``biopython`` C PairwiseAligner, ``biotite``,
``rdkit``, ``scipy``, and ``numpy``), verifying exact output agreement before
timing and reporting replicate medians, IQR, and speedups.

Saves raw JSON measurements to ``--json <path>`` and optionally renders an SVG
chart to ``--svg <path>``.

Run:
    python benchmarks/bio_head_to_head.py --json docs/assets/bio-benchmarks-m1pro.json --svg docs/assets/bench-bio-m1pro.svg
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import time

import numpy as np

import parakeet as par
from examples.bio_algorithms import (
  batched_kabsch_rmsd,
  gotoh_global_align,
  myers_edit_distance,
)
from examples.bio_gallery import (
  single_pass_pileup,
  tanimoto_nearest_hit,
)
from examples.patro_longread_rna import (
  build_unitig_sshash_index,
  load_fasta,
  make_synthetic_isoforms,
  simulate_long_reads,
  stream_map_long_reads,
)


@par.jit
def _batched_myers(pats, txts):
  n = pats.shape[0]
  out = np.empty(n, dtype=np.int64)

  def _one(i):
    out[i] = myers_edit_distance(pats[i], txts[i])

  par.parfor(_one, n)
  return out


@par.jit
def _batched_gotoh_scores(seqs_a, seqs_b, match_s=2, mismatch_s=-1, gap_open=-4, gap_ext=-1):
  n = seqs_a.shape[0]
  out = np.empty(n, dtype=np.int32)

  def _one(i):
    sc, _, _ = gotoh_global_align(seqs_a[i], seqs_b[i], match_s, mismatch_s, gap_open, gap_ext)
    out[i] = sc

  par.parfor(_one, n)
  return out


def _bench(fn, repeats: int = 11, warmup: int = 2) -> dict:
  for _ in range(warmup):
    fn()
  samples = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    fn()
    samples.append((time.perf_counter() - t0) * 1e3)
  arr = np.asarray(samples, dtype=np.float64)
  q25, med, q75 = np.percentile(arr, [25, 50, 75])
  return {
    "median_ms": round(float(med), 4),
    "iqr_ms": round(float(q75 - q25), 4),
    "samples_ms": [round(float(x), 4) for x in samples],
  }


def run_all(fasta_path: str | None = None, repeats: int = 11) -> dict:
  rng = np.random.default_rng(0)
  alph = np.frombuffer(b"ACGT", dtype=np.uint8)
  results: dict[str, dict] = {}

  # 1. Myers bit-vector edit distance (1,000 pairs x 256 bp)
  n_ed, L_ed = 1000, 256
  pats = alph[rng.integers(0, 4, size=(n_ed, L_ed))]
  txts = pats.copy()
  # Introduce ~10% edits
  mut_mask = rng.random(txts.shape) < 0.10
  txts[mut_mask] = alph[rng.integers(0, 4, size=int(mut_mask.sum()))]
  pat_strs = [bytes(p).decode("ascii") for p in pats]
  txt_strs = [bytes(t).decode("ascii") for t in txts]

  ref_ed = _batched_myers(pats, txts, _backend="c")
  ed_arms: dict[str, dict] = {
    "Parakeet (multicore)": _bench(lambda: _batched_myers(pats, txts, _backend="multicore"), repeats),
    "Parakeet (c, 1T)": _bench(lambda: _batched_myers(pats, txts, _backend="c"), repeats),
  }
  try:
    import edlib

    got_edlib = np.array([edlib.align(p, t)["editDistance"] for p, t in zip(pat_strs, txt_strs)], dtype=np.int64)
    np.testing.assert_array_equal(ref_ed, got_edlib)
    ed_arms["edlib (C++ Myers)"] = _bench(
      lambda: [edlib.align(p, t)["editDistance"] for p, t in zip(pat_strs, txt_strs)], repeats
    )
  except ImportError:
    pass
  results["edit_distance_1000x256"] = {"label": "Edit distance (1,000 × 256 bp)", "arms": ed_arms}

  # 2. Affine-gap Gotoh global alignment (200 pairs x 128 bp)
  n_al, L_al = 200, 128
  sa = alph[rng.integers(0, 4, size=(n_al, L_al))]
  sb = sa.copy()
  mm = rng.random(sb.shape) < 0.12
  sb[mm] = alph[rng.integers(0, 4, size=int(mm.sum()))]
  sa_strs = [bytes(s).decode("ascii") for s in sa]
  sb_strs = [bytes(s).decode("ascii") for s in sb]

  ref_gotoh = _batched_gotoh_scores(sa, sb, _backend="c")
  gotoh_arms: dict[str, dict] = {
    "Parakeet (multicore)": _bench(lambda: _batched_gotoh_scores(sa, sb, _backend="multicore"), repeats),
    "Parakeet (c, 1T)": _bench(lambda: _batched_gotoh_scores(sa, sb, _backend="c"), repeats),
  }
  try:
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0  # Biopython open_gap includes the first gap char (open + extend = -4 + -1)
    aligner.extend_gap_score = -1.0
    bio_scores = np.array([int(round(aligner.score(x, y))) for x, y in zip(sa_strs, sb_strs)], dtype=np.int32)
    np.testing.assert_array_equal(ref_gotoh, bio_scores)
    gotoh_arms["Biopython (C PairwiseAligner)"] = _bench(
      lambda: [aligner.score(x, y) for x, y in zip(sa_strs, sb_strs)], repeats
    )
  except ImportError:
    pass
  try:
    import biotite.sequence as bseq
    import biotite.sequence.align as balign

    # Custom match=+2, mismatch=-1 matrix matching Gotoh test
    alph_nuc = bseq.NucleotideSequence.alphabet_unambiguous
    score_m = np.full((len(alph_nuc), len(alph_nuc)), -1, dtype=np.int32)
    np.fill_diagonal(score_m, 2)
    smat = balign.SubstitutionMatrix(alph_nuc, alph_nuc, score_m)
    b_sa = [bseq.NucleotideSequence(s) for s in sa_strs]
    b_sb = [bseq.NucleotideSequence(s) for s in sb_strs]
    b_scores = np.array(
      [balign.align_optimal(x, y, smat, gap_penalty=(-5, -1), max_number=1)[0].score for x, y in zip(b_sa, b_sb)],
      dtype=np.int32,
    )
    np.testing.assert_array_equal(ref_gotoh, b_scores)
    gotoh_arms["Biotite (Cython align_optimal)"] = _bench(
      lambda: [balign.align_optimal(x, y, smat, gap_penalty=(-5, -1), max_number=1) for x, y in zip(b_sa, b_sb)],
      max(5, repeats // 2),
      warmup=1,
    )
  except ImportError:
    pass
  results["gotoh_align_200x128"] = {"label": "Affine-gap Gotoh + traceback (200 × 128 bp)", "arms": gotoh_arms}

  # 3. 2048-bit Morgan fingerprint Tanimoto 1-NN (256 queries x 8,192 library compounds)
  nq, nl, nwords = 256, 8192, 32
  q_fp = rng.integers(0, 1 << 63, size=(nq, nwords), dtype=np.uint64)
  l_fp = rng.integers(0, 1 << 63, size=(nl, nwords), dtype=np.uint64)
  ref_tan = tanimoto_nearest_hit(q_fp, l_fp, _backend="multicore")
  tan_arms: dict[str, dict] = {
    "Parakeet (multicore)": _bench(lambda: tanimoto_nearest_hit(q_fp, l_fp, _backend="multicore"), repeats),
    "Parakeet (c, 1T)": _bench(lambda: tanimoto_nearest_hit(q_fp, l_fp, _backend="c"), repeats),
  }
  try:
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect

    def _to_ebv(row):
      bv = ExplicitBitVect(nwords * 64)
      raw = row.tobytes()
      for byte_i, bval in enumerate(raw):
        if bval:
          for bit in range(8):
            if (bval >> bit) & 1:
              bv.SetBit(byte_i * 8 + bit)
      return bv

    q_bvs = [_to_ebv(r) for r in q_fp]
    l_bvs = [_to_ebv(r) for r in l_fp]
    rd_best = np.array([int(np.argmax(DataStructs.BulkTanimotoSimilarity(qb, l_bvs))) for qb in q_bvs], dtype=np.int64)
    np.testing.assert_array_equal(ref_tan, rd_best)
    tan_arms["RDKit (C++ BulkTanimoto)"] = _bench(
      lambda: [int(np.argmax(DataStructs.BulkTanimotoSimilarity(qb, l_bvs))) for qb in q_bvs], repeats
    )
  except ImportError:
    pass
  results["tanimoto_256x8192"] = {"label": "2048-bit Tanimoto 1-NN (256 × 8,192)", "arms": tan_arms}

  # 4. Batched Kabsch 3D protein superposition & RMSD (2,000 structures x 128 residues)
  nb_k, nr_k = 2000, 128
  Q_ref = rng.standard_normal((nr_k, 3))
  P_batch = Q_ref[None, :, :] + 0.4 * rng.standard_normal((nb_k, nr_k, 3))
  ref_rmsd = batched_kabsch_rmsd(P_batch, Q_ref, _backend="multicore")

  def _np_kabsch_batch():
    mu_q = Q_ref.mean(axis=0)
    Qc = Q_ref - mu_q
    mu_p = P_batch.mean(axis=1, keepdims=True)
    Pc = P_batch - mu_p
    H = np.einsum("bni,nj->bij", Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.swapaxes(1, 2) @ U.swapaxes(1, 2)
    neg = np.linalg.det(R) < 0
    if np.any(neg):
      Vt[neg, 2, :] *= -1.0
      R = Vt.swapaxes(1, 2) @ U.swapaxes(1, 2)
    diff = Pc @ R.swapaxes(1, 2) - Qc[None, :, :]
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))

  np.testing.assert_allclose(ref_rmsd, _np_kabsch_batch(), rtol=1e-10, atol=1e-10)
  kabsch_arms: dict[str, dict] = {
    "Parakeet (multicore)": _bench(lambda: batched_kabsch_rmsd(P_batch, Q_ref, _backend="multicore"), repeats),
    "Parakeet (c, 1T)": _bench(lambda: batched_kabsch_rmsd(P_batch, Q_ref, _backend="c"), repeats),
    "NumPy batched SVD": _bench(_np_kabsch_batch, repeats),
  }
  try:
    import biotite.structure as bstruct

    def _biotite_kabsch():
      tr = bstruct.superimpose(Q_ref, P_batch)
      fitted = tr.apply(P_batch)
      return bstruct.rmsd(Q_ref, fitted)

    np.testing.assert_allclose(ref_rmsd, _biotite_kabsch(), rtol=1e-5, atol=1e-5)
    kabsch_arms["Biotite (batched f32)"] = _bench(_biotite_kabsch, repeats)
  except ImportError:
    pass
  results["kabsch_rmsd_2000x128"] = {"label": "Batched Kabsch 3D RMSD (2,000 × 128 CA)", "arms": kabsch_arms}

  # 5. Single-pass 6-channel BAM pileup (2,000,000 alignments -> 100,000 bp contig)
  n_pile, contig_len = 2_000_000, 100_000
  pos = rng.integers(0, contig_len, size=n_pile, dtype=np.int64)
  bases = rng.integers(0, 4, size=n_pile, dtype=np.int64)
  phred = rng.integers(10, 42, size=n_pile, dtype=np.int64)
  is_rev = rng.integers(0, 2, size=n_pile, dtype=np.int64)

  def _np_pileup():
    return (
      np.bincount(pos, weights=(bases == 0).astype(np.int64), minlength=contig_len),
      np.bincount(pos, weights=(bases == 1).astype(np.int64), minlength=contig_len),
      np.bincount(pos, weights=(bases == 2).astype(np.int64), minlength=contig_len),
      np.bincount(pos, weights=(bases == 3).astype(np.int64), minlength=contig_len),
      np.bincount(pos, weights=phred, minlength=contig_len),
      np.bincount(pos, weights=is_rev, minlength=contig_len),
    )

  pile_arms = {
    "Parakeet (multicore)": _bench(
      lambda: single_pass_pileup(pos, bases, phred, is_rev, contig_len, _backend="multicore"), repeats
    ),
    "Parakeet (c, 1T)": _bench(
      lambda: single_pass_pileup(pos, bases, phred, is_rev, contig_len, _backend="c"), repeats
    ),
    "NumPy (6× bincount)": _bench(_np_pileup, repeats),
  }
  results["bam_pileup_2m"] = {"label": "6-channel BAM pileup (2M reads)", "arms": pile_arms}

  # 6. Rob Patro SSHash + Spearfish long-read RNA alignment (1,200 reads, ~3 Mbp)
  if fasta_path and pathlib.Path(fasta_path).exists():
    _, seqs = load_fasta(pathlib.Path(fasta_path))
    ds_name = pathlib.Path(fasta_path).name
  else:
    _, seqs = make_synthetic_isoforms(rng)
    ds_name = "synthetic_30_isoforms"
  idx = build_unitig_sshash_index(seqs, k=15, m=9)
  reads_rag, _, _ = simulate_long_reads(seqs, n_reads=1200, sub_rate=0.015, indel_rate=0.005, seed=7)
  mbp = len(reads_rag.values) / 1e6

  def _run_patro(backend):
    return stream_map_long_reads(
      reads_rag,
      idx["u_vals"],
      idx["u_offs"],
      idx["p_tx"],
      idx["p_pos"],
      idx["p_offs"],
      idx["min_table"],
      idx["b_offs"],
      idx["b_uid"],
      idx["b_mpos"],
      idx["n_tx"],
      _backend=backend,
    )

  patro_arms = {
    "Parakeet (multicore)": _bench(lambda: _run_patro("multicore"), repeats),
    "Parakeet (c, 1T)": _bench(lambda: _run_patro("c"), repeats),
  }
  for arm in patro_arms.values():
    arm["mbp_per_s"] = round(mbp / (arm["median_ms"] / 1e3), 1)
  results["patro_longread_rna"] = {
    "label": f"SSHash + delta-map long-read RNA ({mbp:.2f} Mbp, {idx['n_tx']} isoforms)",
    "dataset": ds_name,
    "unitig_collapse_x": round(idx["raw_bp"] / max(1, idx["unitig_bp"]), 2),
    "arms": patro_arms,
  }

  return {
    "machine": f"{platform.machine()} / {platform.system()} ({os.cpu_count()} cores)",
    "python": platform.python_version(),
    "numpy": np.__version__,
    "parakeet": par.__version__,
    "repeats": repeats,
    "workloads": results,
  }


def write_svg(report: dict, svg_path: pathlib.Path) -> None:
  """Render clean horizontal bar + replicate dot + IQR plot matching Parakeet benchmark SVGs."""
  workloads = report["workloads"]
  panels = [
    ("edit_distance_1000x256", "Edit distance (1,000 × 256 bp)"),
    ("gotoh_align_200x128", "Affine-gap Gotoh (200 × 128 bp)"),
    ("tanimoto_256x8192", "2048-bit Tanimoto 1-NN (256 × 8,192)"),
    ("kabsch_rmsd_2000x128", "Batched Kabsch 3D RMSD (2,000 × 128)"),
    ("bam_pileup_2m", "6-channel BAM pileup (2M reads)"),
    ("patro_longread_rna", "SSHash + delta-map RNA (3.0 Mbp)"),
  ]

  W, H = 960, 560
  cols, rows = 2, 3
  pad_x, pad_y = 24, 58
  cell_w = (W - pad_x * 2 - 24) / cols
  cell_h = (H - pad_y - 36) / rows

  def color_for(name: str) -> str:
    if name.startswith("Parakeet (multicore)"):
      return "#0d9488"
    if name.startswith("Parakeet"):
      return "#14b8a6"
    if "edlib" in name or "RDKit" in name:
      return "#2563eb"
    if "Biopython" in name or "Biotite" in name:
      return "#7c3aed"
    return "#64748b"

  lines = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    f'<text x="{pad_x}" y="26" font-family="Inter, -apple-system, sans-serif" font-size="16" font-weight="700" fill="#0f172a">Bioinformatics Head-to-Head Benchmarks (Lower is Better)</text>',
    f'<text x="{pad_x}" y="44" font-family="Inter, -apple-system, sans-serif" font-size="11" fill="#475569">{report["machine"]} | Python {report["python"]} | Parakeet {report["parakeet"]} | n={report["repeats"]} warmed replicates (median, IQR, individual dots)</text>',
  ]

  for idx, (key, title) in enumerate(panels):
    col = idx % cols
    row = idx // cols
    x0 = pad_x + col * (cell_w + 24)
    y0 = pad_y + row * cell_h
    wdata = workloads[key]
    arms = list(wdata["arms"].items())
    max_med = max(a["median_ms"] for _, a in arms)
    all_s = [s for _, a in arms for s in a["samples_ms"]]
    max_val = max(max_med * 1.18, float(np.percentile(all_s, 92)), 1e-3)

    lines.append(
      f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h - 14:.1f}" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>'
    )
    lines.append(
      f'<text x="{x0 + 12:.1f}" y="{y0 + 20:.1f}" font-family="Inter, -apple-system, sans-serif" font-size="12" font-weight="700" fill="#0f172a">{title}</text>'
    )

    bar_left = x0 + 174
    bar_max_w = cell_w - 282
    step_y = min(28, (cell_h - 46) / max(1, len(arms)))
    for i, (arm_name, stats) in enumerate(arms):
      cy = y0 + 42 + i * step_y
      med = stats["median_ms"]
      iqr = stats["iqr_ms"]
      bw = max(2.0, min(bar_max_w, (med / max_val) * bar_max_w))
      col_hex = color_for(arm_name)
      lines.append(
        f'<text x="{bar_left - 8:.1f}" y="{cy + 4:.1f}" text-anchor="end" font-family="Inter, -apple-system, sans-serif" font-size="10.2" fill="#334155">{arm_name}</text>'
      )
      lines.append(
        f'<rect x="{bar_left:.1f}" y="{cy - 7:.1f}" width="{bw:.1f}" height="14" rx="3" fill="{col_hex}" fill-opacity="0.82"/>'
      )
      # Replicate dots (clamped to axis right edge)
      for s in stats["samples_ms"]:
        sx = bar_left + min(bar_max_w, (s / max_val) * bar_max_w)
        lines.append(f'<circle cx="{sx:.1f}" cy="{cy:.1f}" r="1.8" fill="#0f172a" fill-opacity="0.45"/>')
      val_txt = f"{med:.2f} ms" if med >= 1.0 else f"{med:.3f} ms"
      if "mbp_per_s" in stats:
        val_txt += f" ({stats['mbp_per_s']:.0f} Mbp/s)"
      elif iqr > 0:
        val_txt += f" (±{iqr:.2f})"
      lines.append(
        f'<text x="{bar_left + bw + 6:.1f}" y="{cy + 4:.1f}" font-family="ui-monospace, Menlo, monospace" font-size="9.3" font-weight="600" fill="#0f172a">{val_txt}</text>'
      )

  lines.append("</svg>")
  svg_path.parent.mkdir(parents=True, exist_ok=True)
  svg_path.write_text("\n".join(lines) + "\n")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--fasta", default="/tmp/real_human_isoforms.fa")
  ap.add_argument("--repeats", type=int, default=11)
  ap.add_argument("--json", default=None)
  ap.add_argument("--svg", default=None)
  args = ap.parse_args()

  report = run_all(args.fasta, repeats=args.repeats)
  print(f"\nMachine: {report['machine']} | Python {report['python']} | Parakeet {report['parakeet']}\n")
  for _key, w in report["workloads"].items():
    print(f"=== {w['label']} ===")
    for name, s in w["arms"].items():
      extra = f"  ({s['mbp_per_s']} Mbp/s)" if "mbp_per_s" in s else ""
      print(f"  {name:32s}: median={s['median_ms']:8.3f} ms   IQR={s['iqr_ms']:6.3f} ms{extra}")
  if args.json:
    p = pathlib.Path(args.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote JSON -> {p}")
  if args.svg:
    p = pathlib.Path(args.svg)
    write_svg(report, p)
    print(f"Wrote SVG  -> {p}")


if __name__ == "__main__":
  main()
