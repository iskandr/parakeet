"""Compiled FASTA and CSV byte parsing with ``par.Alphabet.mask`` and ``par.split_bytes``.

Parses raw byte buffers into ``RaggedArray[uint8]`` lines, counts GC bases via
an L1-resident 256-byte categorical lookup table, and parses float32 scores
with zero ``PyObject`` allocations.

Run:
    python -m examples.fasta_io
"""

from __future__ import annotations

import time

import numpy as np

import parakeet as par

GC = par.Alphabet.mask("GC", ignore_case=True)


@par.jit
def fasta_gc_and_scores(fasta_buf, score_csv_buf):
  """Compute GC fraction from a FASTA byte buffer and mean score from a CSV buffer."""
  lines = par.split_bytes(fasta_buf, ord("\n"), skip_empty=True)
  gc_total = 0
  base_total = 0
  for k in range(len(lines)):
    line = lines[k]
    if len(line) > 0 and line[0] != ">":
      base_total += len(line)
      gc_total += np.sum(GC[line].astype(np.int64))
  scores = par.parse_float32(par.split_bytes(score_csv_buf, ord(","), skip_empty=True))
  return gc_total, base_total, np.mean(scores)


def main():
  seq_line = b"ACGTACGTGCGCATTAGCTA" * 4 + b"\n"
  fasta_buf = (b">read\n" + seq_line) * 2000
  csv_buf = b",".join(f"{0.01 * (i % 100):.2f}".encode() for i in range(50_000))

  gc, bases, mean_s = fasta_gc_and_scores(fasta_buf, csv_buf)
  assert int(bases) == 2000 * 80
  assert int(gc) == 2000 * 40
  assert abs(float(mean_s) - 0.495) < 1e-3

  t0 = time.perf_counter()
  for _ in range(10):
    gc, bases, mean_s = fasta_gc_and_scores(fasta_buf, csv_buf)
  dt = (time.perf_counter() - t0) / 10
  total_mb = (len(fasta_buf) + len(csv_buf)) / 1e6
  print(
    f"FASTA+CSV parse ({total_mb:.2f} MB): gc={int(gc)}/{int(bases)}, "
    f"mean_score={float(mean_s):.4f}, {dt * 1e3:.2f} ms ({total_mb / dt:.1f} MB/s)"
  )
  return int(gc), int(bases), float(mean_s)


if __name__ == "__main__":
  main()
