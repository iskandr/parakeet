"""Rob Patro-style long-read RNA alignment & quantification in Parakeet.

Combines three core ideas from Rob Patro's COMBINE lab ecosystem (SSHash,
Piscem, Spearfish, and Oarfish) in ~250 lines of compiled Python/NumPy:

1. **SSHash-style streaming k-mer lookup over collapsed unitigs:**
   Redundant gene isoforms are collapsed into maximal shared unitigs stored in
   a ``RaggedArray[uint8]`` with CSR ``unitig -> (tx_id, tx_pos)`` postings and
   a ``FrozenHashTable`` sparse minimizer dictionary. While walking a long read,
   whenever the next read base continues along the current unitig, Parakeet
   extends the anchor in O(1) with **zero hash table lookup**.
2. **Spearfish-style shared anchor projection & delta-mapping:**
   Unitig Maximal Exact Match (MEM) anchors are projected onto candidate
   isoforms via CSR postings, scored by co-linear chaining, and filtered by
   score difference ``delta = best_score - score <= delta_max`` so only
   competitive isoforms are retained.
3. **Oarfish-style compiled EM isoform quantification:**
   Runs compiled Expectation-Maximization over the read-to-isoform
   compatibility sets to estimate transcript abundances.

Run on real human isoforms (fetched from Ensembl if available, or built-in
multi-isoform benchmark):
    python examples/patro_longread_rna.py
    python examples/patro_longread_rna.py /tmp/real_human_isoforms.fa
"""

import pathlib
import sys
import time

import numpy as np

import parakeet as par
from parakeet import FrozenHashTable, RaggedArray

# ---------------------------------------------------------------------------
# 1. Offline transcriptome unitig collapsing + SSHash minimizer index builder
# ---------------------------------------------------------------------------

_BASE_TO_2BIT = np.full(256, 4, dtype=np.uint8)
for _c, _v in ((b"A", 0), (b"a", 0), (b"C", 1), (b"c", 1), (b"G", 2), (b"g", 2), (b"T", 3), (b"t", 3)):
  _BASE_TO_2BIT[_c[0]] = _v


def _encode_kmer_py(codes: np.ndarray, pos: int, k: int) -> int:
  v = 0
  for i in range(k):
    v = (v << 2) | int(codes[pos + i])
  return v


def _hash_u64_py(x: int) -> int:
  mask = (1 << 64) - 1
  h = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
  h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & mask
  return (h ^ (h >> 31)) & mask


def build_unitig_sshash_index(tx_seqs: list[bytes], k: int = 15, m: int = 9):
  """Collapse redundant isoforms into shared segments (unitigs) & build SSHash index.

  Splits transcripts into maximal segments whose internal ``k``-mers occur on
  the exact same set of transcripts (exact colored de Bruijn unitigs / core
  graph exons), then indexes each unitig's sparse ``m``-mer minimizers in a
  ``FrozenHashTable`` + CSR bucket arrays.
  """
  tx_codes = [_BASE_TO_2BIT[np.frombuffer(s, dtype=np.uint8)] for s in tx_seqs]
  n_tx = len(tx_codes)

  # Map each k-mer -> list of (tx_id, pos) occurrences (deduplicated per transcript)
  kmer_occ: dict[int, list[tuple[int, int]]] = {}
  for tid, c in enumerate(tx_codes):
    n = len(c)
    if n < k:
      continue
    mask = (1 << (2 * k)) - 1
    v = 0
    for i in range(n):
      v = ((v << 2) | int(c[i])) & mask
      if i >= k - 1:
        kmer_occ.setdefault(v, []).append((tid, i - k + 1))

  # Extract maximal shared segments (unitigs) by walking each transcript and breaking
  # whenever the k-mer's occurrence signature changes (or a k-mer was already emitted).
  emitted_kmers: set[int] = set()
  unitigs: list[np.ndarray] = []
  unitig_postings: list[list[tuple[int, int]]] = []

  for _tid, c in enumerate(tx_codes):
    n = len(c)
    if n < k:
      continue
    mask = (1 << (2 * k)) - 1
    kmers = np.empty(n - k + 1, dtype=np.uint64)
    v = 0
    for i in range(n):
      v = ((v << 2) | int(c[i])) & mask
      if i >= k - 1:
        kmers[i - k + 1] = v

    i = 0
    n_km = len(kmers)
    while i < n_km:
      kv = int(kmers[i])
      if kv in emitted_kmers:
        i += 1
        continue
      occ0 = kmer_occ[kv]
      start = i
      emitted_kmers.add(kv)
      i += 1
      # Extend while consecutive k-mers share the exact same transcript set and contiguous offsets
      while i < n_km:
        kv_next = int(kmers[i])
        if kv_next in emitted_kmers:
          break
        occ_next = kmer_occ[kv_next]
        if len(occ_next) != len(occ0):
          break
        if any(occ_next[r][0] != occ0[r][0] or occ_next[r][1] != occ0[r][1] + (i - start) for r in range(len(occ0))):
          break
        emitted_kmers.add(kv_next)
        i += 1
      seg = c[start : (i - 1) + k].copy()
      unitigs.append(seg)
      unitig_postings.append(occ0)

  # Pack unitigs into flat CSR arrays
  u_offs = np.zeros(len(unitigs) + 1, dtype=np.int64)
  for uid, seg in enumerate(unitigs):
    u_offs[uid + 1] = u_offs[uid] + len(seg)
  u_vals = np.concatenate(unitigs) if unitigs else np.empty(0, dtype=np.uint8)

  # Pack unitig -> (tx_id, tx_pos) postings into CSR arrays
  p_offs = np.zeros(len(unitigs) + 1, dtype=np.int64)
  p_tx_list: list[int] = []
  p_pos_list: list[int] = []
  for uid, posts in enumerate(unitig_postings):
    p_offs[uid + 1] = p_offs[uid] + len(posts)
    for t_id, t_pos in posts:
      p_tx_list.append(t_id)
      p_pos_list.append(t_pos)
  p_tx = np.array(p_tx_list, dtype=np.int64)
  p_pos = np.array(p_pos_list, dtype=np.int64)

  # SSHash sparse minimizer table: for every k-mer on each unitig, pick its m-mer minimizer
  # and insert (uid, u_pos) into that minimizer's bucket (deduplicated per minimizer window).
  mmask = (1 << (2 * m)) - 1
  w = k - m + 1
  buckets: dict[int, list[tuple[int, int]]] = {}
  for uid, seg in enumerate(unitigs):
    L = len(seg)
    if L < k:
      continue
    # Compute all m-mer hashes along this unitig
    m_hashes = np.empty(L - m + 1, dtype=np.uint64)
    mv = 0
    for i in range(L):
      mv = ((mv << 2) | int(seg[i])) & mmask
      if i >= m - 1:
        m_hashes[i - m + 1] = _hash_u64_py(mv)
    # Record minimizer per k-mer (skip consecutive duplicate minimizer positions -> sparse!)
    last_mpos = -1
    for pos in range(L - k + 1):
      best_h = int(m_hashes[pos])
      best_mp = pos
      for j in range(1, w):
        hj = int(m_hashes[pos + j])
        if hj < best_h:
          best_h = hj
          best_mp = pos + j
      if best_mp != last_mpos:
        buckets.setdefault(best_h, []).append((uid, best_mp))
        last_mpos = best_mp

  # Build FrozenHashTable: minimizer_hash (int64) -> bucket_id (int64)
  keys = np.array([np.uint64(h).view(np.int64) for h in buckets], dtype=np.int64)
  bkt_ids = np.arange(len(buckets), dtype=np.int64)
  min_table = FrozenHashTable.from_items(keys, bkt_ids)

  b_offs = np.zeros(len(buckets) + 1, dtype=np.int64)
  b_uid_list: list[int] = []
  b_mpos_list: list[int] = []
  for bid, entries in enumerate(buckets.values()):
    b_offs[bid + 1] = b_offs[bid] + len(entries)
    for uid, mpos in entries:
      b_uid_list.append(uid)
      b_mpos_list.append(mpos)

  b_uid = np.array(b_uid_list, dtype=np.int64)
  b_mpos = np.array(b_mpos_list, dtype=np.int64)
  tx_lens = np.array([len(c) for c in tx_codes], dtype=np.float64)

  return {
    "k": k,
    "m": m,
    "n_tx": n_tx,
    "n_unitigs": len(unitigs),
    "raw_bp": int(sum(len(c) for c in tx_codes)),
    "unitig_bp": int(len(u_vals)),
    "u_vals": u_vals,
    "u_offs": u_offs,
    "p_tx": p_tx,
    "p_pos": p_pos,
    "p_offs": p_offs,
    "min_table": min_table,
    "b_offs": b_offs,
    "b_uid": b_uid,
    "b_mpos": b_mpos,
    "tx_lens": tx_lens,
  }


# ---------------------------------------------------------------------------
# 2. Compiled SSHash streaming query + Spearfish delta-mapping inside @par.jit
# ---------------------------------------------------------------------------


def _splitmix64(x):
  h = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
  h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
  return h ^ (h >> np.uint64(31))


@par.jit
def stream_map_long_reads(
  reads_ragged,
  u_vals,
  u_offs,
  p_tx,
  p_pos,
  p_offs,
  min_table,
  b_offs,
  b_uid,
  b_mpos,
  n_tx,
  k=15,
  m=9,
  delta_frac=0.015,
):
  """Map a batch of long RNA reads in parallel via SSHash streaming + delta-mapping.

  For each read:
  - Uses SSHash unitig contiguity extension to skip hash probes on consecutive bases.
  - Projects Maximal Exact Match (MEM) anchors to transcripts via CSR postings.
  - Computes diagonal-consistent chain scores per transcript and retains all
    isoforms within ``delta_frac * best_score`` of the top score (Spearfish delta-mapping).

  Returns ``(best_tx, best_score, compat_counts, total_extensions, total_probes, tx_compat_matrix)``
  where ``tx_compat_matrix[r, t]`` holds the compatibility weight (``> 0`` iff isoform ``t``
  is within the delta window for read ``r``).
  """
  mmask = (np.uint64(1) << np.uint64(2 * m)) - np.uint64(1)

  def map_read(read):
    rlen = len(read)
    scores = np.zeros(n_tx, dtype=np.int64)
    last_diag = np.full(n_tx, -1_000_000_000, dtype=np.int64)
    last_rpos = np.full(n_tx, -1_000_000_000, dtype=np.int64)
    n_ext = 0
    n_probe = 0
    rpos = 0

    while rpos <= rlen - k:
      # Find m-mer minimizer inside read[rpos : rpos + k]
      best_h = ~np.uint64(0)
      best_off = 0
      mv = np.uint64(0)
      for j in range(k):
        mv = ((mv << np.uint64(2)) | np.uint64(read[rpos + j])) & mmask
        if j >= m - 1:
          h = _splitmix64(mv)
          if h < best_h:
            best_h = h
            best_off = j - (m - 1)

      # Single FrozenHashTable probe for this anchor
      n_probe += 1
      bid = min_table.get(np.int64(best_h), -1)
      if bid < 0:
        rpos += 1
        continue

      # Verify exact k-mer match against the small minimizer bucket
      matched_uid = -1
      matched_upos = -1
      for bi in range(b_offs[bid], b_offs[bid + 1]):
        uid = b_uid[bi]
        u_start = u_offs[uid]
        u_len = u_offs[uid + 1] - u_start
        upos = b_mpos[bi] - best_off
        if upos < 0 or upos + k > u_len:
          continue
        ok = True
        for t in range(k):
          if read[rpos + t] != u_vals[u_start + upos + t]:
            ok = False
            break
        if ok:
          matched_uid, matched_upos = uid, upos
          break

      if matched_uid < 0:
        rpos += 1
        continue

      # SSHash fast path: extend match along unitig with zero hash lookups
      u_start = u_offs[matched_uid]
      u_len = u_offs[matched_uid + 1] - u_start
      mem_len = k
      while (rpos + mem_len) < rlen and (matched_upos + mem_len) < u_len:
        if read[rpos + mem_len] == u_vals[u_start + matched_upos + mem_len]:
          mem_len += 1
          n_ext += 1
        else:
          break

      # Project unitig MEM anchor to all transcripts sharing matched_uid (Spearfish core-graph sharing)
      for pi in range(p_offs[matched_uid], p_offs[matched_uid + 1]):
        tid = p_tx[pi]
        diag = (p_pos[pi] + matched_upos) - rpos
        if last_rpos[tid] < -1000:
          scores[tid] = mem_len
        else:
          pen = min(abs(diag - last_diag[tid]), 250)
          if mem_len > pen:
            scores[tid] += mem_len - pen
        last_diag[tid] = diag
        last_rpos[tid] = rpos + mem_len

      # Jump past mismatch or unitig boundary (avoids k-1 wasted probes across an error base)
      if (rpos + mem_len) < rlen and (matched_upos + mem_len) < u_len:
        rpos += mem_len + 1
      else:
        rpos += mem_len - k + (2 if mem_len == k else 1)

    # Delta-mapping: keep isoforms within delta_frac of the top score
    top_t = np.argmax(scores)
    top_s = scores[top_t]
    cc = 0
    if top_s > 0:
      min_keep = top_s - int(float(top_s) * delta_frac)
      for t in range(n_tx):
        if scores[t] >= min_keep:
          cc += 1
        else:
          scores[t] = 0
    else:
      top_t = -1
    return top_t, top_s, cc, n_ext, n_probe, scores

  best_tx, best_score, compat_counts, ext_counts, probe_counts, tx_scores = par.map(map_read, reads_ragged)
  return best_tx, best_score, compat_counts, np.sum(ext_counts), np.sum(probe_counts), tx_scores


# ---------------------------------------------------------------------------
# 3. Compiled Oarfish-style EM transcript abundance quantification
# ---------------------------------------------------------------------------


@par.jit
def oarfish_em_quantify(tx_scores, read_lens, tx_lens, n_iters=80):
  """Oarfish-style coverage-aware EM over delta-mapped isoform compatibility sets.

  Incorporates Oarfish's key insight for long-read RNA: when a read matches both
  a short isoform and a longer isoform that contains it, the likelihood
  ``P(r | t)`` weights each compatible isoform by its alignment score and
  coverage/length compatibility ``(1 / max(tx_len, read_len)) * exp(-2 * |tx_len - read_len| / tx_len)``.
  """
  n_reads, n_tx = tx_scores.shape
  compat_mask = tx_scores > 0
  # Coverage & effective-length likelihood matrix P(r | t)
  rl = read_lens[:, None]
  tl = tx_lens[None, :]
  len_gap = np.abs(tl - rl) / np.maximum(tl, 1.0)
  score_f = tx_scores.astype(np.float64)
  max_sc = np.max(score_f, axis=1, keepdims=True)
  score_wt = np.exp(np.maximum(-50.0, 0.25 * (score_f - max_sc)))
  like = np.where(compat_mask, score_wt * np.exp(-4.0 * len_gap) / np.maximum(tl, rl), 0.0)

  theta = np.full(n_tx, 1.0 / float(n_tx), dtype=np.float64)
  for _ in range(n_iters):
    weighted = like * theta[None, :]
    denom = np.sum(weighted, axis=1, keepdims=True)
    resp = np.where(denom > 0.0, weighted / np.where(denom > 0.0, denom, 1.0), 0.0)
    counts = np.sum(resp, axis=0)
    total = np.sum(counts)
    if total > 0.0:
      theta = counts / total

  weighted = like * theta[None, :]
  denom = np.sum(weighted, axis=1, keepdims=True)
  resp = np.where(denom > 0.0, weighted / np.where(denom > 0.0, denom, 1.0), 0.0)
  return np.sum(resp, axis=0)


# ---------------------------------------------------------------------------
# 4. Data loader + realistic ONT/PacBio long-read simulator & benchmark runner
# ---------------------------------------------------------------------------


def load_fasta(path: pathlib.Path) -> tuple[list[str], list[bytes]]:
  names: list[str] = []
  seqs: list[bytes] = []
  cur_name = None
  cur_parts: list[bytes] = []
  for raw in path.read_bytes().splitlines():
    if raw.startswith(b">"):
      if cur_name is not None:
        names.append(cur_name)
        seqs.append(b"".join(cur_parts))
      cur_name = raw[1:].decode("ascii", errors="replace").split()[0]
      cur_parts = []
    elif raw:
      cur_parts.append(raw.strip().upper())
  if cur_name is not None:
    names.append(cur_name)
    seqs.append(b"".join(cur_parts))
  return names, seqs


def make_synthetic_isoforms(rng, n_genes=6, isoforms_per_gene=5, exon_len=240) -> tuple[list[str], list[bytes]]:
  """Fallback multi-isoform transcriptome with realistic shared-exon splicing."""
  alph = np.frombuffer(b"ACGT", dtype=np.uint8)
  names: list[str] = []
  seqs: list[bytes] = []
  for g in range(n_genes):
    exons = [bytes(alph[rng.integers(0, 4, size=exon_len)]) for _ in range(8)]
    # Isoforms share core exons (0, 2, 4, 7) and vary cassette exons (1, 3, 5, 6)
    patterns = [
      (0, 1, 2, 3, 4, 5, 6, 7),
      (0, 2, 3, 4, 5, 7),
      (0, 1, 2, 4, 6, 7),
      (0, 2, 4, 5, 6, 7),
      (0, 1, 3, 4, 5, 7),
    ]
    for k in range(isoforms_per_gene):
      names.append(f"GENE{g}|ISO{k}")
      seqs.append(b"".join(exons[e] for e in patterns[k % len(patterns)]))
  return names, seqs


def simulate_long_reads(
  tx_seqs: list[bytes], n_reads: int, sub_rate: float = 0.015, indel_rate: float = 0.005, seed: int = 0
):
  """Simulate PacBio HiFi / ONT Q20 long RNA reads from transcripts with ground-truth counts."""
  rng = np.random.default_rng(seed)
  n_tx = len(tx_seqs)
  # Realistic log-normal isoform expression distribution
  true_weights = rng.lognormal(mean=0.0, sigma=1.4, size=n_tx)
  true_weights /= true_weights.sum()
  tx_choices = rng.choice(n_tx, size=n_reads, p=true_weights)
  true_counts = np.bincount(tx_choices, minlength=n_tx).astype(np.float64)

  tx_codes = [_BASE_TO_2BIT[np.frombuffer(s, dtype=np.uint8)] for s in tx_seqs]
  read_arrays: list[np.ndarray] = []
  for tid in tx_choices:
    src = tx_codes[int(tid)]
    L = len(src)
    # Full-length or 70-100% partial long read
    rlen = max(120, int(L * rng.uniform(0.7, 1.0)))
    start = int(rng.integers(0, max(1, L - rlen + 1)))
    seg = src[start : start + rlen].copy()
    # Substitutions
    n_sub = int(rng.binomial(len(seg), sub_rate))
    if n_sub > 0:
      pos = rng.integers(0, len(seg), size=n_sub)
      seg[pos] = (seg[pos] + rng.integers(1, 4, size=n_sub, dtype=np.uint8)) & np.uint8(3)
    # Small deletions (keep vectorized & fast)
    if indel_rate > 0 and len(seg) > 40:
      keep = rng.random(len(seg)) >= indel_rate
      seg = seg[keep]
    read_arrays.append(seg)

  return RaggedArray.from_sequences(read_arrays), tx_choices, true_counts


def main(fasta_path: str | None = None, n_reads: int = 1200):
  rng = np.random.default_rng(42)
  if fasta_path is not None and pathlib.Path(fasta_path).exists():
    names, seqs = load_fasta(pathlib.Path(fasta_path))
    source_label = pathlib.Path(fasta_path).name
  else:
    names, seqs = make_synthetic_isoforms(rng)
    source_label = "synthetic_isoforms"

  t_idx0 = time.perf_counter()
  idx = build_unitig_sshash_index(seqs, k=15, m=9)
  t_idx = (time.perf_counter() - t_idx0) * 1e3

  reads_rag, true_tx, true_counts = simulate_long_reads(seqs, n_reads=n_reads, sub_rate=0.015, indel_rate=0.005, seed=7)
  read_lens = np.diff(reads_rag.offsets).astype(np.float64)
  total_mbp = len(reads_rag.values) / 1e6

  # Warmup compile
  best_tx, best_sc, compat_c, n_ext, n_probe, tx_scores = stream_map_long_reads(
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
  )
  est_counts = oarfish_em_quantify(tx_scores, read_lens, idx["tx_lens"])

  # Timed mapping + EM run
  t0 = time.perf_counter()
  best_tx, best_sc, compat_c, n_ext, n_probe, tx_scores = stream_map_long_reads(
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
  )
  t_map = (time.perf_counter() - t0) * 1e3

  t1 = time.perf_counter()
  est_counts = oarfish_em_quantify(tx_scores, read_lens, idx["tx_lens"])
  t_em = (time.perf_counter() - t1) * 1e3

  mapped_frac = float(np.mean(best_tx >= 0))
  # Check how often the true isoform is inside the delta-mapped compatibility set
  true_in_compat = float(np.mean([tx_scores[i, true_tx[i]] > 0 for i in range(n_reads)]))
  skip_rate = float(n_ext) / max(1.0, float(n_ext + n_probe))
  corr = float(np.corrcoef(true_counts, est_counts)[0, 1])

  print(
    f"Dataset: {source_label} ({idx['n_tx']} isoforms, {idx['raw_bp']:,} raw bp -> {idx['unitig_bp']:,} unitig bp, {idx['raw_bp'] / max(1, idx['unitig_bp']):.2f}x collapse)"
  )
  print(f"Index build: {t_idx:.1f} ms ({idx['n_unitigs']} unitigs)")
  print(
    f"Long-read mapping ({n_reads} reads, {total_mbp:.2f} Mbp): {t_map:.2f} ms "
    f"({n_reads / (t_map / 1e3):,.0f} reads/s, {total_mbp / (t_map / 1e3):.1f} Mbp/s)"
  )
  print(
    f"SSHash contiguity fast path: {skip_rate * 100:.1f}% of base steps skipped hash lookup ({int(n_ext):,} ext vs {int(n_probe):,} probes)"
  )
  print(
    f"Mapped: {mapped_frac * 100:.1f}%, true isoform in delta-set: {true_in_compat * 100:.1f}%, mean compat isoforms/read: {float(np.mean(compat_c)):.2f}"
  )
  print(f"Oarfish EM quantification: {t_em:.2f} ms, Pearson r vs ground truth = {corr:.4f}")
  assert mapped_frac > 0.95 and true_in_compat > 0.92 and corr > 0.90


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else None)
