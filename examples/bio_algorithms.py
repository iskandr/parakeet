"""Classic bioinformatics algorithms compiled in Parakeet.

Covers seven foundational algorithms across sequence alignment, sketching,
compressed string indexing, probabilistic sequence modeling, and structural
biology:

1. 3-state affine-gap Gotoh global alignment with traceback -> CIGAR runs
2. Myers O(N * ceil(M/64)) bit-vector Levenshtein edit distance (multi-word uint64)
3. O(N) monotone-deque minimizer sketching over hashed canonical 2-bit k-mers
4. FM-index / BWT backward search (C array + sampled Occ checkpoints + LF-map)
5. CIGAR run-length expansion via par.flatmap (IndexExpand)
6. Profile HMM Viterbi & Forward-Backward posterior decoding in log space
7. Batched Kabsch 3D protein superposition & RMSD using compiled np.linalg.svd

Run:
    python examples/bio_algorithms.py
"""

import numpy as np

import parakeet as par

# ---------------------------------------------------------------------------
# 1. 3-state affine-gap Gotoh global alignment + traceback -> CIGAR (op, len)
#    Ops: 0 = M (match/mismatch), 1 = I (insertion in seq_a), 2 = D (deletion)
# ---------------------------------------------------------------------------


@par.jit
def gotoh_global_align(seq_a, seq_b, match_score=2, mismatch_score=-1, gap_open=-4, gap_extend=-1):
  """3-state affine-gap Gotoh global alignment with full traceback.

  Gap of length ``k >= 1`` costs ``gap_open + k * gap_extend``.
  Returns ``(score, cigar_ops, cigar_lens)``.
  """
  n = len(seq_a)
  m = len(seq_b)
  neg_inf = np.int32(-1_000_000_000)

  M = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)
  Ix = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)  # gap in seq_b (consumes seq_a -> op 1)
  Iy = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)  # gap in seq_a (consumes seq_b -> op 2)

  # Backpointers: 0 -> from M, 1 -> from Ix, 2 -> from Iy
  tb_M = np.zeros((n + 1, m + 1), dtype=np.uint8)
  tb_Ix = np.zeros((n + 1, m + 1), dtype=np.uint8)
  tb_Iy = np.zeros((n + 1, m + 1), dtype=np.uint8)

  M[0, 0] = np.int32(0)
  for i in range(1, n + 1):
    Ix[i, 0] = np.int32(gap_open + i * gap_extend)
    tb_Ix[i, 0] = np.uint8(0 if i == 1 else 1)
  for j in range(1, m + 1):
    Iy[0, j] = np.int32(gap_open + j * gap_extend)
    tb_Iy[0, j] = np.uint8(0 if j == 1 else 2)

  for i in range(1, n + 1):
    ai = seq_a[i - 1]
    for j in range(1, m + 1):
      # State Ix (gap in seq_b, advance i)
      from_m = M[i - 1, j] + np.int32(gap_open + gap_extend)
      from_ix = Ix[i - 1, j] + np.int32(gap_extend)
      if from_m >= from_ix:
        Ix[i, j] = from_m
        tb_Ix[i, j] = np.uint8(0)
      else:
        Ix[i, j] = from_ix
        tb_Ix[i, j] = np.uint8(1)

      # State Iy (gap in seq_a, advance j)
      from_m_y = M[i, j - 1] + np.int32(gap_open + gap_extend)
      from_iy = Iy[i, j - 1] + np.int32(gap_extend)
      if from_m_y >= from_iy:
        Iy[i, j] = from_m_y
        tb_Iy[i, j] = np.uint8(0)
      else:
        Iy[i, j] = from_iy
        tb_Iy[i, j] = np.uint8(2)

      # State M (align ai with bj)
      sub = np.int32(match_score) if ai == seq_b[j - 1] else np.int32(mismatch_score)
      best_prev = M[i - 1, j - 1]
      src = np.uint8(0)
      if Ix[i - 1, j - 1] > best_prev:
        best_prev = Ix[i - 1, j - 1]
        src = np.uint8(1)
      if Iy[i - 1, j - 1] > best_prev:
        best_prev = Iy[i - 1, j - 1]
        src = np.uint8(2)
      M[i, j] = best_prev + sub
      tb_M[i, j] = src

  # Pick best terminal state
  best_score = M[n, m]
  state = np.int64(0)
  if Ix[n, m] > best_score:
    best_score = Ix[n, m]
    state = np.int64(1)
  if Iy[n, m] > best_score:
    best_score = Iy[n, m]
    state = np.int64(2)

  # Traceback into reversed raw ops then run-length encode to CIGAR
  raw_ops = []
  i = n
  j = m
  while i > 0 or j > 0:
    if state == 0:
      raw_ops.append(np.uint8(0))
      state = np.int64(tb_M[i, j])
      i -= 1
      j -= 1
    elif state == 1:
      raw_ops.append(np.uint8(1))
      state = np.int64(tb_Ix[i, j])
      i -= 1
    else:
      raw_ops.append(np.uint8(2))
      state = np.int64(tb_Iy[i, j])
      j -= 1

  n_raw = len(raw_ops)
  cigar_ops = []
  cigar_lens = []
  if n_raw > 0:
    cur_op = raw_ops[n_raw - 1]
    cur_len = np.int64(1)
    for k in range(n_raw - 2, -1, -1):
      op = raw_ops[k]
      if op == cur_op:
        cur_len += 1
      else:
        cigar_ops.append(cur_op)
        cigar_lens.append(cur_len)
        cur_op = op
        cur_len = np.int64(1)
    cigar_ops.append(cur_op)
    cigar_lens.append(cur_len)

  return best_score, np.array(cigar_ops, dtype=np.uint8), np.array(cigar_lens, dtype=np.int64)


# ---------------------------------------------------------------------------
# 2. Myers bit-vector Levenshtein edit distance (multi-word uint64)
# ---------------------------------------------------------------------------


@par.jit
def myers_edit_distance(pattern, text):
  """Exact Levenshtein edit distance in ``O(len(text) * ceil(len(pattern) / 64))``.

  Uses Myers' 1999 bit-parallel algorithm over 64-bit unsigned words, supporting
  arbitrary pattern lengths (single-word fast path and multi-word chaining).
  """
  m = len(pattern)
  n = len(text)
  if m == 0:
    return np.int64(n)
  if n == 0:
    return np.int64(m)

  n_words = (m + 63) // 64
  # Pattern bitmask table: peq[byte, word] has bit b set iff pattern[word*64 + b] == byte
  peq = np.zeros((256, n_words), dtype=np.uint64)
  one = np.uint64(1)
  for idx in range(m):
    w = idx // 64
    b = np.uint64(idx % 64)
    peq[pattern[idx], w] = peq[pattern[idx], w] | (one << b)

  all_ones = ~np.uint64(0)
  pv = np.full(n_words, all_ones, dtype=np.uint64)
  mv = np.zeros(n_words, dtype=np.uint64)

  last_bit = one << np.uint64((m - 1) % 64)
  score = np.int64(m)
  msb_shift = np.uint64(63)

  for j in range(n):
    ch = text[j]
    ph_in = np.uint64(1)  # vertical delta at row 0 is +1 (cost of deleting text prefix)
    mh_in = np.uint64(0)
    carry = np.uint64(0)
    for w in range(n_words):
      eq = peq[ch, w]
      pv_w = pv[w]
      mv_w = mv[w]
      xv = eq | mv_w
      xh_in = eq | mh_in
      # Multi-word addition: (xh_in & pv_w) + pv_w + carry
      a = xh_in & pv_w
      s1 = a + pv_w
      c1 = np.uint64(1) if s1 < a else np.uint64(0)
      s2 = s1 + carry
      c2 = np.uint64(1) if s2 < s1 else np.uint64(0)
      carry = c1 | c2
      xh = (s2 ^ pv_w) | xh_in

      ph = mv_w | ~(xh | pv_w)
      mh = pv_w & xh

      if w == n_words - 1:
        if (ph & last_bit) != np.uint64(0):
          score += 1
        elif (mh & last_bit) != np.uint64(0):
          score -= 1

      ph_out = ph >> msb_shift
      mh_out = mh >> msb_shift
      ph_shifted = (ph << one) | ph_in
      mh_shifted = (mh << one) | mh_in

      pv[w] = mh_shifted | ~(xv | ph_shifted)
      mv[w] = ph_shifted & xv

      ph_in = ph_out
      mh_in = mh_out

  return score


# ---------------------------------------------------------------------------
# 3. O(N) Monotone-deque minimizer sketching over canonical 2-bit k-mers
# ---------------------------------------------------------------------------


def _hash_u64(x):
  """SplitMix64 avalanche hash for uniform minimizer ordering."""
  h = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
  h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
  return h ^ (h >> np.uint64(31))


@par.jit
def minimizer_sketch(seq_bytes, k=15, w=10):
  """Extract ``(w, k)`` canonical minimizers in ``O(len(seq))`` via a monotone deque.

  Returns ``(hashes, positions, strands)`` deduplicated across consecutive windows.
  """
  n = len(seq_bytes)
  if n < k:
    return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

  # Base -> 2-bit code (A=0, C=1, G=2, T=3; invalid=4)
  lut = np.full(256, np.uint8(4), dtype=np.uint8)
  lut[ord("A")] = np.uint8(0)
  lut[ord("a")] = np.uint8(0)
  lut[ord("C")] = np.uint8(1)
  lut[ord("c")] = np.uint8(1)
  lut[ord("G")] = np.uint8(2)
  lut[ord("g")] = np.uint8(2)
  lut[ord("T")] = np.uint8(3)
  lut[ord("t")] = np.uint8(3)

  mask = (np.uint64(1) << np.uint64(2 * k)) - np.uint64(1)
  shift_rc = np.uint64(2 * (k - 1))

  # Circular monotone deque of size w (stores (hash, pos, strand))
  dq_h = np.empty(w, dtype=np.uint64)
  dq_p = np.empty(w, dtype=np.int64)
  dq_s = np.empty(w, dtype=np.uint8)
  head = np.int64(0)
  tail = np.int64(0)
  count = np.int64(0)

  out_h = []
  out_p = []
  out_s = []

  fwd = np.uint64(0)
  rev = np.uint64(0)
  valid_run = np.int64(0)
  last_emitted_pos = np.int64(-1)

  for i in range(n):
    code = lut[seq_bytes[i]]
    if code >= np.uint8(4):
      valid_run = np.int64(0)
      fwd = np.uint64(0)
      rev = np.uint64(0)
      head = np.int64(0)
      tail = np.int64(0)
      count = np.int64(0)
      continue

    c64 = np.uint64(code)
    fwd = ((fwd << np.uint64(2)) | c64) & mask
    rev = (rev >> np.uint64(2)) | ((np.uint64(3) - c64) << shift_rc)
    valid_run += 1

    if valid_run >= k:
      kpos = i - k + 1
      if fwd <= rev:
        canon = fwd
        strand = np.uint8(0)
      else:
        canon = rev
        strand = np.uint8(1)
      h = _hash_u64(canon)

      # Evict expired elements from front of deque
      min_pos = kpos - w + 1
      while count > 0 and dq_p[head] < min_pos:
        head = (head + 1) % w
        count -= 1

      # Pop strictly larger hashes from back (keep earliest on ties -> >)
      while count > 0:
        back = (tail - 1 + w) % w
        if dq_h[back] > h:
          tail = back
          count -= 1
        else:
          break

      dq_h[tail] = h
      dq_p[tail] = kpos
      dq_s[tail] = strand
      tail = (tail + 1) % w
      count += 1

      if valid_run >= k + w - 1:
        best_pos = dq_p[head]
        if best_pos != last_emitted_pos:
          out_h.append(dq_h[head])
          out_p.append(best_pos)
          out_s.append(dq_s[head])
          last_emitted_pos = best_pos

  return np.array(out_h, dtype=np.uint64), np.array(out_p, dtype=np.int64), np.array(out_s, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 4. FM-index / BWT backward search (C array + sampled Occ checkpoints)
# ---------------------------------------------------------------------------


def build_fm_index(text_codes, alphabet_size=5, occ_stride=32):
  """Build compact FM-index arrays ``(bwt, C, occ, sa, occ_stride)`` from integer-coded text.

  ``text_codes`` should end with sentinel ``0`` (strictly smaller than all bases
  ``1..alphabet_size-1``).
  """
  t = np.asarray(text_codes, dtype=np.uint8)
  n = len(t)
  # Suffix array via Python/NumPy sort of suffix rotations (offline index build)
  sa = np.array(sorted(range(n), key=lambda i: t[i:].tobytes()), dtype=np.int64)
  bwt = t[(sa - 1) % n]
  counts = np.bincount(bwt, minlength=alphabet_size).astype(np.int64)
  C = np.zeros(alphabet_size + 1, dtype=np.int64)
  np.cumsum(counts, out=C[1:])
  n_chk = (n // occ_stride) + 1
  occ = np.zeros((n_chk, alphabet_size), dtype=np.int64)
  running = np.zeros(alphabet_size, dtype=np.int64)
  for i in range(n):
    if i % occ_stride == 0:
      occ[i // occ_stride] = running
    running[bwt[i]] += 1
  if n % occ_stride == 0:
    occ[n // occ_stride] = running
  return bwt, C, occ, sa, occ_stride


def _fm_occ(bwt, occ, occ_stride, c, pos):
  chk = pos // occ_stride
  base = occ[chk, c]
  start = chk * occ_stride
  for i in range(start, pos):
    if bwt[i] == c:
      base += 1
  return base


@par.jit
def fm_backward_search_batch(bwt, C, occ, occ_stride, queries_ragged):
  """Batch FM-index backward search over a ``RaggedArray[uint8]`` of queries.

  Returns ``(lo, hi)`` suffix-array intervals (``hi - lo`` is the exact occurrence count).
  """
  n_q = len(queries_ragged)
  n_bwt = len(bwt)
  lo_out = np.empty(n_q, dtype=np.int64)
  hi_out = np.empty(n_q, dtype=np.int64)

  def _search_one(q_idx):
    q = queries_ragged[q_idx]
    lo = np.int64(0)
    hi = np.int64(n_bwt)
    for k in range(len(q) - 1, -1, -1):
      c = np.int64(q[k])
      lo = C[c] + _fm_occ(bwt, occ, occ_stride, c, lo)
      hi = C[c] + _fm_occ(bwt, occ, occ_stride, c, hi)
      if lo >= hi:
        break
    lo_out[q_idx] = lo
    hi_out[q_idx] = hi

  par.parfor(_search_one, n_q)
  return lo_out, hi_out


# ---------------------------------------------------------------------------
# 5. CIGAR run-length expansion via par.flatmap (IndexExpand)
# ---------------------------------------------------------------------------


@par.jit
def expand_cigar(ops, lengths):
  """Expand CIGAR ``(ops, lengths)`` runs into flat per-base operations via ``IndexExpand``."""
  return par.flatmap(lambda op, n: n, lambda op, n, k: op, ops, lengths).values


# ---------------------------------------------------------------------------
# 6. Profile HMM Viterbi & Forward-Backward posterior decoding (log space)
# ---------------------------------------------------------------------------


def _logaddexp2(a, b):
  if a < b:
    hi = b
    lo = a
  else:
    hi = a
    lo = b
  if hi <= -1e29:
    return hi
  return hi + np.log(1.0 + np.exp(lo - hi))


@par.jit
def hmm_viterbi_and_posterior(obs, log_init, log_trans, log_emit):
  """Profile / hidden Markov model Viterbi path + Forward-Backward posteriors.

  - ``obs``: ``(T,)`` int64 observation symbols
  - ``log_init``: ``(S,)`` float64 initial log-probabilities
  - ``log_trans``: ``(S, S)`` float64 transition log-probabilities
  - ``log_emit``: ``(S, V)`` float64 emission log-probabilities

  Returns ``(viterbi_path, viterbi_logp, total_logp, posteriors)``.
  """
  T = len(obs)
  S = len(log_init)
  neg_inf = -1e30

  # Viterbi DP
  vit = np.full((T, S), neg_inf, dtype=np.float64)
  bp = np.zeros((T, S), dtype=np.int64)
  o0 = obs[0]
  for s in range(S):
    vit[0, s] = log_init[s] + log_emit[s, o0]

  for t in range(1, T):
    ot = obs[t]
    for s in range(S):
      best_v = neg_inf
      best_p = np.int64(0)
      for prev in range(S):
        cand = vit[t - 1, prev] + log_trans[prev, s]
        if cand > best_v:
          best_v = cand
          best_p = np.int64(prev)
      vit[t, s] = best_v + log_emit[s, ot]
      bp[t, s] = best_p

  path = np.empty(T, dtype=np.int64)
  best_end = np.int64(0)
  best_end_v = vit[T - 1, 0]
  for s in range(1, S):
    if vit[T - 1, s] > best_end_v:
      best_end_v = vit[T - 1, s]
      best_end = np.int64(s)
  path[T - 1] = best_end
  for t in range(T - 2, -1, -1):
    path[t] = bp[t + 1, path[t + 1]]

  # Forward log-probabilities
  fwd = np.full((T, S), neg_inf, dtype=np.float64)
  for s in range(S):
    fwd[0, s] = log_init[s] + log_emit[s, o0]
  for t in range(1, T):
    ot = obs[t]
    for s in range(S):
      acc = neg_inf
      for prev in range(S):
        acc = _logaddexp2(acc, fwd[t - 1, prev] + log_trans[prev, s])
      fwd[t, s] = acc + log_emit[s, ot]

  total_logp = neg_inf
  for s in range(S):
    total_logp = _logaddexp2(total_logp, fwd[T - 1, s])

  # Backward log-probabilities
  bwd = np.full((T, S), neg_inf, dtype=np.float64)
  for s in range(S):
    bwd[T - 1, s] = 0.0
  for t in range(T - 2, -1, -1):
    onext = obs[t + 1]
    for s in range(S):
      acc = neg_inf
      for nxt in range(S):
        acc = _logaddexp2(acc, log_trans[s, nxt] + log_emit[nxt, onext] + bwd[t + 1, nxt])
      bwd[t, s] = acc

  post = np.exp(fwd + bwd - total_logp)
  return path, best_end_v, total_logp, post


# ---------------------------------------------------------------------------
# 7. Kabsch 3D protein superposition & RMSD using compiled np.linalg.svd
# ---------------------------------------------------------------------------


@par.jit
def kabsch_rmsd(P, Q):
  """Optimal rigid-body Kabsch superposition of ``P`` onto ``Q`` (both ``(N, 3)``).

  Uses compiled ``np.linalg.svd`` and ``np.linalg.det`` with proper-rotation
  reflection correction (``det(R) == +1``). Returns ``(rmsd, R, P_aligned)``.
  """
  mu_p = np.mean(P, axis=0)
  mu_q = np.mean(Q, axis=0)
  Pc = P - mu_p
  Qc = Q - mu_q
  H = Pc.T @ Qc
  U, _, Vt = np.linalg.svd(H)
  R = Vt.T @ U.T
  if np.linalg.det(R) < 0.0:
    Vt2 = Vt.copy()
    Vt2[2, :] = -Vt2[2, :]
    R = Vt2.T @ U.T
  P_aligned = Pc @ R.T + mu_q
  diff = P_aligned - Q
  rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
  return rmsd, R, P_aligned


@par.jit
def batched_kabsch_rmsd(P_batch, Q_ref):
  """Parallel Kabsch RMSD of a batch ``(B, N, 3)`` of structures against ``Q_ref (N, 3)``."""
  B = P_batch.shape[0]
  rmsds = np.empty(B, dtype=np.float64)
  mu_q = np.mean(Q_ref, axis=0)
  Qc = Q_ref - mu_q

  def _one(b):
    P = P_batch[b]
    Pc = P - np.mean(P, axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
      Vt2 = Vt.copy()
      Vt2[2, :] = -Vt2[2, :]
      R = Vt2.T @ U.T
    diff = Pc @ R.T - Qc
    rmsds[b] = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

  par.parfor(_one, B)
  return rmsds


def main():
  # 1. Gotoh alignment + CIGAR expansion
  a = np.frombuffer(b"ACGTACGTAAAACGT", dtype=np.uint8)
  b = np.frombuffer(b"ACGTCGTAAAAGTCGT", dtype=np.uint8)
  score, ops, lens = gotoh_global_align(a, b)
  flat_ops = expand_cigar(ops, lens)
  assert len(flat_ops) == int(np.sum(lens))
  print(f"Gotoh score={int(score)}, CIGAR runs={len(ops)}, expanded len={len(flat_ops)}")

  # 2. Myers bit-vector edit distance (including >64 bp multi-word pattern!)
  p = np.frombuffer(b"ACGT" * 25 + b"TTAA", dtype=np.uint8)
  t = np.frombuffer(b"ACGT" * 12 + b"GG" + b"ACGT" * 13 + b"TTGA", dtype=np.uint8)
  ed = int(myers_edit_distance(p, t))
  assert ed == 3, ed
  print(f"Myers edit distance (104bp): {ed}")

  # 3. Minimizer sketch
  rng = np.random.default_rng(0)
  bases = np.frombuffer(b"ACGT", dtype=np.uint8)
  seq = bases[rng.integers(0, 4, size=500)]
  mh, mp, _ = minimizer_sketch(seq, 15, 10)
  assert len(mh) > 0 and np.all(np.diff(mp) > 0)
  print(f"Minimizer sketch (500bp, k=15, w=10): {len(mh)} minimizers")

  # 4. FM-index backward search
  text_codes = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 0], dtype=np.uint8)
  bwt, C, occ, _, stride = build_fm_index(text_codes)
  queries = par.RaggedArray.from_sequences([np.array([1, 2, 3], dtype=np.uint8), np.array([4, 1], dtype=np.uint8)])
  lo, hi = fm_backward_search_batch(bwt, C, occ, stride, queries)
  assert list(hi - lo) == [2, 2], (lo, hi)
  print(f"FM-index batch counts: {(hi - lo).tolist()}")

  # 5. Profile HMM Viterbi + Forward-Backward
  obs = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
  log_init = np.log(np.array([0.5, 0.5]))
  log_trans = np.log(np.array([[0.8, 0.2], [0.2, 0.8]]))
  log_emit = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
  path, vlp, tlp, post = hmm_viterbi_and_posterior(obs, log_init, log_trans, log_emit)
  np.testing.assert_allclose(post.sum(axis=1), np.ones(len(obs)), rtol=1e-10)
  assert list(path) == [0, 0, 1, 1, 1, 0, 0]
  print(f"HMM Viterbi path={path.tolist()}, logP={float(tlp):.3f}")

  # 6. Kabsch 3D RMSD
  Q = rng.standard_normal((128, 3))
  theta = 0.7
  R_true = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
  P = Q @ R_true.T + np.array([3.5, -2.1, 1.4])
  rmsd, R_est, _ = kabsch_rmsd(P, Q)
  assert float(rmsd) < 1e-12, rmsd
  print(f"Kabsch 3D RMSD (exact rigid transform): {float(rmsd):.2e}")


if __name__ == "__main__":
  main()
