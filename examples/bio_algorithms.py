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

from enum import IntEnum

import numpy as np

import parakeet as par

# ---------------------------------------------------------------------------
# 1. 3-state affine-gap Gotoh global alignment + traceback -> CIGAR (op, len)
# ---------------------------------------------------------------------------


class CigarOp(IntEnum):
  MATCH = 0  # match / mismatch (consumes seq_a and seq_b)
  INS = 1  # gap in seq_b (consumes seq_a)
  DEL = 2  # gap in seq_a (consumes seq_b)


@par.jit
def gotoh_global_align(seq_a, seq_b, match_score=2, mismatch_score=-1, gap_open=-4, gap_extend=-1):
  """3-state affine-gap Gotoh global alignment with full traceback.

  Gap of length ``k >= 1`` costs ``gap_open + k * gap_extend``.
  Returns ``(score, cigar_ops, cigar_lens)``.
  """
  n, m = len(seq_a), len(seq_b)
  neg_inf = np.int32(-1_000_000_000)

  M = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)
  Ix = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)
  Iy = np.full((n + 1, m + 1), neg_inf, dtype=np.int32)

  tb_M = np.zeros((n + 1, m + 1), dtype=np.uint8)
  tb_Ix = np.zeros((n + 1, m + 1), dtype=np.uint8)
  tb_Iy = np.zeros((n + 1, m + 1), dtype=np.uint8)

  M[0, 0] = 0
  for i in range(1, n + 1):
    Ix[i, 0] = gap_open + i * gap_extend
    tb_Ix[i, 0] = CigarOp.MATCH if i == 1 else CigarOp.INS
  for j in range(1, m + 1):
    Iy[0, j] = gap_open + j * gap_extend
    tb_Iy[0, j] = CigarOp.MATCH if j == 1 else CigarOp.DEL

  for i in range(1, n + 1):
    ai = seq_a[i - 1]
    for j in range(1, m + 1):
      # State Ix (gap in seq_b, advance i)
      from_m = M[i - 1, j] + (gap_open + gap_extend)
      from_ix = Ix[i - 1, j] + gap_extend
      if from_m >= from_ix:
        Ix[i, j], tb_Ix[i, j] = from_m, CigarOp.MATCH
      else:
        Ix[i, j], tb_Ix[i, j] = from_ix, CigarOp.INS

      # State Iy (gap in seq_a, advance j)
      from_m_y = M[i, j - 1] + (gap_open + gap_extend)
      from_iy = Iy[i, j - 1] + gap_extend
      if from_m_y >= from_iy:
        Iy[i, j], tb_Iy[i, j] = from_m_y, CigarOp.MATCH
      else:
        Iy[i, j], tb_Iy[i, j] = from_iy, CigarOp.DEL

      # State M (align ai with seq_b[j - 1])
      sub = match_score if ai == seq_b[j - 1] else mismatch_score
      best_prev, src = M[i - 1, j - 1], CigarOp.MATCH
      if Ix[i - 1, j - 1] > best_prev:
        best_prev, src = Ix[i - 1, j - 1], CigarOp.INS
      if Iy[i - 1, j - 1] > best_prev:
        best_prev, src = Iy[i - 1, j - 1], CigarOp.DEL
      M[i, j], tb_M[i, j] = best_prev + sub, src

  # Pick best terminal state
  best_score, state = M[n, m], int(CigarOp.MATCH)
  if Ix[n, m] > best_score:
    best_score, state = Ix[n, m], int(CigarOp.INS)
  if Iy[n, m] > best_score:
    best_score, state = Iy[n, m], int(CigarOp.DEL)

  # Traceback into reversed raw ops then run-length encode to CIGAR
  raw_ops = []
  i, j = n, m
  while i > 0 or j > 0:
    if state == CigarOp.MATCH:
      raw_ops.append(np.uint8(CigarOp.MATCH))
      state = int(tb_M[i, j])
      i -= 1
      j -= 1
    elif state == CigarOp.INS:
      raw_ops.append(np.uint8(CigarOp.INS))
      state = int(tb_Ix[i, j])
      i -= 1
    else:
      raw_ops.append(np.uint8(CigarOp.DEL))
      state = int(tb_Iy[i, j])
      j -= 1

  cigar_ops = []
  cigar_lens = []
  if len(raw_ops) > 0:
    cur_op = raw_ops[len(raw_ops) - 1]
    cur_len = 1
    for k in range(len(raw_ops) - 2, -1, -1):
      op = raw_ops[k]
      if op == cur_op:
        cur_len += 1
      else:
        cigar_ops.append(cur_op)
        cigar_lens.append(cur_len)
        cur_op, cur_len = op, 1
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
  m, n = len(pattern), len(text)
  if m == 0:
    return n
  if n == 0:
    return m

  n_words = (m + 63) // 64
  peq = np.zeros((256, n_words), dtype=np.uint64)
  one = np.uint64(1)
  for idx in range(m):
    peq[pattern[idx], idx // 64] |= one << np.uint64(idx % 64)

  pv = np.full(n_words, ~np.uint64(0), dtype=np.uint64)
  mv = np.zeros(n_words, dtype=np.uint64)

  last_bit = one << np.uint64((m - 1) % 64)
  score = m
  msb_shift = np.uint64(63)

  for ch in text:
    ph_in, mh_in, carry = one, np.uint64(0), np.uint64(0)
    for w in range(n_words):
      eq = peq[ch, w]
      pv_w, mv_w = pv[w], mv[w]
      xv = eq | mv_w
      xh_in = eq | mh_in

      # Multi-word addition: (xh_in & pv_w) + pv_w + carry
      a = xh_in & pv_w
      s1 = a + pv_w
      s2 = s1 + carry
      carry = np.uint64(1) if (s1 < a or s2 < s1) else np.uint64(0)
      xh = (s2 ^ pv_w) | xh_in

      ph = mv_w | ~(xh | pv_w)
      mh = pv_w & xh

      if w == n_words - 1:
        if ph & last_bit:
          score += 1
        elif mh & last_bit:
          score -= 1

      ph_out, mh_out = ph >> msb_shift, mh >> msb_shift
      ph_shifted = (ph << one) | ph_in
      mh_shifted = (mh << one) | mh_in

      pv[w] = mh_shifted | ~(xv | ph_shifted)
      mv[w] = ph_shifted & xv
      ph_in, mh_in = ph_out, mh_out

  return score


# ---------------------------------------------------------------------------
# 3. O(N) Monotone-deque minimizer sketching over canonical 2-bit k-mers
# ---------------------------------------------------------------------------

DNA = par.Alphabet("ACGT", unknown=4)


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
  if len(seq_bytes) < k:
    return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)

  mask = (np.uint64(1) << np.uint64(2 * k)) - np.uint64(1)
  shift_rc = np.uint64(2 * (k - 1))

  # Circular monotone deque of size w storing (hash, position, strand)
  dq_h = np.empty(w, dtype=np.uint64)
  dq_p = np.empty(w, dtype=np.int64)
  dq_s = np.empty(w, dtype=np.uint8)
  head = tail = count = 0

  out_h, out_p, out_s = [], [], []
  fwd = rev = np.uint64(0)
  valid_run = 0
  last_emitted_pos = -1

  for i, ch in enumerate(seq_bytes):
    code = DNA[ch]
    if code >= 4:
      valid_run = head = tail = count = 0
      fwd = rev = np.uint64(0)
      continue

    c64 = np.uint64(code)
    fwd = ((fwd << np.uint64(2)) | c64) & mask
    rev = (rev >> np.uint64(2)) | ((np.uint64(3) - c64) << shift_rc)
    valid_run += 1

    if valid_run >= k:
      kpos = i - k + 1
      canon, strand = (fwd, np.uint8(0)) if fwd <= rev else (rev, np.uint8(1))
      h = _hash_u64(canon)

      min_pos = kpos - w + 1
      while count > 0 and dq_p[head] < min_pos:
        head = (head + 1) % w
        count -= 1

      while count > 0:
        back = (tail - 1 + w) % w
        if dq_h[back] > h:
          tail = back
          count -= 1
        else:
          break

      dq_h[tail], dq_p[tail], dq_s[tail] = h, kpos, strand
      tail = (tail + 1) % w
      count += 1

      if valid_run >= k + w - 1 and dq_p[head] != last_emitted_pos:
        out_h.append(dq_h[head])
        out_p.append(dq_p[head])
        out_s.append(dq_s[head])
        last_emitted_pos = dq_p[head]

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
  return occ[chk, c] + np.sum(bwt[chk * occ_stride : pos] == c)


@par.jit
def fm_backward_search_batch(bwt, C, occ, occ_stride, queries):
  """Batch FM-index backward search over a ``RaggedArray[uint8]`` of queries.

  Returns ``(lo, hi)`` suffix-array intervals (``hi - lo`` is the exact occurrence count).
  """
  n_bwt = len(bwt)

  def search_one(q):
    lo, hi = 0, n_bwt
    for k in range(len(q) - 1, -1, -1):
      c = int(q[k])
      lo = C[c] + _fm_occ(bwt, occ, occ_stride, c, lo)
      hi = C[c] + _fm_occ(bwt, occ, occ_stride, c, hi)
      if lo >= hi:
        break
    return lo, hi

  return par.map(search_one, queries)


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


def _logsumexp(x, axis=0):
  m = np.max(x, axis=axis, keepdims=True)
  safe_m = np.where(m <= -1e29, 0.0, m)
  out = np.squeeze(safe_m, axis=axis) + np.log(np.sum(np.exp(x - safe_m), axis=axis))
  return np.where(np.squeeze(m, axis=axis) <= -1e29, -1e30, out)


@par.jit
def hmm_viterbi_and_posterior(obs, log_init, log_trans, log_emit):
  """Profile / hidden Markov model Viterbi path + Forward-Backward posteriors.

  - ``obs``: ``(T,)`` int64 observation symbols
  - ``log_init``: ``(S,)`` float64 initial log-probabilities
  - ``log_trans``: ``(S, S)`` float64 transition log-probabilities
  - ``log_emit``: ``(S, V)`` float64 emission log-probabilities

  Returns ``(viterbi_path, viterbi_logp, total_logp, posteriors)``.
  """
  T, S = len(obs), len(log_init)

  # Viterbi DP
  vit = np.empty((T, S), dtype=np.float64)
  bp = np.zeros((T, S), dtype=np.int64)
  vit[0] = log_init + log_emit[:, obs[0]]
  for t in range(1, T):
    scores = vit[t - 1, :, None] + log_trans
    bp[t] = np.argmax(scores, axis=0)
    vit[t] = np.max(scores, axis=0) + log_emit[:, obs[t]]

  path = np.empty(T, dtype=np.int64)
  path[T - 1] = np.argmax(vit[T - 1])
  best_end_v = vit[T - 1, path[T - 1]]
  for t in range(T - 2, -1, -1):
    path[t] = bp[t + 1, path[t + 1]]

  # Forward log-probabilities
  fwd = np.empty((T, S), dtype=np.float64)
  fwd[0] = log_init + log_emit[:, obs[0]]
  for t in range(1, T):
    fwd[t] = _logsumexp(fwd[t - 1, :, None] + log_trans, axis=0) + log_emit[:, obs[t]]
  total_logp = _logsumexp(fwd[T - 1], axis=0)

  # Backward log-probabilities
  bwd = np.zeros((T, S), dtype=np.float64)
  for t in range(T - 2, -1, -1):
    bwd[t] = _logsumexp(log_trans + (log_emit[:, obs[t + 1]] + bwd[t + 1])[None, :], axis=1)

  post = np.exp(fwd + bwd - total_logp)
  return path, best_end_v, total_logp, post


# ---------------------------------------------------------------------------
# 7. Kabsch 3D protein superposition & RMSD using compiled np.linalg.svd
# ---------------------------------------------------------------------------


@par.jit
def kabsch_rmsd(P, Q):
  """Optimal rigid-body Kabsch superposition of ``P`` onto ``Q`` (both ``(N, 3)``).

  Uses compiled ``np.linalg.svd`` and ``np.linalg.det`` with branchless
  proper-rotation reflection correction (``det(R) == +1``), so ``kabsch_rmsd``
  is differentiable with ``par.grad``. Returns ``(rmsd, R, P_aligned)``.
  """
  mu_p = np.mean(P, axis=0)
  mu_q = np.mean(Q, axis=0)
  Pc, Qc = P - mu_p, Q - mu_q
  U, _, Vt = np.linalg.svd(Pc.T @ Qc)
  sign = np.where(np.linalg.det(Vt.T @ U.T) < 0.0, -1.0, 1.0)
  d = np.array([1.0, 1.0, sign])
  R = (Vt.T * d[None, :]) @ U.T
  P_aligned = Pc @ R.T + mu_q
  diff = P_aligned - Q
  rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
  return rmsd, R, P_aligned


@par.jit
def batched_kabsch_rmsd(P_batch, Q_ref):
  """Parallel Kabsch RMSD of a batch ``(B, N, 3)`` of structures against ``Q_ref (N, 3)``."""
  Qc = Q_ref - np.mean(Q_ref, axis=0)

  def rmsd_one(P):
    Pc = P - np.mean(P, axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    sign = np.where(np.linalg.det(Vt.T @ U.T) < 0.0, -1.0, 1.0)
    d = np.array([1.0, 1.0, sign])
    R = (Vt.T * d[None, :]) @ U.T
    diff = Pc @ R.T - Qc
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))

  return par.map(rmsd_one, P_batch)


# ---------------------------------------------------------------------------
# 8. Differentiable protein backbone Ramachandran dihedrals (phi, psi, omega)
# ---------------------------------------------------------------------------


def _dihedral_angles(p0, p1, p2, p3):
  """Signed torsion angles (radians) for batches of 4-atom quartets ``(K, 3)``."""
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  b1_norm = np.sqrt(np.sum(b1 * b1, axis=-1, keepdims=True))
  b1_unit = b1 / np.maximum(b1_norm, 1e-12)
  # Project b0 and b2 onto the plane perpendicular to the central bond b1
  v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
  w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
  x = np.sum(v * w, axis=-1)
  # Cross product b1_unit x v
  cx = b1_unit[:, 1] * v[:, 2] - b1_unit[:, 2] * v[:, 1]
  cy = b1_unit[:, 2] * v[:, 0] - b1_unit[:, 0] * v[:, 2]
  cz = b1_unit[:, 0] * v[:, 1] - b1_unit[:, 1] * v[:, 0]
  y = cx * w[:, 0] + cy * w[:, 1] + cz * w[:, 2]
  return np.arctan2(y, x)


@par.jit
def backbone_dihedrals(N, CA, C):
  """Compute Ramachandran backbone torsion angles ``(phi, psi, omega)`` from ``(L, 3)`` atoms.

  Returns arrays of length ``L - 1`` (in radians):
  - ``phi[i]`` = dihedral(C[i], N[i+1], CA[i+1], C[i+1])
  - ``psi[i]`` = dihedral(N[i], CA[i], C[i], N[i+1])
  - ``omega[i]`` = dihedral(CA[i], C[i], N[i+1], CA[i+1])
  """
  phi = _dihedral_angles(C[:-1], N[1:], CA[1:], C[1:])
  psi = _dihedral_angles(N[:-1], CA[:-1], C[:-1], N[1:])
  omega = _dihedral_angles(CA[:-1], C[:-1], N[1:], CA[1:])
  return phi, psi, omega


def main():
  # 1. Gotoh alignment + CIGAR expansion
  score, ops, lens = gotoh_global_align(b"ACGTACGTAAAACGT", b"ACGTCGTAAAAGTCGT")
  flat_ops = expand_cigar(ops, lens)
  assert len(flat_ops) == int(np.sum(lens))
  print(f"Gotoh score={int(score)}, CIGAR runs={len(ops)}, expanded len={len(flat_ops)}")

  # 2. Myers bit-vector edit distance (including >64 bp multi-word pattern)
  ed = int(myers_edit_distance(b"ACGT" * 25 + b"TTAA", b"ACGT" * 12 + b"GG" + b"ACGT" * 13 + b"TTGA"))
  assert ed == 3, ed
  print(f"Myers edit distance (104bp): {ed}")

  # 3. Minimizer sketch
  rng = np.random.default_rng(0)
  seq = np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=500)]
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
