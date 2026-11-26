"""Compiled linear algebra functions (``np.linalg.*``) in Parakeet's Python subset.

Provides self-contained, unboxed ``cholesky``, ``solve``, ``inv``, ``det``,
``qr``, ``eigh`` (cyclic Jacobi), and ``svd`` (one-sided Hestenes Jacobi)
that compile directly into native C / multicore loops with zero ``PyObject``
or external LAPACK dependency, work in both ``float32`` and ``float64``, and
compose inside ``par.parfor`` / ``par.map`` for batched small matrices (such
as 3x3 Kabsch protein superposition).
"""

import numpy as np

from . import adverbs as _adverbs
from .core import source

__parakeet_lib__ = True


def _as_float_mat(a):
  if a.ndim != 2:
    raise ValueError("linear algebra routine expects a 2-d array")
  if a.dtype == np.float32:
    return a.copy()
  return a.astype(np.float64)


def _lu_factor(a):
  """In-place LU factorization with partial pivoting; returns ``(lu, piv, sign)``."""
  n = a.shape[0]
  if a.shape[1] != n:
    raise ValueError("matrix must be square")
  lu = _as_float_mat(a)
  piv = np.arange(n, dtype=np.int64)
  sign = 1.0
  for k in range(n):
    best_r = k
    best_v = abs(lu[k, k])
    for r in range(k + 1, n):
      v = abs(lu[r, k])
      if v > best_v:
        best_v = v
        best_r = r
    if best_r != k:
      sign = -sign
      tmp_p = piv[k]
      piv[k] = piv[best_r]
      piv[best_r] = tmp_p
      for c in range(n):
        tmp = lu[k, c]
        lu[k, c] = lu[best_r, c]
        lu[best_r, c] = tmp
    piv_val = lu[k, k]
    if piv_val == 0.0:
      continue
    for r in range(k + 1, n):
      factor = lu[r, k] / piv_val
      lu[r, k] = factor
      for c in range(k + 1, n):
        lu[r, c] = lu[r, c] - factor * lu[k, c]
  return lu, piv, sign


def _lu_solve_vec(lu, piv, b):
  n = lu.shape[0]
  y = np.empty(n, dtype=lu.dtype)
  for i in range(n):
    s = b[piv[i]]
    for j in range(i):
      s = s - lu[i, j] * y[j]
    y[i] = s
  x = np.empty(n, dtype=lu.dtype)
  for i in range(n - 1, -1, -1):
    s = y[i]
    for j in range(i + 1, n):
      s = s - lu[i, j] * x[j]
    if lu[i, i] == 0.0:
      raise ValueError("singular matrix")
    x[i] = s / lu[i, i]
  return x


def _cholesky_2d(a):
  if a.ndim != 2 or a.shape[0] != a.shape[1]:
    raise ValueError("cholesky() expects a square 2-d matrix")
  n = a.shape[0]
  af = _as_float_mat(a)
  L = np.zeros((n, n), dtype=af.dtype)
  for i in range(n):
    for j in range(i + 1):
      s = af[i, j]
      for k in range(j):
        s = s - L[i, k] * L[j, k]
      if i == j:
        if s <= 0.0:
          raise ValueError("Matrix is not positive definite")
        L[i, j] = np.sqrt(s)
      else:
        L[i, j] = s / L[j, j]
  return L


@source(np.linalg.cholesky)
def cholesky(a):
  """Lower-triangular Cholesky decomposition ``L @ L.T == a`` (2-d or batched 3-d)."""
  if a.ndim == 3:
    return _adverbs.map(_cholesky_2d, a)
  return _cholesky_2d(a)


def _det_2d(a):
  lu, _, sign = _lu_factor(a)
  n = lu.shape[0]
  prod = lu.dtype.type(sign)
  for i in range(n):
    prod = prod * lu[i, i]
  return prod


@source(np.linalg.det)
def det(a):
  """Determinant of a square 2-d or batched 3-d matrix via LU with partial pivoting."""
  if a.ndim == 3:
    return _adverbs.map(_det_2d, a)
  return _det_2d(a)


def _solve_2d(a, b):
  lu, piv, _ = _lu_factor(a)
  n = lu.shape[0]
  if b.ndim == 1:
    if b.shape[0] != n:
      raise ValueError("solve() shape mismatch")
    bf = b.astype(lu.dtype)
    return _lu_solve_vec(lu, piv, bf)
  if b.ndim == 2:
    if b.shape[0] != n:
      raise ValueError("solve() shape mismatch")
    m = b.shape[1]
    bf = b.astype(lu.dtype)
    out = np.empty((n, m), dtype=lu.dtype)
    for col in range(m):
      out[:, col] = _lu_solve_vec(lu, piv, bf[:, col])
    return out
  raise ValueError("solve() expects a 1-d or 2-d right-hand side")


@source(np.linalg.solve)
def solve(a, b):
  """Solve ``a @ x = b`` for square 2-d or batched 3-d ``a``."""
  if a.ndim == 3:
    return _adverbs.map(_solve_2d, a, b)
  return _solve_2d(a, b)


def _inv_2d(a):
  lu, piv, _ = _lu_factor(a)
  n = lu.shape[0]
  eye = np.eye(n, dtype=lu.dtype)
  out = np.empty((n, n), dtype=lu.dtype)
  for col in range(n):
    out[:, col] = _lu_solve_vec(lu, piv, eye[:, col])
  return out


@source(np.linalg.inv)
def inv(a):
  """Inverse of a square 2-d or batched 3-d matrix via LU with partial pivoting."""
  if a.ndim == 3:
    return _adverbs.map(_inv_2d, a)
  return _inv_2d(a)


def _qr_2d(a, mode="reduced"):
  af = _as_float_mat(a)
  m, n = af.shape[0], af.shape[1]
  k = min(m, n)
  R = af.copy()
  Q = np.eye(m, dtype=af.dtype)
  for j in range(k):
    norm_sq = af.dtype.type(0.0)
    for i in range(j, m):
      norm_sq = norm_sq + R[i, j] * R[i, j]
    norm = np.sqrt(norm_sq)
    if norm == 0.0:
      continue
    r0 = R[j, j]
    alpha = -norm if r0 >= 0.0 else norm
    v0 = r0 - alpha
    v = np.zeros(m, dtype=af.dtype)
    v[j] = v0
    v_sq = v0 * v0
    for i in range(j + 1, m):
      v[i] = R[i, j]
      v_sq = v_sq + R[i, j] * R[i, j]
    if v_sq == 0.0:
      continue
    beta = af.dtype.type(2.0) / v_sq
    for col in range(j, n):
      dot = af.dtype.type(0.0)
      for i in range(j, m):
        dot = dot + v[i] * R[i, col]
      scale = beta * dot
      for i in range(j, m):
        R[i, col] = R[i, col] - scale * v[i]
    for row in range(m):
      dot = af.dtype.type(0.0)
      for i in range(j, m):
        dot = dot + Q[row, i] * v[i]
      scale = beta * dot
      for i in range(j, m):
        Q[row, i] = Q[row, i] - scale * v[i]
  if mode == "complete":
    return Q, R
  return Q[:, :k], R[:k, :]


@source(np.linalg.qr)
def qr(a, mode="reduced"):
  """Householder QR factorization ``a == Q @ R`` (2-d or batched 3-d)."""
  if a.ndim == 3:
    return _adverbs.map(lambda mat: _qr_2d(mat, mode), a)
  return _qr_2d(a, mode)


def _eigh_2d(a, UPLO="L"):
  if a.ndim != 2 or a.shape[0] != a.shape[1]:
    raise ValueError("eigh() expects a square 2-d matrix")
  n = a.shape[0]
  af = _as_float_mat(a)
  dt = af.dtype
  A = np.empty((n, n), dtype=dt)
  use_upper = UPLO == "U" or UPLO == "u"
  for i in range(n):
    A[i, i] = af[i, i]
    for j in range(i + 1, n):
      val = af[i, j] if use_upper else af[j, i]
      A[i, j] = val
      A[j, i] = val
  V = np.eye(n, dtype=dt)
  eps = dt.type(1e-7) if dt == np.float32 else dt.type(1e-15)
  max_sweeps = 60
  for _ in range(max_sweeps):
    max_off = dt.type(0.0)
    for p in range(n):
      for q in range(p + 1, n):
        apq = abs(A[p, q])
        if apq > max_off:
          max_off = apq
    if max_off <= eps:
      break
    for p in range(n):
      for q in range(p + 1, n):
        apq = A[p, q]
        if abs(apq) <= eps * dt.type(0.1):
          continue
        app = A[p, p]
        aqq = A[q, q]
        tau = (aqq - app) / (dt.type(2.0) * apq)
        root = np.sqrt(dt.type(1.0) + tau * tau)
        t = dt.type(1.0) / (tau + root) if tau >= 0.0 else dt.type(-1.0) / (-tau + root)
        c = dt.type(1.0) / np.sqrt(dt.type(1.0) + t * t)
        s = t * c
        A[p, p] = app - t * apq
        A[q, q] = aqq + t * apq
        A[p, q] = dt.type(0.0)
        A[q, p] = dt.type(0.0)
        for r in range(n):
          if r != p and r != q:
            arp = A[r, p]
            arq = A[r, q]
            nrp = c * arp - s * arq
            nrq = s * arp + c * arq
            A[r, p] = nrp
            A[p, r] = nrp
            A[r, q] = nrq
            A[q, r] = nrq
        for r in range(n):
          vrp = V[r, p]
          vrq = V[r, q]
          V[r, p] = c * vrp - s * vrq
          V[r, q] = s * vrp + c * vrq
  w = np.empty(n, dtype=dt)
  for i in range(n):
    w[i] = A[i, i]
  for i in range(n):
    best = i
    for j in range(i + 1, n):
      if w[j] < w[best]:
        best = j
    if best != i:
      tw = w[i]
      w[i] = w[best]
      w[best] = tw
      for r in range(n):
        tv = V[r, i]
        V[r, i] = V[r, best]
        V[r, best] = tv
  return w, V


@source(np.linalg.eigh)
def eigh(a, UPLO="L"):
  """Symmetric eigendecomposition ``a == V @ diag(w) @ V.T`` (2-d or batched 3-d)."""
  if a.ndim == 3:
    return _adverbs.map(lambda mat: _eigh_2d(mat, UPLO), a)
  return _eigh_2d(a, UPLO)


@source(np.linalg.eigvalsh)
def eigvalsh(a, UPLO="L"):
  """Eigenvalues of a symmetric 2-d or batched 3-d matrix in ascending order."""
  if a.ndim == 3:
    return _adverbs.map(lambda mat: _eigh_2d(mat, UPLO)[0], a)
  w, _ = _eigh_2d(a, UPLO)
  return w


def _svd_tall(af, full_matrices):
  """One-sided Hestenes Jacobi SVD for ``m >= n``; returns ``(U, S, Vt)``."""
  m, n = af.shape[0], af.shape[1]
  dt = af.dtype
  B = af.copy()
  V = np.eye(n, dtype=dt)
  eps = dt.type(1e-6) if dt == np.float32 else dt.type(1e-14)
  for _ in range(60):
    converged = True
    for p in range(n):
      for q in range(p + 1, n):
        alpha = dt.type(0.0)
        beta = dt.type(0.0)
        gamma = dt.type(0.0)
        for i in range(m):
          bp = B[i, p]
          bq = B[i, q]
          alpha = alpha + bp * bp
          beta = beta + bq * bq
          gamma = gamma + bp * bq
        denom = np.sqrt(alpha * beta)
        if denom == 0.0 or abs(gamma) <= eps * denom:
          continue
        converged = False
        zeta = (beta - alpha) / (dt.type(2.0) * gamma)
        root = np.sqrt(dt.type(1.0) + zeta * zeta)
        t = dt.type(1.0) / (zeta + root) if zeta >= 0.0 else dt.type(-1.0) / (-zeta + root)
        c = dt.type(1.0) / np.sqrt(dt.type(1.0) + t * t)
        s = c * t
        for i in range(m):
          bp = B[i, p]
          bq = B[i, q]
          B[i, p] = c * bp - s * bq
          B[i, q] = s * bp + c * bq
        for i in range(n):
          vp = V[i, p]
          vq = V[i, q]
          V[i, p] = c * vp - s * vq
          V[i, q] = s * vp + c * vq
    if converged:
      break

  S = np.empty(n, dtype=dt)
  for j in range(n):
    nsq = dt.type(0.0)
    for i in range(m):
      nsq = nsq + B[i, j] * B[i, j]
    S[j] = np.sqrt(nsq)

  # Sort singular values descending
  for i in range(n):
    best = i
    for j in range(i + 1, n):
      if S[j] > S[best]:
        best = j
    if best != i:
      ts = S[i]
      S[i] = S[best]
      S[best] = ts
      for r in range(m):
        tb = B[r, i]
        B[r, i] = B[r, best]
        B[r, best] = tb
      for r in range(n):
        tv = V[r, i]
        V[r, i] = V[r, best]
        V[r, best] = tv

  u_cols = m if full_matrices else n
  U = np.zeros((m, u_cols), dtype=dt)
  tol = eps * (S[0] if n > 0 else dt.type(1.0))
  n_valid = 0
  for j in range(n):
    if S[j] > tol:
      inv_s = dt.type(1.0) / S[j]
      for i in range(m):
        U[i, j] = B[i, j] * inv_s
      n_valid = j + 1
  # Complete orthonormal basis for any zero singular vectors / extra full_matrices columns
  for col in range(n_valid, u_cols):
    best_e = 0
    best_norm = dt.type(-1.0)
    cand = np.zeros(m, dtype=dt)
    for e in range(m):
      for i in range(m):
        cand[i] = dt.type(1.0) if i == e else dt.type(0.0)
      for prev in range(col):
        dot = dt.type(0.0)
        for i in range(m):
          dot = dot + U[i, prev] * cand[i]
        for i in range(m):
          cand[i] = cand[i] - dot * U[i, prev]
      nsq = dt.type(0.0)
      for i in range(m):
        nsq = nsq + cand[i] * cand[i]
      if nsq > best_norm:
        best_norm = nsq
        best_e = e
    for i in range(m):
      cand[i] = dt.type(1.0) if i == best_e else dt.type(0.0)
    # Two-pass modified Gram-Schmidt for full orthogonality
    for _ in range(2):
      for prev in range(col):
        dot = dt.type(0.0)
        for i in range(m):
          dot = dot + U[i, prev] * cand[i]
        for i in range(m):
          cand[i] = cand[i] - dot * U[i, prev]
    nsq = dt.type(0.0)
    for i in range(m):
      nsq = nsq + cand[i] * cand[i]
    inv_n = dt.type(1.0) / np.sqrt(nsq) if nsq > 0.0 else dt.type(0.0)
    for i in range(m):
      U[i, col] = cand[i] * inv_n

  return U, S, V.T.copy()


def _svd_2d(a, full_matrices=True, compute_uv=True):
  af = _as_float_mat(a)
  m, n = af.shape[0], af.shape[1]
  if m >= n:
    U, S, Vt = _svd_tall(af, full_matrices)
    if compute_uv:
      return U, S, Vt
    return S
  U_t, S, Vt_t = _svd_tall(af.T.copy(), full_matrices)
  if compute_uv:
    return Vt_t.T.copy(), S, U_t.T.copy()
  return S


@source(np.linalg.svd)
def svd(a, full_matrices=True, compute_uv=True):
  """Singular value decomposition ``a == U @ diag(S) @ Vt`` (2-d or batched 3-d)."""
  if a.ndim == 3:
    m, n = a.shape[1], a.shape[2]
    if m >= n:
      if compute_uv:
        return _adverbs.map(lambda mat: _svd_tall(_as_float_mat(mat), full_matrices), a)
      return _adverbs.map(lambda mat: _svd_tall(_as_float_mat(mat), full_matrices)[1], a)
    if compute_uv:

      def _wide_uv(mat):
        U_t, S, Vt_t = _svd_tall(_as_float_mat(mat).T.copy(), full_matrices)
        return Vt_t.T.copy(), S, U_t.T.copy()

      return _adverbs.map(_wide_uv, a)
    return _adverbs.map(lambda mat: _svd_tall(_as_float_mat(mat).T.copy(), full_matrices)[1], a)
  return _svd_2d(a, full_matrices, compute_uv)


def _pinv_2d(a, rcond=1e-15):
  U, S, Vt = _svd_2d(a, False, True)
  dt = S.dtype
  cutoff = dt.type(rcond) * (S[0] if len(S) > 0 else dt.type(0.0))
  inv_s = np.where(S > cutoff, dt.type(1.0) / np.where(S > cutoff, S, dt.type(1.0)), dt.type(0.0))
  return (Vt.T * inv_s[None, :]) @ U.T


@source(np.linalg.pinv)
def pinv(a, rcond=1e-15):
  """Moore-Penrose pseudoinverse via SVD (2-d or batched 3-d)."""
  if a.ndim == 3:
    return _adverbs.map(lambda mat: _pinv_2d(mat, rcond), a)
  return _pinv_2d(a, rcond)


@source(np.linalg.lstsq)
def lstsq(a, b, rcond=-1.0):
  """Minimum-norm least-squares solution to ``a @ x = b`` via SVD.

  Returns ``(x, residuals, rank, s)`` matching NumPy's tuple contract.
  """
  U, S, Vt = _svd_2d(a, False, True)
  dt = S.dtype
  m, n = a.shape[0], a.shape[1]
  eps = dt.type(1.1920929e-7) if dt == np.float32 else dt.type(2.220446049250313e-16)
  rc = dt.type(max(m, n)) * eps if rcond < 0.0 else dt.type(rcond)
  cutoff = rc * (S[0] if len(S) > 0 else dt.type(0.0))
  mask = S > cutoff
  rank = np.sum(mask.astype(np.int64))
  inv_s = np.where(mask, dt.type(1.0) / np.where(mask, S, dt.type(1.0)), dt.type(0.0))
  bf = b.astype(dt)
  if bf.ndim == 1:
    x = Vt.T @ (inv_s * (U.T @ bf))
    diff = a.astype(dt) @ x - bf
    res = np.array([np.sum(diff * diff)], dtype=dt) if m > n and rank == n else np.empty(0, dtype=dt)
    return x, res, rank, S
  x = Vt.T @ (inv_s[:, None] * (U.T @ bf))
  diff = a.astype(dt) @ x - bf
  res = np.sum(diff * diff, axis=0) if m > n and rank == n else np.empty(0, dtype=dt)
  return x, res, rank, S
