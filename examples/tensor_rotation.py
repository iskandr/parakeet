"""Rotate a rank-4 tensor: T'[i,j,k,l] = sum g[a,i] g[b,j] g[c,k] g[d,l] T[a,b,c,d].

Shown in four styles: ``np.einsum`` (``_numpy``), comprehension (``_comp``),
explicit loops (``_loop``), and ``par.imap`` (``_par``).
"""

import numpy as np

import parakeet as par


@par.jit
def rotate_numpy(T, g):
  return np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, T)


@par.jit
def rotate_comp(T, g):
  def element(i, j, k, m):
    outer = (
      g[:, i][:, None, None, None]
      * g[:, j][None, :, None, None]
      * g[:, k][None, None, :, None]
      * g[:, m][None, None, None, :]
    )
    return np.sum(outer * T)

  return np.array([[[[element(i, j, k, m) for m in range(3)] for k in range(3)] for j in range(3)] for i in range(3)])


@par.jit
def rotate_loop(T, g):
  out = np.zeros_like(T)
  for i in range(3):
    for j in range(3):
      for k in range(3):
        for m in range(3):
          total = 0.0
          for a in range(3):
            for b in range(3):
              for c in range(3):
                for d in range(3):
                  total += g[a, i] * g[b, j] * g[c, k] * g[d, m] * T[a, b, c, d]
          out[i, j, k, m] = total
  return out


@par.jit
def rotate_par(T, g):
  def element(i, j, k, m):
    outer = (
      g[:, i][:, None, None, None]
      * g[:, j][None, :, None, None]
      * g[:, k][None, None, :, None]
      * g[:, m][None, None, None, :]
    )
    return np.sum(outer * T)

  return par.imap(element, (3, 3, 3, 3))


def main():
  rng = np.random.default_rng(4)
  T, g = rng.normal(size=(3, 3, 3, 3)), rng.normal(size=(3, 3))
  ref = np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, T)
  for fn in (rotate_numpy, rotate_comp, rotate_loop, rotate_par):
    assert np.allclose(fn(T, g), ref), fn.__name__
  return ref


if __name__ == "__main__":
  print(main()[0, 0])
