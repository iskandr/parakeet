"""Rotate a rank-4 tensor: T'[i,j,k,l] = sum g[ii,i] g[jj,j] g[kk,k] g[ll,l] T[ii,jj,kk,ll].

From http://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy
"""

import numpy as np

from parakeet import jit


@jit
def rotate(T, g):
  def element(i, j, k, l):  # noqa: E741
    total = 0.0
    for ii in range(3):
      for jj in range(3):
        for kk in range(3):
          for ll in range(3):
            total += g[ii, i] * g[jj, j] * g[kk, k] * g[ll, l] * T[ii, jj, kk, ll]
    return total

  return np.array(
    [[[[element(i, j, k, l) for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]  # noqa: E741
  )


def main():
  rng = np.random.default_rng(4)
  T, g = rng.normal(size=(3, 3, 3, 3)), rng.normal(size=(3, 3))
  out = rotate(T, g)
  assert np.allclose(out, np.einsum("ai,bj,ck,dl,abcd->ijkl", g, g, g, g, T))
  return out


if __name__ == "__main__":
  print(main()[0, 0])
