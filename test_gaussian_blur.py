import matplotlib.pyplot as plt
import numpy as np
import testing_helpers

from numpy import uint8
from parakeet import reduce, add, each
from PIL import Image

sausage = Image.open("sausage.jpg")
np_sausage = np.array(sausage)

gaussian = np.array(
  [[0.00000067 ,0.00002292 ,0.00019117 ,0.00038771 ,0.00019117 ,0.00002292 ,0.00000067],
   [0.00002292 ,0.00078633 ,0.00655965 ,0.01330373 ,0.00655965 ,0.00078633 ,0.00002292],
   [0.00019117 ,0.00655965 ,0.05472157 ,0.11098164 ,0.05472157 ,0.00655965 ,0.00019117],
   [0.00038771 ,0.01330373 ,0.11098164 ,0.22508352 ,0.11098164 ,0.01330373 ,0.00038771],
   [0.00019117 ,0.00655965 ,0.05472157 ,0.11098164 ,0.05472157 ,0.00655965 ,0.00019117],
   [0.00002292 ,0.00078633 ,0.00655965 ,0.01330373 ,0.00655965 ,0.00078633 ,0.00002292],
   [0.00000067 ,0.00002292 ,0.00019117 ,0.00038771 ,0.00019117 ,0.00002292 ,0.00000067]])

def gaussian_7x7(X, i, j, d):
  out = 0.0
  it = i-3
  while it < i+4:
    jt = j-3
    while jt < j+4:
      out = out + X[it][jt][d]
      jt = jt + 1
    it = it + 1
  return out / 49.0

def test_sqr_dist():
  iidxs = np.arange(3, len(np_sausage)-3)
  jidxs = np.arange(3, len(np_sausage[0])-3)
  didxs = np.arange(3)
  def do_row(i):
    def do_col(j):
      def do_rbg(d):
        return gaussian_7x7(np_sausage, i, j, d)
      return np.array(map(do_rbg, didxs))
    return np.array(map(do_col, jidxs))
  #blurred = np.array(map(do_row, iidxs))
  def par_do_row(i):
    def do_col(j):
      def do_rbg(d):
        return gaussian_7x7(np_sausage, i, j, d)
      return each(do_rbg, didxs)
    return each(do_col, jidxs)
  par_blurred = each(par_do_row, iidxs)
  par_blurred = par_blurred.astype(uint8)
  #imgplot = plt.imshow(blurred)
  #plt.show(imgplot)
  par_imgplot = plt.imshow(par_blurred)
  plt.show(par_imgplot)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
