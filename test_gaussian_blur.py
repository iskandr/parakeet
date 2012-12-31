import matplotlib.pyplot as plt
import numpy as np
import testing_helpers

from numpy import uint8
from parakeet import reduce, add, each
from PIL import Image

sausage = Image.open("sausage.jpg")
np_sausage = np.array(sausage)
iidxs = np.arange(3, len(np_sausage)-3)
jidxs = np.arange(3, len(np_sausage[0])-3)
didxs = np.arange(3)

gaussian = np.array(
  [[0.00000067 ,0.00002292 ,0.00019117 ,0.00038771 ,0.00019117 ,0.00002292 ,0.00000067],
   [0.00002292 ,0.00078633 ,0.00655965 ,0.01330373 ,0.00655965 ,0.00078633 ,0.00002292],
   [0.00019117 ,0.00655965 ,0.05472157 ,0.11098164 ,0.05472157 ,0.00655965 ,0.00019117],
   [0.00038771 ,0.01330373 ,0.11098164 ,0.22508352 ,0.11098164 ,0.01330373 ,0.00038771],
   [0.00019117 ,0.00655965 ,0.05472157 ,0.11098164 ,0.05472157 ,0.00655965 ,0.00019117],
   [0.00002292 ,0.00078633 ,0.00655965 ,0.01330373 ,0.00655965 ,0.00078633 ,0.00002292],
   [0.00000067 ,0.00002292 ,0.00019117 ,0.00038771 ,0.00019117 ,0.00002292 ,0.00000067]])

def gaussian_7x7(i, j, d):
  out = 0.0
  it = i-3
  while it < i+4:
    jt = j-3
    while jt < j+4:
      out = out + np_sausage[it][jt][d]
      jt = jt + 1
    it = it + 1
  return out / 49.0

def np_blur():
  def do_row(i):
    def do_col(j):
      def do_rbg(d):
        return gaussian_7x7(i, j, d)
      return np.array(map(do_rbg, didxs))
    return np.array(map(do_col, jidxs))
  return np.array(map(do_row, iidxs))

def do_col(i, j):
  def do_gaussian(d):
    return gaussian_7x7(i, j, d)
  return each(do_gaussian, didxs)

def do_par_row(i):
  def do_do_col(j):
    return do_col(i, j)
  return each(do_do_col, jidxs)

def par_blur():
  return each(do_par_row, iidxs)

def test_sqr_dist():
  #blurred = np_blur()
  par_blurred = par_blur()
  print par_blurred
  par_blurred = par_blurred.astype(uint8)
  #imgplot = plt.imshow(blurred)
  #plt.show(imgplot)
  par_imgplot = plt.imshow(par_blurred)
  plt.show(par_imgplot)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
