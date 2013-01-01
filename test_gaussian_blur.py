import matplotlib.pyplot as plt
import numpy as np
import testing_helpers

from parakeet import reduce, add, each
from PIL import Image

sausage = Image.open("sausage.jpg")
np_sausage = np.array(sausage)
iidxs = np.arange(3, len(np_sausage)-3)
jidxs = np.arange(3, len(np_sausage[0])-3)
didxs = np.arange(3)

a = [0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292,
     0.00000067]
b = [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633,
     0.00002292]
c = [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965,
     0.00019117]
d = [0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373,
     0.00038771]

gaussian = np.array([a,b,c,d,c,b,a])

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

plot = False
if not plot:
  iidxs = np.arange(565,615)
  jidxs = np.arange(737,800)

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
  if not plot:
    np_blurred = np_blur().astype(np.uint8)
  par_blurred = par_blur().astype(np.uint8)
  if plot:
    par_imgplot = plt.imshow(par_blurred)
    plt.show(par_imgplot)
  else:
    assert testing_helpers.eq(np_blurred, par_blurred), \
        "Expected %s but got %s" % (np_blurred, par_blurred)

if __name__ == '__main__':
  testing_helpers.run_local_ts()
