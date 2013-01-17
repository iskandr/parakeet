import matplotlib.pyplot as plt
import numpy as np
import testing_helpers

from parakeet import each
from PIL import Image

sausage = Image.open("sausage.jpg")
np_sausage = np.array(sausage)
height = len(np_sausage)
width = len(np_sausage[0])

#new = np_sausage.copy()
#for i in range(100):
#  new = np.append(new, np_sausage).reshape((i+2)*height, width, 3)
#np_sausage = new

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

def gaussian_7x7(i, j):
  out = np.array([0.0, 0.0, 0.0])
  for it in range(0, 7, 1):
    iidx = i + it - 3
    for jt in range(0, 7, 1):
      jidx = j + jt - 3
      out[0] = out[0] + np_sausage[iidx][jidx][0] * gaussian[it][jt]
      out[1] = out[1] + np_sausage[iidx][jidx][1] * gaussian[it][jt]
      out[2] = out[2] + np_sausage[iidx][jidx][2] * gaussian[it][jt]
  return out

def np_blur():
  def do_row(i):
    def do_col(j):
      return gaussian_7x7(i, j)
    return np.array(map(do_col, jidxs[:100]))
  return np.array(map(do_row, iidxs[:100]))

def par_blur():
  def do_row(i):
    def do_col(j):
      return gaussian_7x7(i, j)
    return map(do_col, jidxs)
  return each(do_row, iidxs)

plot = True
def test_blur():
  np_blurred = np_blur().astype(np.uint8)
  #par_blurred = (255*par_blur()).astype(np.uint8)
  #print par_blurred
  if plot:
    par_imgplot = plt.imshow(np_blurred)
    plt.show(par_imgplot)
  else:
    assert testing_helpers.eq(np_blurred, par_blurred[:10,:10]), \
        "Expected %s but got %s" % (np_blurred, par_blurred)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
