import matplotlib.pyplot as plt
import numpy as np
import testing_helpers

from parakeet import each, allpairs
from PIL import Image

sausage = Image.open("sausage.jpg")
np_sausage = np.array(sausage).astype('float64')

iidxs = np.arange(3, len(np_sausage)-3)
jidxs = np.arange(3, len(np_sausage[0])-3)
didxs = np.arange(3)

a = [0.00000067, 0.00002292, 0.00019117, 0.00038771, 
     0.00019117, 0.00002292, 0.00000067]
b = [0.00002292, 0.00078633, 0.00655965, 0.01330373, 
     0.00655965, 0.00078633, 0.00002292]
c = [0.00019117, 0.00655965, 0.05472157, 0.11098164, 
     0.05472157, 0.00655965, 0.00019117]
d = [0.00038771, 0.01330373, 0.11098164, 0.22508352,
     0.11098164, 0.01330373, 0.00038771]

gaussian = np.array([a,b,c,d,c,b,a])

def gaussian_7x7(i, j):
  window = np_sausage[i-3:i+4, j-3:j+4, :]
  red = 0.0
  green = 0.0 
  blue = 0.0
  for it in range(0,7,1):
    for jt in range(0,7,1):
      red = red + window[it,jt,0] * gaussian[it,jt]
      green = green + window[it,jt,1] * gaussian[it,jt]
      blue = blue + window[it,jt,2] * gaussian[it,jt]
  return [red, green, blue]
  
def np_blur(start, stop):
  def do_row(i):
    def do_col(j):
      return gaussian_7x7(i, j)
    return np.array(map(do_col, jidxs[start:stop]))
  return np.array(map(do_row, iidxs[start:stop]))

def par_blur():
  return allpairs(gaussian_7x7, iidxs, jidxs)

plot = False
def test_blur():
  np_blurred_upper_left = np_blur(0,10).astype(np.uint8)
  np_blurred_lower_right = np_blur(-10,None).astype(np.uint8)
  par_blurred = par_blur().astype(np.uint8)
  if plot:
    par_imgplot = plt.imshow(par_blurred)
    plt.show(par_imgplot)
  else:
    assert testing_helpers.eq(np_blurred_upper_left, par_blurred[:10,:10]), \
        "Expected (upper left) %s but got %s" % \
        (np_blurred_upper_left, par_blurred[:10, :10])
    assert testing_helpers.eq(np_blurred_lower_right, par_blurred[-10:, -10:]), \
        "Expected (lower right) %s but got %s" % \
        (np_blurred_lower_right, par_blurred[-10:, -10:])
    assert np.sum(np.sum( (par_blurred - np_sausage[3:-3, 3:-3]) ** 2)) > 0

if __name__ == '__main__':
  testing_helpers.run_local_tests()
