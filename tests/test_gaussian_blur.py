import matplotlib.pyplot as plt
import numpy as np
import time

import testing_helpers

from parakeet import allpairs
from PIL import Image

try:
  sausage = Image.open("../data/sausage.jpg")
except:
  print "Failed to load test image"
  sausage = np.random.random(200,200, 3)

np_sausage = np.array(sausage).astype('float64')
height = len(np_sausage)
width = len(np_sausage[0])

repeat_img = True
if repeat_img:
  new = np_sausage.copy()
  for i in range(10):
    new = np.append(new, np_sausage).reshape((i+2)*height, width, 3)
  np_sausage = new

def gaussian_kernel(size):
  size = int(size)
  x, y = np.mgrid[-size:size+1, -size:size+1]
  g = np.exp(-(x**2/float(size)+y**2/float(size)))
  return g / g.sum()

s = 5
gaussian = gaussian_kernel(s)
iidxs = np.arange(s, len(np_sausage)-s)
jidxs = np.arange(s, len(np_sausage[0])-s)
didxs = np.arange(3)

def gaussian_conv(i, j):
  window = np_sausage[i-s:i+s+1, j-s:j+s+1, :]
  red = 0.0
  green = 0.0
  blue = 0.0
  for it in range(0,2*s+1,1):
    for jt in range(0,2*s+1,1):
      red = red + window[it,jt,0] * gaussian[it,jt]
      green = green + window[it,jt,1] * gaussian[it,jt]
      blue = blue + window[it,jt,2] * gaussian[it,jt]
  return [red, green, blue]

def np_blur(start, stop):
  def do_row(i):
    def do_col(j):
      return gaussian_conv(i, j)
    return np.array(map(do_col, jidxs[start:stop]))
  return np.array(map(do_row, iidxs[start:stop]))

def par_blur():
  return allpairs(gaussian_conv, iidxs, jidxs)

plot = False
def test_blur():
  np_blurred_upper_left = np_blur(0,10).astype(np.uint8)
  np_blurred_lower_right = np_blur(-10,None).astype(np.uint8)

  start = time.time()
  par_blurred = par_blur().astype(np.uint8)
  par_time = time.time() - start
  print "Parakeet total time:", par_time

  #par_blurred_2 = par_blur().astype(np.uint8)
  if plot:
    par_imgplot = plt.imshow(par_blurred)
    plt.show(par_imgplot)
  else:
    assert testing_helpers.eq(np_blurred_upper_left, par_blurred[:10,:10]), \
        "Expected (upper left) %s but got %s" % \
        (np_blurred_upper_left, par_blurred[:10, :10])
    assert testing_helpers.eq(np_blurred_lower_right, par_blurred[-10:,-10:]), \
        "Expected (lower right) %s but got %s" % \
        (np_blurred_lower_right, par_blurred[-10:, -10:])
    assert np.sum(np.sum((par_blurred - np_sausage[s:-s, s:-s]) ** 2)) > 0

if __name__ == '__main__':
  testing_helpers.run_local_tests()
