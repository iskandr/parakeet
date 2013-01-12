#!/usr/bin/python
import numpy as np
import time

from ctypes import *

import runtime

class vm_args_t(Structure):
  _fields_ = [("a", POINTER(c_double)),
              ("b", POINTER(c_double)),
              ("out", POINTER(c_double)),
              ("m", c_int),
              ("n", c_int),
              ("k", c_int)]

class py_c_2d_struct_t(Structure):
  _fields_ = [("d1", c_int64),
              ("d2", c_int64)]

class array_2d_t(Structure):
  _fields_ = [("data", POINTER(c_double)),
              ("shape", POINTER(py_c_2d_struct_t)),
              ("strides", POINTER(py_c_2d_struct_t)),
              ("offset", c_int64),
              ("nelts", c_int64)]

class par_args_t(Structure):
  _fields_ = [("a", POINTER(array_2d_t)),
              ("b", POINTER(array_2d_t)),
              ("o", POINTER(array_2d_t))]

libVM = cdll.LoadLibrary("./vm.so")
libVM.make_array.restype = POINTER(c_double)

#m = 24000
#n = 7200
#k = 1200
m = 5000
n = 5000
k = 5000
a = libVM.make_array(m, k)
b = libVM.make_array(n, k)
o = libVM.make_array(m, n)

ll = False

if ll:
  args_t = POINTER(par_args_t) * 8
  args = args_t()

  def make_ll_2d_array(x, y, p):
    shape = py_c_2d_struct_t(x, y)
    strides = py_c_2d_struct_t(y, 1)
    return pointer(array_2d_t(p, pointer(shape), pointer(strides), 0, x*y))
  a_ll = make_ll_2d_array(m, k, a)
  b_ll = make_ll_2d_array(n, k, b)
  o_ll = make_ll_2d_array(m, n, o)
  arg = pointer(par_args_t(a_ll, b_ll, o_ll))
  for i in range(8):
    args[i] = arg
else:
  args_t = POINTER(vm_args_t) * 8
  args = args_t()
  arg = pointer(vm_args_t(a, b, o, m, n, k))
  for i in range(8):
    args[i] = arg

num_tiles = 3
tile_sizes_t = c_int64 * num_tiles
tile_sizes = tile_sizes_t()
dl_sizes = tile_sizes_t()
dl_sizes[0] = 26
dl_sizes[1] = 26
dl_sizes[2] = 26
ml_sizes = tile_sizes_t()
ml_sizes[0] = 73
ml_sizes[1] = 127
ml_sizes[2] = 127

# fixed:
tile_sizes[0] = 48
tile_sizes[1] = 73
tile_sizes[2] = 73

fn_name = "vm_tiled_unrolled"

r = runtime.Runtime()
print "Launching " + fn_name
start = time.time()

fixed = False
if fixed:
  print "Tile sizes:", list(tile_sizes)
  r.run_job_with_fixed_tiles(getattr(libVM, fn_name),
                             cast(args, c_void_p), m, tile_sizes)
else:
  r.run_compiled_job(getattr(libVM, fn_name),
                     cast(args, c_void_p), m, dl_sizes, ml_sizes)
stop = time.time()
r.cleanup()
print "Time to run job:", stop - start, "secs"

check = False
if check:
  npa = np.reshape(np.fromiter(a, dtype=np.float, count=m*k), (m, k))
  npb = np.reshape(np.fromiter(b, dtype=np.float, count=n*k), (n, k))
  npo = np.reshape(np.fromiter(o, dtype=np.float, count=m*n), (m, n))
  print "Output:", npo

  npbt = npb.T
  start = time.time()
  npr = np.dot(npa, npbt)
  stop = time.time()
  print "Time for numpy:", stop - start, "secs"
  print npr

  if (npr == npo).all():
    print "Passed"
  else:
    print "Failed"
