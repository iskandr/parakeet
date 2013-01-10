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
m = 3000
n = 3000
k = 3000
a = libVM.make_array(m, k)
b = libVM.make_array(n, k)
o = libVM.make_array(m, n)

ll = True

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
tile_sizes[0] = 60
tile_sizes[1] = 60
tile_sizes[2] = 60

r = runtime.Runtime()
print "Launching parallel job"
start = time.time()
#r.run_job(libVM, cast(args, c_void_p), m, [n, k], [None, None])
#r.run_untiled_job(libVM.vm3, cast(args, c_void_p), m)
r.run_job_with_fixed_tiles(libVM.AllPairsAllPairsdotOpt,
                           cast(args, c_void_p), m, tile_sizes)
stop = time.time()
r.cleanup()
print "Time to run job:", stop - start, "secs"

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
