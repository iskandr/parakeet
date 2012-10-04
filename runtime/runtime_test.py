#!/usr/bin/python

from ctypes import *
import numpy as np
import runtime, time

class vm_args_t(Structure):
  _fields_ = [("a", POINTER(c_double)),
              ("b", POINTER(c_double)),
              ("out", POINTER(c_double)),
              ("m", c_int),
              ("n", c_int),
              ("k", c_int),
              ("tile_sizes", POINTER(c_int))]

libVM = cdll.LoadLibrary("vm.so")
libVM.make_array.restype = POINTER(c_double)

m = 10000
n = 800
k = 800
a = libVM.make_array(m, k)
b = libVM.make_array(n, k)
o = libVM.make_array(m, n)

tile_sizes = (c_int * 2)(1,1)
args = pointer(vm_args_t(a, b, o, m, n, k, cast(tile_sizes, POINTER(c_int))))
job = runtime.make_job(0, 10000, 8)

print "Launching parallel job"
start = time.time()
runtime.launch_job(cast(libVM.vm, c_void_p), cast(args, c_void_p), job)
runtime.wait_for_job()
stop = time.time()
runtime.free_job(job)
runtime.cleanup()
print "Time to run job:", stop - start, "secs"

npa = np.reshape(np.fromiter(a, dtype=np.float, count=m*k), (m, k))
npb = np.reshape(np.fromiter(b, dtype=np.float, count=n*k), (n, k))
npo = np.reshape(np.fromiter(o, dtype=np.float, count=m*n), (m, n))

start = time.time()
print np.dot(npa, npb)
stop = time.time()
print "Time for numpy:", stop - start, "secs"
