from parakeet import jit 
from parakeet.testing_helpers import run_local_tests, expect 
import time 

def identity(x,y,z):
  return x 

jit_identity = jit(identity)

def test_call_overhead_identity():
  n = 2000
  x = 3 
  start_t = time.time()
  for i in xrange(n):
    identity(x,x,x)
  python_time = time.time() - start_t 
  print "Python time for %d calls: %f" % (n, python_time)
  # warm up!
  jit_identity(x,x,x)
  start_t = time.time()
  for i in xrange(n):
    jit_identity(x,x,x)
  parakeet_time = time.time() - start_t
  print "Parakeet time for %d calls: %f" % (n, parakeet_time)
  print "Slowdown = %f" % (parakeet_time / python_time )
  assert parakeet_time < 1000 * python_time, "Excessive call overhead: %f Python vs. %f Parakeet" % (python_time, parakeet_time)


if __name__ == "__main__":
  run_local_tests()
  

  
