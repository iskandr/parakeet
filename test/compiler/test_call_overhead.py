from parakeet import jit 
from parakeet.testing_helpers import run_local_tests, expect 
import time 

def identity(x):
  return x 

jit_identity = jit(identity)

def test_call_overhead_identity():
  n = 10000
  x = 3 
  start_t = time.time()
  for i in xrange(n):
    identity(x)
  python_time = time.time() - start_t 
  print "Python time for %d calls: %f" % (n, python_time)
  # warm up!
  jit_identity(x)
  start_t = time.time()
  for i in xrange(n):
    jit_identity(x)
  parakeet_time = time.time() - start_t
  print "Parakeet time for %d calls: %f" % (n, parakeet_time)
  assert parakeet_time < python_time 


if __name__ == "__main__":
  run_local_tests()
  

  
