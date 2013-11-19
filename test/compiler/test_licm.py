
from parakeet.testing_helpers import expect, run_local_tests

def volatile_licm_mistake():
  i = 0
  x = [0]
  while i < 10:
    alloc = [1]
    if i == 5:
      x = alloc
    else:
      alloc[0] = 100
    i = i + 1
  return x[0]

def test_volatile_licm_mistake():
  expect(volatile_licm_mistake, [], volatile_licm_mistake()) #np.array([1]))

def licm_nested_loops():
  
  total = 0
  a = [1,2,3,4,5]
  for _ in range(3):
    for _ in range(2):
      for j in range(len(a)):
        a[j] *= 10
    total += a[1]
  return total   

def test_licm_nested_loops():
  expect(licm_nested_loops, [], licm_nested_loops())

def loop_invariant_alloc():
  b = [1,2,3]
  total = 0
  for i in xrange(3):
    a = [0,0,0]
    for j in xrange(3):
      a[j] = b[j]
    total += a[i]
  return total 
  
def test_loop_invariant_alloc():
  expect(loop_invariant_alloc, [], loop_invariant_alloc())
  

if __name__ == "__main__":
  run_local_tests()