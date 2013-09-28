
from parakeet.testing_helpers import expect, run_local_tests 


def is_prime(n):
  for i in xrange(2,n-1):
    if n % i == 0:
      return False 
  return True 

def test_is_prime():
  expect(is_prime, [2], True)  
  expect(is_prime, [3], True)
  expect(is_prime, [4], False)
  expect(is_prime, [5], True)
  expect(is_prime, [6], False) 

if __name__ == '__main__':
    run_local_tests()
