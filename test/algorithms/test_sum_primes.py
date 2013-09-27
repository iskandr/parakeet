from parakeet.testing_helpers import expect, run_local_tests 

def sum_primes(n):
  total = 0
  count = 0
  for i in xrange(2, n):
    found_divisor = False
    for j in xrange(2,i-1):
      if i % j == 0:
        found_divisor = True
    if not found_divisor:
      total += i
      count += 1
  return total, count 

def test_sum_primes():
  expect(sum_primes, [20], sum_primes(20))  

if __name__ == '__main__':
  run_local_tests()
