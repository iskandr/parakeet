from parakeet.testing_helpers import expect, run_local_tests

def for_range_loop(start, stop, step):
  x = 0
  for _ in range(start, stop, step):
    x = x + 1
  return x

def test_for_range_loop():
  expect(for_range_loop, [1,10,2], for_range_loop(1,10,2))

def test_for_backwards():
  expect(for_range_loop, [10,1,-1], for_range_loop(10,1,-1))

def count_loop(init, count):
  x = init
  while x < count:
    x = x + 1
  return x

def test_count_loop():
  expect(count_loop, [0, 300], 300)
  expect(count_loop, [0.0, 400], 400.0)
  expect(count_loop, [0.0, 500.0], 500.0)

def nested_double_count(x):
  total = 0
  i = 0
  while i < x:
    j = 0
    total = total + 1
    while j < x:
      total = total + 1
      j = j + 1
    i = i + 1
  return total

def test_nested_double_count():
  expect(nested_double_count, [10], 110)

def nested_mult(x,y):
  total_count = 0
  i = 0
  while i < x:
    j = 0
    while j < y:
      total_count = total_count + 1
      j = j + 1
    i = i + 1
  return total_count

def test_nested_mult():
  expect(nested_mult, [10, 11], 110)

def conditional_loop(x):
  i = 0
  if x:
    while i < 10:
      i = i + 1
  return i

def test_conditional_loop():
  expect(conditional_loop, [True], 10)
  expect(conditional_loop, [False], 0)

if __name__ == '__main__':
  run_local_tests()
