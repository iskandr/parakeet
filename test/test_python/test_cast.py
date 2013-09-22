from parakeet.testing_helpers import expect, run_local_tests

def implicit_to_float(x):
  return x + 0.5

def test_implicit_to_float():
  expect(implicit_to_float, [1], 1.5)
  expect(implicit_to_float, [True], 1.5)

def implicit_to_bool(b):
  if b:
    return 10
  else:
    return -10

def test_implicit_to_bool():
  expect(implicit_to_bool, [1], 10)
  expect(implicit_to_bool, [2], 10)
  expect(implicit_to_bool, [0], -10)
  expect(implicit_to_bool, [1.0], 10)
  expect(implicit_to_bool, [2.0], 10)
  expect(implicit_to_bool, [0.0], -10)

if __name__ == '__main__':
    run_local_tests()
