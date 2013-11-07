from parakeet.testing_helpers import expect, run_local_tests

def test_float_to_int():
  expect(int, [1.2], 1)

def test_int_to_float():
  expect(float, [1], 1.0)
  
def test_bool_to_float():
  expect(float, [True], 1.0)

def test_float_to_long():
  expect(long, [-1.0], -1)
  
if __name__ == '__main__':
  run_local_tests()
