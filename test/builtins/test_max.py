from parakeet.testing_helpers import expect, run_local_tests

def test_max_int():
  expect(max, [3,4], 4)

def test_max_float():
  expect(max, [3.0,4.0], 4.0)
  
def test_max_int_float():
  expect(max, [3, 4.0], 4.0)

def test_max_bool():
  expect(max, [False, True], True)
  
if __name__ == '__main__':
  run_local_tests()
