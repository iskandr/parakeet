from parakeet.testing_helpers import expect, run_local_tests

def test_min_int():
  expect(min, [-3, 4], -3)

def test_min_float():
  expect(min, [-3.0, 4.0], -3.0)


if __name__ == '__main__':
  run_local_tests()
