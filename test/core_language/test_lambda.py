from parakeet.testing_helpers import expect, run_local_tests

def call_const_lambda():
  return (lambda: 1)()

def test_call_const_lambda():
  expect(call_const_lambda, [], 1)

def call_identity_lambda(x):
  return (lambda y: y)(x)

def test_call_identity_lambda():
  expect(call_identity_lambda, [1], 1)

def call_closure_lambda(x):
  return (lambda y: x + y)(x)

def test_call_closure_lambda():
  expect(call_closure_lambda, [1], 2)

if __name__ == "__main__":
  run_local_tests()
