from parakeet.testing_helpers import expect, run_local_tests 
import numpy as np 

def always_1():
  return 1

def test_always_1():
  expect(always_1, [], 1)

def always_neg10():
  return -10

def test_always_neg10():
  expect(always_neg10, [], -10)

def always_true():
  return True

def test_always_true():
  expect(always_true, [], True, valid_types = (np.bool, np.bool8, np.bool_, bool))

def always_false():
  return False 

def test_always_false():
  expect(always_false, [], False, valid_types = (np.bool, np.bool8, np.bool_, bool))
  
def add1(x):
  return x + 1

def test_add1():
  expect(add1, [1],  2)

def call_add1(x):
  return add1(x)

def test_call_add1():
  expect(call_add1, [1],  2)

def call_nested_ident(x):
  def ident(y):
    return y
  return ident(x)

def test_nested_ident():
  expect(call_nested_ident, [1], 1)

global_val = 5
def use_global(x):
  return x + global_val

def test_use_global():
  expect(use_global, [3], 8)

def use_if_exp(x):
  return 1 if x < 10 else 2

def test_if_exp():
  expect(use_if_exp, [9], 1)
  expect(use_if_exp, [10], 2)

def simple_branch(x):
  if x < 10:
    return 1
  else:
    return 2

def test_simple_branch():
  expect(simple_branch, [9], 1)
  expect(simple_branch, [10], 2)

def simple_merge(x):
  if x == 0:
    y = 1
  else:
    y = x
  return y

def test_simple_merge():
  expect(simple_merge, [2], 2)
  expect(simple_merge, [0], 1)

def one_sided_merge(x,b):
  if b:
      x = 1
  return x

def test_one_sided_merge():
  expect(one_sided_merge, [100,True], 1)
  expect(one_sided_merge, [100,False], 100)

def if_true_const():
  if True:
    return 1
  else:
    return 2

def test_if_true_const():
  expect(if_true_const, [], 1)




if __name__ == '__main__':
  run_local_tests()
