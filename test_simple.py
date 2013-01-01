from testing_helpers import expect, run_local_tests

def always1():
  return 1

def test_always1():
  expect(always1, [], 1)

def always_neg10():
  return -10

def test_always_neg10():
  expect(always_neg10, [], -10)

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
  expect(simple_merge, [0], 1)
  expect(simple_merge, [2], 2)

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

def varargs_return(*x):
  return x

def test_varargs_return():
  expect(varargs_return, [1,2], (1,2))

def varargs_add(*x):
  return x[0] + x[1]

def test_varargs_add():
  expect(varargs_add, [1,2], 3)

def call_varargs_add(x,y):
  local_tuple = (x,y)
  return varargs_add(*local_tuple)

def test_call_varargs_add():
  expect(call_varargs_add, [1,2], 3)
  expect(call_varargs_add, [True,2.0], 3.0)

def add_defaults(x = 1, y = 2):
    return x + y

def test_add_defaults():
  expect(add_defaults, [], 3)
  expect(add_defaults, [10], 12)
  expect(add_defaults, [10, 20], 30)
  expect(add_defaults, [10, 20.0], 30.0)

def call_add_defaults():
  return add_defaults(10)

def test_call_add_defaults():
  expect(call_add_defaults, [], 12)

def call_add_defaults_with_names():
  return add_defaults(y = 10, x = 20)

def test_call_defaults_with_names():
  expect(call_add_defaults_with_names, [], 30)

def sub(x,y):
  return x - y

def call_pos_with_names():
  return sub(y = 10, x = 20)

def test_call_pos_with_names():
  expect(call_pos_with_names, [], 10)

if __name__ == '__main__':
  run_local_tests()
