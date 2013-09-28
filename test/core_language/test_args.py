from parakeet.testing_helpers import expect, run_local_tests

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
  
def tuple_default(x = (1,2)):
  return x[0] + x[1]

def test_tuple_default():
  expect(tuple_default, [], 3)
  expect(tuple_default, [(1,3)], 4)

def default_closure(x = 1, y=(1,2)):
  def inner(z, r = 0):
    return y[0] + y[1] + z + r
  return inner(x)

def test_default_closure(): 
  expect(default_closure, [], 4)
  expect(default_closure, [10], 13)
  expect(default_closure, [10, (20,30)], 60)

def default_none(x, y = None):
  if y is None:
    return x * 2
  else:
    return x + y

def test_default_none():
  expect(default_none, [10], 20)
  expect(default_none, [10, None], 20)
  expect(default_none, [10, 1], 11)

def add1(x):
  return x + 1

def add2(x): 
  return x + 2 


def fn_as_default(f = add1):
  return f(1)

def test_fn_as_default():
  expect(fn_as_default, [], 2)
  expect(fn_as_default, [add2], 3)

def lambda_closure(x):
  y = 1.0
  def g(f, x):
    return f(x)
  return g(lambda z: z + y, x)

def test_lambda_closure():
  expect(lambda_closure, [1], 2.0) 

if __name__ == '__main__':
  run_local_tests()
