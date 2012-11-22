from testing_helpers import expect, run_local_tests

def always1():
  return 1

def test_always1():
  expect(always1, [], 1)

def add1(x):
  return x + 1

def test_add1():
  expect(add1, [1],  2) 

def call_add1(x):
  return add1(x)

def test_call_add1():
  expect(call_add1, [1],  2) 

def call_nested_ident(x):
  def ident(x):
    return x
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
 
def count_loop(init, count):
  x = init
  while x < count:
    x = x + 1
  return x
 
def test_count_loop():
  expect(count_loop, [0, 300], 300 )
  expect(count_loop, [0.0, 400], 400.0)
  expect(count_loop, [0.0, 500.0], 500.0)
 
def if_true():
  if True:
    return 1
  else:
    return 2

def test_if_true():
  expect(if_true, [], 1)
 


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


if __name__ == '__main__':
  run_local_tests()
  
