import numpy as np 

from treelike.testing_helpers import run_local_tests, expect 

from parakeet import jit 

int_array = np.array([10,20,30])
bool_array = np.array([True, True, True])
float_array = np.array([1.0, 2.0, 3.0])
arrays = [int_array, bool_array, float_array]

def unary(fn): 
  for x in arrays:
    expect(fn, [x], fn(x))

def test_sin():
  unary(np.sin)

def test_cos():
  unary(np.cos)
  
def test_tan():
  unary(np.tan)
  
def test_arcsin():
  unary(np.arcsin)
  
def test_arccos():
  unary(np.arccos)
  
def test_arctan():
  unary(np.arctan)
  
def test_arctan2():
  unary(np.arctan2)
  
def test_hypot():
  unary(np.hypot)
  
def test_sinh():
  unary(np.sinh)
  
def test_cosh():
  unary(np.cosh)
  
def test_tanh():
  unary(np.tanh)
  
def test_arcsinh():
  unary(np.arcsinh)
  
def test_arccosh():
  unary(np.arccosh)
  
def test_arctanh():
  unary(np.arctanh)
  
def test_deg2rad():
  unary(np.deg2rad)
  
def test_rad2deg():
  unary(np.rad2deg)
  
  
if __name__ == '__main__':
  run_local_tests() 