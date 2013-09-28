import numpy as np 

from parakeet.testing_helpers import run_local_tests, expect 


int_array = np.array([2,3])
bool_array = np.array([True, True])
float_array = np.array([1.0, 2.0])
arrays = [int_array, bool_array, float_array]


def unary(fn): 
  for x in arrays:
    try:
      expected = fn(x)
    except:
      expected = fn(x.astype('int'))
    if expected.dtype == 'float16':
      expected = fn(x.astype('int'))
    expect(fn, [x], expected, fn.__name__ + "-" + str(x.dtype) + str(len(x.shape)))

def binary(fn):
  for x in arrays:
    for y in arrays:
      expect(fn, [x,y], fn(x,y), x.dtype)
      
def test_add():
  binary(np.add)

def test_subtract():
  binary(np.subtract)

def test_multiply():
  binary(np.multiply)
  
def test_divide():
  binary(np.divide)
  
def test_logaddexp():
  binary(np.logaddexp)
  
def test_logaddexp2():
  binary(np.logaddexp2)
  
def test_true_divide():
  binary(np.true_divide)
  
def test_floor_divide():
  binary(np.floor_divide)

def test_negative():
  unary(np.negative)

def test_power():
  binary(np.power)

def test_remainder():
  binary(np.remainder)
  
def test_mod():
  binary(np.mod)
  
def test_fmod():
  binary(np.fmod)
  
def test_absolute():
  unary(np.absolute) 
  
def test_rint():
  unary(np.rint)

def test_sign():
  unary(np.sign)
  
def test_conj():
  unary(np.conj)
    
def test_exp():
  unary(np.exp)

def test_exp2():
  unary(np.exp2)
  
def test_log():
  unary(np.log)
  
def test_log2():
  unary(np.log2)

def test_log10():
  unary(np.log10)

def test_expm1():
  unary(np.expm1)
  
def test_log1p():
  unary(np.log1p)
  
def test_sqrt():
  unary(np.sqrt)
  
def test_square():
  unary(np.square)
  
def test_reciprocal():
  unary(np.reciprocal)
  
def test_ones_like():
  unary(np.ones_like)
  
      
if __name__ == '__main__':
  run_local_tests() 
  
