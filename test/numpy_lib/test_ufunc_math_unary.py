
import numpy as np 

from parakeet.testing_helpers import run_local_tests, expect 

int_vec = np.array([2,3])
int_mat = np.array([int_vec, int_vec])

bool_vec = np.array([True, True])
bool_mat = np.array([bool_vec, bool_vec])

float32_vec = np.array([1.0, 2.0], dtype='float32')
float32_mat = np.array([float32_vec, float32_vec])

float64_vec = np.array([1.0, 10.0])
float64_mat = np.array([float64_vec, float64_vec])

vecs = [int_vec, bool_vec, float32_vec, float64_vec]
mats = [int_mat, bool_mat, float32_mat, float64_mat]


def unary(fn): 
  for x in mats:
    try:
      expected = fn(x)
    except:
      expected = fn(x.astype('int'))
    if expected.dtype == 'float16':
      expected = fn(x.astype('int'))
    expect(fn, [x], expected, fn.__name__ + "-" + str(x.dtype) + str(len(x.shape)))

def test_negative():
  unary(np.negative)

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
