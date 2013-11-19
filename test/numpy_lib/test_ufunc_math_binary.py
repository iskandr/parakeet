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

def binary(fn):
  for x in mats:
    for y in mats:
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
def test_power():
  binary(np.power)

def test_remainder():
  binary(np.remainder)
  
def test_mod():
  binary(np.mod)
  
def test_fmod():
  binary(np.fmod)
  
      
if __name__ == '__main__':
  run_local_tests() 
  
