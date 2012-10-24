from parakeet import expect 
def loop_dot(x,y):
  i = 0
  result = 0
  n = x.shape[0]
  while i < n:
      result = result + x[i] * y[i]
      i = i + 1
  return result

import numpy as np 
int_vec = np.array([1,2,3,4,5])
float_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

def test_loopdot():
    expect(loop_dot, [int_vec, int_vec], np.dot(int_vec, int_vec))
    expect(loop_dot, [float_vec, float_vec], np.dot(float_vec, float_vec))
    expect(loop_dot, [float_vec, int_vec], np.dot(float_vec, int_vec))


if __name__ == '__main__':
    import testing_helpers
    testing_helpers.run_local_tests()
    
