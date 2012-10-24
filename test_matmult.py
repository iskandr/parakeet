import parakeet as par 

import numpy as np 
bool_vec = np.array([True, False, True, False, True])
int_vec = np.array([1,2,3,4,5])
float_vec = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

def allpairs_inputs(parakeet_fn, python_fn, inputs):
  for x in inputs:
    for y in inputs:
      expect(parakeet_fn, [x,y], python_fn(x,y))


from parakeet import expect  
def loop_dot(x,y):
  n = x.shape[0]
  result = x[0] * y[0]
  i = 1
  while i < n:
      result = result + x[i] * y[i]
      i = i + 1
  return result




def test_loopdot():
  allpairs_inputs(loop_dot, np.dot, [bool_vec, int_vec, float_vec])

def adverb_dot(x,y):
  return par.sum(x*y)

def test_adverb_dot():
  allpairs_inputs(adverb_dot, np.dot, [bool_vec, int_vec, float_vec])

if __name__ == '__main__':
    import testing_helpers
    testing_helpers.run_local_tests()
    
