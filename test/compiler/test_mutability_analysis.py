import numpy as np 



from parakeet.analysis import mutability_analysis
from parakeet import make_array_type, Int64, Bool, ptr_type, specialize
from parakeet.transforms.pipeline import lowering 
from parakeet.testing_helpers import run_local_tests

def f(x,y):
    x[0] = True
    return y

def test_mutable_array():
  bool_vec = np.array([True, False])
  int_vec = np.array([1,2])
  typed, _, =  specialize(f, [bool_vec, int_vec])
   
  mutable_types = mutability_analysis.find_mutable_types(typed)
  int_array_t = make_array_type(Int64, 1)
  bool_array_t = make_array_type(Bool, 1)
  assert int_array_t not in mutable_types, \
      "Didn't expect %s in mutable_types %s" % (int_array_t, mutable_types)
  assert bool_array_t in mutable_types, \
      "Expected %s in mutable_types %s" % (bool_array_t, mutable_types)
      
  lowered = lowering.apply(typed)
  mutable_types = mutability_analysis.find_mutable_types(lowered)
  ptr_bool_t = ptr_type(Bool)
  ptr_int_t = ptr_type(Int64) 
  assert ptr_int_t not in mutable_types, \
      "Didn't expect %s in lowered mutable types %s" % \
      (ptr_int_t, mutable_types)
  assert ptr_bool_t in mutable_types, \
      "Expected %s in lowered mutable_types %s" % (ptr_bool_t, mutable_types)

"""
# Removed this test when I made attributes immutable 
def f():
  x = slice(1, 2, 3)
  x.start = 10
  x.stop = 20
  y = slice(1,2,3)
  if 3 < y.step:
    y.step = y.step + 1
  else:
    y.step = 0

def test_mutable_slice():
  _, typed, _, _ =  testing_helpers.specialize_and_compile(f, [])
  mutable_types = mutability_analysis.find_mutable_types(typed)

  assert len(mutable_types) == 1, mutable_types
  lowered = lowering.apply(typed)
  mutable_types = mutability_analysis.find_mutable_types(lowered)
  assert len(mutable_types) == 1, mutable_types
"""
if __name__ == '__main__':
  run_local_tests()
