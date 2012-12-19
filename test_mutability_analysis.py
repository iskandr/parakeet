
import testing_helpers 
import mutability_analysis
import lowering 

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
  lowered = lowering.lower(typed)
  mutable_types = mutability_analysis.find_mutable_types(lowered)
  assert len(mutable_types) == 1, mutable_types
  
if __name__ == '__main__':
  testing_helpers.run_local_tests()
