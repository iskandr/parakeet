import numpy as np 
from parakeet import testing_helpers, frontend, analysis, type_inference, transforms

def infer_ranges(python_fn, input_types):
  untyped = frontend.translate_function_value(python_fn)
  typed = type_inference.specialize(untyped, input_types)

  vra = analysis.ValueRangeAnalyis()
  
  vra.visit_fn(typed)
  return vra.ranges 

def expect_const(k, v, c):
  assert v.__class__ is analysis.value_range_analysis.Interval, \
    "Expected %s to be const interval but got %s" % (k,v) 
  assert v.lower == c, "Expected %s to be const value %s not %s" % (k, c,v)
  assert v.upper == c, "Expected %s to be const value %s not %s" % (k, c,v)
  

def expect_range(k, v, lower = None, upper = None):
  assert v.__class__ is analysis.value_range_analysis.Interval, \
    "Expected %s to be interval but got %s" % (k,v)
  if lower is not None:
    assert v.lower == lower, \
      "Expected %s's lower value %s but got %s" % (k, lower,v)
  if upper is not None: 
    assert v.upper == upper, \
      "Expected %s's upper value %s but got %s" % (k, upper,v)

def incr_loop():
  x = 1
  for i in xrange(10):
    x = x + 1
  return x 

def test_incr_loop():
  ranges = infer_ranges(incr_loop, [])
  assert len(ranges) == 4, "Wrong number of variables in %s" % ranges 
  for (k, v) in sorted(ranges.items(), key = lambda (k,v): k):
    if k.startswith("i"):
      expect_range(k, v, 0, 10)
    elif k.startswith("x"):
      assert v.__class__ is analysis.value_range_analysis.Interval, \
        "Expected %s to be interval but got %s" % (k,v)
      assert v.lower in (1,2), "Expected %s to have lower bound of 1 or 2 but got %s" % v 
    else:
      assert False, "Unexpected variables %s = %s" % (k,v)

def const():
  x = 1 
  return x 

def test_const():
  ranges = infer_ranges(const, [])
  assert len(ranges) == 1, "Too many variables: %s" % ranges 
  k,v = ranges.items()[0]
  expect_const(k,v, 1)

def nested_loops():
  x = 0
  y = 0
  z = 0 
  for i in xrange(10):
    x = x + i
    for j in xrange(3,12,1):
      y = y + x
      for k in xrange(3):
        z = z + y
  q = z
  return q

def test_nested_loops():
  ranges = infer_ranges(nested_loops, [])    
  print ranges 
  for k, v in ranges.iteritems():
    if k.startswith("i"):
      expect_range(k,v,0,10)
    elif k.startswith("j"):
      expect_range(k,v,3,12)
    elif k.startswith("x"):
      try:
        expect_range(k,v,0,None)
      except:
        expect_range(k,v,3,np.inf)
    elif k.startswith("q"):
      expect_range(k,v,None,np.inf)
if __name__ == "__main__":
  testing_helpers.run_local_tests()

