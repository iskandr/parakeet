from parakeet import  testing_helpers
import math 

def test_erfc():
  testing_helpers.expect(math.erfc, [0], math.erfc(0))
  testing_helpers.expect(math.erfc, [0.5], math.erfc(0.5))
  testing_helpers.expect(math.erfc, [1], math.erfc(1))

def test_erf():
  testing_helpers.expect(math.erf, [0], math.erf(0))
  testing_helpers.expect(math.erf, [0.5], math.erf(0.5))
  testing_helpers.expect(math.erf, [1], math.erf(1))
  
if __name__ == "__main__":
    testing_helpers.run_local_tests()
