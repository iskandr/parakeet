import numpy as np 
from parakeet import testing_helpers 

def test_simple_linspace():
  testing_helpers.expect(np.linspace, [0.2, 0.9], np.linspace(0.2, 0.9))

def test_linspace_with_count():
  testing_helpers.expect(np.linspace, [-0.2, 0.9, 30], np.linspace(-0.2, 0.9, 30))

if __name__ == "__main__":
  testing_helpers.run_local_tests()


