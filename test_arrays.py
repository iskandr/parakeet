import numpy as np 
from parakeet import expect
def create_const(x):
    return [x,x,x,x]

def test_create_const():
    expect(create_const, [1],  np.array([1,1,1,1]))
    expect(create_const, [1.0], np.array([1.0, 1.0, 1.0, 1.0]))
    expect(create_const, [True], np.array([True, True, True, True]))

if __name__ == '__main__':
  import testing_helpers
  testing_helpers.run_local_tests()

