
# Rosenbrock function derivative
# 
# Copied from https://github.com/numfocus/python-benchmarks/blob/master/rosen_der/rosen_der_python.py
# Original authors: Travis Oliphant (NumPy version) & Serge Guelton (loops version)
#

import numpy as np 
from parakeet import jit 

@jit 
def rosenbrock_derivative(x):
  der = np.empty_like(x)
  der[1:-1] = (+ 200 * (x[1:-1] - x[:-2] ** 2)
               - 400 * (x[2:] - x[1:-1] ** 2) * x[1:-1]
               - 2 * (1 - x[1:-1]))
  der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
  der[-1] = 200 * (x[-1] - x[-2] ** 2)
  return der

N = 12
x = np.arange(N) / float(N)
print "Input: ", x
print "Deriv(Rosenbrock):", rosenbrock_derivative(x)

