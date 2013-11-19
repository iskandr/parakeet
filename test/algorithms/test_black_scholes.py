from parakeet import jit 
from parakeet.testing_helpers import eq, run_local_tests, expect 

from numpy import exp, log, sqrt


def CND(x):
  a1 = 0.31938153
  a2 = -0.356563782
  a3 = 1.781477937
  a4 = -1.821255978
  a5 = 1.330274429
  L = abs(x)
  K = 1.0 / (1.0 + 0.2316419 * L)
  w = 1.0 - 1.0/sqrt(2*3.141592653589793)* exp(-1*L*L/2.) * (a1*K +
      a2*K*K + a3*K*K*K + a4*K*K*K*K + a5*K*K*K*K*K)
  if x<0:
    w = 1.0-w
  return w

def black_scholes(CallFlag,S,X,T,r,v):
  d1 = ((r+v*v/2.)*T+log(S/X))/(v*sqrt(T))
  d2 = d1-v*sqrt(T)
  z = exp(-1.0*r*T) * X
  if CallFlag:
    return S*CND(d1) - z*CND(d2)
  else:
    return z*CND(-1.0*d2) - S*CND(-1.0*d1)

black_scholes_parakeet = jit(black_scholes)

def test_black_scholes():
  x1 = (False, 10.0, 10.0, 2.0, 2.0, 2.0)
  x2 = (True, 10.0, 10.0, 2.0, 2.0, 2.0)
  xs = [x1, x2]
  for x in xs:
    expect(black_scholes, x, black_scholes(*x))

if __name__ == '__main__':
  run_local_tests()
