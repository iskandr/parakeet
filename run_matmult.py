import numpy as np
from parakeet import allpairs, run


def dot(x,y):
  return sum(x*y)

def adverb_matmult(X,Y):
  return allpairs(dot, X, Y.T.copy())

X = np.array([[1,2],[3,4]])
Y = np.array([[10,1],[-1,10]])

Z = adverb_matmult(X, Y)
E = np.dot(X,Y)
assert (Z == E).all(), \
        "Expected:\n  %s but got:\n  %s" % (E, Z)
