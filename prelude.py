

from prims import *

from external_api import map, reduce 

# from adverbs import map, reduce, scan 
def len(arr):
  return arr.shape[0]
    
def shape(arr):
  return arr.shape 

def sum(x):
  # should be reduce(add, add, x[1:], x[0])
  return reduce(add, add, x, 0)

def prod(x):
  # should be reduce(add, add, x[1:], x[0])
  return reduce(multiply, multiply, x, 1)

def mean(x):
  return sum(x) / len(x)
