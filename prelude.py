

from prims import *

from external_api import each, seq_reduce 

# from adverbs import map, reduce, scan 
def len(arr):
  return arr.shape[0]
    
def shape(arr):
  return arr.shape 

def sum(x):
  # should be reduce(add, add, x[1:], x[0])
  return seq_reduce(add, add, x, 0)

def prod(x):
  # should be reduce(add, add, x[1:], x[0])
  return seq_reduce(multiply, multiply, x, 1)

def mean(x):
  return sum(x) / len(x)
