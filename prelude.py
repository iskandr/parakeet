from prims import *

# from adverbs import map, reduce, scan 
def len(arr):
  return arr.shape[0]
    
def shape(arr):
  return arr.shape 

def reduce(f, combine, xs, init):
  start_idx = 0
  n = len(xs)
  while start_idx < n:
    init = f(init, xs[start_idx])
    start_idx += 1
  return init 

def sum(x):
  return reduce(add, add, x[1:], x[0])

def prod(x):
  return reduce(multiply, multiply, x[1:], x[0])

def mean(x):
  return sum(x) / len(x)
