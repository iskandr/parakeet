
from prims import *
from lib_simple import len 
from adverb_api  import each, reduce 

# from adverbs import map, reduce, scan 


def sum(x):
  # should be reduce(add, add, x[1:], x[0])
  return reduce(add, x)

def prod(x):
  # should be reduce(add, add, x[1:], x[0])
  return reduce(multiply,  x)

def mean(x):
  return sum(x) / len(x)

