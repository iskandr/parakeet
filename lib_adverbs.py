
from prims import *
from lib_simple import len 
from adverb_api  import each, reduce 

# from adverbs import map, reduce, scan 


def sum(x):
  return reduce(add, x, init = 0)

def prod(x):
  return reduce(multiply,  x, init = 1)

def mean(x):
  return sum(x) / len(x)

