"""Simple library functions which don't depend on adverbs"""

__all__ = ["identity", "len", "shape"]

def identity(x):
  return x

# TODO: I'm not sure using builtin names is a good idea.
def len(arr):
  return arr.shape[0]

def shape(arr):
  return arr.shape
