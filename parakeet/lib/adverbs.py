
from .. frontend import macro, staged_macro, jit,  translate_function_value 
from .. syntax import none, zero_i64 
from .. syntax import Map, Reduce, Scan, IndexMap, IndexReduce, ParFor, AllPairs 
from .. syntax import none

@jit 
def identity(x):
  return x

@macro 
def parfor(shape, fn):
  return ParFor(fn = fn, shape = shape)

@staged_macro("axis")
def map(f, *args, **kwds):
  axis = kwds.get('axis', zero_i64)
  return Map(fn = f, args = args, axis = axis)
each = map

@staged_macro("axis") 
def allpairs(f, x, y, axis = 0):
  return AllPairs(fn = f, args = (x,y), axis = axis)

@staged_macro("axis")
def reduce(f, *args, **kwds):
  axis = kwds.get('axis', none)
  init = kwds.get('init', none)
  
  ident = translate_function_value(identity)
  return Reduce(fn = ident, 
                combine = f, 
                args = args,
                init = init,
                axis = axis)

@staged_macro("axis")
def scan(f, *args, **kwds):
  axis = kwds.get('axis', zero_i64)
  init = kwds.get('init', none)
  
  ident = translate_function_value(identity)
  return Scan(fn = ident,  
                     combine = f,
                     emit = ident, 
                     args = args,
                     init = init,
                     axis = axis)
@macro
def imap(fn, shape):
  return IndexMap(shape = shape, fn = fn)


@macro
def ireduce(combine, shape, map_fn = identity, init = None):
  return IndexReduce(fn = map_fn, combine=combine, shape = shape, init = init)