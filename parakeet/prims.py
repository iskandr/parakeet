

import numpy as np
import ndtypes 

from ndtypes import Bool, FloatT, BoolT, Float32, Float64, ScalarT 

prim_lookup_by_value = {}

def find_prim_from_python_value(fn):
  return prim_lookup_by_value[fn]

prim_lookup_by_op_name = {}

def op_name(op):
  return op.__class__.__name__

def is_ast_op(op):
  return op_name(op) in prim_lookup_by_op_name

def find_ast_op(op):
  name = op_name(op)
  if name in prim_lookup_by_op_name:
    return prim_lookup_by_op_name[name]
  else:
    raise RuntimeError("Operator not implemented: %s" % name)

def is_prim(numpy_fn):
  return numpy_fn in prim_lookup_by_value


class Prim(object):
  def __init__(self, fn, python_op_name = None, symbol = None,
               name = None, nin = None, nout = None, 
               extra_signatures = [], 
               doc = None):
    if doc is not None:
      self.__doc__ = doc 
      
    self.fn = fn
    prim_lookup_by_value[fn] = self
    self.symbol = symbol
    self.python_op_name = python_op_name

    if python_op_name is not None:
      prim_lookup_by_op_name[python_op_name] = self

    if name:
      self.name = name
    else:
      self.name = fn.__name__

    if nin:
      self.nin = nin
    elif hasattr(fn, 'nin'):
      self.nin = fn.nin
    else:
      assert hasattr(fn, 'func_code')
      self.nin = fn.func_code.co_argcount

    if nout:
      self.nout = nout
    elif hasattr(fn, 'nout'):
      self.nout = fn.nout
    else:
      self.nout = 1

    self._create_type_table()
    for sig in extra_signatures:
      self._add_signature(sig)
    # table mapping mismatching types i.e. (Int32, Float64) to (Float64, Float64)
    self._upcast_types = {}


  def _add_signature(self, signature):
    # numpy type signatures look like 'ff->f' where each character
    # represents a single type

    arg_codes, result_code = signature.split('->')
    try:
      parakeet_types = [ndtypes.from_char_code(c) for c in arg_codes]
      input_types = tuple(parakeet_types)
      result_type = ndtypes.from_char_code(result_code)
      self.type_table[input_types] = result_type
    except:
      # print "Signature %s failed  for %s" % (signature , self.fn)
      pass
  
  
  def _create_type_table(self):
    # for now only support ufuncs which describe their own type behavior
    if hasattr(self.fn, 'types'):
      "Primitive function %s doesn't supply type signatures" % self.name

      self.type_table = {}
      for signature in self.fn.types:
        self._add_signature(signature)

  def __eq__(self, other):
    return self.fn == other.fn

  def __hash__(self):
    return hash(self.name)

  def __call__(self, *args, **kwds):
    return self.fn(*args, **kwds)

  def __repr__(self):
    return "prim(%s)" % self.name

  def _signature_distance(self, types1, types2):
    dist = 0
    for (t1, t2) in zip(types1, types2):
      if t1 != t2:
        assert isinstance(t1, ScalarT), "Expected scalar type but got %s" % t1
        assert isinstance(t2, ScalarT), "Expected scalar type but got %s" % t2
        # penalty just for being a different type   
        dist += 1 
        
        size_difference = t2.nbytes - t1.nbytes
        
        # if we're downcasting, this sucks 
        if size_difference < 0:
          dist += 10000
        # going from int to float of same type is mildly unsafe  
        elif size_difference > 0:
          dist += np.log2(1 + size_difference)
        elif size_difference == 0:
          if isinstance(t2, FloatT) and not isinstance(t1, FloatT):
            dist += 10
        # can't go from float to int 
        if isinstance(t1, FloatT) and not isinstance(t2, FloatT):
          dist += 1000
        # but going to from int to float is only minor penalty...
        elif isinstance(t2, FloatT) and not isinstance(t1, FloatT):
          dist += 1

        # can't go from int to bool 
        if isinstance(t1, BoolT) and not isinstance(t2, BoolT):
          dist += 1
        elif isinstance(t2, BoolT) and not isinstance(t1, BoolT):
          dist += 1000
    return dist 
  
  def expected_input_types(self, arg_types):
    """Given some argument types, return the desired upcast types"""
    # by default we just figure out the common type and expect every arg to be
    # of that type
    n_inputs = len(arg_types)
    assert n_inputs == self.nin, \
        "Incorrect number of argument types for %s, expected %s but given %d" \
        % (self.name, self.nin, n_inputs)
    
    arg_types = tuple(arg_types)
    if arg_types in self.type_table:
      return arg_types
    elif arg_types in self._upcast_types:
      return self._upcast_types[arg_types]
    else:
      assert all(isinstance(t, ScalarT) for t in arg_types), \
        "Prim %s expects scalar inputs but given %s" % (self, arg_types)
      # search over all possible signatures to figure out 
      best_upcast_types = None
      best_distance = np.inf 
      for candidate_types in self.type_table:
        dist = self._signature_distance(arg_types, candidate_types)
        if dist < best_distance:
          best_distance = dist
          best_upcast_types = candidate_types
      self._upcast_types[arg_types] = best_upcast_types
      return best_upcast_types  
    #common_type = combine_type_list(arg_types)
    #return [common_type] * n_inputs

  def result_type(self, arg_types):
    """
    Given some argument types, look up the result type in the type_table we
    generated from numpy's given signatures
    """
    key = tuple(arg_types)
    if key not in self.type_table:
      raise RuntimeError("Primitives %s doesn't support input types %s, candidates: %s" % (self.name, key, self.type_table))
    else:
      return self.type_table[key]

class Float(Prim):
  """Always returns a float"""
  def expected_input_types(self, arg_types):
    assert all(isinstance(t, ScalarT) for t in arg_types), \
        "Prim %s expects scalar inputs but given %s" % (self, arg_types)
    max_nbytes = max(t.nbytes for t in arg_types)
    if max_nbytes <= 4:
      upcast_types = [Float32.combine(t) for t in arg_types]
    else:
      upcast_types = [Float64.combine(t) for t in arg_types]
    return Prim.expected_input_types(self, upcast_types)
  
  def result_type(self, arg_types):
    t = Prim.result_type(self, arg_types)
    return t.combine(Float32)
  
class Arith(Prim):
  """Basic arithmetic operators"""
  pass

class Logical(Prim):
  """Expects boolean inputs, returns a boolean"""
  def expected_input_types(self, arg_types):
    return [Bool] * len(arg_types)

class Bitwise(Prim):
  """Takes any two identical scalar types, returns the same"""
  pass

class Cmp(Prim):
  """Takes two arguments of any type, returns a boolean"""
  pass

class Round(Prim):
  """
  Rounding operations
  """
  pass

class_list = [Cmp, Bitwise, Logical, Arith, Float, Round]

abs = Float(np.abs, doc = "Absolute value")
sqrt = Float(np.sqrt)

exp = Float(np.exp)
exp2 = Float(np.exp2)
expm1 = Float(np.expm1)

log = Float(np.log)
log10 = Float(np.log10)
log2 = Float(np.log2)
log1p = Float(np.log1p)

cos = Float(np.cos)
cosh = Float(np.cosh)
arccos = Float(np.arccos)
arccosh = Float(np.arccosh)

sin = Float(np.sin)
sinh = Float(np.sinh)
arcsin = Float(np.arcsin)
arcsinh = Float(np.arcsinh)

# TODO: figure out how to derive type table for this:
# sinc = Float(np.sinc)

tan = Float(np.tan)
tanh = Float(np.tanh)
arctan = Float(np.arctan)
arctan2 = Float(np.arctan2)
arctanh = Float(np.arctanh)

logical_and = Logical(np.logical_and, "And")
logical_not = Logical(np.logical_not, "Not")
logical_or = Logical(np.logical_or, "Or")
#logical_xor = Logical(np.logical_xor, 'BitXor')

bitwise_not = Bitwise(np.bitwise_not, 'Invert', '!')
bitwise_and = Bitwise(np.bitwise_and, 'BitAnd', '&')
bitwise_or = Bitwise(np.bitwise_or, 'BitOr', '|')
bitwise_xor = Bitwise(np.bitwise_xor, 'BitXor', '^')

# Adding booleans in Python results in an integer, 
# but add *arrays* of booleans gives a boolean result
# ----
# Since Parakeet unifies the scalar and broadcast behavior of 
# primitive functions, I had to pick one of these two behaviors
# and went with Boolean + Boolean = Integer (since np.mean(bool) is useful) 
add = Arith(np.add, 'Add', '+')

subtract = Arith(np.subtract, 'Sub', '-')
multiply = Arith(np.multiply, 'Mult', '*')


divide = Arith(np.divide, 'Div', '/', extra_signatures = ['??->?'])

remainder = Arith(np.remainder, 'Mod', '%', extra_signatures = ['??->?'])
mod = remainder 
fmod = Arith(np.fmod, doc = "Return the element-wise remainder of division. C-style modulo.")


# used to be Arith but easier if result is always floating point 
power = Float(np.power, 'Pow', '**')
# power_int = Arith(np.power, extra_signatures = [''])


negative = Arith(np.negative, 'USub', '-', None, 1, 1)
maximum = Arith(np.maximum, None, None, 'maximum', 2, 1)
minimum = Arith(np.minimum, None, None, 'minimum', 2, 1)


equal = Cmp(np.equal, 'Eq', '==')
not_equal = Cmp(np.not_equal, 'NotEq', '!=')
less = Cmp(np.less, 'Lt', '<')
less_equal = Cmp(np.less_equal, 'LtE', '<=')
greater = Cmp(np.greater, 'Gt', '>')
greater_equal = Cmp(np.greater_equal, 'GtE', '>=')

is_ = Cmp(lambda x,y: x is y, 'Is', 'is')

trunc = Round(np.trunc)
rint = Round(np.rint)
floor = Round(np.floor)
ceil = Round(np.ceil)
round = Round(np.round) 


