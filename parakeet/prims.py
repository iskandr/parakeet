import core_types
import names
import numpy as np
import syntax

from args import FormalArgs

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

_untyped_prim_wrappers = {}
def prim_wrapper(p):
  """Given a primitive, return an untyped function which calls that prim"""
  if p in _untyped_prim_wrappers:
    return _untyped_prim_wrappers[p]
  else:
    fn_name = names.fresh(p.name)

    args_obj = FormalArgs()

    arg_vars = []
    for name in names.fresh_list(p.nin):
      args_obj.add_positional(name)
      arg_vars.append(syntax.Var(name))
    body = [syntax.Return(syntax.PrimCall(p, arg_vars))]
    fundef = syntax.Fn(fn_name, args_obj, body, [])
    _untyped_prim_wrappers[p] = fundef

    return fundef

class Prim(object):
  def __init__(self, fn, python_op_name = None, symbol = None,
               name = None, nin = None, nout = None):
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
    self.wrapper = prim_wrapper(self)

  def _create_type_table(self):
    # for now only support ufuncs which describe their own type behavior
    if hasattr(self.fn, 'types'):
      "Primitive function %s doesn't supply type signatures" % self.name

      self.type_table = {}
      for signature in self.fn.types:
        # numpy type signatures look like 'ff->f' where each character
        # represents a single type

        arg_codes, result_code = signature.split('->')
        try:
          input_types = tuple([core_types.from_char_code(c) for c in arg_codes])

          result_type = core_types.from_char_code(result_code)
          self.type_table[input_types] = result_type
        except:
          pass

  def __eq__(self, other):
    return self.fn == other.fn

  def __hash__(self):
    return hash(self.name)

  def __call__(self, *args, **kwds):
    return self.fn(*args, **kwds)

  def __repr__(self):
    return "prim(%s)" % self.name

  def expected_input_types(self, arg_types):
    """Given some argument types, return the desired upcast types"""
    # by default we just figure out the common type and expect every arg to be
    # of that type
    n_inputs = len(arg_types)
    assert n_inputs == self.nin, \
        "Incorrect number of argument types for %s, expected %s but given %d" \
        % (self.name, self.nin, n_inputs)
    common_type = core_types.combine_type_list(arg_types)
    return [common_type] * n_inputs

  def result_type(self, arg_types):
    """
    Given some argument types, look up the result type in the type_table we
    generated from numpy's given signatures
    """
    key = tuple(arg_types)
    if key not in self.type_table:
      raise RuntimeError("Primitives %s doesn't support input types %s || %s" %
                         (self.name, key, self.type_table))
    else:
      return self.type_table[key]

class Float(Prim):
  """Always returns a float"""
  pass

class Arith(Prim):
  """Basic arithmetic operators"""
  pass

class Logical(Prim):
  """Expects boolean inputs, returns a boolean"""
  pass

class Bitwise(Prim):
  """Takes any two identical scalar types, returns the same"""
  pass

class Cmp(Prim):
  """Takes two arguments of any type, returns a boolean"""
  pass

class_list = [Cmp, Bitwise, Logical, Arith, Float]

sqrt = Float(np.sqrt)
exp = Float(np.exp)
log = Float(np.log)
sqrt = Float(np.sqrt)
log10 = Float(np.log10)
log2 = Float(np.log2)
cos = Float(np.cos)
cosh = Float(np.cosh)
sin = Float(np.sin)
sinh = Float(np.sinh)
# TODO: figure out how to derive type table for this:
# sinc = Float(np.sinc)
tan = Float(np.tan)
tanh = Float(np.tanh)


logical_and = Logical(np.logical_and, "And")
logical_not = Logical(np.logical_not, "Not")
logical_or = Logical(np.logical_or, "Or")
#logical_xor = Logical(np.logical_xor, 'BitXor')

bitwise_not = Bitwise(np.bitwise_not, 'Invert', '!')
bitwise_and = Bitwise(np.bitwise_and, 'BitAnd', '&')
bitwise_or = Bitwise(np.bitwise_or, 'BitOr', '|')
bitwise_xor = Bitwise(np.bitwise_xor, 'BitXor', '^')

add = Arith(np.add, 'Add', '+')
subtract = Arith(np.subtract, 'Sub', '-')
multiply = Arith(np.multiply, 'Mult', '*')
divide = Arith(np.divide, 'Div', '/')
mod = Arith(np.mod, 'Mod', '%')
power = Arith(np.power, 'Pow', '**')
#negative = Arith(np.negative, 'USub', '-', None, 1, 1)

equal = Cmp(np.equal, 'Eq', '==')
not_equal = Cmp(np.not_equal, 'NotEq', '!=')
less = Cmp(np.less, 'Lt', '<')
less_equal = Cmp(np.less_equal, 'LtE', '<=')
greater = Cmp(np.greater, 'Gt', '>')
greater_equal = Cmp(np.greater_equal, 'GtE', '>=')
