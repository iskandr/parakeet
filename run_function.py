import ast_conversion
import llvm_backend
import syntax
 
import type_inference

from llvm.ee import GenericValue

import llvm_types
from llvm_context import opt_context
import core_types
import type_conv
import ctypes


def python_to_generic_value(x, t):
  if isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.real(llvm_t, x)
  elif isinstance(t, core_types.IntT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int(llvm_t, x)
  elif isinstance(t, core_types.PtrT):
    return GenericValue.pointer(x)
  else:
    ctypes_obj = type_conv.from_python(x)
    return GenericValue.pointer(ctypes.addressof(ctypes_obj))

def ctypes_to_generic_value(cval, t):
  if isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.real(llvm_t, cval.value)
  elif isinstance(t, core_types.IntT):
    llvm_t = llvm_types.llvm_value_type(t)
    return GenericValue.int(llvm_t, cval.value)
  elif isinstance(t, core_types.PtrT):
    return GenericValue.pointer(ctypes.addressof(cval.contents))
  else:
    return GenericValue.pointer(ctypes.addressof(cval))

def generic_value_to_python(gv, t):
  if isinstance(t, core_types.IntT):
    return t.dtype.type(gv.as_int())
  elif isinstance(t, core_types.FloatT):
    llvm_t = llvm_types.ctypes_scalar_to_lltype(t.ctypes_repr)
    return t.dtype.type(gv.as_real(llvm_t))
  else:
    addr = gv.as_pointer()
    struct = t.ctypes_repr.from_address(addr)

    return t.to_python(struct)

class CompiledFn:
  def __init__(self, llvm_fn, parakeet_fn,
               exec_engine = opt_context.exec_engine):
    self.llvm_fn = llvm_fn
    self.parakeet_fn = parakeet_fn
    self.exec_engine = exec_engine

  def __call__(self, *args):
    actual_types = map(type_conv.typeof, args)
    expected_types = self.parakeet_fn.input_types
    assert actual_types == expected_types, \
      "Arg type mismatch, expected %s but got %s" % \
      (expected_types, actual_types)

    # calling conventions are that output must be preallocated by the caller'
    ctypes_inputs = [t.from_python(v) for (v,t) in zip(args, expected_types)]
    gv_inputs = [ctypes_to_generic_value(cv, t) for (cv,t) in
                 zip(ctypes_inputs, expected_types)]
    gv_return = self.exec_engine.run_function(self.llvm_fn, gv_inputs)
    return generic_value_to_python(gv_return, self.parakeet_fn.return_type)

def specialize_and_compile(fn, args):
  """
  Translate, specialize, optimize, and compile the given 
  function for the types of the supplies arguments. 
  
  Return the untyped, typed, and compiled representation,
  along with all the arguments needed to actually execute. 
  """
  if isinstance(fn, syntax.Fn):
    untyped = fn
  else:
    # translate from the Python AST to Parakeet's untyped format
    untyped  = ast_conversion.translate_function_value(fn)
  all_args = untyped.python_nonlocals() + list(args)

  # get types of all inputs
  input_types = [type_conv.typeof(arg) for arg in all_args]

  # propagate types through function representation and all
  # other functions it calls
  typed = type_inference.specialize(untyped, input_types)

  # compile to native code
  llvm_fn, parakeet_fn, exec_engine = \
    llvm_backend.compile_fn(typed)
  compiled_fn_wrapper = CompiledFn(llvm_fn, parakeet_fn, exec_engine)
  return untyped, typed, compiled_fn_wrapper, all_args

def run(fn, *args):
  """
  Given a python function, run it in Parakeet on the supplied args
  """
  _, _,  compiled, all_args = specialize_and_compile(fn, args)
  return compiled(*all_args)


