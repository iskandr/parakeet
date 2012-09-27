import numpy as np

from llvm import *
from llvm.core import *
from llvm.ee import *
from llvm.ee import GenericValue as gv
import llvm.core as llcore
from llvm.core import Type as lltype
import llvm.passes as passes

void_t = lltype.void()
int1_t = lltype.int(1)
int8_t = lltype.int(8)
int16_t = lltype.int(16)
int32_t = lltype.int(32)
int64_t = lltype.int(64)
float32_t = lltype.float()
float64_t = lltype.double()

ptr_int32_t = lltype.pointer(int32_t)

numpy_scalar_to_llvm_types = {
  np.int8 : int8_t,
  np.int16 : int16_t,
  np.int32 : int32_t,
  np.int64 : int64_t,
  np.float32 : float32_t,
  np.float64 : float64_t,
}

def dtype_to_lltype(dt):
  return numpy_scalar_to_llvm_types[dt]

def to_lltype(t):
  elt_t = dtype_to_lltype(t.dtype())
  if isinstance(t, types.Scalar):
    return elt_t
  elif isinstance(t, types.Tuple):
    llvm_elt_types = map(to_llvm_type, t.elt_types)
    return lltype.struct(llvm_elt_types)
  else:
    arr_t = lltype.pointer(elt_t)
    # arrays are a pointer to their data and
    # pointers to shape and strides arrays
    return lltype.struct([arr_t, ptr_int32_t, ptr_int32_t])

# we allocate heap slots for output scalars before entering the
# function
def to_llvm_output_type(t):
  llvm_type = to_llvm_type(t)
  if isinstance(t, types.Scalar):
    return lltype.pointer(llvm_type)
  else:
    return llvm_type

global_module = llcore.Module.new("global_module")

def llvm_compile(specialized):
  llvm_input_types = map(to_llvm_type, specialized.input_types)
  llvm_output_types = map(to_llvm_output_type, specialized.output_types)
  llvm_fn_t = lltype.function(void_t, llvm_input_types + llvm_output_types)
  fn = llcore.add_function(llvm_fn_t, specialized.name)
  n_inputs = specialized.input_arity()
  for i, arg in enumerate(fn.args):
    if i < n_inputs:
      arg.name = specialized.input_ids[i]
    else:
      arg.name = specialized.output_ids[i - n_inputs]

  bb = fn.append_basic_block("entry")
  builder = Builder.new(bb)
  return fn

def scalar_to_generic_value(x):
  if isinstance(x, int) or isinstance(x, long):
    return gv.int(int64_t, x)
  elif isinstance(x, float):
    return gv.real(float64_t, x)
  elif isinstance(x, bool):
    return gv.int(int8_t, x)
  # if it's a numpy scalar integer
  elif isinstance(x, np.integer):
    return gv.int(dtype_to_lltype(x.dtype()), x)
  elif isinstance(x, np.floating):
    return gv.real(dtype_to_lltype(x.dtype()), x)
  else:
    raise RuntimeError("Don't know how to convert value " + str(x))

# Given a list of Python values, return a list of generic values
# which will be understood by the LLVM execution engine
# The returned list might possibly be longer since array arguments
# are passed in as the data pointer followed by shape and strides pointers
def convert_args_to_generic_values(python_values):

def llvm_run(llvm_fn, python_values):
  inputs = []
  for v in python_values:
    inputs.append(v.data)
    if isinstance(v, np.ndarray)
      # convert the shape and strides tuples
      # into uniform arrays so we can pass them
      # as an int* rather than PyObject*
      shape = np.array(v.shape)
      strides = np.array(v.strides)
      # the generated LLVM code will expect
      # the shape and strides of an array to simply
      # be passed after the array's data pointer
      inputs.append(shape.data)
      inputs.append(strides.data)
    else:
      assert isinstance(v, tuple)


