import llvm.core as llcore
 
from .. ndtypes  import ScalarT, FloatT, Int32, Int64
from llvm_types import llvm_value_type
 
def const(python_scalar, parakeet_type):
  assert isinstance(parakeet_type, ScalarT)
  llvm_type = llvm_value_type(parakeet_type)
  if isinstance(parakeet_type, FloatT):
    return llcore.Constant.real(llvm_type, float(python_scalar))
  else:
    return llcore.Constant.int(llvm_type, int(python_scalar))

def int32(x):
  """Make LLVM constants of type int32"""
  return const(x, Int32)

def int64(x):
  return const(x, Int64)

def zero(llvm_t):
  """
  Make a zero constant of either int or real type. 
  Doesn't (yet) work for vector constants! 
  """
  if isinstance(llvm_t, llcore.IntegerType):
    return llcore.Constant.int(llvm_t, 0)
  else:
    return llcore.Constant.real(llvm_t, 0.0)
  
def one(llvm_t):
  """
  Make a constant 1 of either int or real type. 
  Doesn't (yet) work for vector constants!
  """
  if isinstance(llvm_t, llcore.IntegerType):
    return llcore.Constant.int(llvm_t, 1)
  else:
    return llcore.Constant.real(llvm_t, 1.0)
  
  