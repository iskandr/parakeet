
import llvm.core as llc   

import llvm_types
from .. import prims
from .. ndtypes import Float32, Float64, Int32, Int64
from llvm_context import global_context

  

signed_int_comparisons = {
  prims.equal : llc.ICMP_EQ, 
  prims.not_equal : llc.ICMP_NE, 
  prims.greater : llc.ICMP_SGT, 
  prims.greater_equal : llc.ICMP_SGE, 
  prims.less : llc.ICMP_SLT, 
  prims.less_equal : llc.ICMP_SLE
}

unsigned_int_comparisons = {
  prims.equal : llc.ICMP_EQ, 
  prims.not_equal : llc.ICMP_NE, 
  prims.greater : llc.ICMP_UGT, 
  prims.greater_equal : llc.ICMP_UGE, 
  prims.less : llc.ICMP_ULT, 
  prims.less_equal : llc.ICMP_ULE
}  

float_comparisons = { 
  prims.equal : llc.FCMP_OEQ, 
  prims.not_equal : llc.FCMP_ONE, 
  prims.greater : llc.FCMP_OGT, 
  prims.greater_equal : llc.FCMP_OGE, 
  prims.less : llc.FCMP_OLT, 
  prims.less_equal : llc.FCMP_OLE                     
}


signed_binops = { 
  prims.add : 'add',  
  prims.subtract : 'sub', 
  prims.multiply : 'mul', 
  prims.divide : 'sdiv',
}

unsigned_binops = { 
  prims.add : 'add',  
  prims.subtract : 'sub', 
  prims.multiply : 'mul', 
  prims.divide : 'udiv', 
  prims.mod : 'urem', 
}

float_binops = { 
  prims.add : 'fadd',  
  prims.subtract : 'fsub', 
  prims.multiply : 'fmul', 
  prims.divide : 'fdiv', 
}

# Note: there is no division instruction between booleans
# so b1 / b2 should be translated to int(b1) / int(b2) 
bool_binops = { 
  prims.add : 'or_',
  prims.multiply : 'and_', 
  prims.subtract : 'xor',
  prims.divide : 'and_',
  
  prims.mod : 'urem'
  #prims.logical_not : 'not_'
}

int32_fn_t = llc.Type.function(llvm_types.int32_t, [llvm_types.int32_t])
int64_fn_t = llc.Type.function(llvm_types.int64_t, [llvm_types.int64_t])
float32_fn_t = llc.Type.function(llvm_types.float32_t, [llvm_types.float32_t])
float64_fn_t = llc.Type.function(llvm_types.float64_t, [llvm_types.float64_t])

binary_float32_fn_t = llc.Type.function(llvm_types.float32_t, 
                                        [llvm_types.float32_t, llvm_types.float32_t])
binary_float64_fn_t = llc.Type.function(llvm_types.float64_t, 
                                        [llvm_types.float64_t, llvm_types.float64_t])

def float32_fn(name):
  return global_context.module.add_function(float32_fn_t, name)

def float64_fn(name):
  return global_context.module.add_function(float64_fn_t, name)



#import llvmmath 
#mathlib = llvmmath.get_default_math_lib()
#linker = llvmmath.linking.get_linker(mathlib)

_llvm_intrinsics = set(['sqrt', 
                        'powi', 
                        'sin', 
                        'cos',
                        'pow',
                        'exp', 
                        'exp2', 
                        'log', 
                        'log10', 
                        'log2', 
                        'fma', 
                        'fabs', 
                        'floor', 
                        'ceil', 
                        'trunc', 
                        'rint', 
                        'nearbyint',                         
                      ])

from ..c_backend.c_prims import _float_fn_names

def get_float_op(prim, t, _float_decls = {}):
  key = (prim, t)
  if key in _float_decls:
    return _float_decls[key]
  
  assert prim in _float_fn_names, \
    "Unsupported float primitive %s" % prim 
  assert t in (Float32, Float64), \
    "Invalid type %s, expected Float32 or Float64" % t
  
  prim_name = _float_fn_names[prim]
  
  if t == Float32:
    fn_t = float32_fn_t if prim.nin == 1 else binary_float32_fn_t
    if prim_name in _llvm_intrinsics:
      fn_name = "llvm.%s.f32" % prim_name
    else:
      fn_name = prim_name + "f" 
  else:
    assert t == Float64
    fn_t = float64_fn_t if prim.nin == 1 else binary_float64_fn_t
    if prim_name in _llvm_intrinsics:
      fn_name = "llvm.%s.f64" % prim_name 
    else:
      fn_name = prim_name 
  llvm_value = global_context.module.get_or_insert_function(fn_t, fn_name)
  llvm_value.add_attribute(llc.ATTR_NO_UNWIND)
  llvm_value.add_attribute(llc.ATTR_READONLY)
  _float_decls[key] = llvm_value
  return llvm_value 

