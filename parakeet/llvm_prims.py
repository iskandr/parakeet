

import prims
import llvm.core as llc   

  

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


import core_types 
import llvm_types

from llvm_context import global_context

float32_fn_t = llc.Type.function(llvm_types.float32_t, [llvm_types.float32_t])
float64_fn_t = llc.Type.function(llvm_types.float64_t, [llvm_types.float64_t])

def float32_fn(name):
  return global_context.module.add_function(float32_fn_t, name)

def float64_fn(name):
  return global_context.module.add_function(float64_fn_t, name)

float_unary_ops_list = [ 
  prims.tan, prims.tanh, prims.cos, prims.cosh, prims.sin, prims.sinh, 
  prims.log, prims.log10, prims.sqrt,
  prims.exp 
]

float_unary_ops = {}

def get_float_unary_op(prim, t):
  key = (prim, t)
  if key in float_unary_ops:
    return float_unary_ops[key]
  assert prim in float_unary_ops_list, \
    "Unsupported float primitive %s" % prim 
  assert t in (core_types.Float32, core_types.Float64), \
    "Invalid type %s, expected Float32 or Float64" % t
  fn_t = float32_fn_t if t == core_types.Float32 else float64_fn_t    
  llvm_value = global_context.module.add_function(fn_t, prim.name)
  float_unary_ops[key] = llvm_value 
  return llvm_value 
