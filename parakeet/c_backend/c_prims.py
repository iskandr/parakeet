from .. import prims
from ..ndtypes import FloatT, Float32, Float64 


# names used in math library 
_float_fn_names = {
  prims.tan : 'tan', 
  prims.tanh : 'tanh',
  prims.arctan : 'atan', 
  prims.arctanh : 'atanh',
  prims.arctan2 : 'atan2',  
   
  prims.cos : 'cos', 
  prims.cosh : 'cosh', 
  prims.arccos : 'acos', 
  prims.arccosh : 'acosh', 
  
  prims.sin : 'sin', 
  prims.sinh : 'sinh',
  prims.arcsin : 'asin',
  prims.arcsinh : 'asinh', 
  
  prims.log : 'log',
  prims.log2 : 'log2', 
  prims.log10 : 'log10',
  prims.log1p : 'log1p', 
   
   
  prims.exp : 'exp',  
  prims.exp2 : 'exp2', 
  prims.power : 'pow', 
  prims.expm1 : 'expm1', 
  prims.sqrt : 'sqrt', 
  
  prims.abs : 'fabs', 
  
  prims.ceil : 'ceil', 
  prims.floor : 'floor', 
  prims.round : 'round', 
  
} 
 
def float_prim(p, t):
  assert p in _float_fn_names
  assert isinstance(t, FloatT)
  name = _float_fn_names[p]
  if t == Float32:
    return name + "f" 
  else:
    return name 
  