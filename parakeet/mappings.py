import math  
import numpy as np
import lib, prims   
from syntax import Zip, Len


function_mappings = {
  # PYTHON BUILTINS          
  zip : Zip, 
  map : lib.map, 
  reduce : lib.reduce, 
  tuple : lib.builtin_tuple, 
  range : lib.arange, 
  xrange : lib.arange, 
  float : lib.numpy_types.float64, 
  int : lib.numpy_types.int64, 
  long : lib.numpy_types.int64, 
  bool : lib.numpy_types.bool, 
  len : Len,  
  min : lib.builtin_min,
  max : lib.builtin_max,
  all : lib.reduce_all, 
  any : lib.reduce_any, 
  sum : lib.reduce_sum, 
  abs : prims.abs,  
  pow : prims.power, 
  
  # TYPES 
  np.int8 : lib.numpy_types.int8, 
  np.int16 : lib.numpy_types.int16, 
  np.int32 : lib.numpy_types.int32, 
  np.int64 : lib.numpy_types.int64, 
  np.uint8 : lib.numpy_types.uint8,
  np.uint16 : lib.numpy_types.uint16, 
  np.uint32 : lib.numpy_types.uint32, 
  np.uint64 : lib.numpy_types.uint64,  
  np.bool : lib.numpy_types.bool, 
  np.bool8 : lib.numpy_types.bool, 
  np.bool_ : lib.numpy_types.bool, 
  np.float32 : lib.numpy_types.float32, 
  np.float64 : lib.numpy_types.float64, 
  
  np.rank : lib.rank, 
  np.alen : Len, 
  np.real : lib.real, 
  # np.imag : lib.imag
  np.size : lib.size, 
  
  np.minimum : prims.minimum, 
  np.maximum : prims.maximum, 
  
  # COMPARISONS
  np.greater : prims.greater, 
  np.greater_equal : prims.greater_equal, 
  np.equal : prims.equal, 
  np.not_equal : prims.not_equal, 
  np.less : prims.less, 
  np.less_equal : prims.less_equal, 
  
  # REDUCTIONS 
  np.min : lib.reduce_min, 
  np.max : lib.reduce_max,
  
  np.argmin : lib.argmin, 
  np.argmax : lib.argmax, 
  
  np.all : lib.reduce_all, 
  
  np.any : lib.reduce_any, 
  np.sum : lib.reduce_sum, 
  np.prod : lib.prod, 
  np.mean : lib.mean, 
  
  
  np.abs : prims.abs, 

  
  
  # ARRAY CONSTRUCTORS 
  np.array : lib.array, 
  np.tile : lib.tile, 
  np.arange  : lib.arange, 
  
  np.empty_like : lib.empty_like, 
  np.empty : lib.empty, 
  np.zeros_like : lib.zeros_like, 
  np.zeros : lib.zeros, 
  np.ones : lib.ones, 
  np.ones_like : lib.ones_like, 
  
  # ARITHMETIC 
  np.add : prims.add, 
  np.subtract : prims.subtract, 
  np.multiply : prims.multiply,
   
  np.divide : prims.divide, 
  np.floor_divide : lib.floor_divide, 
  np.true_divide : lib.true_divide, 
   
  np.mod : prims.remainder,
  np.remainder : prims.remainder,
  np.fmod : prims.fmod,
  # np.modf : prims.modf,
  
  np.logical_and : prims.logical_and, 
  np.logical_not : prims.logical_not, 
  np.logical_or : prims.logical_or,  
    
  math.sqrt : prims.sqrt, 
  np.sqrt : prims.sqrt, 
  
  np.sign : lib.sign, 
  np.reciprocal : lib.reciprocal, 
  np.conjugate : lib.conjugate,
  
  # ROUNDING
  
  np.trunc : prims.trunc, 
  math.trunc : prims.trunc,
   
  np.rint : prims.rint,
  
  np.floor : prims.floor, 
  math.floor : prims.floor,
   
  np.ceil : prims.ceil,
  math.ceil : prims.ceil, 
  
  np.round : prims.round,
 
  # LOGS AND EXPONENTIATION 
  np.square : lib.square,
  math.pow : prims.power, 
  np.power : prims.power,
  
  np.exp : prims.exp,
  math.exp : prims.exp, 
   
  np.exp2 : prims.exp2,
   
  np.expm1 : prims.expm1,
  math.expm1 : prims.expm1, 
   
  np.log : prims.log, 
  math.log : prims.log, 
  
  np.log10 : prims.log10,
  math.log10 : prims.log10, 
   
  np.log2 : prims.log2,
   
  np.log1p : prims.log1p,
  math.log1p : prims.log1p, 
  
  np.logaddexp : lib.logaddexp, 
  np.logaddexp2 : lib.logaddexp2, 
  
  # TRIG 
  np.cos : prims.cos, 
  math.cos : prims.cos, 
  np.sin : prims.sin, 
  math.sin : prims.sin, 
  np.tan : prims.tan, 
  math.tan : prims.tan, 
  np.arccos : prims.arccos, 
  math.acos: prims.arccos, 
  np.arcsin : prims.arcsin, 
  math.asin : prims.arcsin, 
  np.arctan : prims.arctan, 
  math.atan : prims.arctan, 
  np.arctan2 : prims.arctan2, 
  math.atan2 : prims.arctan2, 
  math.cosh : prims.cosh, 
  np.cosh : prims.cosh, 
  np.sinh : prims.sinh,           
  math.sinh : prims.sinh, 
  np.tanh : prims.tanh, 
  math.tanh : prims.tanh, 
  math.atanh : prims.arctanh,
  np.arctanh : prims.arctanh,  
  math.asinh : prims.arcsinh, 
  np.arcsinh : prims.arcsinh, 
  math.acosh : prims.arccosh, 
  np.arccosh : prims.arccosh,
  np.rad2deg : lib.rad2deg, 
  np.deg2rad : lib.deg2rad,  
  # np.hypot : lib.hypot,
  
  np.where : lib.where,
  np.linspace : lib.linspace, 
  np.vdot : lib.vdot, 
  np.dot : lib.dot,
  np.linalg.norm : lib.linalg.norm,  
}

property_mappings = {
  'dtype' : lib.get_elt_type,                
  # 'imag' : lib.imag,      
  'itemsize' : lib.itemsize,
  'real' :  lib.identity, # ain't no complex numbers yet 
  'shape' : lib.shape, 
  'size' : lib.size,   
  # 'strides' : lib.strides, 
  'ndim' : lib.rank, 
  'T' : lib.transpose, 
}


method_mappings = {
  'fill' : lib.fill, 
  'any' : lib.reduce_any, 
  'all' : lib.reduce_all, 
  'argmax' : lib.argmax, 
  # 'argsort' : lib.argsort, 
  'copy' : lib.copy, 
  'cumprod' : lib.cumprod, 
  'cumsum' : lib.cumsum, 
  'dot' : lib.dot, 
  'fill' : lib.fill, 
  'flatten' : lib.ravel, 
  # 'diagonal' : lib.diagonal, 
  
  'mean' : lib.mean, 
  
  'min' : lib.reduce_min,
  'max' : lib.reduce_max,
  
  'ravel' : lib.ravel, 
  'transpose' : lib.transpose,
  'sum' : lib.reduce_sum,
  }
