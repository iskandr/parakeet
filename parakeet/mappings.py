import math  
import numpy as np
import lib, prims   

property_mappings = {
  'dtype' : lib.elt_type,                
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
  'any' : lib.any_, 
  'all' : lib.all_, 
  'argmax' : lib.argmax, 
  # 'argsort' : lib.argsort, 
  'copy' : lib.copy, 
  'cumprod' : lib.cumprod, 
  'cumsum' : lib.cumsum, 
  # 'diagonal' : lib.diagonal, 
  'min' : lib.reduce_min, 
  'max' : lib.reduce_max, 
  'ravel' : lib.ravel, 
  'flatten' : lib.ravel, 
  'sum' : lib.sum_, 
}

function_mappings = {
                     
  map : lib.map, 
  reduce : lib.reduce, 
 
  np.array : lib.identity, 
  tuple : lib.tuple_, 
 
  int : lib.int64, 
  long : lib.int64, 
  np.int8 : lib.int8, 
  np.int16 : lib.int16, 
  np.int32 : lib.int32, 
  np.int64 : lib.int64, 
  
  float : lib.float64, 
  np.float32 : lib.float32, 
  np.float64 : lib.float64, 
  
  bool : lib.bool, 
  np.bool : lib.bool, 
  np.bool8 : lib.bool, 
  np.bool_ : lib.bool, 
  
  np.rank : lib.rank, 
  len : lib.len_, 
  np.alen : lib.len_, 
  np.real : lib.real, 
  # np.imag : lib.imag
  np.size : lib.size, 
  
  min : lib.builtin_min,
  max : lib.builtin_max, 
  np.min : lib.reduce_min, 
  np.max : lib.reduce_max, 
  np.minimum : prims.minimum, 
  np.maximum : prims.maximum, 
  
  np.argmin : lib.argmin, 
  np.argmax : lib.argmax, 
  
  all : lib.all_, 
  np.all : lib.all_, 
  any : lib.any_, 
  np.any : lib.any_, 
  
  sum : lib.sum_, 
  np.sum : lib.sum_, 
  np.prod : lib.prod, 
  np.mean : lib.mean, 
  
  abs : prims.abs, 
  np.abs : prims.abs, 
  
  range : lib.arange, 
  xrange : lib.arange, 
  np.arange  : lib.arange, 
  
  np.empty_like : lib.empty_like, 
  np.empty : lib.empty, 
  np.zeros_like : lib.zeros_like, 
  np.zeros : lib.zeros, 
  np.ones : lib.ones, 
  np.ones_like : lib.ones_like, 
  
  np.add : prims.add, 
  np.subtract : prims.subtract, 
  np.multiply : prims.multiply, 
  np.divide : prims.divide, 
  
  np.logical_and : prims.logical_and, 
  np.logical_not : prims.logical_not, 
  np.logical_or : prims.logical_or,  
  
  np.cos : prims.cos, 
  math.cos : prims.cos, 
  np.sin : prims.sin, 
  math.sin : prims.sin, 
  np.tan : prims.tan, 
  math.tan : prims.tan, 
  
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
  
  math.sqrt : prims.sqrt, 
  np.sqrt : prims.sqrt, 
 
}
