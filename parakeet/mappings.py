import numpy as np 
import lib 
import prims
import math  

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
  'any' : lib.any, 
  'all' : lib.all, 
  'argmax' : lib.argmax, 
  # 'argsort' : lib.argsort, 
  'copy' : lib.copy, 
  'cumprod' : lib.cumprod, 
  'cumsum' : lib.cumsum, 
  # 'diagonal' : lib.diagonal, 
  'min' : lib.min, 
  'max' : lib.max, 
  'ravel' : lib.ravel, 
  'flatten' : lib.ravel, 
}

function_mappings = {
  int : lib.Int64, 
  bool : lib.bool8, 
  float : lib.float64, 
  
  np.rank : lib.rank, 
  len : lib.alen, 
  np.alen : lib.alen, 
  np.real : lib.real, 
  # np.imag : lib.imag
  np.size : lib.size, 
  
  min : lib._prim_min,
  max : lib._prim_max, 
  np.min : lib.min, 
  np.max : lib.max, 
  np.minimum : lib.minimum, 
  np.maximum : lib.maximum, 
  
  all : lib.all, 
  np.all : lib.all, 
  any : lib.any, 
  np.any : lib.any, 
  
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
  
}
