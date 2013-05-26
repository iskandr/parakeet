import numpy as np 
import lib 
import prims 

property_mappings = {
  'dtype' : lib.elt_type,                
  'imag' : lib.imag,      
  'itemsize' : lib.itemsize,
  'real' :  lib.real, 
  'shape' : lib.shape, 
  'size' : lib.size,   
  'strides' : lib.strides, 
  'T' : lib.transpose,     
}

method_mappings = {
  'fill' : lib.fill, 
  'any' : lib.any, 
  'all' : lib.all, 
  'argmax' : lib.argmax, 
  'argsort' : lib.argsort, 
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
}
