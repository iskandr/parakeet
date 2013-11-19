from dsltools import Traversal
 
import types 
import numpy as np 

def transform_value(x):
  """
  Replace arrays with their shapes, 
  and recursively replace any instances of arrays
  in data structures like tuples also with their shapes
  """
  if isinstance(x, np.ndarray):
    return x.shape 
  elif isinstance(x, list):
    return np.array(x).shape 
  elif isinstance(x, tuple):
    return tuple(transform_value(elt) for elt in x)

  elif isinstance(x, (int, long, float, complex, types.NoneType)):
    return x
  else:
    return None

class EvalShape(Traversal):

  def __init__(self, input_values):
    
    self.inputs = []
    for x in input_values:
      y = transform_value(x)
      if hasattr(y, '__iter__'):
        self.inputs.extend(y)
      else:
        self.inputs.append(y)
  
  def visit_Var(self, v):
    return self.inputs[v.num]  
  
  def visit_Const(self, v):
    return v.value 

  def visit_AnyScalar(self, v):
    return None 
    
  def visit_Shape(self, v):
    dims = self.visit_tuple(v.dims)
    assert all(isinstance(d, int) for d in dims)
    return dims
    
  def visit_Dim(self, v):
    return self.visit(v.array)[v.dim]
    
  def visit_Tuple(self, v):
    return self.visit_tuple(v.elts)
    
  def visit_Sub(self, v):
    return self.visit(v.x) - self.visit(v.y)
  
  def visit_Add(self, v):
    return self.visit(v.x) + self.visit(v.y)
  
  def visit_Mult(self, v):
    return self.visit(v.x) * self.visit(v.y)  
  
  def visit_Div(self, v):
    return self.visit(v.x) / self.visit(v.y)  
  
  def visit_Mod(self, v):
    return self.visit(v.x) % self.visit(v.y)  
  
  def visit_Closure(self, v):
    return v.fn, self.visit_tuple(v.args)

def eval_shape(symbolic_shape, input_values):
  evaluator = EvalShape(input_values)
  result = evaluator.visit(symbolic_shape)
  if not isinstance(result, tuple):
    return () 
  else:
    assert all(isinstance(elt, int) for elt in result)
    return result
   

def result_shape(typed_fn, input_values):
  import shape_inference
  symbolic_shape = shape_inference.call_shape_expr(typed_fn)
  return eval_shape(symbolic_shape, input_values)

      
  