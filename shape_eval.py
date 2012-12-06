from traversal import Traversal 
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
  else:
    assert isinstance(x, (int, long, float, complex, types.NoneType)), \
      "Unexpected value " + str(x)
    return x

class EvalShape(Traversal):

  def __init__(self, input_values):
    
    self.inputs = []
    for x in input_values:
      y = transform_value(x)
      if hasattr(y, '__iter__'):
        self.inputs.extend(y)
      else:
        self.inputs.append(y)
  
  def visit_Input(self, v):
    return self.inputs[v.pos]  
  
  def eval_Const(self, v):
    return v.value 
    
  def eval_Shape(self, v):
    return self.visit_tuple(v.dims)
    
  def eval_Dim(self, v):
    return self.visit(v.array)[v.dim]
    
  def eval_Tuple(self, v):
    return self.visit_tuple(v.elts)
    
  def eval_Sub(self, v):
    return self.visit(v.x) - self.visit(v.y)
  
  def eval_Add(self, v):
    return self.visit(v.x) + self.visit(v.y)
  
  def eval_Mult(self, v):
    return self.visit(v.x) * self.visit(v.y)  
  
  def eval_Div(self, v):
    return self.visit(v.x) / self.visit(v.y)  
  
  def eval_Mod(self, v):
    return self.visit(v.x) % self.visit(v.y)  
  
  def eval_Closure(self, v):
    return v.fn, self.eval_tuple(v.args)

def eval_shape(symbolic_shape, input_values):
  evaluator = EvalShape(input_values)
  return evaluator.visit(symbolic_shape)
    
def eval_shapes(symbolic_shapes, input_values):
  return [eval_shape()]
      
  