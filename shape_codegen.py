
from traversal import Traversal 

class ShapeCodegen(Traversal):
  def __init__(self, codegen, inputs):
    self.codegen = codegen 
    self.arrays = arrays 
    
  def visit_Input(self, v):
    return self.arrays[v.pos]  
  
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
      
  