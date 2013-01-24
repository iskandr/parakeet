import shape 
import syntax  


from shape_inference import shape_env
from transform import Transform

class ShapeElimination(Transform): 
  def pre_apply(self, fn):
    self.shape_env = shape_env(fn)

#    print "Inferred shapes"
#    for (k,v) in sorted(self.shape_env.items()):
#      print "  %s => %s" % (k,v)
  def transform_Var(self, expr):
    if expr.name in self.shape_env:
      v = self.shape_env[expr.name]
      if v.__class__ is shape.Const:
        return  syntax.Const(value = v.value, type = expr.type)
    return expr
  
  def transform_lhs(self, lhs):
    return lhs