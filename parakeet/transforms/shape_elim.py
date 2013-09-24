from .. import syntax 
from ..shape_inference import shape_env, shape
from ..transforms import Transform


class ShapeElimination(Transform): 
  def pre_apply(self, fn):
    self.shape_env = shape_env(fn)

  def transform_Var(self, expr):
    if expr.name in self.shape_env:
      v = self.shape_env[expr.name]
      if v.__class__ is shape.Const:
        print "SHAPE ELIM", expr, v 
        return  syntax.Const(value = v.value, type = expr.type)
    return expr
  
  def transform_lhs(self, lhs):
    return lhs