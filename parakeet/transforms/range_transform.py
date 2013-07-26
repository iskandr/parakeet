from ..analysis import ValueRangeAnalyis
from ..syntax import Var, Const 
from ..syntax.helpers import true, false 
from transform import Transform

class RangeTransform(Transform):
  """
  Base class for transforms which use value ranges gathered from analysis
  """
  
  def pre_apply(self, old_fn):
    analysis = ValueRangeAnalyis()
    analysis.visit_fn(old_fn)
    self.ranges = analysis.ranges 
  
  def get(self, x):
    if x.__class__ is Var and x.name in self.ranges:
      return self.ranges[x.name]
    elif x.__class__ is Const:
      return (x.value, x.value)
    else:
      return None   
  
  def cmp_eq(self, x, y):
    if x is None or y is None:
      return None
    lx, ux = x
    ly, uy = y 
    if lx == ly and ux == uy:
      return true 
    if ux < ly or uy < lx:
      return false
    return None   
  
  def cmp_lt(self, x, y):
    if x is None or y is None:
      return None
    lx, ux = x
    ly, uy = y
    if ux < ly:
      return true 
    elif lx > uy:
      return false
    else:
      return None 
  
  def cmp_lte(self, x, y):
    if x is None or y is None:
      return None
    lx, ux = x
    ly, uy = y
    if ux < ly:
      return true 
    elif lx > uy:
      return false
    else:
      return None 