from ..builder import mk_cast_fn
from ..ndtypes import Int64 

from transform import Transform


class IndexifyArrayConstructors(Transform):
  """
  Given first-order array constructors, turn them into IndexMaps
  """

  def transform_Range(self, expr):
    nelts = self.elts_in_range(expr.start, expr.stop, expr.step)
    caster = mk_cast_fn(Int64, expr.type.elt_type)
    return self.imap(caster, nelts)
    
    
    