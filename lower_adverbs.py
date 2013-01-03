import syntax_helpers

from adverb_semantics import AdverbSemantics
from transform import Transform, apply_pipeline

class LowerAdverbs(AdverbSemantics, Transform):
  def transform_TypedFn(self, expr):
    return lower_adverbs(expr)

  def transform_Map(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis)

  def transform_Reduce(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    init = self.transform_if_expr(expr.init)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_if_expr(expr.init)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_scan(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    assert len(args) == 2
    x,y = self.transform_expr_list(args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)

import config
from simplify import Simplify
from dead_code_elim import DCE
import inline
from  clone_function import CloneFunction
_cache = {}

def lower_adverbs(fn):
  if fn.name in _cache:
    return _cache[fn.name]
  else:
    pipeline = [CloneFunction, LowerAdverbs, Simplify, DCE]
    if config.opt_inline:
      pipeline.append(inline.Inliner)
      pipeline.append(Simplify)
      pipeline.append(DCE)
    new_fn = apply_pipeline(fn, pipeline)
    _cache[fn.name] = new_fn
    _cache[new_fn.name] = new_fn
    return new_fn
