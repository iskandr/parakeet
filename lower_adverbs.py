import core_types
import syntax
import syntax_helpers
import transform
import type_inference

from adverb_semantics import AdverbSemantics
from transform import Transform

class CodegenSemantics(Transform):
  # Can't put type inference related methods inside Transform
  # since this create a cyclic dependency with InsertCoercions

  def invoke_type(self, closure, args):
    closure_t = closure.type
    arg_types = syntax_helpers.get_types(args)
    assert all( isinstance(t, core_types.Type) for t in arg_types), \
      "Invalid types: %s" % (arg_types, )
    return type_inference.invoke_result_type(closure_t, arg_types)

  def invoke(self, closure, args):
    call_result_t = self.invoke_type(closure, args)
    call = syntax.Invoke(closure, args, type = call_result_t)
    return self.assign_temp(call, "invoke_result")

  def size_along_axis(self, value, axis):
    return self.shape(value, axis)

  def check_equal_sizes(self, sizes):
    pass

  none = syntax_helpers.none
  null_slice = syntax_helpers.slice_none

class LowerAdverbs(CodegenSemantics, AdverbSemantics):
  def transform_Map(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis)

  def transform_Reduce(self, expr):
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    init = self.transform_expr(expr.init) if expr.init else None 
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_expr(expr.init) if expr.init else None 
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    assert len(args) == 2
    x,y = args 
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)

  def post_apply(self, fn):
    print "POST ADVERB LOWERING"
    print fn 
def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
