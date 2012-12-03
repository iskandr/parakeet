import core_types
import syntax
import syntax_helpers
import transform
import type_inference
from args import ActualArgs
from adverb_semantics import AdverbSemantics
from simplify_invoke import SimplifyInvoke

class CodegenSemantics(SimplifyInvoke):
  # Can't put type inference related methods inside Transform
  # since this creates a cyclic dependency with RewriteTyped

  def invoke_type(self, closure, args):
    closure_t = closure.type
    arg_types = syntax_helpers.get_types(args)
    assert all( isinstance(t, core_types.Type) for t in arg_types), \
      "Invalid types: %s" % (arg_types, )
    return type_inference.invoke_result_type(closure_t, arg_types)

  def invoke(self, closure, args):
    call_result_t = self.invoke_type(closure, args)
    untyped_fn, args, arg_types = self.linearize_invoke(closure, args)
    print "untyped_fn", untyped_fn

    typed_fn = type_inference.specialize(untyped_fn, arg_types)
    print "high level", typed_fn
    import lowering
    fn = lowering.lower(typed_fn)
    print "lower level", fn
    call = syntax.Call(fn, args, type = call_result_t)
    return self.assign_temp(call, "call_result")

  def size_along_axis(self, value, axis):
    return self.shape(value, axis)

  def check_equal_sizes(self, sizes):
    pass

  none = syntax_helpers.none
  null_slice = syntax_helpers.slice_none

class LowerAdverbs(CodegenSemantics, AdverbSemantics):
  def transform_TypedFn(self, expr):
    import lowering
    return lowering.lower(expr)

  def transform_Map(self, expr):
    print "TRANSFORM MAP", expr
    print expr.fn
    fn, args, arg_types = self.linearize_invoke(expr.fn, expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    print expr
    for t in arg_types:
      print t
    for arg in args:
      print "arg_type:", arg.type
    return self.eval_map(fn, args, axis)

  def transform_Reduce(self, expr):
    print "TRANSFORM REDUCE"
    fn, args, _ = self.linearize_invoke(expr.fn, expr.args)
    combine = self.transform_expr(expr.combine)
    init = self.transform_expr(expr.init) if expr.init else None
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn, args, _ = self.linearize_invoke(expr.fn, expr.args)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_expr(expr.init) if expr.init else None
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_scan(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn = self.transform_expr(expr.fn)
    if isinstance(expr.args, ActualArgs):
      args = expr.args.positional
    else:
      args = expr.arg
    assert len(args) == 2
    x,y = self.transform_expr_list(args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_allpairs(fn, x, y, axis)

  def pre_apply(self, fn):
    print "before adverb lowering", fn

def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
