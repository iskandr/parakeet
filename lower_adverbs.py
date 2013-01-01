import closure_type
import core_types
import syntax
import syntax_helpers
import type_inference

from adverb_semantics import AdverbSemantics
from transform import MemoizedTransform, apply_pipeline

class CodegenSemantics(MemoizedTransform):
  # Can't put type inference related methods inside Transform
  # since this creates a cyclic dependency with RewriteTyped

  def invoke_type(self, closure, args):
    closure_t = closure.type
    arg_types = syntax_helpers.get_types(args)
    assert all(isinstance(t, core_types.Type) for t in arg_types), \
        "Invalid types: %s" % (arg_types, )
    return type_inference.invoke_result_type(closure_t, arg_types)

  def invoke(self, fn, args):
    if fn.__class__ is syntax.TypedFn:
      closure_args = []
    else:
      assert isinstance(fn.type, closure_type.ClosureT), \
        "Unexpected function %s with type: %s" % (fn, fn.type)
      closure_args = self.closure_elts(fn)
      arg_types = syntax_helpers.get_types(args)
      fn = type_inference.specialize(fn.type, arg_types)


    lowered_fn = lower_adverbs(fn)
    combined_args = closure_args + args
    call = syntax.Call(lowered_fn, combined_args, type = lowered_fn.return_type)
    return self.assign_temp(call, "call_result")

  def size_along_axis(self, value, axis):
    return self.shape(value, axis)

  def check_equal_sizes(self, sizes):
    pass

  none = syntax_helpers.none
  null_slice = syntax_helpers.slice_none

class LowerAdverbs(AdverbSemantics, CodegenSemantics):
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
