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

  def rank(self, value):
    return value.type.rank

  def accumulator(self, v):
    return self.loop_counter("acc", v)

  def get_acc(self, (var_before, var_after, merge)):
    return var_before

  def set_acc(self, (var_before, var_after, merge), v):
    self.assign(var_after, v)

  def const_int(self, x):
    return syntax_helpers.const_int(x)

  def array(self, size, elt):
    assert isinstance(size, syntax.Const)
    return self.alloc_array(self, elt.type, size)

  def shift_array(self, arr, offset):
    assert False
    # return arr[offset:]

  def setidx(self, arr, idx, v):
    #print "arr", arr
    #print "idx", idx
    #print "value", v

    # arr[idx] = v
    assert False

  def check_equal_sizes(self, sizes):
    pass

  def slice_value(self, start, stop, step):
    return syntax.Slice(start, stop, step)

  none = syntax_helpers.none
  null_slice = syntax_helpers.slice_none
  identity_function = None #function_registry.identity_function
  trivial_combiner = None #function_registry.return_second

class LowerAdverbs(CodegenSemantics, AdverbSemantics):
  def transform_Map(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_map(fn, args, axis)

  def transform_Reduce(self, expr):

    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    init = self.transform_expr(expr.init)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, init, args, axis)

  def transform_Scan(self, expr):
    fn = self.transform_expr(expr.fn)
    combine = self.transform_expr(expr.combine)
    emit = self.transform_expr(expr.emit)
    init = self.transform_expr(expr.init)
    args = self.transform_expr_list(expr.args)
    axis = syntax_helpers.unwrap_constant(expr.axis)
    return self.eval_reduce(fn, combine, emit, init, args, axis)

  def transform_AllPairs(self, expr):
    fn, args, axis = self.adverb_prelude(expr)

    if all( arg.type.rank == 0 for arg in args ):
      return syntax.Invoke(expr.fn, args, type = expr.type)

    x, y = args
    nx = self.shape(x, axis)
    ny = self.shape(y, axis)

    elt_t = expr.type.elt_type
    array_result = self.alloc_array(elt_t, (nx, ny))

    i, i_after, merge_i = self.loop_counter("i")
    cond_i = self.lt(i, nx)
    self.blocks.push()

    j, j_after, merge_j = self.loop_counter("j")
    cond_j = self.lt(j, ny)
    self.blocks.push()

    nested_args = [self.index_along_axis(x, axis, i),
                   self.index_along_axis(y, axis, j)]
    invoke = self.invoke(fn, nested_args)
    indices = self.tuple([i, j], "indices")
    output_idx = syntax.Index(array_result, indices, type = invoke.type)
    self.assign(output_idx, invoke)

    self.assign(j_after, self.add(j, syntax_helpers.one_i64))
    inner_body = self.blocks.pop()
    self.blocks += syntax.While(cond_j, inner_body, merge_j )

    self.assign(i_after, self.add(i, syntax_helpers.one_i64))

    outer_body = self.blocks.pop()
    self.blocks += syntax.While(cond_i, outer_body, merge_i)
    return array_result

def lower_adverbs(fn):
  return transform.cached_apply(LowerAdverbs, fn)
