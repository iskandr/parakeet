import array_type
import closure_type
import core_types
import names
import prims
import shape_codegen
import shape_inference
import syntax
import syntax_helpers

from array_type import ArrayT, SliceT
from core_types import ScalarT, Int32, Int64, NoneT, Type, StructT
from closure_type import ClosureT, make_closure_type
from nested_blocks import NestedBlocks
from syntax import AllocArray, ForLoop, Comment 
from syntax import Var, Assign, Closure, Attribute, PrimCall
from syntax import Index, Const, TypedFn, Struct, ClosureElt, Cast
from syntax import TupleProj, Tuple, Alloc, Slice, While, Fn, If, Return
from syntax import ArrayView
from syntax_helpers import get_types, wrap_constants, wrap_if_constant, \
                           one_i64, zero, zero_i64, \
                           one, const_int, const_bool
from tuple_type import TupleT, make_tuple_type

class Codegen(object):
  def __init__(self):
    self.type_env = {}
    self.blocks = NestedBlocks()

    # cut down the number of created nodes by
    # remembering which tuple variables we've created
    # and looking up their elements directly
    self.tuple_elt_cache = {}

  def comment(self, text):
    self.blocks.append(Comment(text))
    
  def fresh_var(self, t, prefix = "temp"):
    assert t is not None, "Type required for new variable %s" % prefix
    ssa_id = names.fresh(prefix)
    self.type_env[ssa_id] = t
    return Var(ssa_id, type = t)

  def fresh_i32(self, prefix = "temp"):
    return self.fresh_var(Int32, prefix)

  def fresh_i64(self, prefix = "temp"):
    return self.fresh_var(Int64, prefix)

  def insert_stmt(self, stmt):
    self.blocks.append_to_current(stmt)

  def assign(self, lhs, rhs):
    self.insert_stmt(Assign(lhs, rhs))

  def temp_name(self, expr):
    c = expr.__class__
    if c is PrimCall:
      return expr.prim.name
    elif c is Attribute:
      if expr.value.__class__ is Var:
        return names.original(expr.value.name) + "_" + expr.name
      else:
        return expr.name
    elif c is Index:
      idx_t = expr.index.type
      if isinstance(idx_t, SliceT) or \
         (isinstance(idx_t, TupleT) and \
          any(isinstance(elt_t, SliceT) for elt_t in idx_t.elt_types)):
        return "slice"
      else:
        return "elt"
    else:
      return "temp"

  def is_simple(self, expr):
    c = expr.__class__
    return c is Var or \
           c is Const or \
           (c is Tuple and len(expr.elts) == 0) or \
           (c is Struct and len(expr.args) == 0) or \
           (c is Closure and len(expr.args) == 0)



  def assign_temp(self, expr, name = None):
    if self.is_simple(expr):
      return expr

    if name is None:
      name = self.temp_name(expr)
    var = self.fresh_var(expr.type, name)
    self.assign(var, expr)
    return var

  def int(self, x):
    return const_int(x)

  def bool(self, x):
    return const_bool(x)

  def zero(self, t = Int32, name = "counter"):
    return self.assign_temp(zero(t), name)

  def zero_i32(self, name = "counter"):
    return self.zero(t = Int32, name = name)

  def zero_i64(self, name = "counter"):
    return self.zero(t = Int64, name = name)

  def cast(self, expr, t):
    assert isinstance(t, ScalarT), \
        "Can't cast %s to non-scalar type %s" % (expr, t)
    if expr.type == t:
      return expr
    else:
      return self.assign_temp(Cast(expr, type = t), "cast_%s" % t)

  def index(self, arr, idx, temp = True, name = None):
    """Index into array or tuple differently depending on the type"""

    temp = temp or name is not None
    arr_t = arr.type

    if isinstance(arr_t, ScalarT):
      # even though it's not correct externally, it's
      # often more convenient to treat indexing
      # into scalars as the identity function.
      # Just be sure to catch this as an error in
      # the user's code earlier in the pipeline.
      return arr
    if isinstance(arr_t, TupleT):
      if isinstance(idx, Const):
        idx = idx.value

      assert isinstance(idx, int), \
          "Index into tuple must be an integer, got %s" % idx
      if isinstance(idx, Const):
        idx = idx.value
      proj = self.tuple_proj(arr, idx)
      if temp:
        return self.assign_temp(proj, "tuple_elt%d" % idx if name is None else name)
      else:
        return proj

    if self.is_tuple(idx):
      indices = self.tuple_elts(idx)
    elif hasattr(idx, '__iter__'):
      indices = tuple(map(wrap_if_constant,idx))
    else:
      indices = (wrap_if_constant(idx),)

    n_required = arr_t.rank
    n_indices = len(indices)
    if n_indices < n_required:
      # all unspecified dimensions are considered fully sliced
      extra = (syntax_helpers.slice_none,) * (n_required - n_indices)
      indices = indices + extra

    if len(indices) > 1:
      idx = self.tuple(indices, "index_tuple" if name is None else name)
    else:
      idx = indices[0]

    t = arr_t.index_type(idx.type)
    idx_expr = Index(arr, idx, type=t)
    if temp:
      return self.assign_temp(idx_expr, "array_elt" if name is None else name)
    else:
      return idx_expr

  def index_along_axis(self, arr, axis, idx, name=None):
    if arr.type.__class__ is not ArrayT:
      return arr
    assert isinstance(axis, int), \
        "Axis must be a known constant int, got: " + str(axis)
    indices = []
    for i in xrange(arr.type.rank):
      if i == axis:
        indices.append(syntax_helpers.wrap_if_constant(idx))
      else:
        indices.append(syntax_helpers.slice_none)

    index_tuple = self.tuple(indices, "indices")


    result_t = arr.type.index_type(index_tuple.type)
    idx_expr = Index(arr, index_tuple, type=result_t)
    if name:
      return self.assign_temp(idx_expr, name)
    else:
      return idx_expr

  def setidx(self, arr, idx, v):
    self.assign(self.index(arr, idx, temp=False), v)

  def prim(self, prim_fn, args, name=None):
    args = wrap_constants(args)
    arg_types = get_types(args)
    upcast_types = prim_fn.expected_input_types(arg_types)
    result_type = prim_fn.result_type(upcast_types)
    upcast_args = [self.cast(x, t) for (x,t) in zip(args, upcast_types)]
    prim_call = PrimCall(prim_fn, upcast_args, type=result_type)
    if name:
      return self.assign_temp(prim_call, name)
    else:
      return prim_call

  def pick_first(self, x, y):
    """Return x but first cast it to the common type of both args"""

    return self.cast(x, x.type.combine(y.type))

  def pick_second(self, x, y):
    """Return y but first cast it to the common type of both args"""

    return self.cast(y, x.type.combine(y.type))

  def pick_const(self, x, y, c):
    """Return a constant cast to the common type of both args"""

    return self.cast(syntax_helpers.wrap_if_constant(c), x.type.combine(y.type))

  def add(self, x, y, name = None):
    if syntax_helpers.is_zero(x):
      return self.pick_second(x,y)
    elif syntax_helpers.is_zero(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value + y.value)
    else:
      return self.prim(prims.add, [x,y], name)

  def sub(self, x, y, name = None):
    if syntax_helpers.is_zero(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value - y.value)
    else:
      return self.prim(prims.subtract, [x,y], name)

  def mul(self, x, y, name = None):
    if syntax_helpers.is_one(x):
      return self.pick_second(x,y)
    elif syntax_helpers.is_one(y):
      return self.pick_first(x,y)
    elif syntax_helpers.is_zero(x) or syntax_helpers.is_zero(y):
      return self.pick_const(x, y, 0)
    else:
      return self.prim(prims.multiply, [x,y], name)

  def div(self, x, y, name = None):
    if syntax_helpers.is_one(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value / y.value)
    else:
      return self.prim(prims.divide, [x,y], name)

  def mod(self, x, y, name = None):
    if syntax_helpers.is_one(y):
      return self.pick_const(x, y, 0)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value % y.value)
    else:
      return self.prim(prims.mod, [x,y], name)

  def lt(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(False)
    else:
      return self.prim(prims.less, [x,y], name)

  def lte(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.less_equal, [x,y], name)

  def gt(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(False)
    else:
      return self.prim(prims.greater, [x,y], name)

  def gte(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.greater_equal, [x,y], name)

  def eq(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.equal, [x,y], name)

  def neq(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return syntax_helpers.const_bool(False)
    return self.prim(prims.not_equal, [x,y], name)

  def min(self, x, y, name = None):
    assert x.type == y.type, \
        "Type mismatch between %s and %s" % (x, y)
    if x.__class__ is Const and y.__class__ is Const:
      return x if x.value < y.value else y 
    
    if name is None:
      name = "min_temp"
    result = self.fresh_var(x.type, name)
    cond = self.lte(x, y)
    merge = {result.name:(x,y)}
    self.blocks += syntax.If(cond, [], [], merge)
    return result 

  def max(self, x, y, name = None):
    assert x.type == y.type, \
        "Type mismatch between %s and %s" % (x, y)
    if x.__class__ is Const and y.__class__ is Const:
      return x if x.value > y.value else y 
    if name is None:
      name = "min_temp"
    result = self.fresh_var(x.type, name)
    cond = self.gte(x, y)
    merge = {result.name:(x,y)}
    self.blocks += syntax.If(cond, [], [], merge)
    return result 

  def attr(self, obj, field, name = None):
    if name is None:
      name = field
    obj_t = obj.type
    if obj.__class__ is Struct:
      pos = obj_t.field_pos(name)
      result =  obj.args[pos]
    elif obj.__class__ is ArrayView:
      return getattr(obj, field)
    else:
      assert isinstance(obj_t, StructT), \
        "Can't get attribute '%s' from type %s" % (field, obj_t)
      field_t = obj.type.field_type(field)
      result = Attribute(obj, field, type = field_t)
    if name:
      return self.assign_temp(result, name)
    else:
      return result

  def is_none(self, x):
    return x.type.__class__ is NoneT

  def is_array(self, x):
    return x.type.__class__ is  ArrayT

  def elt_type(self, x):
    if isinstance(x, Type):
      try:
        return x.elt_type
      except:
        return x
    elif self.is_array(x):
      return x.type.elt_type
    else:
      return x.type

  def shape(self, array, dim = None):
    if isinstance(array.type, ArrayT):
      shape = self.attr(array, "shape")
      if dim is None:
        return shape
      else:
        dim_t = shape.type.elt_types[dim]
        dim_value = TupleProj(shape, dim, type = dim_t)
        return self.assign_temp(dim_value, "dim%d" % dim)
    else:
      return self.tuple([])

  def strides(self, array, dim = None):
    assert array.type.__class__ is ArrayT
    strides = self.attr(array, "strides")
    if dim is None:
      return strides
    else:
      elt_t = strides.type.elt_types[dim]
      elt_value = TupleProj(strides, dim, type = elt_t)
      return self.assign_temp(elt_value, "stride%d" % dim)

  def tuple(self, elts, name = "tuple", explicit_struct = False):
    if not isinstance(elts, (list, tuple)):
      elts = [elts]
    tuple_t = make_tuple_type(get_types(elts))
    if explicit_struct:
      tuple_expr = Struct(elts, type = tuple_t)
    else:
      tuple_expr = Tuple(elts, type = tuple_t)
    if name:
      result_var = self.assign_temp(tuple_expr, name)
      # cache the simple elements so we can look them up directly
      for (i, elt) in enumerate(elts):
        if self.is_simple(elt):
          self.tuple_elt_cache[(result_var.name, i)] = elt
      return result_var
    else:
      return tuple_expr

  def is_tuple(self, x):
    try:
      return x.type.__class__ is TupleT
    except:
      return False

  def concat_tuples(self, x, y, name = "concat_tuple"):
    if self.is_tuple(x):
      x_elts = self.tuple_elts(x)
    else:
      x_elts = (x,)
    if self.is_tuple(y):
      y_elts = self.tuple_elts(y)
    else:
      y_elts = (y,)

    elts = []
    elts.extend(x_elts)
    elts.extend(y_elts)
    return self.tuple(elts, name = name)

  def tuple_proj(self, tup, idx, explicit_struct = False):
    assert isinstance(idx, (int, long))
    if isinstance(tup, Tuple):
      return tup.elts[idx]
    elif isinstance(tup, tuple):
      return tup[idx]
    elif tup.__class__ is Var and (tup.name, idx) in self.tuple_elt_cache:
      return self.tuple_elt_cache[(tup.name, idx)]
    elif explicit_struct:
      return Attribute(tuple, "elt%d" % idx, type = tup.type.elt_types[idx])
    else:
      return TupleProj(tup, idx, type = tup.type.elt_types[idx])

  def tuple_elts(self, tup, explicit_struct = False):
    nelts = len(tup.type.elt_types)
    return tuple([self.tuple_proj(tup, i, explicit_struct = explicit_struct)
                  for i in xrange(nelts)])

  def closure_elt(self, clos, idx):
    assert isinstance(idx, (int, long))

    if isinstance(clos, Closure):
      return clos.args[idx]
    else:
      return ClosureElt(clos, idx, type = clos.type.arg_types[idx])

  def closure_elts(self, clos):
    if clos.__class__ is TypedFn:
      return []
    return [self.closure_elt(clos, i)
            for i in xrange(len(clos.type.arg_types))]

  def get_fn(self, maybe_clos):
    if maybe_clos.__class__ is Closure:
      return maybe_clos.fn
    elif maybe_clos.type.__class__ is ClosureT:
      return maybe_clos.type.fn
    else:
      return maybe_clos

  def closure(self, maybe_fn, extra_args, name = None):
    fn = self.get_fn(maybe_fn)
    old_closure_elts = self.closure_elts(maybe_fn)
    closure_elts = old_closure_elts + extra_args
    if len(closure_elts) == 0:
      return fn 
    closure_elt_types = [elt.type for elt in closure_elts]
    closure_t = make_closure_type(fn, closure_elt_types)
    result = Closure(fn, closure_elts, type = closure_t)
    if name:
      return self.assign_temp(result, name)
    else:
      return result

  def prod(self, elts, name = None):
    if self.is_tuple(elts):
      elts = self.tuple_elts(elts)
    if len(elts) == 0:
      return one_i64
    else:
      result = elts[0]
      for e in elts[1:]:
        result = self.mul(result, e, name = name)
      return result

  def alloc_array(self, elt_t, dims, name = "array", explicit_struct = False):
    """
    Given an element type and sequence of expressions denoting each dimension
    size, generate code to allocate an array and its shape/strides metadata. For
    now I'm assuming that all arrays are in row-major, eventually we should make
    the layout an option.
    """

    if self.is_tuple(dims):
      shape = dims
      dims = self.tuple_elts(shape)
    else:
      if not isinstance(dims, (list, tuple)):
        dims = [dims]
      shape = self.tuple(dims, "shape", explicit_struct = explicit_struct)
    rank = len(dims)
    array_t = array_type.make_array_type(elt_t, rank)
    if explicit_struct:
      nelts = self.prod(dims, name = "nelts")
      ptr_t = core_types.ptr_type(elt_t)

      ptr_var = self.assign_temp(Alloc(elt_t, nelts, type = ptr_t), "data_ptr")
      stride_elts = [syntax_helpers.const(1)]

      # assume row-major for now!
      for d in reversed(dims[1:]):
        next_stride = self.mul(stride_elts[0], d, "dim")
        stride_elts = [next_stride] + stride_elts
      strides = self.tuple(stride_elts, "strides", explicit_struct = True)
      array = Struct([ptr_var, shape, strides, zero_i64, nelts], type = array_t)
    else:
      array = AllocArray(shape, type = array_t)
    return self.assign_temp(array, name)

  def return_type(self, fn):
    if isinstance(fn, TypedFn):
      return fn.return_type
    else:
      assert isinstance(fn.type, closure_type.ClosureT), \
          "Unexpected fn type: %s" % fn.type
      assert isinstance(fn.type.fn, TypedFn), \
          "Unexpected fn: %s" % fn.type.fn
      return fn.type.fn.return_type

  def invoke_type(self, closure, args):
    import type_inference
    closure_t = closure.type
    arg_types = syntax_helpers.get_types(args)
    assert all(isinstance(t, core_types.Type) for t in arg_types), \
        "Invalid types: %s" % (arg_types, )
    return type_inference.invoke_result_type(closure_t, arg_types)

  def invoke(self, fn, args):
    import type_inference
    if fn.__class__ is syntax.TypedFn:
      closure_args = []
    else:
      assert isinstance(fn.type, closure_type.ClosureT), \
          "Unexpected function %s with type: %s" % (fn, fn.type)
      closure_args = self.closure_elts(fn)
      arg_types = syntax_helpers.get_types(args)
      fn = type_inference.specialize(fn.type, arg_types)

    import pipeline
    lowered_fn = pipeline.loopify(fn)
    combined_args = closure_args + args
    call = syntax.Call(lowered_fn, combined_args, type = lowered_fn.return_type)

    return self.assign_temp(call, "call_result")

  def size_along_axis(self, value, axis):
    return self.shape(value, axis)

  def check_equal_sizes(self, sizes):
    pass

  none = syntax_helpers.none
  null_slice = syntax_helpers.slice_none

  # TODO: get rid of that leading underscore to enable this function once
  # shape inference works for all the weird and wacky constructs in our
  # syntax zoo
  def _create_output_array(self, fn, args, extra_dims, name = "output"):
    """
    Given a function and its argument, use shape inference to figure out the
    result shape of the array and preallocate it.  If the result should be a
    scalar, just return a scalar variable.
    """
    try:
      inner_shape_tuple = self.call_shape(fn, args)
    except:
      print "Shape inference failed when calling %s with %s" % (fn, args)
      import sys
      print sys.exc_info()[0]
      print "Error %s ==> %s" % (sys.exc_info()[:2])

      raise

    if self.is_tuple(extra_dims):
      outer_shape_tuple = extra_dims
    elif isinstance(extra_dims, (list, tuple)):
      outer_shape_tuple = self.tuple(extra_dims)
    else:
      outer_shape_tuple = self.tuple((extra_dims,) if extra_dims else ())

    shape = self.concat_tuples(outer_shape_tuple, inner_shape_tuple)
    elt_t = self.elt_type(self.return_type(fn))
    if len(shape.type.elt_types) > 0:
      return self.alloc_array(elt_t, shape, name)
    else:
      return self.fresh_var(elt_t, name)

  def rank(self, value):
    if self.is_array(value):
      return value.type.rank
    else:
      return 0

  def slice_value(self, start, stop, step):
    slice_t = array_type.make_slice_type(start.type, stop.type, step.type)
    return Slice(start, stop, step, type = slice_t)

  def loop_counter(self, name = "i", start_val = syntax_helpers.zero_i64):
    """
    Generate three SSA variables to use as the before/during/after values
    of a loop counter throughout some loop.

    By default initialize the counter to zero, but optionally start at different
    values using the 'start_val' keyword.
    """

    start_val = syntax_helpers.wrap_if_constant(start_val)
    counter_type = start_val.type

    counter_before = self.assign_temp(start_val, name + "_before")
    counter = self.fresh_var(counter_type, name)
    counter_after = self.fresh_var(counter_type, name + "_after")
    merge = {counter.name:(counter_before, counter_after)}
    return counter, counter_after, merge

  def loop(self, start, niters, loop_body,
            return_stmt = False,
            while_loop = False):
    if while_loop:
      i, i_after, merge = self.loop_counter("i", start)
      cond = self.lt(i, niters)
      self.blocks.push()
      loop_body(i)
      self.assign(i_after, self.add(i, syntax_helpers.one_i64))
      body = self.blocks.pop()
      loop_stmt = While(cond, body, merge)
    else:
      var_t = start.type
      var = self.fresh_var(var_t, "i")
      self.blocks.push()
      loop_body(var)
      body = self.blocks.pop()
      loop_stmt = ForLoop(var, start, niters, one(var_t), body, {})

    if return_stmt:
      return loop_stmt
    else:
      self.blocks += loop_stmt

  def call_shape(self, maybe_clos, args):
    fn = self.get_fn(maybe_clos)
    closure_args = self.closure_elts(maybe_clos)
    combined_args = closure_args + args

    if isinstance(fn, Fn):
      # if we're given an untyped function, first specialize it
      import type_inference
      fn = type_inference.specialize(fn, get_types(combined_args))
    abstract_shape = shape_inference.call_shape_expr(fn)
    return shape_codegen.make_shape_expr(self, abstract_shape, combined_args)

  class Accumulator:
    def __init__(self, acc_type, fresh_var, assign):
      self.acc_type = acc_type
      self.fresh_var = fresh_var
      self.assign = assign
      self.start_var = fresh_var(acc_type, "acc")
      self.curr_var = self.start_var

    def get(self):
      return self.curr_var

    def update(self, new_value):
      new_var = self.fresh_var(self.acc_type, "acc")
      self.assign(new_var, new_value)
      self.curr_var = new_var

  def accumulate_loop(self, start, stop, loop_body, init, return_stmt = False):
    acc = self.Accumulator(init.type, self.fresh_var, self.assign)
    def loop_body_with_acc(i):
      loop_body(acc, i)
    loop_stmt = self.loop(start, stop, loop_body_with_acc, return_stmt = True)
    loop_stmt.merge[acc.start_var.name] = (init, acc.curr_var)
    if return_stmt:
      return loop_stmt, acc.start_var
    else:
      self.blocks += loop_stmt
      return acc.start_var

  def nelts(self, array):
    shape_elts = self.tuple_elts(self.shape(array))
    return self.prod(shape_elts, name = "nelts")

  def array_copy(self, src, dest, return_stmt = False):
    assert self.is_array(dest)
    # nelts = self.nelts(dest)
    shape = self.shape(dest)
    dims = self.tuple_elts(shape)
    rank = len(dims)
    index_vars = []
    def create_loops():
      i = len(index_vars)
      def loop_body(index_var):
        index_vars.append(index_var)
        if i+1 == rank:
          index_tuple = self.tuple(index_vars, "idx")
          lhs = self.index(dest, index_tuple, temp=False)
          rhs = self.index(src, index_tuple, temp=True)
          self.assign(lhs, rhs)
        else:
          self.blocks += create_loops()
      start = syntax_helpers.zero_i64
      stop = dims[i]
      if i > 0 or return_stmt:
        return self.loop(start, stop, loop_body, True)
      else:
        return self.loop(start, stop, loop_body, return_stmt)

    return create_loops()
