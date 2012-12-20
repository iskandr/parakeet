import array_type
import core_types
import names
import prims
import shape_codegen
import shape_inference
import syntax
import syntax_helpers
import tuple_type

from core_types import Int32, Int64
from nested_blocks import NestedBlocks
from syntax_helpers import get_types, wrap_constants, wrap_if_constant, \
                           zero, zero_i64

class Codegen(object):
  def __init__(self):
    self.type_env = {}
    self.blocks = NestedBlocks()

  def fresh_var(self, t, prefix = "temp"):
    assert t is not None, "Type required for new variable %s" % prefix
    ssa_id = names.fresh(prefix)
    self.type_env[ssa_id] = t
    return syntax.Var(ssa_id, type = t)

  def fresh_i32(self, prefix = "temp"):
    return self.fresh_var(Int32, prefix)

  def fresh_i64(self, prefix = "temp"):
    return self.fresh_var(Int64, prefix)

  def insert_stmt(self, stmt):
    self.blocks.append_to_current(stmt)

  def assign(self, lhs, rhs):
    self.insert_stmt(syntax.Assign(lhs, rhs))

  def assign_temp(self, expr, name = "temp"):
    if isinstance(expr, syntax.Var):
      return expr
    else:
      var = self.fresh_var(expr.type, name)
      self.assign(var, expr)
      return var

  def int(self, x):
    return syntax_helpers.const_int(x)

  def bool(self, x):
    return syntax_helpers.const_bool(x)

  def zero(self, t = Int32, name = "counter"):
    return self.assign_temp(zero(t), name)

  def zero_i32(self, name = "counter"):
    return self.zero(t = Int32, name = name)

  def zero_i64(self, name = "counter"):
    return self.zero(t = Int64, name = name)

  def cast(self, expr, t):
    assert isinstance(t, core_types.ScalarT), \
        "Can't cast %s to non-scalar type %s" % (expr, t)
    if expr.type == t:
      return expr
    else:
      return self.assign_temp(syntax.Cast(expr, type = t), "cast_%s" % t)

  def index(self, arr, idx, temp = True):
    """
    Index into array or tuple differently depending on the type
    """
    arr_t = arr.type

    if isinstance(arr_t, core_types.ScalarT):
      # even though it's not correct externally, it's
      # often more convenient to treat indexing
      # into scalars as the identity function.
      # Just be sure to catch this as an error in
      # the user's code earlier in the pipeline.
      return arr
    if isinstance(arr_t, tuple_type.TupleT):
      if isinstance(idx, syntax.Const):
        idx = idx.value

      assert isinstance(idx, int), \
          "Index into tuple must be an integer, got %s" % idx
      if isinstance(idx, syntax.Const):
        idx = idx.value
      proj = self.tuple_proj(arr, idx)
      if temp:
        return self.assign_temp(proj, "tuple_elt%d" % idx)
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
      idx = self.tuple(indices, "index_tuple")
    else:
      idx = indices[0]

    t = arr_t.index_type(idx.type)
    idx_expr = syntax.Index(arr, idx, type = t)
    if temp:
      return self.assign_temp(idx_expr, "array_elt")
    else:
      return idx_expr

  def index_along_axis(self, arr, axis, idx, name = None):
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
    idx_expr = syntax.Index(arr, index_tuple, type = result_t)
    if name:
      return self.assign_temp(idx_expr, name)
    else:
      return idx_expr

  def setidx(self, arr, idx, v):
    self.assign(self.index(arr, idx, temp=False), v)

  def prim(self, prim_fn, args, name = None):
    args = wrap_constants(args)
    arg_types = get_types(args)
    upcast_types = prim_fn.expected_input_types(arg_types)
    result_type = prim_fn.result_type(upcast_types)
    upcast_args = [self.cast(x, t) for (x,t) in zip(args, upcast_types)]
    prim_call = syntax.PrimCall(prim_fn, upcast_args, type = result_type)
    if name:
      return self.assign_temp(prim_call, name)
    else:
      return prim_call

  def pick_first(self, x, y):
    """
    Return x but first cast it to the common type of both args
    """
    return self.cast(x, x.type.combine(y.type))

  def pick_second(self, x, y):
    """
    Return y but first cast it to the common type of both args
    """
    return self.cast(y, x.type.combine(y.type))

  def pick_const(self, x, y, c):
    """
    Return a constant cast to the common type of both args
    """
    return self.cast(syntax_helpers.wrap_if_constant(c), x.type.combine(y.type))

  def add(self, x, y, name = None):
    if syntax_helpers.is_zero(x):
      return self.pick_second(x,y)
    elif syntax_helpers.is_zero(y):
      return self.pick_first(x,y)
    else:
      return self.prim(prims.add, [x,y], name)

  def sub(self, x, y, name = None):
    if syntax_helpers.is_zero(y):
      return self.pick_first(x,y)
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
    else:
      return self.prim(prims.divide, [x,y], name)

  def mod(self, x, y, name = None):
    if syntax_helpers.is_one(y):
      return self.pick_const(x, y, 0)
    else:
      return self.prim(prims.mod, [x,y], name)

  def lt(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(False)
    else:
      return self.prim(prims.less, [x,y], name)

  def lte(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.less_equal, [x,y], name)

  def gt(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(False)
    else:
      return self.prim(prims.greater, [x,y], name)

  def gte(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.greater_equal, [x,y], name)

  def eq(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(True)
    else:
      return self.prim(prims.equal, [x,y], name)

  def neq(self, x, y, name = None):
    if isinstance(x, (syntax.Var, syntax.Const)) and x == y:
      return syntax_helpers.const_bool(False)
    return self.prim(prims.not_equal, [x,y], name)

  def attr(self, obj, field, name = None):
    if name is None:
      name = field
    obj_t = obj.type
    assert isinstance(obj_t, core_types.StructT), \
        "Can't get attribute '%s' from type %s" % (field, obj_t)
    field_t = obj.type.field_type(field)
    attr_expr = syntax.Attribute(obj, field, type = field_t)
    if name:
      return self.assign_temp(attr_expr, name)
    else:
      return attr_expr

  def is_none(self, x):
    return hasattr(x, 'type') and \
      isinstance(x.type, core_types.NoneT)

  def is_array(self, x):
    return hasattr(x, 'type') and \
      isinstance(x.type, array_type.ArrayT)

  def elt_type(self, x):
    if isinstance(x, core_types.Type):
      if hasattr(x, 'elt_type'):
        return x.elt_type
      else:
        return x
    elif self.is_array(x):
      return x.type.elt_type
    else:
      return x.type

  def shape(self, array, dim = None):
    if isinstance(array.type, array_type.ArrayT):
      shape = self.attr(array, "shape")
      if dim is None:
        return shape
      else:
        dim_t = shape.type.elt_types[dim]
        dim_value = syntax.TupleProj(shape, dim, type = dim_t)
        return self.assign_temp(dim_value, "dim%d" % dim)
    else:
      return self.tuple([])

  def strides(self, array, dim = None):
    assert isinstance(array.type, array_type.ArrayT)
    strides = self.attr(array, "strides")
    if dim is None:
      return strides
    else:
      elt_t = strides.type.elt_types[dim]
      elt_value = syntax.TupleProj(strides, dim, type = elt_t)
      return self.assign_temp(elt_value, "stride%d" % dim)

  def tuple(self, elts, name = "tuple"):
    if not isinstance(elts, (list, tuple)):
      elts = [elts]
    tuple_t = tuple_type.make_tuple_type(get_types(elts))
    tuple_t.metadata = self.__class__.__name__
    tuple_expr = syntax.Tuple(elts, type = tuple_t)
    if name:
      return self.assign_temp(tuple_expr, name)
    else:
      return tuple_expr

  def is_tuple(self, x):
    return hasattr(x, 'type') and isinstance(x.type, tuple_type.TupleT)

  def concat_tuples(self, x, y):
    assert self.is_tuple(x)
    assert self.is_tuple(y)
    elts = []
    for i in xrange(len(x.type.elt_types)):
      elts.append(self.tuple_proj(x, i))
    for i in xrange(len(y.type.elt_types)):
      elts.append(self.tuple_proj(y, i))
    return self.tuple(elts)

  def tuple_proj(self, tup, idx):
    assert isinstance(idx, (int, long))
    if isinstance(tup, syntax.Tuple):
      return tup.elts[idx]
    elif isinstance(tup, tuple):
      return tup[idx]
    else:
      return syntax.TupleProj(tup, idx, type = tup.type.elt_types[idx])

  def tuple_elts(self, tup):
    nelts = len(tup.type.elt_types)
    return tuple([self.tuple_proj(tup, i) for i in xrange(nelts)])

  def closure_elt(self, clos, idx):
    assert isinstance(idx, (int, long))

    if isinstance(clos, syntax.Closure):
      return clos.args[idx]
    else:
      return syntax.ClosureElt(clos, idx, type = clos.type.arg_types[idx])

  def closure_elts(self, clos):
    if isinstance(clos, syntax.TypedFn):
      return []
    return [self.closure_elt(clos, i)
            for i in xrange(len(clos.type.arg_types))]

  def get_fn(self, maybe_clos):
    if isinstance(maybe_clos, syntax.Closure):
      return maybe_clos.fn
    else:
      return maybe_clos

  def prod(self, elts, name = None):
    result = elts[0]
    for e in elts[1:]:
      result = self.mul(result, e, name = name)
    return result

  def alloc_array(self, elt_t, dims, name = "temp_array"):
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
      shape = self.tuple(dims, "shape")

    rank = len(dims)
    nelts = self.prod(dims, name = "nelts")

    array_t = array_type.make_array_type(elt_t, rank)
    ptr_t = core_types.ptr_type(elt_t)

    ptr_var = self.assign_temp(syntax.Alloc(elt_t, nelts, type = ptr_t),
                               "data_ptr")

    stride_elts = [syntax_helpers.const(1)]

    # assume row-major for now!
    for d in reversed(dims[1:]):
      next_stride = self.mul(stride_elts[0], d, "dim")
      stride_elts = [next_stride] + stride_elts

    strides = self.tuple(stride_elts, "strides")
    array = syntax.Struct([ptr_var, shape, strides, zero_i64], type = array_t)
    return self.assign_temp(array, name)

  def return_type(self, fn):
    if isinstance(fn, syntax.TypedFn):
      return fn.return_type
    else:
      import closure_type

      assert isinstance(fn.type, closure_type.ClosureT)
      assert isinstance(fn.type.fn, syntax.TypedFn)
      return fn.type.fn.return_type

  # TODO: get rid of that leading underscore to enable this function once
  # shape inference works for all the weird and wacky constructs in our
  # syntax zoo
  def _create_output_array(self, fn, args, extra_dims, name = "output"):
    """
    Given a function and its argument, use shape inference
    to figure out the result shape of the array and preallocate it
    """
    try:
      symbolic_shape = shape_inference.call_shape_expr(fn)
      inner_shape_tuple = shape_codegen.make_shape_expr(self, symbolic_shape,
                                                        args)
      print "-- Shape inference succeeded when calling %s with %s" % \
            (fn, args)
    except:
      print "[Warning] Shape inference failed when calling %s with %s" % \
            (fn, args)
      result = self.invoke(fn, args)
      inner_shape_tuple = self.shape(result)
    if not hasattr(extra_dims, '__iter__'):
      extra_dims = (extra_dims,)
    outer_shape_tuple = self.tuple(extra_dims)
    shape = self.concat_tuples(outer_shape_tuple, inner_shape_tuple)
    elt_t = self.elt_type(self.return_type(fn))
    return self.alloc_array(elt_t, shape, name)

  def rank(self, value):
    if self.is_array(value):
      return value.type.rank
    else:
      return 0

  def slice_value(self, start, stop, step):
    slice_t = array_type.make_slice_type(start.type, stop.type, step.type)
    return syntax.Slice(start, stop, step, type = slice_t)

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
    merge = { counter.name : (counter_before, counter_after) }
    return counter, counter_after, merge

  def loop(self, start, niters, loop_body, return_stmt = False):
    i, i_after, merge = self.loop_counter("i", start)
    cond = self.lt(i, niters)
    self.blocks.push()
    loop_body(i)
    self.assign(i_after, self.add(i, syntax_helpers.one_i64))
    body = self.blocks.pop()
    loop_stmt = syntax.While(cond, body, merge)
    if return_stmt:
      return loop_stmt
    else:
      self.blocks += loop_stmt

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

  def linear_to_indices(self, linear_idx, shape):
    """
    Return tuple of dimension-wise indices from linear index
    """
    dim_sizes = self.tuple_elts(shape)
    rank = len(dim_sizes)
    slice_sizes = [syntax_helpers.one_i64]
    for (i, d) in enumerate(reversed(dim_sizes[:-1])):
      slice_sizes.append(self.mul(slice_sizes[-1], d, "slice_size_%d" % i))
    slice_sizes.reverse()
    remainder = linear_idx
    indices = []
    for i in xrange(rank):
      s = slice_sizes[i]
      indices.append(self.div(remainder, s, "idx%d" % i))
      remainder = self.mod(remainder, s, "rem%d" % i)
    return self.tuple(indices)

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
          create_loops()
      start = syntax_helpers.zero_i64
      stop = dims[i]
      if i > 0 or return_stmt:
        return self.loop(start, stop, loop_body, True)
      else:
        return self.loop(start, stop, loop_body, return_stmt)

    return create_loops()
