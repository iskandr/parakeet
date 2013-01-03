import adverb_semantics
import array_type
import core_types
import tuple_type

import syntax
from syntax_visitor import SyntaxVisitor
from syntax import TypedFn, Fn

import shape
from shape import Var, Const, Shape, Tuple, Closure
from shape import Slice, Scalar, UnknownScalar, Unknown
from shape import unknown_scalar, unknown_value, const
from shape import combine_list, increase_rank
from shape import is_zero, is_one, make_shape
import shape_from_type

class ShapeSemantics(adverb_semantics.AdverbSemantics):
  def size_along_axis(self, value, axis):
    assert isinstance(value, Shape)
    return value.dims[axis]

  #def slice_along_axis(self, arr, axis, idx):
  #  assert False, (arr, axis, idx)

  def is_tuple(self, x):
    return isinstance(x, Tuple)

  def is_none(self, x):
    return isinstance(x, Const) and x.value is None

  def rank(self, value):
    if isinstance(value, Shape):
      return value.rank
    else:
      return 0

  def int(self, x):
    return const(x)

  def bool(self, x):
    return const(x)

  def add(self, x, y):
    if is_zero(x):
      return y
    elif is_zero(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(x.value + y.value)
    else:
      return shape.Add(x,y)

  def sub(self, x, y):
    if is_zero(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(x.value - y.value)
    else:
      return shape.Sub(x, y)

  def div(self, x, y):
    assert not is_zero(y)
    if is_one(y):
      return x
    elif isinstance(x, Const) and isinstance(y, Const):
      return const(int(x.value / y.value))
    else:
      return shape.Div(x, y)

  def shape(self, x):
    if isinstance(x, Shape):
      return Tuple(x.dims)
    else:
      return Tuple(())

  def elt_type(self, x):
    return "DON'T CARE ABOUT ELT TYPES"

  def alloc_array(self, _, dims):
    return make_shape(dims)

  def index(self, arr, idx):
    if isinstance(arr, Scalar):
      return arr
    assert arr.__class__ is Shape
    if isinstance(idx, Scalar):
      indices = [idx]
    elif idx.__class__ is Tuple:
      indices = idx.elts
    result_dims = []
    for (i, curr_idx) in enumerate(indices):
      old_dim = arr.dims[i]
      if curr_idx is None or \
         (isinstance(curr_idx, Scalar) and curr_idx.value is None):
        result_dims.append(old_dim)
      elif isinstance(curr_idx, Scalar):
        pass
      else:
        assert curr_idx.__class__ is Slice, \
          "Unsupported index %s" % curr_idx
        if isinstance(curr_idx.start, Const):
          if curr_idx.start.value is None:
            lower = const(0)
          elif curr_idx.start.value < 0:
            lower = self.sub(old_dim, curr_idx.start)
          else:
            lower = curr_idx.start
        else:
          lower = unknown_scalar

        if isinstance(curr_idx.stop, Const):
          if curr_idx.stop.value is None:
            upper = old_dim
          elif curr_idx.stop.value < 0:
            upper = self.sub(old_dim, curr_idx.stop)
          else:
            upper = curr_idx.stop
        else:
          upper = unknown_scalar

        n = self.sub(upper, lower)
        step = curr_idx.step
        if step and \
            isinstance(step, Const) and \
            step.value is not None and \
            step.value != 1:
          n = self.div(n, step)
        result_dims.append(n)
    n_original = len(arr.dims)
    n_idx= len(indices)
    if n_original > n_idx:
      result_dims.extend(arr.dims[n_idx:])

    return make_shape(result_dims)

  def tuple(self, elts):
    return Tuple(tuple(elts))

  def concat_tuples(self, t1, t2):
    return Tuple(t1.elts + t2.elts)

  def setidx(self, arr, idx, v):
    pass

  def loop(self, start_idx, stop_idx, body):
    body(start_idx)

  class Accumulator(object):
    def __init__(self, v):
      self.v = v

    def update(self, new_v):
      self.v = new_v

    def get(self):
      return self.v

  def accumulate_loop(self, start_idx, stop_idx, body, init):
    acc = self.Accumulator(init)
    body(acc, start_idx)
    return acc.get()

  def check_equal_sizes(self, sizes):
    pass

  def slice_value(self, start, stop, step):
    Slice(start, stop, step)

  def invoke(self, fn, args):
    if fn.__class__ is Closure:
      args = tuple(fn.args) + tuple(args)
      fn = fn.fn
    return symbolic_call(fn, args)

  none = None
  null_slice = slice(None, None, None)

  def identity_function(self, x):
    return x

shape_semantics = ShapeSemantics()

class ShapeInference(SyntaxVisitor):
  def __init__(self):
    self._clear()

  def _clear(self):
    self.value_env = {}
    self.equivalence_classes = {}

  def unify_scalar_var(self, x, y):
    assert isinstance(x, Var), "Expected scalar variable, but got: " + str(x)
    assert isinstance(y, Scalar), "Expected scalar, but got: " + str(y)
    if isinstance(y, UnknownScalar):
      return x
    equivs = self.equivalence_classes.get(x, set([]))
    equivs.add(y)
    for var in equivs:
      self.equivalence_classes[var] = equivs
    if isinstance(y, Const):
      for var in equivs:
        self.value_env[var] = y
      return y
    else:
      return var

  def unify_scalar_pairs(self, xs, ys):
    result_elts = []
    for xi, yi in zip(xs.elts, ys.elts):
      result_elts.append(self.unify_scalars(xi, yi))
    return result_elts

  def unify_scalar_list(self, values):
    assert len(values) > 0
    acc = unknown_scalar
    for v in values:
      acc = self.unify_scalars(acc, v)
    return acc

  def unify_scalars(self, x, y):
    if isinstance(x, Unknown):
      return y
    elif isinstance(y, Unknown):
      return x
    elif isinstance(x, Var):
      return self.unify_scalar_var(x, y)
    elif isinstance(y, Var):
      return self.unify_scalar_var(y, x)
    else:
      raise RuntimeError("Unsupported by unify: %s, %s" % (x,y))

  def visit_merge_loop_start(self, merge):
    for (k, (l, _)) in merge.iteritems():
      self.value_env[k] = self.visit_expr(l)

  def visit_merge(self, merge):
    for (k, (l,r)) in merge.iteritems():
      new_l = self.visit_expr(l)
      new_r = self.visit_expr(r)
      self.value_env[k] = new_l.combine(new_r)

  def visit_expr(self, expr):
    abstract_shape = SyntaxVisitor.visit_expr(self, expr)
    assert abstract_shape is not None, \
        "Unsupported expression in shape inference: %s" % expr.node_type()
    return abstract_shape

  def visit_Alloc(self, expr):
    # alloc doesn't return an array but rather
    # a pointer whose shape properties
    # we don't yet care about here
    return unknown_value

  def visit_Struct(self, expr):
    if isinstance(expr.type, array_type.ArrayT):
      shape_tuple = self.visit_expr(expr.args[1])
      return make_shape(shape_tuple.elts)
    elif isinstance(expr.type, tuple_type.TupleT):
      return Tuple(self.visit_expr_list(expr.args))
    else:
      assert False, "Unexpected struct: %s" % (expr,)

  def visit_Fn(self, fn):
    return Closure(fn, [])

  def visit_TypedFn(self, fn):
    return Closure(fn, [])

  def visit_Slice(self, expr):
    start = self.visit_expr(expr.start)
    stop = self.visit_expr(expr.stop)
    step = self.visit_expr(expr.step)
    return Slice(start, stop, step)

  def visit_Const(self, expr):
    return const(expr.value)

  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    assert clos.__class__ is Closure, \
        "Unexpected closure shape %s for expression %s" % (clos, expr)
    return clos.args[expr.index]

  def visit_TupleProj(self, expr):
    t = self.visit_expr(expr.tuple)
    assert isinstance(t, Tuple)
    return t.elts[expr.index]

  def visit_Attribute(self, expr):
    v = self.visit_expr(expr.value)
    name = expr.name
    if isinstance(v, Shape) and name =='shape':
      return Tuple(v.dims)
    elif isinstance(v, Tuple) and name.startswith('elt'):
      idx = int(name[3:])
      return v[idx]

    try:
      t = expr.value.type.field_type(name)
      if isinstance(t, core_types.ScalarT):
        return unknown_scalar
    except:
      return unknown_value

  def visit_PrimCall(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
    return shape.combine_list(arg_shapes, preserve_const = False)

  def visit_Var(self, expr):
    name = expr.name
    if name in self.value_env:
      return self.value_env[name]
    elif name in self.equivalence_classes:
      for other_name in self.equivalence_classes[name]:
        if other_name in self.value_env:
          return self.value_env[other_name]
    raise RuntimeError("Unknown variable: %s" %  expr)

  def visit_Tuple(self, expr):
    return Tuple(self.visit_expr_list(expr.elts))

  def visit_AllocArray(self, expr):
    shape_tuple = self.visit_expr(expr.shape)
    return make_shape(shape_tuple.elts)
     
  def visit_Array(self, expr):
    elts = self.visit_expr_list(expr.elts)
    elt = combine_list(elts)
    n = len(elts)
    res = increase_rank(elt, 0, const(n))
    return res

  def visit_Call(self, expr):
    fn = self.visit_expr(expr.fn)
    args = self.visit_expr_list(expr.args)
    return symbolic_call(fn, args)

  def visit_Closure(self, clos):
    assert not isinstance(clos.fn, str), \
        "[ShapeInference] Function names in closures not supported: " + clos.fn
    fn = self.visit_expr(clos.fn)
    closure_arg_shapes = self.visit_expr_list(clos.args)
    if fn.__class__ is Closure:
      closure_arg_shapes = tuple(fn.args) + tuple(closure_arg_shapes)
      fn = fn.fn
    return Closure(fn, closure_arg_shapes)

  def visit_Index(self, expr):
    arr = self.visit_expr(expr.value)
    idx = self.visit_expr(expr.index)
    if arr.__class__ is Tuple and idx.__class__ is Const:
      return arr[idx.value]
    elif arr.__class__ is Shape:
      if isinstance(idx, Scalar):
        return shape.lower_rank(arr, 0)
      elif idx.__class__ is Shape:
        assert len(idx.dims) <= len(arr.dims), \
            "Can't index into rank %d array with rank %d indices" % \
            (len(arr.dims), len(idx.dims))
        dims = [d for d in arr.dims]
        for (i,d) in enumerate(idx.dims):
          dims[i] = d
        return shape.make_shape(dims)
      else:
        return shape_semantics.index(arr, idx)
    assert False, \
        "Can't index (%s) with array shape %s and index shape %s" % \
        (expr, arr, idx)

  def visit_Map(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
    fn = self.visit_expr(expr.fn)
    res = shape_semantics.eval_map(fn, arg_shapes, expr.axis)
    return res

  def visit_Reduce(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    arg_shapes = self.visit_expr_list(expr.args)
    init = self.visit_expr(expr.init) if expr.init else None
    return shape_semantics.eval_reduce(fn, combine, init, arg_shapes, expr.axis)

  def visit_Scan(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    emit = self.visit_expr(expr.emit)
    arg_shapes = self.visit_expr_list(expr.args)
    init = self.visit_expr(expr.init) if expr.init else None
    return shape_semantics.eval_reduce(fn, combine, emit, init, arg_shapes,
                                       expr.axis)

  def visit_AllPairs(self, expr):
    axis = self.visit_expr(expr.axis)
    arg_shapes = self.visit_expr_list(expr.args)
    fn = self.visit_expr(expr.fn)
    return shape_semantics.eval_allpairs(fn, arg_shapes, axis)

  def visit_TiledMap(self, expr):
    fn = self.visit_expr(expr.fn)
    args = self.visit_expr_list(expr.args)
    return symbolic_call(fn, args)

  def visit_TiledReduce(self, expr):
    args = self.visit_expr_list(expr.args)
    fn = self.visit_expr(expr.fn)
    return symbolic_call(fn, args)

  def visit_TiledScan(self, expr):
    args = self.visit_expr_list(expr.args)
    fn = self.visit_expr(expr.fn)
    acc_shape = symbolic_call(fn, args)
    emit = self.visit_expr(expr.emit)
    return symbolic_call(emit, [acc_shape])

  def bind(self, lhs, rhs):
    if isinstance(lhs, syntax.Tuple):
      assert isinstance(rhs, Tuple)
      for l,r in zip(lhs.elts, rhs.elts):
        self.bind(l,r)
    elif isinstance(lhs, syntax.Var):
      self.value_env[lhs.name] = rhs

  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)
    self.bind(stmt.lhs, rhs)

  def visit_Return(self, stmt):
    new_value = self.visit_expr(stmt.value)
    old_value = self.value_env.get("$return", unknown_value)
    combined = old_value.combine(new_value)
    self.value_env["$return"] = combined

  def visit_fn(self, fn):
    assert isinstance(fn, syntax.TypedFn)
    self._clear()
    arg_types = [fn.type_env[name] for name in fn.arg_names]
    input_values = shape_from_type.Converter().from_types(arg_types)
    for n,v in zip(fn.arg_names, input_values):
      self.value_env[n] = v
    self.visit_block(fn.body)
    return self.value_env["$return"]

_shape_cache = {}
def call_shape_expr(typed_fn):
  if isinstance(typed_fn, str):
    typed_fn = syntax.TypedFn.registry[typed_fn]

  if typed_fn.name in _shape_cache:
    return _shape_cache[typed_fn.name]
  else:
    shape_inference = ShapeInference()
    abstract_shape = shape_inference.visit_fn(typed_fn)
    assert abstract_shape is not None, \
        "Shape inference returned None for %s" % typed_fn.name
    _shape_cache[typed_fn.name] = abstract_shape
    return abstract_shape

def bind(lhs, rhs, env):
  if isinstance(lhs, Var):
    env[lhs] = rhs
  elif isinstance(lhs, Shape):
    assert isinstance(rhs, Shape), "Expected %s, got %s" % (lhs, rhs)
    bind_pairs(lhs.dims, rhs.dims, env)
  elif isinstance(lhs, Closure):
    assert isinstance(rhs, Closure)
    bind_pairs(lhs.args, rhs.args, env)
  elif isinstance(lhs, Tuple):
    assert isinstance(rhs, Tuple)
    bind_pairs(lhs.elts, rhs.elts, env)
  else:
    raise RuntimeError("Unexpected shape LHS: %s" % lhs)

def bind_pairs(xs, ys, env):
  assert len(xs) == len(ys), \
      "Can't bind %s and %s due to unequal lengths" % (xs, ys)
  for (x,y) in zip(xs,ys):
    bind(x,y,env)

def subst(x, env):
  if isinstance(x, Var):
    assert x in env, "Unknown variable %s" % x
    return env[x]
  elif isinstance(x, Scalar):
    return x
  elif isinstance(x, Shape):
    return make_shape(subst_list(x.dims, env))
  elif isinstance(x, Tuple):
    return shape_semantics.tuple(subst_list(x.elts, env))
  elif isinstance(x, Closure):
    return Closure(x.fn, subst_list(x.args, env))
  else:
    raise RuntimeError("Unexpected abstract expression: %s" % x)

def subst_list(xs, env):
  return [subst(x, env) for x in xs]

def symbolic_call(fn, abstract_inputs):
  # result in terms of variables like input0, (shape: input1, input2), etc..
  if fn.__class__ is Closure:
    closure_elts = tuple(fn.args)
    fn = fn.fn
  else:
    closure_elts = ()
  abstract_result_value = call_shape_expr(fn)
  conv = shape_from_type.Converter()
  shape_formals = conv.from_types(fn.input_types)
  env = {}
  bind_pairs(shape_formals, closure_elts + tuple(abstract_inputs), env)
  return subst(abstract_result_value, env)
