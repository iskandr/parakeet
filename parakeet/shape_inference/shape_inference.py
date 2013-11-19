
from .. import syntax, prims 
from ..analysis import OffsetAnalysis, SyntaxVisitor
from ..ndtypes import  ArrayT, ScalarT, SliceT, TupleT, NoneT
from ..syntax import unwrap_constant, Expr  

import shape
import shape_from_type
from shape import (Var, Const, Shape, Tuple, Closure, Slice, Scalar, Unknown, 
                   ConstSlice, Struct, AnyScalar, Add, Mult, Div, Sub, Mod, 
                   any_scalar, unknown_value, const, any_value, combine_list, 
                   increase_rank, make_shape, is_zero, is_one, Ptr,
                   dims, combine_dims     
                   ) 

class ShapeInferenceFailure(Exception):
  def __init__(self, value, fn):
    self.value = value 
    self.fn = fn 
  
  def __str__(self):
    return "Couldn't infer shape of %s in function %s" % (self.value, self.fn.name)

  
counter = 0
class ShapeInference(SyntaxVisitor):
  def size_along_axis(self, value, axis):
    assert isinstance(value, Shape)
    return value.dims[axis]

  
  def is_tuple(self, x):
    return isinstance(x, Tuple)

  def is_none(self, x):
    return isinstance(x, Const) and x.value is None

  def rank(self, value):
    if isinstance(value, Shape):
      return value.rank
    else:
      return 0
  
  def max_rank(self, values):
    return max(self.rank(v) for v in values)
  
  def int(self, x):
    return const(x)

  def bool(self, x):
    return const(x)


  _scalar_shape_classes = (Const, Var, Add, Sub, Mult, Div, Mod,  AnyScalar)
  def add(self, x, y):
    cx = x.__class__ 
    cy = y.__class__ 

    if cx in self._scalar_shape_classes and cy in self._scalar_shape_classes:
      if is_zero(x):
        return y
      elif is_zero(y):
        return x
      elif cx is Const and cy is Const: 
        return const(x.value + y.value)
      elif cx is AnyScalar or cy is AnyScalar:
        return any_scalar
      else:
        return Add(x,y)
    else:
      return any_value 

  def sub(self, x, y):
    cx = x.__class__ 
    cy = y.__class__ 
    if cx in self._scalar_shape_classes and cy in self._scalar_shape_classes:
      if is_zero(y):
        return x
      elif cx is Const and cy is Const: 
        return const(x.value - y.value)
      elif cx is AnyScalar or cy is AnyScalar:
        return any_scalar
      elif x == y: 
        return Const(0)
      else:
        return Sub(x, y)
    else:
      return any_value 

  def mul(self, x, y):
    cx = x.__class__ 
    cy = y.__class__ 

    if cx in self._scalar_shape_classes and cy in self._scalar_shape_classes:
      if is_zero(x) or is_zero(y):
        return const(0)
      elif is_one(x):
        return y
      elif is_one(y):
        return x
      elif cx is AnyScalar or cy is AnyScalar:
        return any_scalar
      else:
        return Mult(x,y)
    else:
      return any_value
    
  def div(self, x, y):
    assert not is_zero(y), "Encountered divide by zero during shape inference"
    cx = x.__class__ 
    cy = y.__class__ 

    if cx in self._scalar_shape_classes and cy in self._scalar_shape_classes:
      if is_one(y):
        return x
      elif cx is AnyScalar or cy is AnyScalar:
        return any_scalar
      elif cx is Const and cy is Const:
        return const(int(x.value / y.value))
      elif x == y:
        return const(1)
      else:
        return Div(x, y)
    else:
      return any_value 
      

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
    if isinstance(idx, (Scalar, Slice, ConstSlice)):
      indices = [idx]
    elif idx.__class__ is Tuple:
      indices = idx.elts
    else:
      assert False, "Unexpected index: %s" % (idx,)
    result_dims = []
    for (i, curr_idx) in enumerate(indices):
      old_dim = arr.dims[i]
      if curr_idx is None or \
         (isinstance(curr_idx, Const) and curr_idx.value is None):
        result_dims.append(old_dim)
      elif isinstance(curr_idx, Scalar):
        pass
      elif curr_idx.__class__ is ConstSlice:
        result_dims.append(curr_idx.nelts)
      elif curr_idx.__class__ is Shape:
        if len(curr_idx.dims) == 0:
          # same as unknown scalar 
          pass 
        else:
          assert len(curr_idx.dims) == 1, "Indexing by a multi-dimensional array not yet supported"
          result_dims.append(curr_idx.dims[0])
      else:
        assert curr_idx.__class__ is Slice, "Unsupported index %s" % curr_idx

        if curr_idx.start is None:
          lower = const(0)
        elif isinstance(curr_idx.start, Const):
          if curr_idx.start.value is None:
            lower = const(0)
          elif curr_idx.start.value < 0:
            lower = self.sub(old_dim, curr_idx.start)
          else:
            lower = curr_idx.start
        else:
          lower = any_scalar
         
        if curr_idx.stop is None:
          upper = old_dim 
        elif isinstance(curr_idx.stop, Const):
          if curr_idx.stop.value is None:
            upper = old_dim
          elif curr_idx.stop.value < 0:
            upper = self.sub(old_dim, curr_idx.stop)
          else:
            upper = curr_idx.stop
        else:
          upper = any_scalar

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
  
  def slice_along_axis(self, arr, axis):
    if arr.__class__ is Shape:
      dims = arr.dims[:axis] + arr.dims[(axis+1):]
      if len(dims) > 0:
        return Shape(dims)
      else:
        return any_scalar
    else:
      return arr 
  
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
    
    # if all elements of the slice are constant 
    # then we can ignore the exact start/stop/step 
    # and track only the number of elements in the 
    # slice
    if start.__class__ is Const and \
       stop.__class__ is Const and \
       stop.value is not None and \
       step.__class__ is Const:
      start_val = start.value
      if start_val is None:
        start_val = 0
      step_val = step.value
      if step_val is None:
        step_val = 1
      nelts = (stop.value - start_val) / step_val
      # TODO: 
      # Properly handle negative slicing 
      if nelts >= 0:
        return ConstSlice(nelts)
    return Slice(start, stop, step)

  def call(self, fn, args):
    if fn.__class__ is Closure:
      args = tuple(fn.args) + tuple(args)
      fn = fn.fn
    return symbolic_call(fn, args)

  def invoke(self, fn, args):
    return self.call(fn, args)
  none = None
  null_slice = slice(None, None, None)

  def identity_function(self, x):
    return x
  
  def visit_fn(self, fn):
    assert isinstance(fn, syntax.TypedFn), "Expected typed function, got %s" % fn 
    self.fn = fn 
    self.value_env = {}
    self.equivalence_classes = {}

    self.known_offsets = OffsetAnalysis().visit_fn(fn)

    arg_types = [fn.type_env[name] for name in fn.arg_names]
    input_values = shape_from_type.Converter().from_types(arg_types)
    for n,v in zip(fn.arg_names, input_values):
      self.value_env[n] = v
    self.visit_block(fn.body)

  def unify_scalar_var(self, x, y):
    """
    Unification is different than combining in that it imposes constraints on
    the program. If, for example, we're unifying some scalar that's reached the
    top of the lattice (and thus know nothing about it statically), with a
    scalar known to be some constant-- then the result is we expect both
    variables to be equal to that constant.
    """

    assert isinstance(x, Var), "Expected scalar variable, but got: " + str(x)
    assert isinstance(y, Scalar), "Expected scalar, but got: " + str(y)
    if y == any_scalar:
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
    acc = any_scalar
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
  
  def visit_merge_loop_repeat(self, merge):
    self.visit_merge(merge)

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
    return Ptr(any_scalar)

  def visit_Cast(self, expr):
    return any_scalar 
  
  def visit_TypeValue(self, expr):
    return unknown_value
  
  def visit_Struct(self, expr):
    if isinstance(expr.type, ArrayT):
      shape_tuple = self.visit_expr(expr.args[1])
      return make_shape(shape_tuple.elts)
    elif isinstance(expr.type, TupleT):
      return Tuple(self.visit_expr_list(expr.args))
    elif isinstance(expr.type, SliceT):
      start, stop, step = self.visit_expr_list(expr.args)
      return Slice(start, stop, step)
    else:
      return unknown_value

  def visit_Fn(self, fn):
    return Closure(fn, [])


  def visit_UntypedFn(self, fn):
    return Closure(fn, [])
  
  def visit_TypedFn(self, fn):
    return Closure(fn, [])

  def shape_from_tuple(self, expr):  
    shape_tuple = self.visit_expr(expr)
    if shape_tuple.__class__ is Tuple:
      return make_shape(tuple(shape_tuple.elts))
    elif shape_tuple.__class__ is Const:
      return make_shape((shape_tuple.value,))
    else:
      return make_shape( (any_scalar,) * expr.type.rank)

  def tuple_from_shape(self, expr):
    shape = self.visit_expr(expr)
    if shape.__class__ is Shape:
      return Tuple(tuple(shape.dims))
    elif shape.__class__ is Const:
      return Tuple( (shape.value,) )
    else:
      return Tuple( (any_scalar,) * expr.type.rank) 
   
  def visit_ArrayView(self, expr):
    return self.shape_from_tuple(expr.shape)
  
  def visit_Reshape(self, expr):
    return self.shape_from_tuple(expr.shape)
  
  def visit_Shape(self, expr):
    return self.tuple_from_shape(expr.array)
  
  def visit_Transpose(self, expr):
    shape = self.visit_expr(expr.array)
    if shape.__class__ is Shape:
      return Shape(tuple(reversed(shape.dims)))
    else:
      return shape 
  
  def visit_AllocArray(self, expr):
    return self.shape_from_tuple(expr.shape)
    
  def visit_Array(self, expr):
    elts = self.visit_expr_list(expr.elts)
    elt = combine_list(elts)
    n = len(elts)
    res = increase_rank(elt, 0, const(n))
    return res

  
  def visit_ConstArray(self, expr):
    return self.shape_from_tuple(expr.shape)
  
  def visit_ConstArrayLike(self, expr):
    return self.visit_expr(expr.array)
  
  def ravel(self, shape):        
    if isinstance(shape, Shape):
      nelts = const(1)
      for dim in shape.dims:
        nelts = self.mul(nelts, dim)
      return Shape((nelts,))
    else:
      return any_value 
  

  def visit_Ravel(self, expr):
    shape = self.visit_expr(expr.array)
    return self.ravel(shape)
  
  def visit_Range(self, expr):
    start = self.visit_expr(expr.start)
    stop = self.visit_expr(expr.stop)
    step = self.visit_expr(expr.step)
    slice_value = self.slice_value(start, stop, step)
    if slice_value.__class__ is ConstSlice:
      return Shape( (slice_value.nelts,))
    else:
      return Shape( (any_scalar,) )

  
  def visit_Slice(self, expr):
    step = self.visit_expr(expr.step)
    if expr.start.__class__ is syntax.Var and \
       expr.stop.__class__ is syntax.Var and \
       step.__class__ is Const:
      start_name = expr.start.name

      stop_name = expr.stop.name
      offsets = self.known_offsets.get(stop_name, [])
      step = step.value if step.value else 1
      for (other_var, offset) in offsets:

        if other_var == start_name:
          nelts = (offset + step - 1) /  step
          # assert False, (start_name, stop_name, offsets)
          return ConstSlice(nelts)

    start = self.visit_expr(expr.start)
    stop = self.visit_expr(expr.stop)
    return self.slice_value(start, stop, step)

  def visit_Const(self, expr):
    return Const(expr.value)

  def visit_ClosureElt(self, expr):
    clos = self.visit_expr(expr.closure)
    assert clos.__class__ is Closure, \
        "Unexpected closure shape %s for expression %s" % (clos, expr)
    return clos.args[expr.index]

  def visit_TupleProj(self, expr):
    t = self.visit_expr(expr.tuple)
    assert isinstance(t, Tuple), "Expected tuple type but got %s : %s" % (t, type(t))
    return t.elts[expr.index]

  def visit_Attribute(self, expr):
    v = self.visit_expr(expr.value)
    name = expr.name

    if v.__class__ is Shape:
      if name == 'shape':
        return Tuple(v.dims)
      elif name == 'strides':
        return Tuple((any_scalar,) * len(v.dims) )
      elif name in ('offset', 'size', 'nelts'):
        return any_scalar
      elif name == 'data':
        return Ptr(any_scalar)
    elif v.__class__ is Tuple:
      if name.startswith('elt'):
        idx = int(name[3:])
      else:
        idx = int(name)
      return v[idx]
    
    elif v.__class__ is Slice:
      return getattr(v, name)
    
    elif v.__class__ is Closure:
      if name.startswith('elt'):
        idx = int(name[3:])
      elif name.startswith('closure_elt'):
        idx = int(name[len('closure_elt'):])
      else:
        idx = int(name)
      return v.args[idx]
        
      
    elif v.__class__ is Struct:
      return v.values[v.fields.index(name)]

    t = expr.value.type.field_type(name)
    
    if isinstance(t, ScalarT):
      return any_scalar
    else:
      return any_value

  def visit_PrimCall(self, expr):
    
    p = expr.prim
    args = self.visit_expr_list(expr.args)
    if p == prims.add: 
      return self.add(args[0], args[1])
    elif p == prims.subtract:
      return self.sub(args[0], args[1])
    elif p == prims.multiply:
      return self.mul(args[0], args[1])
    elif p == prims.divide:
      return self.div(args[0], args[1])
    else:
      result = shape.combine_list(args, preserve_const = False)
      if result.__class__ is Shape:
        return result
      else:
        # once a scalar passes through some prim, it's not longer the same value!
        return any_scalar 

  def visit_Select(self, expr):
    cond = self.visit_expr(expr.cond)
    falseval = self.visit_expr(expr.true_value)
    trueval = self.visit_expr(expr.false_value)

    return cond.combine(falseval).combine(trueval)
    
  def visit_Var(self, expr):
    name = expr.name
    if name in self.value_env:
      return self.value_env[name]
    elif name in self.equivalence_classes:
      for other_name in self.equivalence_classes[name]:
        if other_name in self.value_env:
          return self.value_env[other_name]
    raise RuntimeError("Unknown variable: %s in function %s" %  (expr, self.fn.name))

  def visit_Tuple(self, expr):
    return Tuple(self.visit_expr_list(expr.elts))

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
        return self.index(arr, idx)
      
    elif arr.__class__ is Ptr:
      assert isinstance(arr.elt_shape, Scalar)
      assert isinstance(idx, Scalar)
      
      return any_scalar
    
    if isinstance(arr, Scalar):
      assert False, "Expected %s to be array, shape inference found scalar" % (arr,)
    elif arr == shape.any_value:
      raise ShapeInferenceFailure(expr, self.fn)
    assert False, \
        "Can't index (%s) with array shape %s and index shape %s" % \
        (expr, arr, idx)


  
  def visit_IndexMap(self, expr):
    shape_tuple = self.visit_expr(expr.shape)
    clos = self.visit_expr(expr.fn)
    if isinstance(clos.fn.input_types[-1], TupleT):
      elt_result = symbolic_call(clos, [shape_tuple])
    else:
      elt_result = symbolic_call(clos, shape_tuple.elts)
    return make_shape(combine_dims(shape_tuple, elt_result))
    
    
  def visit_IndexReduce(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    bounds = self.visit_expr(expr.shape)
    elt_shape = symbolic_call(fn, [bounds])
    init_shape = elt_shape if self.expr_is_none(expr.init) else self.visit_expr(expr.init) 
    return symbolic_call(combine, [init_shape, elt_shape])

  def visit_IndexScan(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    emit = self.visit_expr(expr.emit)
    bounds = self.visit_expr(expr.shape)
    elt_shape = symbolic_call(fn, [bounds])
    init_shape = elt_shape if self.expr_is_none(expr.init) else self.visit_expr(expr.init) 
    acc_shape = symbolic_call(combine, [init_shape, elt_shape])
    output_elt_shape = symbolic_call(emit, [acc_shape])
    return make_shape(combine_dims(bounds, output_elt_shape))


  def normalize_axes(self, axis, args):
    if isinstance(axis, Expr):
      axis = unwrap_constant(axis)
    if isinstance(axis,tuple):
      axes = axis 
    else:
      axes = (axis,) * len(args)
    
    assert len(axes) == len(args), \
      "Mismatch between args %s and axes %s" % (args, axis)
    return axes 
  
  def adverb_elt_shapes(self, arg_shapes, axes):
    """
    Slice into array shapes along the specified axis 
    """
    elt_shapes = []
    for arg_shape, axis in zip(arg_shapes, axes):
      if axis is None:
        elt_shapes.append(any_scalar)
      elif axis < self.rank(arg_shape):
        elt_shapes.append(self.slice_along_axis(arg_shape, axis))
      else:
        elt_shapes.append(arg_shape)
    return elt_shapes 

  def inner_map_result_shape(self, elt_result, arg_shapes, axes):
    max_rank = self.max_rank(arg_shapes)    
    for i, arg_shape in enumerate(arg_shapes):
      r = self.rank(arg_shape)
      if r == max_rank:
        axis = axes[i]
        if axis is None:
          combined_dims = dims(arg_shape) + dims(elt_result)
          if len(combined_dims) > 0:
            return Shape(combined_dims)
          else:
            return any_scalar 
        else:
          return increase_rank(elt_result, 0, arg_shape.dims[axis])
    return elt_result
    
  def outer_map_result_shape(self, elt_result, arg_shapes, axes):
    result_dims = list(dims(elt_result))
    for i, arg_shape in enumerate(arg_shapes):
      r = self.rank(arg_shape)
      if r > 0:
        axis = axes[i]
        if axis is None:
          result_dims.extend(arg_shape.dims)
        else:
          result_dims.append(arg_shape.dims[axis])
    return make_shape(result_dims)
    
  def visit_Map(self, expr):
    arg_shapes = self.visit_expr_list(expr.args)
    fn = self.visit_expr(expr.fn)
    axes = self.normalize_axes(expr.axis, expr.args)
    elt_shapes = self.adverb_elt_shapes(arg_shapes, axes)    
    elt_result = symbolic_call(fn, elt_shapes)
    return self.inner_map_result_shape(elt_result, arg_shapes, axes)
  
  def expr_is_none(self, expr):
    return expr is None or expr.type.__class__ is NoneT
  
  def visit_Reduce(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    arg_shapes = self.visit_expr_list(expr.args)
    axes = self.normalize_axes(expr.axis, expr.args)
    elt_shapes = self.adverb_elt_shapes(arg_shapes, axes)
    elt_result = symbolic_call(fn, elt_shapes)
    init = elt_result if self.expr_is_none(expr.init) else self.visit_expr(expr.init) 
    return symbolic_call(combine, [init, elt_result])
      
  def visit_Scan(self, expr):
    fn = self.visit_expr(expr.fn)
    combine = self.visit_expr(expr.combine)
    arg_shapes = self.visit_expr_list(expr.args)
    axes = self.normalize_axes(expr.axis, expr.args)
    elt_shapes = self.adverb_elt_shapes(arg_shapes, axes)
    elt_result = symbolic_call(fn, elt_shapes)
    init = elt_result if self.expr_is_none(expr.init) else self.visit_expr(expr.init) 
    acc_shape = symbolic_call(combine, [init, elt_result])
    emit = self.visit_expr(expr.emit)
    emit_shape = symbolic_call(emit, [acc_shape])
    return self.inner_map_result_shape(emit_shape, arg_shapes, axes)
    
  def visit_OuterMap(self, expr):
    fn = self.visit_expr(expr.fn)
    arg_shapes = self.visit_expr_list(expr.args)
    axes = self.normalize_axes(expr.axis, expr.args)
    elt_shapes = self.adverb_elt_shapes(arg_shapes, axes)    
    elt_result = symbolic_call(fn, elt_shapes)
    return self.outer_map_result_shape(elt_result, arg_shapes, axes)

  def visit_Assign(self, stmt):
    rhs = self.visit_expr(stmt.rhs)
    if stmt.lhs.__class__ in (syntax.Var, syntax.Tuple):
      bind_syntax(stmt.lhs, rhs, self.value_env)
    
  def visit_Return(self, stmt):
    new_value = self.visit_expr(stmt.value)
    old_value = self.value_env.get("$return", unknown_value)
    combined = old_value.combine(new_value)
    self.value_env["$return"] = combined

  def visit_ForLoop(self, stmt):
    self.value_env[stmt.var.name] = any_scalar
    SyntaxVisitor.visit_ForLoop(self, stmt)
    # visit body a second time in case first-pass fixed values relative 
    # to initial value of iteration vars 
    self.visit_block(stmt.body)
  
  def visit_While(self, stmt):
    SyntaxVisitor.visit_While(self, stmt)
    # visit body a second time in case first-pass fixed values relative 
    # to initial value of iteration vars 
    self.visit_block(stmt.body)
    
    
_shape_env_cache = {}
def shape_env(typed_fn):
  key = typed_fn.cache_key
  
  if key in _shape_env_cache:
    return _shape_env_cache[key]

  shape_inference = ShapeInference()
  shape_inference.visit_fn(typed_fn)
  env = shape_inference.value_env
  _shape_env_cache[key] = env
  return env

_shape_cache = {}
def call_shape_expr(typed_fn):
  key = typed_fn.cache_key
  if key in _shape_cache:
    return _shape_cache[key]
  env = shape_env(typed_fn)
  abstract_shape = env.get("$return", Const(None))
  _shape_cache[key] = abstract_shape
  return abstract_shape

def bind_syntax(lhs, rhs, env):
  if isinstance(lhs, syntax.Tuple):
    assert isinstance(rhs, Tuple), "Expected tuple on RHS of binding %s = %s" % (lhs,rhs)
    for l,r in zip(lhs.elts, rhs.elts):
      bind_syntax(l, r, env)
  elif isinstance(lhs, syntax.Var):
    env[lhs.name] = rhs

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
    if rhs == any_value: 
      bind_pairs(lhs.elts, [any_value for _ in lhs.elts], env)
    elif lhs == unknown_value:
      bind_pairs(lhs.elts, [unknown_value for _ in lhs.elts], env)
    else:
      assert isinstance(rhs, Tuple), "Expected tuple on RHS of binding %s = %s" % (lhs,rhs)
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
    return Tuple(tuple((subst_list(x.elts, env))))
  elif isinstance(x, Closure):
    return Closure(x.fn, subst_list(x.args, env))
  elif isinstance(x, Ptr):
    return Ptr(subst(x.elt_shape, env))
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
