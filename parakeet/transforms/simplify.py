
from dsltools import ScopedDict

from .. import prims, syntax 
from .. analysis.collect_vars import collect_var_names
from .. analysis.mutability_analysis import TypeBasedMutabilityAnalysis
from .. analysis.use_analysis import use_count
from .. ndtypes import ArrayT,  ClosureT, NoneT, ScalarT, TupleT, ImmutableT, NoneType, SliceT, FnT

from .. syntax import (AllocArray, Assign, ExprStmt, Expr, 
                       Const, Var, Tuple, TupleProj, Closure, ClosureElt, Cast,
                       Slice, Index, Array, ArrayView, Attribute, Struct, Select, 
                       PrimCall, Call, TypedFn, UntypedFn, 
                       OuterMap, Map, Reduce, Scan, IndexMap, IndexReduce, 
                       IndexScan, FilterReduce)
from .. syntax.helpers import (collect_constants, is_one, is_zero, is_false, is_true, all_constants,
                               get_types, 
                               slice_none_t, const_int, one, none, true, false, slice_none, 
                               zero_i64) 
import subst
import transform 
from transform import Transform


# classes of expressions known to have no side effects
# and to be unaffected by changes in mutable state as long
# as all their arguments are SSA variables or constants
#
# examples of unsafe expressions:
#  - Index: the underlying array is mutable, thus the expression depends on
#    any data modifications
#  - Call: Unless the function is known to contain only safe expressions it
#    might depend on mutable state or modify it itself

class Simplify(Transform):
  def __init__(self):
    transform.Transform.__init__(self)
    # associate var names with any immutable values
    # they are bound to
    self.bindings = ScopedDict()

    # which expressions have already been computed
    # and stored in some variable?
    self.available_expressions = ScopedDict()

  def pre_apply(self, fn):
    ma = TypeBasedMutabilityAnalysis()

    # which types have elements that might
    # change between two accesses?
    self.mutable_types = ma.visit_fn(fn)
    self.use_counts = use_count(fn)
    
  def immutable_type(self, t):
    return t not in self.mutable_types


  _immutable_classes = set([Const,  Var, 
                            Closure, ClosureElt, 
                            Tuple, TupleProj, 
                            Cast, PrimCall, 
                            TypedFn, UntypedFn, 
                            ArrayView, 
                            Slice, 
                            Map, Reduce, Scan, OuterMap, 
                            IndexMap, IndexReduce, IndexScan, 
                            ])
  
  def immutable(self, expr):
    """
    TODO: make all this mutability/immutability stuff sane 
    """
    klass = expr.__class__ 
    
    
    result = (klass in self._immutable_classes and 
                (all(self.immutable(c) for c in expr.children()))) or \
             (klass is Attribute and isinstance(expr.type, ImmutableT))
    return result 
    
                      
  def temp(self, expr, name = None, use_count = 1):
    """
    Wrapper around Codegen.assign_name which also updates bindings and
    use_counts
    """
    if self.is_simple(expr):
      return expr 
    else:
      new_var = self.assign_name(expr, name = name)
      self.bindings[new_var.name] = expr
      self.use_counts[new_var.name] = use_count
      return new_var

  def transform_expr(self, expr):
    if self.is_simple(expr):
      if expr.type == NoneType:
        return none 
      else:
        return Transform.transform_expr(self, expr)
    stored = self.available_expressions.get(expr)
    if stored is not None: 
      return stored
    return Transform.transform_expr(self, expr)

    
  def transform_Var(self, expr):
    t = expr.type 
    if t.__class__ is NoneT:
      return none 
    elif t.__class__ is SliceT and \
         t.start_type == NoneType and \
         t.stop_type == NoneType and \
         t.step_type == NoneType:
      return slice_none 
    
    name = expr.name
    prev_expr = expr

    while name in self.bindings:
      prev_expr = expr 
        
      expr = self.bindings[name]
      if expr.__class__ is Var:
        name = expr.name
      else:
        break
    c = expr.__class__ 
    
    if c is Var or c is Const:

      return expr
    else:

      return prev_expr
  
  def transform_Cast(self, expr):
    
    v = self.transform_expr(expr.value)
    if v.type == expr.type:
      return v
    elif v.__class__ is Const and isinstance(expr.type, ScalarT):
      return Const(expr.type.dtype.type(v.value), type = expr.type)
    elif self.is_simple(v):
      expr.value = v
      return expr
    else:
      expr.value = self.assign_name(v)
      return expr

  def transform_Attribute(self, expr):
    v = self.transform_expr(expr.value)
    
    if v.__class__ is Var and v.name in self.bindings:
      stored_v = self.bindings[v.name]
      c = stored_v.__class__
      if c is Var or c is Struct:
        v = stored_v
      elif c is ArrayView:
        if expr.name == 'shape':
          return self.transform_expr(stored_v.shape)
        elif expr.name == 'strides':
          return self.transform_expr(stored_v.strides)
        elif expr.name == 'data':
          return self.transform_expr(stored_v.data)
      elif c is AllocArray:
        if expr.name == 'shape':
          return self.transform_expr(stored_v.shape)
      elif c is Slice:
        if expr.name == "start":
          return self.transform_expr(stored_v.start)
        elif expr.name == "stop":
          return self.transform_expr(stored_v.stop)
        else:
          assert expr.name == "step", "Unexpected attribute for slice: %s" % expr.name  
          return self.transform_expr(stored_v.step)
    if v.__class__ is Struct:
      idx = v.type.field_pos(expr.name)
      return v.args[idx]
    elif v.__class__ is not Var:
      v = self.temp(v, "struct")
    if expr.value == v:
      return expr
    else:
      return Attribute(value = v, name = expr.name, type = expr.type)
  
  def transform_Closure(self, expr):
    expr.args = tuple(self.transform_simple_exprs(expr.args))
    return expr

  def transform_Tuple(self, expr):
    expr.elts = tuple( self.transform_simple_exprs(expr.elts))
    return expr

  def transform_TupleProj(self, expr):
    idx = expr.index
    assert isinstance(idx, int), \
        "TupleProj index must be an integer, got: " + str(idx)
    new_tuple = self.transform_expr(expr.tuple)

    if new_tuple.__class__ is Var and new_tuple.name in self.bindings:
      tuple_expr = self.bindings[new_tuple.name]
      
      if tuple_expr.__class__ is Tuple:
        assert idx < len(tuple_expr.elts), \
          "Too few elements in tuple %s : %s, elts = %s" % (expr, tuple_expr.type, tuple_expr.elts)
        return tuple_expr.elts[idx]
      elif tuple_expr.__class__ is Struct:
        assert idx < len(tuple_expr.args), \
          "Too few args in closure %s : %s, elts = %s" % (expr, tuple_expr.type, tuple_expr.elts) 
        return tuple_expr.args[idx]
    

    #if not self.is_simple(new_tuple):
    #  complex_expr = new_tuple 
    #  new_tuple = self.assign_name(complex_expr, "tuple")
    #  print "MADE ME A NEW TUPLE", complex_expr, new_tuple 
    expr.tuple = new_tuple
    return expr

  def transform_ClosureElt(self, expr):
    idx = expr.index
    assert isinstance(idx, int), \
        "ClosureElt index must be an integer, got: " + str(idx)
    new_closure = self.transform_expr(expr.closure)

    if new_closure.__class__ is Var and new_closure.name in self.bindings:
      closure_expr = self.bindings[new_closure.name]
      if closure_expr.__class__ is Closure:
        return closure_expr.args[idx]

    if not self.is_simple(new_closure):
      new_closure = self.assign_name(new_closure, "closure")
    expr.closure = new_closure
    return expr

  def transform_Call(self, expr):
    fn = self.transform_expr(expr.fn)
    args = self.transform_simple_exprs(expr.args)
    if fn.type.__class__ is ClosureT:
      closure_elts = self.closure_elts(fn)
      combined_args = tuple(closure_elts) + tuple(args)
      if fn.type.fn.__class__ is TypedFn:
        fn = fn.type.fn
      else:
        assert isinstance(fn.type.fn, UntypedFn)
        from .. type_inference import specialize 
        fn = specialize(fn, get_types(combined_args))
      assert fn.return_type == expr.type
      return Call(fn, combined_args, type = fn.return_type)
    else:
      expr.fn = fn
      expr.args = args
      return expr

  def transform_if_simple_expr(self, expr):
    if isinstance(expr, Expr):
      return self.transform_simple_expr(expr)
    else:
      return expr 
    
  def transform_simple_expr(self, expr, name = None):
    if name is None: name = "temp"
    result = self.transform_expr(expr)
    if not self.is_simple(result):
      return self.assign_name(result, name)
    else:
      return result
  
  def transform_simple_exprs(self, args):
    return [self.transform_simple_expr(x) for x in args]

  def transform_Array(self, expr):
    expr.elts = tuple(self.transform_simple_exprs(expr.elts))
    return expr
  
  def transform_Slice(self, expr):
    expr.start = self.transform_simple_expr(expr.start)
    expr.stop = self.transform_simple_expr(expr.stop)
    expr.step = self.transform_simple_expr(expr.step)
    return expr 

  
  def transform_index_expr(self, expr):
    if expr.__class__ is Tuple:
      new_elts = []
      for elt in expr.elts:
        new_elt = self.transform_expr(elt)
        if not self.is_simple(new_elt) and new_elt.type.__class__ is not SliceT:
          new_elt = self.temp(new_elt, "index_tuple_elt")
        new_elts.append(new_elt)
      expr.elts = tuple(new_elts)
      return expr 
    else:
      return self.transform_expr(expr) 
  
  def transform_Index(self, expr):
    expr.value = self.transform_expr(expr.value)

    expr.index = self.transform_index_expr(expr.index)

    if expr.value.__class__ is Array and expr.index.__class__ is Const:
      assert isinstance(expr.index.value, (int, long)) and \
             len(expr.value.elts) > expr.index.value
      return expr.value.elts[expr.index.value]
    
    # take expressions like "a[i][j]" and turn them into "a[i,j]" 
    if expr.value.__class__ is Index: 
      base_array = expr.value.value
      if isinstance(base_array.type, ArrayT):
        base_index = expr.value.index 
        if isinstance(base_index.type, TupleT):
          indices = self.tuple_elts(base_index)
        else:
          assert isinstance(base_index.type, ScalarT), \
            "Unexpected index type %s : %s in %s" % (base_index, base_index.type, expr)
          indices = [base_index]
        if isinstance(expr.index.type, TupleT):
          indices = tuple(indices) + tuple(self.tuple_elts(expr.index))
        else:
          assert isinstance(expr.index.type, ScalarT), \
            "Unexpected index type %s : %s in %s" % (expr.index, expr.index.type, expr)
          indices = tuple(indices) + (expr.index,)
        expr = self.index(base_array, self.tuple(indices))
        return self.transform_expr(expr)
    if expr.value.__class__ is not Var:
      expr.value = self.temp(expr.value, "array")
    return expr

  def transform_Struct(self, expr):
    new_args = self.transform_simple_exprs(expr.args)
    return syntax.Struct(new_args, type = expr.type)

  def transform_Select(self, expr):
    cond = self.transform_expr(expr.cond)
    trueval = self.transform_expr(expr.true_value)
    falseval = self.transform_expr(expr.false_value)
    if is_true(cond):
      return trueval 
    elif is_false(cond):
      return falseval
    elif trueval == falseval:
      return trueval 
    else:
      expr.cond = cond 
      expr.false_value = falseval 
      expr.true_value = trueval 
      return expr    
  
  def transform_PrimCall(self, expr):
    args = self.transform_simple_exprs(expr.args)
    prim = expr.prim
    if all_constants(args):
      return syntax.Const(value = prim.fn(*collect_constants(args)),
                          type = expr.type)
    
    if len(args) == 1:
      x = args[0]
      if prim == prims.logical_not:
        if is_false(x):
          return true 
        elif is_true(x):
          return false 
    if len(args) == 2:
      x,y = args 
      
      if prim == prims.add:
        if is_zero(x):
          return y
        elif is_zero(y):
          return x
        if y.__class__ is Const and y.value < 0:
          expr.prim = prims.subtract
          expr.args = (x, Const(value = -y.value, type = y.type))
          
          return expr 
        elif x.__class__ is Const and x.value < 0:
          expr.prim = prims.subtract
          expr.args = (y, Const(value = -x.value, type = x.type)) 
          return expr 
        
      elif prim == prims.subtract:
        if is_zero(y):
          return x
        elif is_zero(x) and y.__class__ is Var:
           
          stored = self.bindings.get(y.name)
          
          # 0 - (a * b) --> -a * b |or| a * -b 
          if stored and stored.__class__ is PrimCall and stored.prim == prims.multiply:
            
            a,b = stored.args
            if a.__class__ is Const:
              expr.prim = prims.multiply
              neg_a = Const(value = -a.value, type = a.type)
              expr.args = [neg_a, b]
              return expr 
            elif b.__class__ is Const:
              expr.prim = prims.multiply
              neg_b = Const(value = -b.value, type = b.type)
              expr.args = [a, neg_b]
              return expr 
            
      elif prim == prims.multiply:
      
        if is_one(x):
          return y
        elif is_one(y):
          return x
        elif is_zero(x):
          return x
        elif is_zero(y):
          return y
        
      elif prim == prims.divide and is_one(y):
        return x
      
      elif prim == prims.power:
      
        if is_one(y):
          return self.cast(x, expr.type)
        elif is_zero(y):
          return one(expr.type)
        elif y.__class__ is Const and y.value == 2:
          return self.cast(self.mul(x, x, "sqr"), expr.type)
      elif prim == prims.logical_and:
        if is_true(x):
          return y
        elif is_true(y):
          return x  
        elif is_false(x) or is_false(y):
          return false 
      elif prim == prims.logical_or:
        if is_true(x) or is_true(y):
          return true
        elif is_false(x):
          return y 
        elif is_false(y):
          return x 
    expr.args = args
    return expr 
  
  def transform_Map(self, expr):

    expr.args = self.transform_simple_exprs(expr.args)
    expr.fn = self.transform_expr(expr.fn)
    expr.axis = self.transform_if_expr(expr.axis)
    
    
    max_rank = max(self.rank(arg) for arg in expr.args)
    # if an axis is the Python value None, turn it into the IR expression for None
    if max_rank == 1 and self.is_none(expr.axis): expr.axis = zero_i64
    elif expr.axis is None: expr.axis = none   
    return expr  
  
  def transform_OuterMap(self, expr):
    expr.args = self.transform_simple_exprs(expr.args)
    expr.fn = self.transform_expr(expr.fn)
    expr.axis = self.transform_if_expr(expr.axis)
    max_rank = max(self.rank(arg) for arg in expr.args)
    # if an axis is the Python value None, turn it into the IR expression for None
    if max_rank == 1 and self.is_none(expr.axis): expr.axis = zero_i64
    elif expr.axis is None: expr.axis = none   
    return expr  
    
  def transform_shape(self, expr):
    if isinstance(expr, Tuple):
      expr.elts = tuple(self.transform_simple_exprs(expr.elts))
      return expr 
    else:
      return self.transform_simple_expr(expr)
  
  def transform_ParFor(self, stmt):
    stmt.bounds = self.transform_shape(stmt.bounds)
    stmt.fn = self.transform_expr(stmt.fn)
    return stmt
  
  def transform_Reduce(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.init = self.transform_if_simple_expr(expr.init)
    expr.args = self.transform_simple_exprs(expr.args)
    # if an axis is the Python value None, turn it into the IR expression for None
    max_rank = max(self.rank(arg) for arg in expr.args)
    if max_rank == 1 and self.is_none(expr.axis): expr.axis = zero_i64
    elif expr.axis is None: expr.axis = none   
    return expr  
  
  def transform_Scan(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.emit = self.transform_expr(expr.emit)
    expr.init = self.transform_if_simple_expr(expr.init)
    expr.args = self.transform_simple_exprs(expr.args)
    max_rank = max(self.rank(arg) for arg in expr.args)
    if max_rank == 1 and self.is_none(expr.axis): expr.axis = zero_i64
    elif expr.axis is None: expr.axis = none   
    return expr  
   
  
  def transform_IndexMap(self, expr):
    expr.fn = self.transform_expr(expr.fn)
    expr.shape = self.transform_shape(expr.shape)
    return expr 
  
  def transform_IndexReduce(self, expr):
    expr.fn = self.transform_if_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.init = self.transform_if_simple_expr(expr.init)
    expr.shape = self.transform_shape(expr.shape)
    return expr 
  
  def transform_IndexScan(self, expr):
    expr.fn = self.transform_if_expr(expr.fn)
    expr.combine = self.transform_expr(expr.combine)
    expr.emit = self.transform_if_expr(expr.emit)
    expr.init = self.transform_if_simple_expr(expr.init)
    expr.shape = self.transform_shape(expr.shape)
    return expr 
  
     
  def transform_ConstArray(self, expr):
    expr.shape = self.transform_shape(expr.shape)
    expr.value = self.transform_simple_expr(expr.value)
    return expr
  
  def transform_ConstArrayLike(self, expr):
    expr.array = self.transform_simple_expr(expr.array)
    expr.value = self.transform_simple_expr(expr.value)
  
  def temp_in_block(self, expr, block, name = None):
    """
    If we need a temporary variable not in the current top scope but in a
    particular block, then use this function. (this function also modifies the
    bindings dictionary)
    """
    if name is None:
      name = "temp"
    var = self.fresh_var(expr.type, name)
    block.append(Assign(var, expr))
    self.bindings[var.name] = expr
    return var

  def set_binding(self, name, value):
    assert value.__class__ is not Var or \
        value.name != name, \
        "Can't set name %s bound to itself" % name
    self.bindings[name] = value

  def bind_var(self, name, rhs):
    if rhs.__class__ is Var:
      old_val = self.bindings.get(rhs.name)
      if old_val and self.is_simple(old_val):
        self.set_binding(name, old_val)
      else:
        self.set_binding(name, rhs)
    else:
      self.set_binding(name, rhs)

  def bind(self, lhs, rhs):
    lhs_class = lhs.__class__
    if lhs_class is Var:
      self.bind_var(lhs.name, rhs)
    elif lhs_class is Tuple and rhs.__class__ is Tuple:
      assert len(lhs.elts) == len(rhs.elts)
      for lhs_elt, rhs_elt in zip(lhs.elts, rhs.elts):
        self.bind(lhs_elt, rhs_elt)

  def transform_lhs_Index(self, lhs):
    lhs.index = self.transform_index_expr(lhs.index)
    if lhs.value.__class__ is Var:
      stored = self.bindings.get(lhs.value.name)
      if stored and stored.__class__ is Var:
        lhs.value = stored
    else:
      lhs.value = self.assign_name(lhs.value, "array")
    return lhs

  def transform_lhs_Attribute(self, lhs):
    # lhs.value = self.transform_expr(lhs.value)
    return lhs

  def transform_ExprStmt(self, stmt):
    """Don't run an expression unless it possibly has a side effect"""

    v = self.transform_expr(stmt.value)
    if self.immutable(v):
      return None
    else:
      stmt.value = v
      return stmt

  def transform_Assign(self, stmt):
    
    lhs = stmt.lhs
    rhs = self.transform_expr(stmt.rhs)
    lhs_class = lhs.__class__
    rhs_class = rhs.__class__

    if lhs_class is Var:
      if lhs.type.__class__ is NoneT and self.use_counts.get(lhs.name,0) == 0:
        return self.transform_stmt(ExprStmt(rhs))
      elif self.immutable(rhs):
        
        self.bind_var(lhs.name, rhs)
        if rhs_class is not Var and rhs_class is not Const:
          self.available_expressions.setdefault(rhs, lhs)
    elif lhs_class is Tuple:
      self.bind(lhs, rhs)

    elif lhs_class is Index:
      if rhs_class is Index and \
         lhs.value == rhs.value and \
         lhs.index == rhs.index:
        # kill effect-free writes like x[i] = x[i]
        return None
      elif rhs_class is Var and \
           lhs.value.__class__ is Var and \
           lhs.value.name == rhs.name and \
           lhs.index.type.__class__ is TupleT and \
           all(elt_t == slice_none_t for elt_t in lhs.index.type.elt_types):
        # also kill x[:] = x
        return None
      else:
        lhs = self.transform_lhs_Index(lhs)
        # when assigning x[j] = [1,2,3]
        # just rewrite it as a sequence of element assignments 
        # to avoid 
        if lhs.type.__class__ is ArrayT and \
           lhs.type.rank == 1 and \
           rhs.__class__ is Array:
          lhs_slice = self.assign_name(lhs, "lhs_slice")
          for (elt_idx, elt) in enumerate(rhs.elts):
            lhs_idx = self.index(lhs_slice, const_int(elt_idx), temp = False)
            self.assign(lhs_idx, elt)
          return None
        elif not self.is_simple(rhs):
          rhs = self.assign_name(rhs)
    else:
      assert lhs_class is Attribute
      assert False, "Considering making attributes immutable"
      lhs = self.transform_lhs_Attribute(lhs)

    if rhs_class is Var and \
       rhs.name in self.bindings and \
       self.use_counts.get(rhs.name, 1) == 1:
      self.use_counts[rhs.name] = 0
      rhs = self.bindings[rhs.name]
    stmt.lhs = lhs
    stmt.rhs = rhs
    return stmt

  def transform_block(self, stmts, keep_bindings = False):
    self.available_expressions.push()
    self.bindings.push()
    
    new_stmts = Transform.transform_block(self, stmts)
    
    self.available_expressions.pop()
    if not keep_bindings:
      self.bindings.pop()
    return new_stmts

  def enter_loop(self, phi_nodes):
    result = {}
    for (k, (left,right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      if new_left == right:
        self.set_binding(k, new_left)
      else:
        result[k] = (new_left, right)
    return result 
  
  def transform_merge(self, phi_nodes, left_block, right_block):
    result = {}
    for (k, (left, right)) in phi_nodes.iteritems():
      new_left = self.transform_expr(left)
      new_right = self.transform_expr(right)

      if not isinstance(new_left, (Const, Var)):
        new_left = self.temp_in_block(new_left, left_block)
      if not isinstance(new_right, (Const, Var)):
        new_right = self.temp_in_block(new_right, right_block)

      if new_left == new_right:
        # if both control flows yield the same value then
        # we don't actually need the phi-bound variable, we can just
        # replace the left value everywhere
        self.assign(Var(name= k, type = new_left.type), new_left)
        self.set_binding(k, new_left)
      else:
        result[k] = new_left, new_right
    return result

  def transform_If(self, stmt):
    stmt.true = self.transform_block(stmt.true, keep_bindings = True)
    stmt.false = self.transform_block(stmt.false, keep_bindings=True)
    stmt.merge = self.transform_merge(stmt.merge,
                                      left_block = stmt.true,
                                      right_block = stmt.false)
    self.bindings.pop()
    self.bindings.pop()
    stmt.cond = self.transform_simple_expr(stmt.cond, "cond")
    if len(stmt.true) == 0 and len(stmt.false) == 0 and len(stmt.merge) <= 2:
      for (lhs_name, (true_expr, false_expr)) in stmt.merge.items():
        lhs_type = self.lookup_type(lhs_name)
        lhs_var = Var(name = lhs_name, type = lhs_type)
        assert true_expr.type == false_expr.type, \
          "Unexpcted type mismatch: %s != %s" % (true_expr.type, false_expr.type)
        rhs = Select(stmt.cond, true_expr, false_expr, type = true_expr.type)
        self.bind_var(lhs_name, rhs)
        self.assign(lhs_var, rhs)
      return None 
    return stmt

  def transform_loop_condition(self, expr, outer_block, loop_body, merge):
    """Normalize loop conditions so they are just simple variables"""

    if self.is_simple(expr):
      return self.transform_expr(expr)
    else:
      loop_carried_vars = [name for name in collect_var_names(expr)
                           if name in merge]
      if len(loop_carried_vars) == 0:
        return expr

      left_values = [merge[name][0] for name in loop_carried_vars]
      right_values = [merge[name][1] for name in loop_carried_vars]

      left_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars,
                                                  left_values)))
      if not self.is_simple(left_cond):
        left_cond = self.temp_in_block(left_cond, outer_block, name = "cond")

      right_cond = subst.subst_expr(expr, dict(zip(loop_carried_vars,
                                                   right_values)))
      if not self.is_simple(right_cond):
        right_cond = self.temp_in_block(right_cond, loop_body, name = "cond")

      cond_var = self.fresh_var(left_cond.type, "cond")
      merge[cond_var.name] = (left_cond, right_cond)
      return cond_var

    
  def transform_While(self, stmt):
    merge = self.enter_loop(stmt.merge)
    stmt.body = self.transform_block(stmt.body)
    stmt.merge = self.transform_merge(merge,
                                      left_block = self.blocks.current(),
                                      right_block = stmt.body)
    stmt.cond = \
        self.transform_loop_condition(stmt.cond,
                                      outer_block = self.blocks.current(),
                                      loop_body = stmt.body,
                                      merge = stmt.merge)
    return stmt

  def transform_ForLoop(self, stmt):
    

    merge = self.enter_loop(stmt.merge)
    stmt.body = self.transform_block(stmt.body) 
    stmt.merge = self.transform_merge(merge,
                                      left_block = self.blocks.current(),
                                      right_block = stmt.body)
    stmt.start = self.transform_simple_expr(stmt.start, 'start')
    stmt.stop = self.transform_simple_expr(stmt.stop, 'stop')
    if self.is_none(stmt.step):
      stmt.step = one(stmt.start.type)
    else:
      stmt.step = self.transform_simple_expr(stmt.step, 'step')

    # if a loop is only going to run for one iteration, might as well get rid of
    # it
    if stmt.start.__class__ is Const and \
       stmt.stop.__class__ is Const and \
       stmt.step.__class__ is Const:
      if stmt.start.value >= stmt.stop.value:
        for (var_name, (input_value, _)) in stmt.merge.iteritems():
          var = Var(var_name, input_value.type)
          self.blocks.append(Assign(var, input_value))
        return None
      elif stmt.start.value + stmt.step.value >= stmt.stop.value:
        for (var_name, (input_value, _)) in stmt.merge.iteritems():
          var = Var(var_name, input_value.type)
          self.blocks.append(Assign(var, input_value))
        self.assign(stmt.var, stmt.start)
        self.blocks.top().extend(stmt.body)
        return None
    return stmt

  def transform_Return(self, stmt):
    new_value = self.transform_expr(stmt.value)
    if new_value != stmt.value:
      stmt.value = new_value
    return stmt
