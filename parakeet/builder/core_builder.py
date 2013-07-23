from treelike import NestedBlocks

from .. import syntax, names 
from ..ndtypes import (make_array_type, make_tuple_type, 
                       Int32, Int64, SliceT, TupleT, ScalarT, StructT, NoneT, 
                       ArrayT, FnT, ClosureT) 
from ..syntax import (ArrayView, Assign, Attribute, Cast, Const, Closure,  Comment, Expr, 
                      Index, PrimCall,   Return, Struct,  Tuple, TupleProj, Var, TypedFn)
from ..syntax.helpers import (wrap_if_constant, 
                              const_bool, const_int, const_float, get_types,  
                              one_i64, zero_i64, 
                              zero)

class CoreBuilder(object):
  none = syntax.none
  null_slice = syntax.slice_none
  
  def __init__(self, type_env = None, blocks = None):
    if type_env is None:
      type_env = {}
    self.type_env = type_env 
    if blocks is None: 
      blocks = NestedBlocks()
      
    self.blocks = blocks 

    # cut down the number of created nodes by
    # remembering which tuple variables we've created
    # and looking up their elements directly
    self.tuple_elt_cache = {}

  def comment(self, text):
    self.blocks.append(Comment(text))
  
  def assign(self, lhs, rhs):
    self.insert_stmt(Assign(lhs, rhs))
    
  def return_(self, value):
    self.blocks += [Return(value)]  
    
  def fresh_var(self, t, prefix = "temp"):
    assert prefix is not None
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

  def temp_name(self, expr):
    c = expr.__class__
    if c is PrimCall:
      return expr.prim.name
    elif c is Attribute:
      if expr.value.__class__ is Var:
        return names.original(expr.value.name) + "_" + expr.name
      else:
        return expr.name
    elif c is Attribute:
      if expr.value.__class__ is Var:
        return "%s_%s" % (expr.value.name, expr.name)
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
    elif c is TupleProj:
      if expr.tuple.__class__ is Var:
        original = names.original(expr.tuple.name)
        return "%s%d_elt%d" % (original, names.versions[original], expr.index)
      else:
        return "tuple_elt%d" % expr.index
    elif c is Var:
      return names.refresh(expr.name) 
    else:
      return "temp"

  def is_simple(self, expr):
    c = expr.__class__
    return c is Var or \
           c is Const or \
           (c is Tuple and len(expr.elts) == 0) or \
           (c is Struct and len(expr.args) == 0) or \
           (c is Closure and len(expr.args) == 0)



  def assign_name(self, expr, name = None):
    #if self.is_simple(expr):
    #  return expr
    if name is None:
      name = self.temp_name(expr)
    
    var = self.fresh_var(expr.type, names.refresh(name))
    self.assign(var, expr)
    return var

  def int(self, x, name = None):
    if name is None: 
      return const_int(x)
    else:
      return self.assign_name(const_int(x), name)

  def bool(self, x, name = None):
    if name is None: 
      return const_bool(x)
    else:
      return self.assign_name(const_bool(x), name)

  def zero(self, t = Int32, name = None):
    if name is None:
      return zero(t)
    else:
      return self.assign_name(zero(t), name)

  def zero_i32(self, name = None):
      return self.zero(t = Int32, name = name)

  def zero_i64(self, name = None):
    return self.zero(t = Int64, name = name)

  def cast(self, expr, t):
    if expr.type == t:
      return expr
    assert isinstance(t, ScalarT), \
        "Can't cast %s : %s to non-scalar type %s" % (expr, expr.type, t)
    return self.assign_name(Cast(expr, type = t), "cast_%s" % t)
  


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
      return self.assign_name(result, name)
    else:
      return result

  def is_none(self, x):
    return x is None or (isinstance(x, Expr) and x.type.__class__ is NoneT)


  def ravel(self, x):
    assert self.is_array(x)
    if x.type.rank == 1:
      return x
    print "[Warning] Ravel will fail for discontiguous arrays"
    nelts = self.nelts(x)
    shape = self.tuple((nelts,), 'shape')
    strides = self.tuple((self.int(1),), "strides")
    data = self.attr(x, 'data', 'data_ptr')
    offset = self.attr(x, 'offset')
    t = make_array_type(x.type.elt_type, 1)
    return ArrayView(data, shape, strides, offset, nelts, type = t)
  


  def tuple(self, elts, name = "tuple", explicit_struct = False):
    if not isinstance(elts, (list, tuple)):
      elts = [elts]
    tuple_t = make_tuple_type(get_types(elts))
    if explicit_struct:
      tuple_expr = Struct(elts, type = tuple_t)
    else:
      tuple_expr = Tuple(elts, type = tuple_t)
    if name:
      result_var = self.assign_name(tuple_expr, name)
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
    
  def is_array(self, x):
    return x.type.__class__ is  ArrayT
  
  def is_fn(self, x):
    return x.__class__ in (TypedFn, Closure) or isinstance(x.type, (FnT, ClosureT)) 
  
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
