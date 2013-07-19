from .. import prims 
from ..syntax import Const, PrimCall, Var, If  
from ..syntax.helpers import (zero, one, 
                              is_zero, is_one, const_bool, 
                              wrap_constants, get_types, wrap_if_constant)

from core_builder import CoreBuilder 

class ArithBuilder(CoreBuilder):
  def prim(self, prim_fn, args, name=None):
    args = wrap_constants(args)
    arg_types = get_types(args)
    upcast_types = prim_fn.expected_input_types(arg_types)
    result_type = prim_fn.result_type(upcast_types)
    upcast_args = [self.cast(x, t) for (x,t) in zip(args, upcast_types)]
    prim_call = PrimCall(prim_fn, upcast_args, type=result_type)
    if name:
      return self.assign_name(prim_call, name)
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

    return self.cast(wrap_if_constant(c), x.type.combine(y.type))

  def add(self, x, y, name = None):
    if is_zero(x):
      return self.pick_second(x,y)
    elif is_zero(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value + y.value)
    else:
      return self.prim(prims.add, [x,y], name)

  def sub(self, x, y, name = None):
    if is_zero(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value - y.value)
    else:
      return self.prim(prims.subtract, [x,y], name)

  def mul(self, x, y, name = None):
    if is_one(x):
      return self.pick_second(x,y)
    elif is_one(y):
      return self.pick_first(x,y)
    elif is_zero(x) or is_zero(y):
      return self.pick_const(x, y, 0)
    else:
      return self.prim(prims.multiply, [x,y], name)

  def div(self, x, y, name = None):
    if is_one(y):
      return self.pick_first(x,y)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value / y.value)
    else:
      return self.prim(prims.divide, [x,y], name)

  def safediv(self, x, y, name = None):
    top = self.add(x, y)
    top = self.sub(top, one(top.type))
    return self.div(top, y, name = name)


  def mod(self, x, y, name = None):
    if is_one(y):
      return self.pick_const(x, y, 0)
    elif x.__class__ is Const and y.__class__ is Const:
      return self.pick_const(x, y, x.value % y.value)
    else:
      return self.prim(prims.mod, [x,y], name)

  def lt(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(False)
    else:
      return self.prim(prims.less, [x,y], name)

  def lte(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(True)
    else:
      return self.prim(prims.less_equal, [x,y], name)

  def gt(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(False)
    else:
      return self.prim(prims.greater, [x,y], name)

  def gte(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(True)
    else:
      return self.prim(prims.greater_equal, [x,y], name)

  def eq(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(True)
    else:
      return self.prim(prims.equal, [x,y], name)

  def neq(self, x, y, name = None):
    if isinstance(x, (Var, Const)) and x == y:
      return const_bool(False)
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
    self.blocks += If(cond, [], [], merge)
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
    self.blocks += If(cond, [], [], merge)
    return result   
