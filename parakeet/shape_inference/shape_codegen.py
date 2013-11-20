from dsltools import Traversal 

from ..syntax import const  
from ..ndtypes import ArrayT, ClosureT, ScalarT, TupleT 
from shape import Closure, Scalar, Var, AnyScalar, any_scalar, computable_dim

class ArgConverter(Traversal):
  def __init__(self, codegen):
    self.codegen = codegen
    self.var_counter = 0
    self.env = {}

  def fresh_var(self):
    n = self.var_counter
    self.var_counter += 1
    return Var(n)

  def bind(self, scalar_value):
    v = self.fresh_var()
    self.env[v] = scalar_value

  def convert(self, x):
    t = x.type
    if isinstance(t, ScalarT):
      self.bind(x)
    elif isinstance(t, ArrayT):
      shape = self.codegen.shape(x)
      shape_elts = self.codegen.tuple_elts(shape)
      return self.convert_list(shape_elts)
    elif isinstance(t, TupleT):
      elts = self.codegen.tuple_elts(x)
      self.convert_list(elts)
    elif isinstance(t, ClosureT):
      closure_elts = self.codegen.closure_elts(x)
      return Closure(t.fn, self.convert_list(closure_elts))
    else:
      assert False, "[shape_codegen] Not supported %s : %s" % (x,x.type)

  def convert_list(self, xs):
    for x in xs:
      self.convert(x)

class ShapeCodegen(Traversal):
  def __init__(self, codegen, exprs):
    self.codegen = codegen
    conv = ArgConverter(codegen)
    self.exprs = exprs
    conv.convert_list(exprs)
    self.env = conv.env

  def visit_Var(self, v):
    return self.env[v]

  def visit_Const(self, v):
    return const(v.value)

  def visit_Shape(self, v):
    assert len(v.dims) > 0, "Encountered empty shape"
    assert all(computable_dim(d) for d in v.dims), \
        "Symbolic shape '%s' has non-constant dimensions" % (v,)
    return self.codegen.tuple([self.visit(d) for d in v.dims])

  def visit_Dim(self, v):
    return self.codegen.tuple_proj(self.visit(v.array), v.dim)

  def visit_AnyScalar(self, v):    
    assert False, "Can't generate shape expression for unknown scalar"

  def visit_Tuple(self, v):
    # a tuple is effectively a scalar
    return self.codegen.tuple([])
    # return self.codegen.tuple([self.visit(e) for e in v.elts])

  def binop(self, op_name, v):
    if v.x.__class__ is AnyScalar or v.y.__class__ is AnyScalar:
      return any_scalar 
    x = self.visit(v.x)
    y = self.visit(v.y)
    op = getattr(self.codegen, op_name)
   
    return op(x,y)

  def visit_Sub(self, v):
    return self.binop('sub', v)

  def visit_Add(self, v):
    return self.binop('add', v)

  def visit_Mult(self, v):
    return self.binop('mult', v)

  def visit_Div(self, v):
    return self.binop('div', v)

  def visit_Mod(self, v):
    return self.binop('mod', v)

  def visit_Closure(self, v):
    assert False, "Unexpected closure in result shape: %s" % (v,)

def make_shape_expr(codegen, symbolic_shape, input_exprs):
  
  """
  Given a codegen object we're currently using to create a function, and a
  symbolic result shape of a function call (along with the input expressions
  that went into the function) generate a code expression for the shape of the
  result
  """

  if isinstance(symbolic_shape, Scalar):
    return codegen.tuple([])

  shape_codegen = ShapeCodegen(codegen, input_exprs)
  return shape_codegen.visit(symbolic_shape)
