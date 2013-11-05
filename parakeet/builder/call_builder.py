
from ..ndtypes import FnT, ClosureT, Type, make_closure_type, NoneType 
from ..syntax import UntypedFn, TypedFn, Var, Call, Closure, ClosureElt, Return, FormalArgs   
from ..syntax.helpers import get_types, zero_i64, none 
from ..syntax.adverb_helpers import max_rank

from core_builder import CoreBuilder
from parakeet.syntax.stmt import ExprStmt

class CallBuilder(CoreBuilder):
  
  """
  Builder for all things related to calling functions
  """
  
  def closure_elt(self, clos, idx, name = None):
    assert isinstance(idx, (int, long))

    if isinstance(clos, Closure):
      result = clos.args[idx]
    else:
      result = ClosureElt(clos, idx, type = clos.type.arg_types[idx])
    if name is None:
      return result
    return self.assign_name(result, name)
  
  def closure_elts(self, clos):
    if clos.__class__ is TypedFn:
      return ()
    return tuple(self.closure_elt(clos, i)
                 for i in xrange(len(clos.type.arg_types)))

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
    closure_elts = old_closure_elts + tuple(extra_args)
    if len(closure_elts) == 0:
      return fn 
    closure_elt_types = [elt.type for elt in closure_elts]
    closure_t = make_closure_type(fn, closure_elt_types)
    result = Closure(fn, closure_elts, type = closure_t)
    if name:
      return self.assign_name(result, name)
    else:
      return result
  

  def return_type(self, fn):
    if isinstance(fn, TypedFn):
      return fn.return_type
    else:
      assert isinstance(fn.type, ClosureT), \
          "Unexpected fn type: %s" % fn.type
      assert isinstance(fn.type.fn, TypedFn), \
          "Unexpected fn: %s" % fn.type.fn
      return fn.type.fn.return_type


  def input_vars(self, fn):
    assert isinstance(fn, TypedFn), "Expected TypedFn, got %s" % fn 
    
    return [Var(arg_name, t) 
            for arg_name, t in 
            zip(fn.arg_names, fn.input_types)]
  
  def input_types(self, closure):
    fn = self.get_fn(closure)
    closure_args = self.closure_elts(closure)
    return fn.input_types[len(closure_args):]
  
  def invoke_type(self, closure, args):
    from .. type_inference import invoke_result_type 
    closure_t = closure.type
    arg_types = get_types(args)
    assert all(isinstance(t, Type) for t in arg_types), \
        "Invalid types: %s" % (arg_types, )
    return invoke_result_type(closure_t, arg_types)

  def is_identity_fn(self, fn):
    if fn.__class__ is TypedFn and len(fn.arg_names) == 1:
      input_name = fn.arg_names[0]
    elif fn.__class__ is UntypedFn:
      args = fn.args 
      if isinstance(args, (list, tuple)):
        input_name = args[0]
      else:
        assert isinstance(args, FormalArgs), "Unexpected args %s" % (args,)
        if args.n_args == 1 and len(args.positional) == 1:
          input_name = args.positional[0]  
        else:
          return False 
    else:
      return False 
    
    if isinstance(input_name, Var):
      input_name = input_name.name
    else:
      assert isinstance(input_name, str), "Unexpected input %s" % (input_name,)    
    if len(fn.body) == 1:
      stmt = fn.body[0]
      if stmt.__class__ is Return:
        expr = stmt.value 
        if expr.__class__ is Var:
          return expr.name == input_name
    return False 
    
  def invoke(self, fn, args, loopify = False, lower = False, name = None):
    

    if isinstance(fn, UntypedFn) or isinstance(fn.type, FnT):
      closure_args = [] 
    else: 
      assert isinstance(fn.type, ClosureT), "Unexpected function %s with type: %s" % (fn, fn.type)
      closure_args = self.closure_elts(fn)
      fn = self.get_fn(fn)
      
    
    args =  tuple(closure_args) + tuple(args)
    
    if isinstance(fn, UntypedFn):
      arg_types = get_types(args)
      from .. import type_inference
      fn = type_inference.specialize(fn, arg_types)
      
    if loopify or lower: 
      from  ..transforms import pipeline
      if loopify: 
        fn = pipeline.loopify(fn)
      if lower: 
        fn = pipeline.lowering(fn)
    
    # don't generate Call nodes for identity function 
    if self.is_identity_fn(fn):
      assert len(args) == 1
      return args[0]
    
  
    call = Call(fn, args, type = fn.return_type)

    if fn.return_type == NoneType:
      self.insert_stmt(ExprStmt(call))
      return none
    else:
      if name is None:
        name = "call_result"
      return self.assign_name(call, name)

  def call(self, fn, args, name = None):
    return self.invoke(fn, args, name = name) 

  def call_shape(self, maybe_clos, args):
    from ..shape_inference import call_shape_expr, shape_codegen
    fn = self.get_fn(maybe_clos)
    closure_args = self.closure_elts(maybe_clos)
    combined_args = tuple(closure_args) + tuple(args)
    if isinstance(fn, UntypedFn):
      # if we're given an untyped function, first specialize it
      from ..type_inference import specialize       
      fn = specialize(fn, get_types(combined_args))
      from ..transforms import pipeline 
      fn = pipeline.high_level_optimizations(fn)
    abstract_shape = call_shape_expr(fn)
    return shape_codegen.make_shape_expr(self, abstract_shape, combined_args)

