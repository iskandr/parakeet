
from ..ndtypes import ClosureT, Type, make_closure_type, NoneType 
from ..syntax import UntypedFn, TypedFn, Var, Call, Closure, ClosureElt, Return, FormalArgs   
from ..syntax.helpers import get_types, zero_i64, none 
from ..syntax.adverb_helpers import max_rank

from core_builder import CoreBuilder
from parakeet.syntax.stmt import ExprStmt

class CallBuilder(CoreBuilder):
  
  """
  Builder for all things related to calling functions
  """
  
  def closure_elt(self, clos, idx):
    assert isinstance(idx, (int, long))

    if isinstance(clos, Closure):
      return clos.args[idx]
    else:
      return ClosureElt(clos, idx, type = clos.type.arg_types[idx])

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
    
  def invoke(self, fn, args, loopify = False, lower = False):
    #import type_inference
    if fn.__class__ is TypedFn:
      closure_args = []
    else:
      assert isinstance(fn.type, ClosureT), \
          "Unexpected function %s with type: %s" % (fn, fn.type)
      #if isinstance(fn.type.fn, UntypedFn):
      #  arg_types = get_types(args)
      #  from .. import type_inference
      #  fn = type_inference.specialize(fn.type, arg_types)
      closure_args = self.closure_elts(fn)
      fn = self.get_fn(fn)
    if loopify or lower : 
      from  ..transforms import pipeline
      if loopify: 
        fn = pipeline.loopify(fn)
      if lower: 
        fn = pipeline.lowering(fn)
        
    combined_args = tuple(closure_args) + tuple(args)
    if isinstance(fn, UntypedFn):
      combined_arg_types = get_types(combined_args)
      from .. import type_inference
      fn = type_inference.specialize(fn, combined_arg_types)
    
    # don't generate Call nodes for identity function 
    if self.is_identity_fn(fn):
      assert len(combined_args) == 1
      return combined_args[0]
    
    call = Call(fn, combined_args, type = fn.return_type)
    if fn.return_type == NoneType:
      self.insert_stmt(ExprStmt(call))
      return none
    else:
      return self.assign_name(call, "call_result")

  def call(self, fn, args):
    return self.invoke(fn, args) 

  def call_shape(self, maybe_clos, args):
    from ..shape_inference import call_shape_expr, shape_codegen
    fn = self.get_fn(maybe_clos)
    closure_args = self.closure_elts(maybe_clos)
    combined_args = tuple(closure_args) + tuple(args)
     
    if isinstance(fn, UntypedFn):
      # if we're given an untyped function, first specialize it
      from ..type_inference import specialize

       
      fn = specialize(fn, get_types(combined_args))
    abstract_shape = call_shape_expr(fn)
    return shape_codegen.make_shape_expr(self, abstract_shape, combined_args)

