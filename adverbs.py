
import syntax  


class Adverb(syntax.Expr):
  _members = ['fn', 'args', 'axis']
    
  def node_init(self):
    assert self.fn is not None
    assert self.args is not None 
    
    
  def __repr__(self):
    args_str = ", ".join(self.args)    
    return "%s(%s, %s, axis = %s)" % \
      (self.node_type(), self.fn, args_str, self.axis)
        
  def __str__(self):
    return repr(self)

class Tiled(object):
  pass 

class Map(Adverb):
  pass

class TiledMap(Map, Tiled):
  pass 

class AllPairs(Adverb):
  def node_init(self):
    if self.axis is None:
      self.axis = 0

class TiledAllPairs(AllPairs, Tiled):
  pass 
  

class Accumulative(Adverb):
  """
  Adverbs such as Reduce and Scan
  which carry an accumulated value
  and require a 'combine' function 
  to merge the accumulators resulting
  from parallel sub-computations. 
  """
  _members = ['init', 'combine']
  def __repr__(self):
    args_str = ", ".join(self.args)    
    return "%s(%s, %s, axis = %s, init = %s, combine = %s)" % \
      (self.node_type(), self.fn, args_str, self.axis, self.init, self.combine)
      
  def node_init(self):
    assert self.init is not None
    assert self.combine is not None 
    
class Reduce(Adverb):
  pass

class TiledReduce(Reduce, Tiled):
  pass 
  
class Scan(Accumulative):
  pass

class TiledScan(Scan, Tiled):
  pass 

import syntax_helpers
import names 


_adverb_wrapper_cache = {}
def untyped_wrapper(adverb_class, untyped_fn, **adverb_params):
  """
  Given an adverb and its untyped function argument, 
  create a fresh function whose body consists of only 
  the evaluation of the adverb.
  
  The wrapper should accept the same number of arguments 
  as the parameterizing function and correctly forward any
  global dependencies by creating a closure.  
  """
  
  fn_args = untyped_fn.args.fresh_copy()
  
  
   
  nonlocals = untyped_fn.nonlocals
  
  # create a local closure to forward nonlocals into the adverb 
  n_nonlocals = len(nonlocals)
  closure = syntax.Closure(untyped_fn.name, fn_args.positional[:n_nonlocals])
  closure_var = syntax.Var(names.fresh("closure"))
  body = [syntax.Assign(closure_var, closure)] 
  
  # the adverb parameters are given as python values, convert them to
  # constant syntax nodes 
  adverb_param_exprs = {}
  for (k,v) in adverb_params.items():
    adverb_param_exprs[k] = syntax_helpers.const(v)
  adverb_name = adverb_class.node_type()
  adverb = adverb_class(closure_var, fn_args.positional[n_nonlocals:], **adverb_param_exprs)
  body += [syntax.Return(adverb)]
  fn_name = names.fresh(adverb_name + "_wrapper")
  return syntax.Fn(fn_name, fn_args, body, nonlocals)
