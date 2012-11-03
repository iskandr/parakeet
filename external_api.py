import ast_conversion 
import llvm_backend
import runtime.runtime
import syntax 
import type_conv
import type_inference 

def specialize_and_compile(fn, args):
  if isinstance(fn, syntax.Fn):
    untyped = fn 
  else:
    # translate from the Python AST to Parakeet's untyped format 
    untyped  = ast_conversion.translate_function_value(fn)
  all_args = untyped.python_nonlocals() + list(args)
  
  # get types of all inputs
  input_types = [type_conv.typeof(arg) for arg in all_args]
  
  # propagate types through function representation and all
  # other functions it calls 
  typed = type_inference.specialize(untyped, input_types)
  
  # compile to native code 
  compiled = llvm_backend.compile_fn(typed)

  return untyped, typed, all_args, compiled 
  
def run(fn, args):
  _, _, all_args, compiled = specialize_and_compile(fn, args)
  return compiled(*all_args)

import adverb_helpers
import adverbs

def create_adverb_hook(adverb_class, default_args = ['x'], default_axis = None):
  def create_wrapper(fundef, **kwds):
    if not isinstance(fundef, syntax.Fn):
      fundef = ast_conversion.translate_function_value(fundef)
    assert len(fundef.args.defaults) == 0
    arg_names = ['fn'] + list(fundef.args.positional)
    return adverb_helpers.untyped_wrapper(adverb_class, arg_names, **kwds)
  def python_hook(fn, *args, **kwds):
    wrapper = create_wrapper(fn, **kwds)
    return run(wrapper, [fn] + list(args))
  # for now we register with the default number of args since our wrappers
  # don't yet support unpacking a variable number of args
  default_wrapper_args = ['fn'] + default_args
  
  default_wrapper = adverb_helpers.untyped_wrapper(adverb_class,
                                                   default_wrapper_args,
                                                   axis=default_axis)
  adverb_helpers.register_adverb(python_hook, default_wrapper)
  return python_hook

seq_each = create_adverb_hook(adverbs.Map)
allpairs = create_adverb_hook(adverbs.AllPairs, default_args = ['x','y'],
                              default_axis = 0)
seq_reduce = create_adverb_hook(adverbs.Reduce, default_args = ['acc', 'x'])
seq_scan = create_adverb_hook(adverbs.Scan, default_args = ['acc', 'x'])

runtime = runtime.Runtime()

def each(fn, *args, **kwds):
  # Create the closure that implements the mapped function
  if not isinstance(fn, syntax.Fn):
    fundef = ast_conversion.translate_function_value(fn)

  # Don't handle outermost axis = None yet  
  assert not axis is None, "Can't handle axis = None in outermost adverbs yet"

  # TODO: Should make sure that all the shapes conform here, 
  # but we don't yet have anything like assertions or error handling
  max_arg = adverb_helpers.max_rank_arg(args)
  niters = self.shape(max_arg, axis)
  
  # UGH generating code into SSA form is annoying 
  counter_before = self.zero_i64("i_before")
  counter = self.fresh_i64("i_loop")
  counter_after = self.fresh_i64("i_after")
  
  merge = { counter.name : (counter_before, counter_after) }
  
  cond = self.lt(counter, niters)
  elt_t = expr.type.elt_type
  array_result = self.alloc_array(elt_t, niters)
  self.blocks.push()
  nested_args = [self.index(arg, counter) for arg in args]
  closure_t = fn.type
  nested_arg_types = syntax_helpers.get_types(nested_args)
  call_result_t = type_inference.invoke_result_type(closure_t,
                                                    nested_arg_types)
  call = syntax.Invoke(fn, nested_args, type = call_result_t)
  call_result = self.assign_temp(call, "call_result")
  output_idx = syntax.Index(array_result, counter, type = call_result.type)
  self.assign(output_idx, call_result)
  self.assign(counter_after, self.add(counter, syntax_helpers.one_i64))

  body = self.blocks.pop()
  self.blocks += syntax.While(cond, body, merge)
  return array_result
  
  # Create the wrapper function that unpacks the args and executes the closure
  wrapper_arg_names = ['fn', 'start', 'stop', 'args', 'tile_sizes']
  
