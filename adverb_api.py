import ctypes 
import numpy as np 

import ast_conversion
import syntax
import core_types 
import type_inference
import adverbs
import adverb_helpers
import type_conv 
import llvm_backend 

from run_function import run 
from runtime import runtime


def create_adverb_hook(adverb_class, default_args = ['x'], default_axis = None):
  def create_wrapper(fundef, **kwds):
    if not isinstance(fundef, syntax.Fn):
      fundef = ast_conversion.translate_function_value(fundef)
    assert len(fundef.args.defaults) == 0
    arg_names = ['fn'] + list(fundef.args.positional)
    return adverb_helpers.untyped_wrapper(adverb_class, arg_names, **kwds)
  def python_hook(fn, *args, **kwds):
    wrapper = create_wrapper(fn, **kwds)
    return run(wrapper, *[fn] + list(args))
  # for now we register with the default number of args since our wrappers
  # don't yet support unpacking a variable number of args
  default_wrapper_args = ['fn'] + default_args
  default_wrapper = \
    adverb_helpers.untyped_wrapper(adverb_class,
                                   default_wrapper_args,
                                   axis=default_axis)
  adverb_helpers.register_adverb(python_hook, default_wrapper)
  return python_hook

each = create_adverb_hook(adverbs.Map)
allpairs = create_adverb_hook(adverbs.AllPairs, default_args = ['x','y'],
                              default_axis = 0)
seq_reduce = create_adverb_hook(adverbs.Reduce, default_args = ['x'])
seq_scan = create_adverb_hook(adverbs.Scan, default_args = ['x'])


try:
  rt = runtime.Runtime()
except:
  print "Warning: Failed to load parallel runtime"
  rt = None 
  
import args, array_type, function_registry, names
_par_wrapper_cache = {}

def gen_par_work_function(adverb_class, fn, arg_types):
  key = (adverb_class, fn.name, tuple(arg_types))
  if key in _par_wrapper_cache:
    return _par_wrapper_cache[key]
  else:
    start_var = syntax.Var(names.fresh("start"))
    stop_var = syntax.Var(names.fresh("stop"))
    args_var = syntax.Var(names.fresh("args"))
    tile_sizes_var = syntax.Var(names.fresh("tile_sizes"))
    inputs = [start_var, stop_var, args_var, tile_sizes_var]

    nested_arg_names = ['fn'] + list(fn.args.positional)
    nested_wrapper = adverb_helpers.untyped_wrapper(adverb_class,
                                                    nested_arg_names, 
                                                    axis = 0)
    # TODO: Closure args should go here.
    unpacked_args = [syntax.Closure(fn.name, [])]
    for i, t in enumerate(arg_types):
      attr = syntax.Attribute(args_var, ("arg%d" % i))
      if isinstance(t, array_type.ArrayT):
        s = syntax.Slice(start_var, stop_var, syntax.Const(1))
        unpacked_args.append(syntax.Index(attr, s))
      else:
        unpacked_args.append(attr)
    nested_closure = syntax.Closure(nested_wrapper.name, [])
    call = syntax.Invoke(nested_closure, unpacked_args)
    body = [syntax.Assign(syntax.Attribute(args_var, "output"), call)]
    fn_name = names.fresh(adverb_class.node_type() + fn.name + "_par_wrapper")
    fundef = syntax.Fn(fn_name, args.Args(positional = inputs), body)
    function_registry.untyped_functions[fn_name] = fundef
    _par_wrapper_cache[key] = fundef
    return fundef

import closure_type 

def translate_fn(python_fn):
  """
  Given a python function, return its closure type 
  and the definition of its untyped representation 
  """
  closure_t = type_conv.typeof(python_fn)
  assert isinstance(closure_t, closure_type.ClosureT)
  untyped = function_registry.untyped_functions[closure_t.fn]
  return closure_t, untyped
 
def par_each(fn, *args, **kwds):
  arg_types = map(type_conv.typeof, args)
  
  
  closure_t, untyped = translate_fn(fn)
  
  # Don't handle outermost axis = None yet
  axis = kwds.get('axis')
  # assert not axis is None, "Can't handle axis = None in outermost adverbs yet"
   
   
  map_result_type = type_inference.infer_map_type(closure_t, arg_types, axis)
  # Create args struct type
  fields = []
  for i, arg_type in enumerate(arg_types):
    fields.append((("arg%d" % i), arg_type))
  fields.append(("output", map_result_type))

  class ParEachArgsType(core_types.StructT):
    _fields_ = fields

  args_t = ParEachArgsType()
  c_args = args_t.ctypes_repr()
  for i, arg in enumerate(args):
    obj = type_conv.from_python(arg)
    field_name = "arg%d" % i
    t = type_conv.typeof(arg)
    if isinstance(t, core_types.StructT):
      setattr(c_args, field_name, ctypes.pointer(obj))
    else:
      setattr(c_args, field_name, obj)
    
  c_args_list = [c_args]
  
  for i in range(rt.dop - 1):
    c_args_new = args_t.ctypes_repr()
    ctypes.memmove(ctypes.byref(c_args_new), ctypes.byref(c_args),
                   ctypes.sizeof(args_t.ctypes_repr))
    c_args_list.append(c_args_new)

  wf = gen_par_work_function(adverbs.Map, untyped, arg_types)
  wf_types = [core_types.Int32, core_types.Int32, args_t, core_types.Int32]
  typed = type_inference.specialize(wf, wf_types)
  compiled = llvm_backend.compile_fn(typed)
  wf_ptr = compiled.exec_engine.get_pointer_to_function(compiled.llvm_fn)
  r = adverb_helpers.max_rank(arg_types)
  for (arg, t) in zip(args, arg_types):
    if t.rank == r:
      max_arg = arg
      break
  num_iters = max_arg.shape[axis]

  # Execute on thread pool
  rt.run_untiled_job(wf_ptr, c_args_list, num_iters)

  # Concatenate results
  output_ptrs = [args_obj.output for args_obj in c_args_list]
  outputs = [typed.return_type.to_python(output_ptr) \
             for output_ptr in output_ptrs]
  #TODO: Have to handle concatenation axis
  return np.concatenate(outputs) 