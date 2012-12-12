import adverbs
import adverb_helpers
import adverb_registry
import adverb_wrapper
import core_types
import ctypes
import llvm_backend
import numpy as np
import syntax
import type_conv
import type_inference

from run_function import run
from runtime import runtime

def one_is_none(f,g):
  return int(f is None) + int(g is None) == 1

def create_adverb_hook(adverb_class,
                         map_fn_name = None,
                         combine_fn_name = None,
                         arg_names = None):
  assert one_is_none(map_fn_name, combine_fn_name), \
    "Invalid fn names: %s and %s" % (map_fn_name, combine_fn_name)
  if arg_names is None:
    data_names = []
    varargs_name = 'xs'
  else:
    data_names = arg_names
    varargs_name = None

  def mk_wrapper(axis):
    """
    An awkward mismatch between treating adverbs as
    functions is that their axis parameter is really
    fixed as part of the syntax of Parakeet.
    Thus, when you're calling an adverb from
    outside Parakeet you can generate new syntax for
    any axis you want, but if you use an adverb
    as a function value within Parakeet:
      r = par.reduce
      return r(f, xs)
    ...then we hackishly force the adverb to go
    along the default axis of 0.
    """
    return adverb_wrapper.untyped_wrapper(
      adverb_class,
      map_fn_name = map_fn_name,
      combine_fn_name = combine_fn_name,
      data_names = data_names,
      varargs_name = varargs_name,
      axis=axis)

  def python_hook(fn, *args, **kwds):
    axis = kwds.get('axis', 0)
    wrapper = mk_wrapper(axis)
    return run(wrapper, *([fn] + list(args)))
  # for now we register with the default number of args since our wrappers
  # don't yet support unpacking a variable number of args
  default_wrapper = mk_wrapper(axis = 0)

  adverb_registry.register(python_hook, default_wrapper)
  return python_hook

each = create_adverb_hook(adverbs.Map, map_fn_name = 'f')
allpairs = create_adverb_hook(adverbs.AllPairs,
                              map_fn_name = 'f',
                              arg_names = ['x', 'y'])
reduce = create_adverb_hook(adverbs.Reduce,
                            combine_fn_name = 'f',
                            arg_names = ['x'])
scan = create_adverb_hook(adverbs.Scan,
                          combine_fn_name = 'f',
                          arg_names = ['x'])

try:
  rt = runtime.Runtime()
except:
  print "Warning: Failed to load parallel runtime"
  rt = None

import array_type,  names
from args import FormalArgs
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
    fn_args_obj = FormalArgs()
    for var in inputs:
      name = var.name 
      fn_args_obj.add_positional(name)

    nested_wrapper = adverb_wrapper.untyped_wrapper(adverb_class,
                                                    map_fn_name = 'fn',
                                                    data_names = fn.args.positional,
                                                    varargs_name = None, 
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
    call = syntax.Call(nested_closure, unpacked_args)
    body = [syntax.Assign(syntax.Attribute(args_var, "output"), call)]
    fn_name = names.fresh(adverb_class.node_type() + fn.name + "_par_wrapper")
    
    fundef = syntax.Fn(fn_name, fn_args_obj, body)

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
  if isinstance(closure_t.fn, str):
    untyped = syntax.Fn.registry[closure_t.fn]
  else:
    untyped = closure_t.fn 
  
  return closure_t, untyped

import llvm_types
from common import list_to_ctypes_array
from llvm.ee import GenericValue
# from run_function import ctypes_to_generic_value, generic_value_to_python

def par_each(fn, *args, **kwds):
  arg_types = map(type_conv.typeof, args)

  closure_t, untyped = translate_fn(fn)

  # Don't handle outermost axis = None yet
  axis = kwds.get('axis', 0)

  # assert not axis is None, "Can't handle axis = None in outermost adverbs yet"
  map_result_type = type_inference.infer_Map(closure_t, arg_types)

  r = adverb_helpers.max_rank(arg_types)
  for (arg, t) in zip(args, arg_types):
    if t.rank == r:
      max_arg = arg
      break
  num_iters = max_arg.shape[axis]

  # Create args struct type
  fields = []
  for i, arg_type in enumerate(arg_types):
    fields.append((("arg%d" % i), arg_type))
  fields.append(("output", map_result_type))

  class ParEachArgsType(core_types.StructT):
    _fields_ = fields
    
    def __hash__(self):
      return hash(tuple(fields))
    def __eq__(self, other):
      return isinstance(other, ParEachArgsType)
    
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

  wf = gen_par_work_function(adverbs.Map, untyped, arg_types)
  wf_types = [core_types.Int32, core_types.Int32, args_t,
              core_types.ptr_type(core_types.Int32)]
  typed = type_inference.specialize(wf, wf_types)
  (llvm_fn, _, exec_engine) = llvm_backend.compile_fn(typed)
  parallel = True
  if parallel:
    c_args_list = [c_args]

    for i in range(rt.dop - 1):
      c_args_new = args_t.ctypes_repr()
      ctypes.memmove(ctypes.byref(c_args_new), ctypes.byref(c_args),
                     ctypes.sizeof(args_t.ctypes_repr))
      c_args_list.append(c_args_new)

    c_args_array = list_to_ctypes_array(c_args_list, pointers = True)
    wf_ptr = exec_engine.get_pointer_to_function(llvm_fn)
    # Execute on thread pool
    rt.run_untiled_job(wf_ptr, c_args_array, num_iters)
    output_ptrs = [args_obj.contents.output for args_obj in c_args_array]

    output_contents = [ptr.contents for ptr in output_ptrs]

    outputs = [map_result_type.to_python(x) for x in output_contents]

    #TODO: Have to handle concatenation axis
    result = np.concatenate(outputs)
  else:
    start = GenericValue.int(llvm_types.int32_t, 0)
    stop = GenericValue.int(llvm_types.int32_t, num_iters)
    fn_args_array =  GenericValue.pointer(ctypes.addressof(c_args))
    dummy_tile_sizes_t = ctypes.c_int * 1
    dummy_tile_sizes = dummy_tile_sizes_t()
    arr_tile_sizes = (dummy_tile_sizes_t * rt.dop)()
    tile_sizes = GenericValue.pointer(ctypes.addressof(arr_tile_sizes))
    gv_inputs = [start, stop, fn_args_array, tile_sizes]
    exec_engine.run_function(llvm_fn, gv_inputs)
    result = map_result_type.to_python(c_args.output.contents)

  return result
