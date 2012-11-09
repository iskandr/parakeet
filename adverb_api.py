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

from common import list_to_ctypes_array
from run_function import ctypes_to_generic_value, generic_value_to_python
from llvm.ee import GenericValue
import llvm_types

def par_each(fn, *args, **kwds):
  arg_types = map(type_conv.typeof, args)

  closure_t, untyped = translate_fn(fn)

  # Don't handle outermost axis = None yet
  axis = kwds.get('axis', 0)

  # assert not axis is None, "Can't handle axis = None in outermost adverbs yet"
  map_result_type = type_inference.infer_map_type(closure_t, arg_types, axis)


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
    print output_ptrs
    for ptr in output_ptrs:
      print ptr.contents
    output_contents = [ptr.contents for ptr in output_ptrs]
    print output_contents

    outputs = [map_result_type.to_python(x) for x in output_contents]
    print outputs

    #TODO: Have to handle concatenation axis
    result = np.concatenate(outputs)
  else:
    start = GenericValue.int(llvm_types.int32_t, 0)
    stop = GenericValue.int(llvm_types.int32_t, num_iters)
    # print "Addr1", ctypes.addressof(c_args)
    # print "Addr2", ctypes.addressof(ctypes.cast(c_args, ctypes.POINTER(args_t.ctypes_repr)).contents)
    fn_args_array =  GenericValue.pointer(ctypes.addressof(c_args))
    dummy_tile_sizes_t = ctypes.c_int * 1
    dummy_tile_sizes = dummy_tile_sizes_t()
    arr_tile_sizes = (dummy_tile_sizes_t * rt.dop)()
    # tile_sizes = ctypes.cast(arr_tile_sizes, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))
    tile_sizes = GenericValue.pointer(ctypes.addressof(arr_tile_sizes))
    gv_inputs = [start, stop, fn_args_array, tile_sizes]
    c_fn_ptr = exec_engine.get_pointer_to_function(llvm_fn)
    c_input_types = [ctypes.c_int, ctypes.c_int,
                     ctypes.POINTER(args_t.ctypes_repr),
                     ctypes.POINTER(ctypes.c_int)]
    c_fn_type = ctypes.CFUNCTYPE(None, *tuple(c_input_types))
    c_fn = c_fn_type(c_fn_ptr)
    print hex(c_fn_ptr)
    c_fn(ctypes.c_int(0), ctypes.c_int(num_iters),
         ctypes.byref(c_args),
         arr_tile_sizes[0])
    #exec_engine.run_function(llvm_fn, gv_inputs)
    print c_args
    print c_args.output
    print c_args.output.contents
    result = map_result_type.to_python(c_args.output.contents)
  print result
  return result
