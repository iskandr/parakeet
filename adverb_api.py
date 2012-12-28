import adverbs
import adverb_helpers
import adverb_registry
import adverb_wrapper
import config
import core_types
import ctypes
import llvm_backend
import lowering
import numpy as np
import run_function
import syntax
import syntax_helpers
import tuple_type
import type_conv
import type_inference

from core_types import Int64
from runtime import runtime

try:
  rt = runtime.Runtime()
except:
  print "Warning: Failed to load parallel runtime"
  rt = None

import array_type, names
from args import ActualArgs, FormalArgs

# TODO: Get rid of this extra level of wrapping.
_lowered_wrapper_cache = {}
def gen_tiled_wrapper(adverb_class, fn, arg_types, nonlocals):
  key = (adverb_class, fn.name, tuple(arg_types))
  if key in _lowered_wrapper_cache:
    return _lowered_wrapper_cache[key]
  else:
    # Generate a wrapper for the payload function, and then type specialize it
    # as well as tile it.  Tiling needs to happen here, as we don't want to
    # tile the outer parallel wrapper function.
    nested_wrapper = \
        adverb_wrapper.untyped_wrapper(adverb_class,
                                       map_fn_name = 'fn',
                                       data_names = fn.args.positional,
                                       varargs_name = None,
                                       axis = 0)
    nonlocal_args = ActualArgs([syntax.Var(names.fresh("arg")) for _ in nonlocals])
    untyped_args = [syntax.Var(names.fresh("arg")) for _ in arg_types]
    fn_args_obj = FormalArgs()
    for arg in nonlocal_args:
      fn_args_obj.add_positional(arg.name)
    for arg in untyped_args:
      fn_args_obj.add_positional(arg.name)
    nested_closure = syntax.Closure(nested_wrapper.name, [])
    call = syntax.Call(nested_closure,
                       [syntax.Closure(fn.name, nonlocal_args)] + untyped_args)
    body = [syntax.Return(call)]
    fn_name = names.fresh(adverb_class.node_type() + fn.name + "_wrapper")
    untyped_wrapper = syntax.Fn(fn_name, fn_args_obj, body)
    nonlocal_types = [type_conv.typeof(arg) for arg in nonlocals]
    all_types = arg_types.prepend_positional(nonlocal_types)
    typed = type_inference.specialize(untyped_wrapper, all_types)
    return lowering.lower(typed, tile=config.opt_tile)

_par_wrapper_cache = {}
def gen_par_work_function(adverb_class, f, nonlocals, nonlocal_types,
                          args_t, arg_types, closure_pos):
  key = (adverb_class, f.name, tuple(arg_types))
  if key in _par_wrapper_cache:
    return _par_wrapper_cache[key]
  else:
    fn = gen_tiled_wrapper(adverb_class, f, arg_types, nonlocals)
    num_tiles = fn.num_tiles
    # Construct a typed parallel wrapper function that unpacks the args struct
    # and calls the (possibly tiled) payload function with its slices of the
    # arguments.
    start_var = syntax.Var(names.fresh("start"), type=Int64)
    stop_var = syntax.Var(names.fresh("stop"), type=Int64)
    args_var = syntax.Var(names.fresh("args"), type=args_t)
    tile_type = tuple_type.make_tuple_type([Int64 for _ in range(num_tiles)])
    tile_sizes_var = syntax.Var(names.fresh("tile_sizes"), type=tile_type)
    inputs = [start_var, stop_var, args_var, tile_sizes_var]

    # Manually unpack the args into types Vars and slice into them.
    slice_t = array_type.make_slice_type(Int64, Int64, Int64)
    arg_slice = \
        syntax.Slice(start_var, stop_var, syntax_helpers.one_i64, type=slice_t)
    def slice_arg(arg):
      indices = [arg_slice]
      for _ in xrange(1, arg.type.rank):
        indices.append(syntax_helpers.slice_none)
      tuple_t = tuple_type.make_tuple_type(syntax_helpers.get_types(indices))
      index_tuple = syntax.Tuple(indices, tuple_t)
      result_t = t.index_type(tuple_t)
      return syntax.Index(arg, index_tuple, type=result_t)
    unpacked_args = []
    i = 0
    for t in nonlocal_types:
      unpacked_args.append(syntax.Attribute(args_var, ("arg%d" % i), type=t))
      i += 1
    for t in arg_types:
      attr = syntax.Attribute(args_var, ("arg%d" % i), type=t)
      if isinstance(t, array_type.ArrayT) and i not in closure_pos:
        # TODO: Handle axis.
        unpacked_args.append(slice_arg(attr))
      else:
        unpacked_args.append(attr)
      i += 1

    # If tiling, pass in the tile params array.
    if config.opt_tile:
      unpacked_args.append(tile_sizes_var)

    # Make a typed closure that calls the payload function with the arg slices.
    closure_t = closure_type.make_closure_type(fn, [])
    nested_closure = syntax.Closure(fn.name, [], type=closure_t)
    return_t = fn.return_type
    call = syntax.Call(nested_closure, unpacked_args, type=return_t)
    output = slice_arg(syntax.Attribute(args_var, "output", type=return_t))
    body = [syntax.Assign(output, call),
            syntax.Return(syntax_helpers.none)]
    type_env = {}
    for arg in inputs:
      type_env[arg.name] = arg.type

    # Construct the typed wrapper.
    parallel_wrapper = \
        syntax.TypedFn(name = names.fresh(adverb_class.node_type() + fn.name +
                                          "_par_wrapper"),
                       arg_names = [var.name for var in inputs],
                       input_types = syntax_helpers.get_types(inputs),
                       body = body,
                       return_type = core_types.NoneType,
                       type_env = type_env)
    lowered = lowering.lower(parallel_wrapper)
    _par_wrapper_cache[key] = lowered

    return lowered, num_tiles

import closure_type
from common import list_to_ctypes_array

def prepare_adverb_args(python_fn, args, kwargs):
  """
  Fetch the function's nonlocals and return an ActualArgs object of both the arg
  values and their types
  """

  closure_t = type_conv.typeof(python_fn)
  assert isinstance(closure_t, closure_type.ClosureT)
  if isinstance(closure_t.fn, str):
    untyped = syntax.Fn.registry[closure_t.fn]
  else:
    untyped = closure_t.fn

  nonlocals = list(untyped.python_nonlocals())
  adverb_arg_values = ActualArgs(args, kwargs)

  # get types of all inputs
  adverb_arg_types = adverb_arg_values.transform(type_conv.typeof)
  return untyped, closure_t, nonlocals, adverb_arg_values, adverb_arg_types

def get_par_args_repr(nonlocals, nonlocal_types, args, arg_types, return_t):
  # Create args struct type
  fields = []
  i = 0
  for arg_type in nonlocal_types:
    fields.append((("arg%d" % i), arg_type))
    i += 1
  for arg_type in arg_types:
    fields.append((("arg%d" % i), arg_type))
    i += 1
  fields.append(("output", return_t))

  class ParArgsType(core_types.StructT):
    _fields_ = fields

    def __hash__(self):
      return hash(tuple(fields))
    def __eq__(self, other):
      return isinstance(other, ParArgsType)

  args_t = ParArgsType()
  c_args = args_t.ctypes_repr()
  i = 0
  for arg in nonlocals:
    obj = type_conv.from_python(arg)
    field_name = "arg%d" % i
    t = type_conv.typeof(arg)
    if isinstance(t, core_types.StructT):
      setattr(c_args, field_name, ctypes.pointer(obj))
    else:
      setattr(c_args, field_name, obj)
    i += 1
  for arg in args:
    obj = type_conv.from_python(arg)
    field_name = "arg%d" % i
    t = type_conv.typeof(arg)
    if isinstance(t, core_types.StructT):
      setattr(c_args, field_name, ctypes.pointer(obj))
    else:
      setattr(c_args, field_name, obj)
    i += 1

  return args_t, c_args

def exec_in_parallel(fn, args_repr, c_args, num_iters, num_tiles):
  (llvm_fn, _, exec_engine) = llvm_backend.compile_fn(fn)

  c_args_list = [c_args]
  for _ in range(rt.dop - 1):
    c_args_new = args_repr.ctypes_repr()
    ctypes.memmove(ctypes.byref(c_args_new), ctypes.byref(c_args),
                   ctypes.sizeof(args_repr.ctypes_repr))
    c_args_list.append(c_args_new)

  c_args_array = list_to_ctypes_array(c_args_list, pointers=True)
  wf_ptr = exec_engine.get_pointer_to_function(llvm_fn)

    # Execute on thread pool
  if config.print_parallel_exec_time:
    import time
    start = time.time()
  rt.run_job_with_dummy_tiles(wf_ptr, c_args_array, num_iters, num_tiles)
  if config.print_parallel_exec_time:
    t = time.time() - start
    print "Time to execute:", t

def par_each(fn, *args, **kwds):
  # Don't handle outermost axis = None yet
  axis = kwds.get('axis', 0)

  untyped, closure_t, nonlocals, args, arg_types = \
      prepare_adverb_args(fn, args, kwds)

  return_t = type_inference.infer_Map(closure_t, arg_types)

  r = adverb_helpers.max_rank(arg_types)
  for (arg, t) in zip(args, arg_types):
    if t.rank == r:
      max_arg = arg
      break
  num_iters = max_arg.shape[axis]

  nonlocal_types = [type_conv.typeof(arg) for arg in nonlocals]
  args_repr, c_args = get_par_args_repr(nonlocals, nonlocal_types, args,
                                        arg_types, return_t)

  # TODO: Use shape inference to determine output shape.
  single_iter_rslt = run_function.run(fn, *[arg[0] for arg in args.positional])
  output_shape = (num_iters,) + single_iter_rslt.shape
  output = np.zeros(output_shape, dtype=array_type.elt_type(return_t).dtype)
  output_obj = type_conv.from_python(output)
  gv_output = ctypes.pointer(output_obj)
  setattr(c_args, "output", gv_output)

  wf, num_tiles = gen_par_work_function(adverbs.Map, untyped,
                                        nonlocals, nonlocal_types,
                                        args_repr, arg_types, [])
  exec_in_parallel(wf, args_repr, c_args, num_iters, num_tiles)

  return output

def par_allpairs(fn, x, y, **kwds):
  # Don't handle outermost axis = None yet
  axis = kwds.get('axis', 0)

  untyped, closure_t, nonlocals, args, arg_types = \
      prepare_adverb_args(fn, [x, y], kwds)

  xtype, ytype = arg_types
  return_t = type_inference.infer_AllPairs(closure_t, xtype, ytype)

  # For now, only split up the larger of the 2 args amongst the threads,
  # passing the other through in toto.
#  if len(args.positional[0]) > len(args.positional[1]):
#    num_iters = len(args.positional[0])
#    closure_pos = [1]
#  else:
#    num_iters = len(args.positional[1])
#    closure_pos = [0]

  # Actually, for now, just split the first one.  Otherwise we'd have to carve
  # the output along axis = 1 and I don't feel like figuring that out.
  num_iters = len(args.positional[0])
  closure_pos = [1]

  nonlocal_types = [type_conv.typeof(arg) for arg in nonlocals]
  args_repr, c_args = get_par_args_repr(nonlocals, nonlocal_types, args,
                                        arg_types, return_t)

  # TODO: Use shape inference to determine output shape.
  single_iter_rslt = run_function.run(fn, x[0], y[0])
  output_shape = (len(x), len(y)) + single_iter_rslt.shape
  output = np.zeros(output_shape, dtype=array_type.elt_type(return_t).dtype)
  output_obj = type_conv.from_python(output)
  gv_output = ctypes.pointer(output_obj)
  setattr(c_args, "output", gv_output)

  wf, num_tiles = gen_par_work_function(adverbs.AllPairs, untyped,
                                        nonlocals, nonlocal_types,
                                        args_repr, arg_types, closure_pos)
  exec_in_parallel(wf, args_repr, c_args, num_iters, num_tiles)

  return output

from adverb_wrapper import untyped_identity_function as ident
from macro import staged_macro
from run_function import run

def one_is_none(f, g):
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
    An awkward mismatch between treating adverbs as functions is that their axis
    parameter is really fixed as part of the syntax of Parakeet. Thus, when
    you're calling an adverb from outside Parakeet you can generate new syntax
    for any axis you want, but if you use an adverb as a function value within
    Parakeet:
      r = par.reduce
      return r(f, xs)
    ...then we hackishly force the adverb to go along the default axis of 0.
    """
    return adverb_wrapper.untyped_wrapper(adverb_class,
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

def get_axis(kwargs):
  axis = kwargs.get('axis', 0)
  return syntax_helpers.unwrap_constant(axis)

# TODO: Called from the outside maybe macros should generate wrapper functions

call_from_python = None
if config.call_from_python_in_parallel:
  call_from_python = par_each

@staged_macro("axis", call_from_python=call_from_python)
def each(f, *xs, **kwargs):
  return adverbs.Map(f, args = xs, axis = get_axis(kwargs))

if config.call_from_python_in_parallel:
  call_from_python = par_allpairs

@staged_macro("axis", call_from_python=call_from_python)
def allpairs(f, x, y, **kwargs):
  return adverbs.AllPairs(fn = f, args = [x,y], axis = get_axis(kwargs))

@staged_macro("axis")
def reduce(f, x, **kwargs):
  axis = get_axis(kwargs)
  init = kwargs.get('init')

  return adverbs.Reduce(fn = ident, combine = f, args = [x], init = init,
                        axis = axis)

@staged_macro("axis")
def scan(f, x, **kwargs):
  axis = get_axis(kwargs)
  init = kwargs.get('init')
  if init is None:
    init = syntax_helpers.none
  return adverbs.Scan(fn = ident, combine = f, emit = ident, args = [x],
                      init = init, axis = axis)
