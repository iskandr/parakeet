import collections 
import distutils 
import imp 

import os

from tempfile import NamedTemporaryFile

from .. import config as root_config 
import config 
from system_info import (python_lib_dir,  
                         windows,  
                         get_source_extension, object_extension, shared_extension,  
                         get_compiler)
from flags import get_compiler_flags, get_linker_flags
from shell_command import CommandFailed, run_cmd 

CompiledPyFn = collections.namedtuple("CompiledPyFn",
                                      ("c_fn", 
                                       "src_filename", 
                                       "module", 
                                       "shared_filename", 
                                       "src", 
                                       "fn_name",
                                       "fn_signature"))

CompiledObject = collections.namedtuple("CompiledObject", 
                                        (
                                         "src_filename",  
                                         "object_filename", 
                                         "fn_name",
                                         ))

  
c_headers = ["stdint.h",  "math.h",  "signal.h"]
core_python_headers = ["Python.h"]
numpy_headers = ['numpy/arrayobject.h', 'numpy/arrayscalars.h']

python_headers = core_python_headers + numpy_headers 

# went to some annoying effort to clean up all the array->flags, &c that have been 
# replaced with PyArray_FLAGS in NumPy 1.7 
global_preprocessor_defs = ["#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"]


def create_module_source(raw_src, fn_name, 
                            extra_headers = [], 
                            declarations = [], 
                            extra_function_sources = [], 
                            print_source = None):
    # when compiling with NVCC, other headers get implicitly included 
  # and cause warnings since Python redefines this constant
  src_lines = list(global_preprocessor_defs) 
  if config.undef_posix_c_source:
    src_lines.append("#undef _XOPEN_SOURCE")
    src_lines.append("#undef _POSIX_C_SOURCE")
  
  for header in extra_headers + c_headers:
    src_lines.append("#include <%s>" % header)
  
  for decl in declarations:
    decl = decl.strip()
    if not decl.endswith(";"):
      decl += ";"
    src_lines.append(decl)
    
  src_lines.extend(extra_function_sources)
  
  src_lines.append(raw_src)
  module_init = """
    \n\n
    static PyMethodDef %(fn_name)sMethods[] = {
      {"%(fn_name)s",  %(fn_name)s, METH_VARARGS,
       "%(fn_name)s"},

      {NULL, NULL, 0, NULL}        /* Sentinel */
    };
  
    PyMODINIT_FUNC
    init%(fn_name)s(void)
    {
      //Py_Initialize();
      Py_InitModule("%(fn_name)s", %(fn_name)sMethods);
      import_array();
    }
  """ % locals()
  src_lines.append(module_init)
  full_src =  "\n".join(src_lines)
  
  if print_source is None: print_source = root_config.print_generated_code  
  if print_source:
    for i, line in enumerate(full_src.splitlines()):
        if config.print_line_numbers: print i+1, " ", line
        else: print line
  return full_src    

def create_source_file(src, 
                         fn_name = None, 
                         src_filename = None, 
                         src_extension = None):
  if fn_name is None: prefix = "parakeet_"
  else: prefix = "parakeet_%s_" % fn_name
  if src_extension is None: src_extension = get_source_extension()
  if src_filename is None:
    src_file = NamedTemporaryFile(suffix = src_extension,  
                                  prefix =  prefix, 
                                  delete = False,
                                  mode = 'w', 
                                  )
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
  
  src_file.write(src)  
  src_file.close()
  return src_file 

def compile_object(src_filename, 
                     fn_name = None,  
                     src_extension = None, 
                     declarations = [],
                     extra_objects = [], 
                     extra_compile_flags = [], 
                     print_commands = None, 
                     compiler = None, 
                     compiler_flag_prefix = None):
  
  
  if print_commands is None:  print_commands = config.print_commands
  if src_extension is None: src_extension = get_source_extension()
  if compiler is None: compiler = get_compiler()
    
  object_name = src_filename.replace(src_extension, object_extension)
  compiler_flags = get_compiler_flags(extra_compile_flags, compiler_flag_prefix)
  
  if isinstance(compiler, (list,tuple)):
    compiler_cmd = list(compiler)
  else:
    compiler_cmd = [compiler]
    
  compiler_cmd += compiler_flags 
  compiler_cmd += ['-c', src_filename, '-o', object_name]
  run_cmd(compiler_cmd, label = "Compile source")
  
  return CompiledObject(src_filename = src_filename, 
                        object_filename = object_name, 
                        fn_name = fn_name)

def link_module(compiler, object_name, shared_name, 
                 extra_objects = [], 
                 extra_link_flags = [], 
                 linker_flag_prefix = None):
  linker_flags = get_linker_flags(extra_link_flags, linker_flag_prefix) 
  
  if isinstance(compiler, (list,tuple)):
    linker_cmd = list(compiler)
  else:
    linker_cmd = [compiler]
  linker_cmd += [object_name] 
  linker_cmd += linker_flags 
  linker_cmd += list(extra_objects) 
  linker_cmd += ['-o', shared_name]

  env = os.environ.copy()
  if not windows:
    env["LD_LIBRARY_PATH"] = python_lib_dir
  run_cmd(linker_cmd, env = env, label = "Linking")

def compile_with_distutils(extension_name, 
                              src_filename,
                              extra_objects = [], 
                              extra_compiler_flags = [],
                              extra_link_flags = [],   
                              print_commands = False):

    # copied largely from pyxbuild 
    from distutils.dist import Distribution
    from distutils.extension import Extension
    
    compiler_flags = get_compiler_flags(extra_compiler_flags)
    linker_flags = get_linker_flags(extra_link_flags)
    
    ext = Extension(name=extension_name, 
                    sources=[src_filename],
                    extra_objects=extra_objects,
                    extra_compile_args=compiler_flags,
                    extra_link_args=linker_flags)
  
    script_args = ['build_ext', '--quiet']
    setup_args = {"script_name": None,
                  "script_args": script_args, 
                  }
    dist = Distribution(setup_args)
    if not dist.ext_modules: dist.ext_modules = []
    dist.ext_modules.append(ext)
    # I have no idea how distutils works or why I have to do any of this 
    config_files = dist.find_config_files()
    try: config_files.remove('setup.cfg')
    except ValueError: pass
    dist.parse_config_files(config_files)
    dist.parse_command_line()
    obj_build_ext = dist.get_command_obj("build_ext")
    
    dist.run_commands()
    shared_name = obj_build_ext.get_outputs()[0]
    return shared_name
  
def compiler_is_gnu(compiler):
  return (compiler.endswith("gcc") or
          compiler.endswith("gcc.exe") or 
          compiler.endswith("g++") or 
          compiler.endswith("g++.exe"))
  
def compile_module_from_source(
      partial_src, 
      fn_name,
      fn_signature = None,  
      src_filename = None,
      src_extension = None, 
      declarations = [],
      extra_function_sources = [], 
      extra_headers = [],  
      extra_objects = [],
      extra_compile_flags = [], 
      extra_link_flags = [], 
      print_source = None, 
      print_commands = None, 
      compiler = None, 
      compiler_flag_prefix = None, 
      linker_flag_prefix = None):
  
  if print_source is None: print_source = root_config.print_generated_code 
  if print_commands is None: print_commands = config.print_commands
  if src_extension is None: src_extension = get_source_extension()
  
  full_src = create_module_source(partial_src, fn_name, 
                                 extra_headers = python_headers + extra_headers, 
                                 declarations = declarations,  
                                 extra_function_sources = extra_function_sources, 
                                 print_source = print_source)
  
  src_file = create_source_file(full_src, 
                                fn_name = fn_name,
                                src_filename = src_filename, 
                                src_extension = src_extension)
  src_filename = src_file.name
  
  if compiler is None: compiler = get_compiler()
  
  try:
    compiled_object = compile_object(src_filename, 
                                     fn_name = fn_name,
                                     src_extension = src_extension,  
                                     extra_objects = extra_objects,
                                     extra_compile_flags = extra_compile_flags, 
                                     print_commands = print_commands, 
                                     compiler = compiler, 
                                     compiler_flag_prefix = compiler_flag_prefix)
  
    object_name = compiled_object.object_filename
    shared_name = src_filename.replace(src_extension, shared_extension)
    link_module(compiler, object_name, shared_name, 
                extra_objects = extra_objects, 
                extra_link_flags = extra_link_flags, 
                linker_flag_prefix = linker_flag_prefix)
    
    if config.delete_temp_files:
      os.remove(object_name)
  except CommandFailed:
    # if normal compilation fails, try distutils instead
    if not compiler_is_gnu(compiler):
      raise 
    if compiler_flag_prefix or linker_flag_prefix:
      raise 
     
    import hashlib
    digest = hashlib.sha224(full_src).hexdigest()
    shared_name = compile_with_distutils(fn_name + "_" + digest, 
                                         src_filename,
                                         extra_objects, 
                                         extra_compile_flags,
                                         extra_link_flags,
                                         print_commands)
    

  if print_commands:
    print "Loading newly compiled extension module %s..." % shared_name
  module =  imp.load_dynamic(fn_name, shared_name)
  
  #on a UNIX-style filesystem it should be OK to delete a file while it's open
  #since the inode will just float untethered from any name
  #If we ever support windows we should find some other way to delete the .dll 
  
  
  c_fn = getattr(module,fn_name)
  
  if config.delete_temp_files:
    os.remove(src_filename)
    # window's can't just untether inodes like a UNIX
    # ...have to eventually think of a plan to clean these things up
    if not windows: os.remove(shared_name)
    
  compiled_fn = CompiledPyFn(c_fn = c_fn, 
                             module = module, 
                             shared_filename =  shared_name,
                             src = full_src, 
                             src_filename = src_filename,
                             fn_name = fn_name, 
                             fn_signature = fn_signature)
  return compiled_fn

