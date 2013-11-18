import collections 
import distutils 
import imp 
import numpy.distutils as npdist 
import numpy.distutils.system_info as np_sysinfo 
import os
import platform
import subprocess  
import time 

from tempfile import NamedTemporaryFile
from .. import config as root_config 
import config 

CompiledPyFn = collections.namedtuple("CompiledPyFn",
                                      ("c_fn", 
                                       "src_filename", 
                                       "module", 
                                       "shared_filename", 
                                       "object_filename",
                                       "src", 
                                       "fn_name",
                                       "fn_signature"))

CompiledObject = collections.namedtuple("CompiledObject", 
                                        ("src",
                                         "src_filename",  
                                         "object_filename", 
                                         "fn_name",
                                         "fn_signature"))

  
c_headers = ["stdint.h",  "math.h",  "signal.h"]
core_python_headers = ["Python.h"]
numpy_headers = ['numpy/arrayobject.h', 'numpy/arrayscalars.h']

python_headers = core_python_headers + numpy_headers 

cpp_defs = [] #"#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"]

config_vars = distutils.sysconfig.get_config_vars()


def get_compiler(_cache = {}):
  if config.compiler_path:
    return config.compiler_path
  if config.pure_c in _cache:
    return _cache[config.pure_c]
  for compiler in [('gcc' if config.pure_c else 'g++'), 
                   'icc', 
                   'clang']:
    path = distutils.spawn.find_executable(compiler)
    if path:
      _cache[config.pure_c] = path
      return path 
  assert False, "No compiler found!"
    

def get_source_extension():
  return ".c" if config.pure_c else ".cpp"

object_extension = ".o"
shared_extension = np_sysinfo.get_shared_lib_extension(True)


mac_os = platform.system() == 'Darwin'
windows = platform.system() == 'Windows'

python_include_dirs = [distutils.sysconfig.get_python_inc()]

numpy_include_dirs = npdist.misc_util.get_numpy_include_dirs()
include_dirs = python_include_dirs + numpy_include_dirs 

def get_opt_flags():
  opt_flags = [config.opt_level] 
  if config.sse2:
    opt_flags.append('-msse2')
  if config.fast_math:
    opt_flags.append('-ffast-math')
  return opt_flags 

def get_compiler_flags(compiler, extra_flags = [], compiler_flag_prefix = None):
  compiler_flags = ['-I%s' % path for path in include_dirs]
  
  def add_flag(flag):
    if compiler_flag_prefix is not None:
      compiler_flags.append(compiler_flag_prefix)
    compiler_flags.append(flag)

  add_flag('-fPIC')
  
  if config.debug:
    # nvcc understands debug mode flags
    compiler_flags.extend(['-g', '-O0'])
  else:
    for flag in get_opt_flags():
      add_flag(flag)

  if not config.pure_c: 
    add_flag('-fpermissive')

  for flag in extra_flags:
    add_flag(flag)
    
  return compiler_flags   

python_lib_dir = distutils.sysconfig.get_python_lib() + "/../../"
python_version = distutils.sysconfig.get_python_version()

def get_linker_flags(compiler, extra_flags = [], linker_flag_prefix = None):
  # for whatever reason nvcc is OK with the -shared linker flag
  # but not with the -fPIC compiler flag 
  linker_flags = ['-shared']
  
  def add_flag(flag):
    if linker_flag_prefix is not None:
      linker_flags.append(linker_flag_prefix)
    linker_flags.append(flag)
    
  add_flag('-lm')
  if windows:
    # crazy stupid hack for exposing Python symbols
    # even though we're compiling a shared library, why does Windows care?
    # why does windows even exist? 
    inc_dir = distutils.sysconfig.get_python_inc()
    base = os.path.split(inc_dir)[0]
    lib_dir = base + "\libs"
    add_flag('-L' + lib_dir)
    libname = 'python' + distutils.sysconfig.get_python_version().replace('.', '')
    add_flag('-l' + libname)
    
  if mac_os:
    add_flag("-Wl,-undefined")
    add_flag("-Wl,dynamic_lookup")
    
  for flag in extra_flags:
    add_flag(flag)
  return linker_flags 




def create_source_file(src, 
                         fn_name = None, 
                         src_filename = None, 
                         declarations = [],
                         extra_function_sources = [], 
                         extra_headers = [], 
                         print_source = None, 
                         src_extension = None):
  
  if print_source is None:
    print_source = root_config.print_generated_code

  if fn_name is None:
    prefix = "parakeet_"
  else:
    prefix = "parakeet_%s_" % fn_name
  if src_extension is None:
    src_extension = get_source_extension()
      
  if src_filename is None:
    src_file = NamedTemporaryFile(suffix = src_extension,  
                                  prefix =  prefix, 
                                  delete = False,
                                  mode = 'w', 
                                  )
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
  
  for d in cpp_defs:
    src_file.write(d)
    src_file.write("\n")
  
  for header in extra_headers + c_headers:
    src_file.write("#include <%s>\n" % header)
  
  for decl in declarations:
    decl = decl.strip()
    if not decl.endswith(";"):
      decl += ";"
    decl += "\n"
    src_file.write(decl)
  
  for other_fn_src in extra_function_sources:
    src_file.write(other_fn_src)
    src_file.write("\n")
  
  src_file.write(src)
  src_file.close()
      
  if print_source:
    with open(src_filename, 'r') as src_file_readonly:
      for i, line in enumerate(src_file_readonly.read().splitlines()):
        if config.print_line_numbers:
          print i+1, " ", line
        else:
          print line   
    #print subprocess.check_output(['cat', '-n',  src_filename])
  
  
  
  return src_file 

def run_cmd(cmd, env = None, label = ""):
  if config.print_commands: 
    print " ".join(cmd)
  if config.print_command_elapsed_time: 
    t = time.time()
  
  # first compile silently
  # if you encounter an error, then recompile with output printing 
  with open(os.devnull, "w") as fnull:
    try:  
      
      subprocess.check_call(cmd, stdout = fnull, stderr = fnull, env = env)
    except:
      print "Parakeet encountered error(s) during compilation: "
      subprocess.check_output(cmd, env = env)
      
    
  if config.print_command_elapsed_time: 
    if label:
      print "%s, elapsed time: %0.4f" % (label, time.time() - t)
    else:
      print "Elapsed time:", time.time() - t 

def compile_object(src, 
                   fn_name = None,  
                   fn_signature = None,
                   src_filename = None, 
                   src_extension = None, 
                   declarations = [],
                   extra_function_sources = [], 
                   extra_headers = python_headers, 
                   extra_objects = [], 
                   extra_compile_flags = [], 
                   print_source = None, 
                   print_commands = None, 
                   compiler = None, 
                   compiler_flag_prefix  = None):
  
  if print_source is None: 
    print_source = root_config.print_generated_code
  if print_commands is None: 
    print_commands = config.print_commands
  if src_extension is None:
    src_extension = get_source_extension()
  if compiler is None:
    compiler = get_compiler()
  
  src_file = create_source_file(src, 
                                fn_name = fn_name,
                                src_filename = src_filename, 
                                declarations = declarations,
                                extra_function_sources = extra_function_sources,  
                                extra_headers = extra_headers,
                                print_source = print_source, 
                                src_extension = src_extension)
  src_filename = src_file.name
  object_name = src_filename.replace(src_extension, object_extension)
  compiler_flags = get_compiler_flags(compiler, extra_compile_flags, compiler_flag_prefix)
  if isinstance(compiler, (list,tuple)):
    compiler_cmd = list(compiler)
  else:
    compiler_cmd = [compiler] 
  compiler_cmd += compiler_flags 
  compiler_cmd += ['-c', src_filename, '-o', object_name]
  
  run_cmd(compiler_cmd, label = "Compile source")
  
  return CompiledObject(src = src, 
                        src_filename = src_filename, 
                        object_filename = object_name, 
                        fn_name = fn_name, 
                        fn_signature = fn_signature)
  
  
def compile_module(src, 
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
  
  if print_source is None:
    print_source = root_config.print_generated_code 
  if print_commands is None:
    print_commands = config.print_commands

  src += """
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
  

  if src_extension is None: src_extension = get_source_extension()
  if compiler is None: compiler = get_compiler()
  compiled_object = compile_object(src, 
                                   fn_name,
                                   src_filename  = src_filename,
                                   src_extension = src_extension,  
                                   declarations = declarations,
                                   extra_function_sources = extra_function_sources, 
                                   extra_headers = python_headers + extra_headers,  
                                   extra_objects = extra_objects,
                                   extra_compile_flags = extra_compile_flags, 
                                   print_source = print_source, 
                                   print_commands = print_commands, 
                                   compiler = compiler, 
                                   compiler_flag_prefix = compiler_flag_prefix)
  
  src_filename = compiled_object.src_filename
  object_name = compiled_object.object_filename
  shared_name = src_filename.replace(src_extension, shared_extension)
  linker_flags = get_linker_flags(compiler, extra_link_flags, linker_flag_prefix) 
  
  if isinstance(compiler, (list,tuple)):
    linker_cmd = list(compiler)
  else:
    linker_cmd = [compiler]
  linker_cmd += [object_name] 
  linker_cmd += linker_flags 
  linker_cmd += list(extra_objects) 
  linker_cmd += ['-o', shared_name]

  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = python_lib_dir
  run_cmd(linker_cmd, env = env, label = "Linking")
  

  if print_commands:
    print "Loading newly compiled extension module %s..." % shared_name
  module =  imp.load_dynamic(fn_name, shared_name)
  
  #on a UNIX-style filesystem it should be OK to delete a file while it's open
  #since the inode will just float untethered from any name
  #If we ever support windows we should find some other way to delete the .dll 
  
  
  c_fn = getattr(module,fn_name)
  
  if config.delete_temp_files:
    os.remove(src_filename)
    os.remove(object_name)
    # window's can't just untether inodes like a UNIX
    # ...have to eventually think of a plan to clean these things up
    if not windows: os.remove(shared_name)
    
  compiled_fn = CompiledPyFn(c_fn = c_fn, 
                             module = module, 
                             shared_filename =  shared_name,
                             object_filename = object_name, 
                             src = src, 
                             src_filename = src_filename,
                             fn_name = fn_name, 
                             fn_signature = fn_signature)
  return compiled_fn

