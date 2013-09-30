import collections 
import distutils 
import imp 
import numpy.distutils as npdist 
import numpy.distutils.system_info as np_sysinfo 
import os
import platform
import subprocess  
import tempfile
import time 

from tempfile import NamedTemporaryFile

from config import (debug, pure_c, fast_math, 
                    print_commands, print_module_source, 
                    print_command_elapsed_time,
                    use_openmp, 
                    delete_temp_files)



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

default_compiler = None
for compiler in [  ('gcc' if pure_c else 'g++'), 'clang']:
  path = distutils.spawn.find_executable(compiler)
  if path:
    default_compiler = path
    break

assert compiler is not None, "No compiler found!"


source_extension = ".c" if pure_c else ".cpp"
object_extension = ".o"
shared_extension = np_sysinfo.get_shared_lib_extension(True)

mac_os = platform.system() == 'Darwin'
if mac_os:
  python_lib_extension = '.dylib'
else:
  python_lib_extension = '.so'

python_include_dirs = [distutils.sysconfig.get_python_inc()]

numpy_include_dirs = npdist.misc_util.get_numpy_include_dirs()
include_dirs = python_include_dirs + numpy_include_dirs 
compiler_flags = ['-I%s' % path for path in include_dirs] + \
                 ['-fPIC', '-Wall', '-Wno-unused-variable']

opt_flags = ['-O3', '-msse2']

if fast_math:
  opt_flags.append('-ffast-math')

if debug:
  compiler_flags.extend(['-g', '-O0'])
else:
  compiler_flags.extend(opt_flags)

if not pure_c: 
  compiler_flags.extend(['-fpermissive'])  


python_lib_dir = distutils.sysconfig.get_python_lib() + "/../../"
python_version = distutils.sysconfig.get_python_version()
python_lib = "python%s" % python_version
python_lib_full = 'lib%s%s' % (python_lib, python_lib_extension)


linker_flags = ['-shared'] + \
               ["-L%s" % python_lib_dir] + \
               ['-lm']  
if mac_os:
  linker_flags.append("-headerpad_max_install_names")
  linker_flags.append("-undefined dynamic_lookup")
else:
  linker_flags.append("-l%s" % python_lib)

if not debug and use_openmp and compiler in ('gcc', 'g++'):
  compiler_flags.append('-fopenmp')
  linker_flags.append('-fopenmp')             



tempdir = None 
# write to in-memory tmpfs if possible
if os.path.isdir("/dev/shm") and os.access('/dev/shm', os.W_OK):
  tempdir = "/dev/shm"

def create_source_file(src, 
                         fn_name = None, 
                         src_filename = None, 
                         forward_declarations = [],
                         extra_function_sources = [], 
                         extra_headers = [], 
                         print_source = print_module_source):

  if fn_name is None:
    prefix = "parakeet_"
  else:
    prefix = "parakeet_%s_" % fn_name
  if src_filename is None:
    src_file = NamedTemporaryFile(suffix = source_extension, 
                                  prefix =  prefix, 
                                  delete = False, 
                                  mode = 'w',
                                  dir = tempdir)
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
  
  for d in cpp_defs:
    src_file.write(d)
    src_file.write("\n")
  
  for header in extra_headers + c_headers:
    src_file.write("#include <%s>\n" % header)
  
  for decl in set(forward_declarations):
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
  if print_source: print subprocess.check_output(['cat', src_filename])
  return src_file 

def run_cmd(cmd, env = None, label = ""):
  if print_commands: print " ".join(cmd)
  if print_command_elapsed_time: t = time.time()
  with open(os.devnull, "w") as fnull:
    with NamedTemporaryFile(prefix="parakeet_compile_err", mode = 'r+') as err_file:
      try:  
        subprocess.check_call(cmd, stdout = fnull, stderr = err_file, env = env)
      except:
        print "Parakeet encountered error(s) during compilation: "
        print err_file.read()
        raise 
    
  if print_command_elapsed_time: 
    if label:
      print "%s, elapsed time: %0.4f" % (label, time.time() - t)
    else:
      print "Elapsed time:", time.time() - t 

def compile_object(src, 
                   fn_name = None,  
                   fn_signature = None,
                   src_filename = None, 
                   forward_declarations = [],
                   extra_function_sources = [], 
                   extra_headers = python_headers, 
                   extra_objects = [], 
                   print_source = print_module_source, 
                   print_commands = print_commands ):
  
  src_file = create_source_file(src, 
                                fn_name = fn_name, 
                                src_filename = src_filename, 
                                forward_declarations = forward_declarations,
                                extra_function_sources = extra_function_sources,  
                                extra_headers = extra_headers,
                                print_source = print_source)
  src_filename = src_file.name
  object_name = src_filename.replace(source_extension, object_extension)
  compiler_cmd = [compiler] + compiler_flags + ['-c', src_filename, '-o', object_name]
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
                     forward_declarations = [],
                     extra_function_sources = [], 
                     extra_headers = [],  
                     extra_objects = [],
                     print_source = print_module_source, 
                     print_commands = print_commands):
  

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
  

  compiled_object = compile_object(src, 
                                   fn_name,
                                   src_filename  = src_filename, 
                                   forward_declarations = forward_declarations,
                                   extra_function_sources = extra_function_sources, 
                                   extra_headers = python_headers + extra_headers,  
                                   extra_objects = extra_objects,
                                   print_source = print_source, 
                                   print_commands = print_commands)
  
  src_filename = compiled_object.src_filename
  object_name = compiled_object.object_filename
  
  shared_name = src_filename.replace(source_extension, shared_extension)
  linker_cmd = [compiler] + linker_flags + [object_name] + list(extra_objects) + ['-o', shared_name]

  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = python_lib_dir
  run_cmd(linker_cmd, env = env, label = "Linking")
  
   
  if mac_os:
    # Annoyingly have to patch up the shared library to point to the correct Python
    change_cmd = ['install_name_tool', '-change', '%s' % python_lib_full, 
                  python_lib_dir + "/%s" % python_lib_full, 
                  shared_name]
    if print_commands: print " ".join(change_cmd)
    subprocess.check_output(change_cmd)
  # delete the .o file since we don't need it anymore 
  # os.remove(object_name)
  
  if print_commands:
    print "Loading newly compiled extension module %s..." % shared_name
  module =  imp.load_dynamic(fn_name, shared_name)
  
  #on a UNIX-style filesystem it should be OK to delete a file while it's open
  #since the inode will just float untethered from any name
  #If we ever support windows we should find some other way to delete the .dll 
  
  
  c_fn = getattr(module,fn_name)
  
  if delete_temp_files:
    os.remove(src_filename)
    os.remove(object_name)
    os.remove(shared_name)
    
  compiled_fn = CompiledPyFn(c_fn = c_fn, 
                             module = module, 
                             shared_filename =  shared_name,
                             object_filename = object_name, 
                             src = src, 
                             src_filename = src_filename,
                             fn_name = fn_name, 
                             fn_signature = fn_signature)
  return compiled_fn

"""

class CompiledFn(object):
  def __init__(self, fn_name, fn_signature, src,
                src_filename= None,  
                c_fn = None, module = None, 
                shared_filename = None, 
                object_filename = None):
    self.fn_name = fn_name 
    self.fn_signature = fn_signature
    self.src = src
     
    self._src_filename = src_filename 
    self._c_fn = c_fn
    self._module = module  
    self._shared_filename = shared_filename
    self._object_filename = object_filename 
    
  @property
  def src_filename(self):
    if self._src_filename: return self._src_filename
    # write source
    # store in self._src_filename
  
  @property
  def c_fn(self):
    if self._c_fn: return self._c_fn
    
    
    
  
  @property
  def shared_filename(self):
    pass
     
  @property
  def object_filename(self):
    pass
  
  @property 
  def module(self):
    pass 
"""  

