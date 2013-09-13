import collections 
import distutils 
import imp 
import numpy.distutils as npdist
import os
import platform
import subprocess  
import tempfile

from config import debug

CompiledPyFn = collections.namedtuple("CompiledPyFn",
                                      ("c_fn", 
                                       "module", 
                                       "shared_filename", 
                                       "object_filename",
                                       "src", 
                                       "src_filename",
                                       "name"))

CompiledObject = collections.namedtuple("CompiledObject", 
                                        ("src",
                                         "src_filename",
                                         "object_filename"))

common_headers = ["stdint.h",  "math.h",  "signal.h"]

python_headers = ["Python.h",  'numpy/arrayobject.h', 'numpy/arrayscalars.h',]

  
  

defs = []#["#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"]
defs = "\n".join(defs + ["\n"])
config_vars = distutils.sysconfig.get_config_vars()

default_compiler = None
for compiler in [ 'clang', 'g++' ]:
  path = distutils.spawn.find_executable(compiler)
  if path:
    default_compiler = path
    break

assert compiler is not None, "No compiler found!"

source_extension = ".cpp"
object_extension = ".o"
shared_extension = npdist.system_info.get_shared_lib_extension(True)

mac_os = platform.system() == 'Darwin'
if mac_os:
  python_lib_extension = '.dylib'
else:
  python_lib_extension = '.so'

python_include_dirs = [distutils.sysconfig.get_python_inc()]

numpy_include_dirs = npdist.misc_util.get_numpy_include_dirs()
include_dirs = python_include_dirs + numpy_include_dirs 
compiler_flags = ['-I%s' % path for path in include_dirs] +  ['-fPIC', '-Wall']

if debug:
  compiler_flags.extend(['-ggdb3', '-O0'])
else:
  compiler_flags.extend(['-O3'])


python_lib_dir = distutils.sysconfig.get_python_lib() + "/../../"
python_version = distutils.sysconfig.get_python_version()
python_lib = "python%s" % python_version
python_lib_full = 'lib%s%s' % (python_lib, python_lib_extension)


linker_flags = ['-shared'] + \
               ["-L%s" % python_lib_dir] + \
               ["-l%s" % python_lib] + ['-ggdb3']
               



def create_source_file(src, fn_name = None, src_filename = None, 
                       forward_declarations = [],
                       extra_headers = [], 
                       print_source = debug):
  
  if fn_name is None:
    prefix = "parakeet_"
  else:
    prefix = "parakeet_%s_" % fn_name
  if src_filename is None:
    src_file = tempfile.NamedTemporaryFile(suffix = source_extension, 
                                           prefix =  prefix, 
                                           delete=False)
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
  
  for d in defs:
    src_file.write(d)
  
  for header in extra_headers + common_headers:
    src_file.write("#include <%s>\n" % header)
  
  for decl in set(forward_declarations):
    src_file.write(decl)
  
  src_file.write(src)
  src_file.close()
  if print_source: print subprocess.check_output(['cat', src_filename])
  return src_file 

def compile_object(src, 
                   fn_name = None,  
                   src_filename = None, 
                   forward_declarations = [],
                   extra_headers = [], 
                   extra_objects = [], 
                   print_source = debug, 
                   print_commands = debug ):
  src_file = create_source_file(src, fn_name, src_filename, 
                                forward_declarations, extra_headers,
                                print_source = print_source)
  src_filename = src_file.name
  object_name = src_filename.replace(source_extension, object_extension)
  compiler_cmd = [compiler] + compiler_flags + ['-c', src_filename, '-o', object_name]
  if print_commands: print " ".join(compiler_cmd)
  print subprocess.check_output(compiler_cmd)
  return CompiledObject(src = src, src_filename = src_filename, object_filename = object_name)
  
  
def compile_module(src, fn_name, 
                     src_filename = None,
                     forward_declarations = [],
                     extra_headers = [],  
                     extra_objects = [],
                     
                     print_source = debug, 
                     print_commands = debug):
  
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
  
  compiled_object = compile_object(src, fn_name, src_filename, 
                                   forward_declarations,
                                   extra_headers = python_headers + extra_headers,  
                                   extra_objects = extra_objects,
                                   print_source = print_source, 
                                   print_commands = print_commands)
  
  src_filename = compiled_object.src_filename
  object_name = compiled_object.object_filename
  
  shared_name = src_filename.replace(source_extension, shared_extension)
  linker_cmd = [compiler] + linker_flags + [object_name] + list(extra_objects) + ['-o', shared_name]
  
  if print_commands: print "LD_LIBRARY_PATH=%s" % python_lib_dir, " ".join(linker_cmd)
  
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = python_lib_dir
  print subprocess.check_call(linker_cmd, env = env)
  
  if mac_os:
    # Annoyingly have to patch up the shared library to point to the correct Python
    change_cmd = ['install_name_tool', '-change', '%s' % python_lib_full, 
                  python_lib_dir + "/%s" % python_lib_full, 
                  shared_name]
    if print_commands: print " ".join(change_cmd)
    print subprocess.check_output(change_cmd)
  if print_commands:
    print "Loading newly compiled extension module %s..." % shared_name
  module =  imp.load_dynamic(fn_name, shared_name)
  c_fn = getattr(module,fn_name)
  compiled_fn = CompiledPyFn(c_fn = c_fn, 
                             module = module, 
                             shared_filename =  shared_name,
                             object_filename = object_name, 
                             src = src, 
                             src_filename = src_filename,
                             name = fn_name)
  return compiled_fn

