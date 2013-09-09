import distutils 
import imp 
import numpy.distutils as npdist
import os
import platform
import subprocess  
import tempfile

header_names = ["Python.h", 
                'numpy/arrayobject.h',
                "stdint.h", 
                "math.h", 
                ]
common_headers = "\n".join("#include <%s>" % header for header in header_names) + "\n"

config_vars = distutils.sysconfig.get_config_vars()

default_compiler = None
for compiler in ['clang', 'g++']:
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

python_lib_dir = distutils.sysconfig.get_python_lib() + "/../../"
python_version = distutils.sysconfig.get_python_version()
python_lib = "python%s" % python_version
python_lib_full = 'lib%s%s' % (python_lib, python_lib_extension)

linker_flags = ['-shared'] + \
               ["-L%s" % python_lib_dir] + \
               ["-l%s" % python_lib]

numpy_include_dirs = npdist.misc_util.get_numpy_include_dirs()
include_dirs = python_include_dirs + numpy_include_dirs 
compiler_flags = ['-I%s' % path for path in include_dirs] +  ['-fPIC']


def compile_module(src, fn_name, 
                     src_filename = None, 
                     print_source = True, 
                     print_commands = True):
  src = common_headers + src
  if src_filename is None:
    src_file = tempfile.NamedTemporaryFile(suffix = source_extension, 
                                           prefix = "parakeet_%s_" % fn_name, 
                                           delete=False)
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
    
  
  src_file.write(src)
  src_file.write("""
  static PyMethodDef %(fn_name)sMethods[] = {
    {"%(fn_name)s",  %(fn_name)s, METH_VARARGS,
     "%(fn_name)s"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
  };
  """ % locals())
  
  src_file.write("""
  PyMODINIT_FUNC
  init%(fn_name)s(void)
  {
    //Py_Initialize();
    Py_InitModule("%(fn_name)s", %(fn_name)sMethods);
    import_array();
  }
  """ % locals())
  
  src_file.close()
  object_name = src_filename.replace(source_extension, object_extension)
  if print_source:
    print subprocess.check_output(['cat', src_filename])
  compiler_cmd = [compiler] + compiler_flags + ['-c', src_filename, '-o', object_name]
  if print_commands: print " ".join(compiler_cmd)
  print subprocess.check_output(compiler_cmd)
  
  shared_name = src_filename.replace(source_extension, shared_extension)
  linker_cmd = [compiler] + linker_flags + [object_name, '-o', shared_name]
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
  return imp.load_dynamic(fn_name, shared_name)
