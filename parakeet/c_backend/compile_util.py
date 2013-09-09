import distutils 
import subprocess
import sys
import ctypes 
import numpy as np 
import tempfile

header_names = ["math.h", "stdint.h", "Python.h", 'numpy/arrayobject.h']
common_headers = "\n".join("#include <%s>" % header for header in header_names) + "\n"

python_include_dirs = [distutils.sysconfig.get_python_inc()]

def get_python_lib_dir():
  return distutils.unixccompiler.sysconfig.get_python_lib() + "/../../"

python_lib_dir = get_python_lib_dir()

version_info = sys.version_info
python_libs = ['python%d.%d' % (version_info.major, version_info.minor)]

linker_flags = ['-shared'] + \
               ["-L%s" % python_lib_dir] + \
               ["-l%s" % lib for lib in python_libs]
                # '-ldl', '-framework', 'CoreFoundation'

numpy_include_dirs = np.distutils.misc_util.get_numpy_include_dirs()

include_dirs = python_include_dirs + numpy_include_dirs 

compiler_flags = ['-I%s' % path for path in include_dirs] +  ['-fPIC']

def get_default_compiler(compilers = ['clang', 'g++']):
  for compiler in compilers:
    path = distutils.spawn.find_executable(compiler)
    if path:
      return path 
  assert False, "No compiler found"
  
def compile_module(src, fn_name, src_filename = None):
  src = common_headers + src
  if src_filename is None:
    src_file = tempfile.NamedTemporaryFile(suffix = ".c", prefix = "parakeet_", delete=False)
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
  compiler = get_default_compiler()
  object_name = src_filename.replace('.c', '.o')
  print subprocess.check_output(['cat', src_filename])
  import sys
  sys.stdout.flush()
  run_compiler = [compiler] + compiler_flags + ['-c', src_filename, '-o', object_name]
  print " ".join(run_compiler)
  subprocess.check_call(run_compiler)
  
  shared_name = src_filename.replace('.c', '.so')
  run_linker = [compiler] + linker_flags + [object_name, '-o', shared_name]
  print " ".join(run_linker)
  subprocess.check_call(run_linker, env = {"LD_LIBRARY_PATH" :  python_lib_dir })
  print subprocess.check_output(['install_name_tool', 
                                 '-change', 
                                  'libpython2.7.dylib', 
                                  python_lib_dir + "/libpython2.7.dylib", 
                                  shared_name])
  import imp 
  m = imp.load_dynamic(fn_name, shared_name)
  return getattr(m, fn_name)
