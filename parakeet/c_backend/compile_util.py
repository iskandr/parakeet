import distutils 
import subprocess
import sys
import ctypes 
import numpy as np 
import tempfile

header_names = ["math.h", "stdint.h", "Python.h", 'numpy/arrayobject.h']
common_headers = "\n".join("#include <%s>" % header for header in header_names) + "\n"

python_include_dirs = [distutils.sysconfig.get_python_inc()]
python_lib_dirs = []

version_info = sys.version_info
python_libs = ['python%d.%d' % (version_info.major, version_info.minor)]
linker_flags = ["-l%s" % lib for lib in python_libs] + ["-L%s" % path for path in python_lib_dirs]

numpy_include_dirs = np.distutils.misc_util.get_numpy_include_dirs()

include_dirs = python_include_dirs + numpy_include_dirs 

compiler_flags = ['-I%s' % path for path in include_dirs]

def get_default_compiler(compilers = ['clang', 'gcc']):
  for compiler in compilers:
    path = distutils.spawn.find_executable(compiler)
    if path:
      return path 
  assert False, "No compiler found"
  
def compile_dll(src, src_filename = None):
  src = common_headers + src
  if src_filename is None:
    src_file = tempfile.NamedTemporaryFile(suffix = ".c", prefix = "parakeet_", delete=False)
    src_filename = src_file.name 
  else:
    src_file = open(src_filename, 'w')
  src_file.write(src)
  src_file.close()
  compiler = get_default_compiler()
  object_name = src_filename.replace('.c', '.o')
  subprocess.check_call([compiler] + compiler_flags + ['-fPIC'] + \
                         ['-c', src_filename, '-o', object_name])
  
  shared_name = src_filename.replace('.c', '.so')
  subprocess.check_call([compiler, '-shared'] + linker_flags + \
                         ['-o', shared_name, object_name])
  
  cdll = ctypes.cdll.LoadLibrary(shared_name)
  return cdll 