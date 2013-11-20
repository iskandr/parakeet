import distutils
import platform  
import numpy.distutils as npdist 
import numpy.distutils.system_info as np_sysinfo 

import config 

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

python_lib_dir = distutils.sysconfig.get_python_lib() + "/../../"
python_version = distutils.sysconfig.get_python_version()
