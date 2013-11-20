import distutils
import os 

from . system_info import include_dirs, windows, mac_os 
from . import config 

def get_opt_flags():
  opt_flags = [config.opt_level] 
  if config.sse2:
    opt_flags.append('-msse2')
  if config.fast_math:
    opt_flags.append('-ffast-math')
  return opt_flags 

def get_compiler_flags(extra_flags = [], compiler_flag_prefix = None):
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


def get_linker_flags(extra_flags = [], linker_flag_prefix = None):
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

