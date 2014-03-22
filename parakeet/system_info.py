import os
import subprocess
import platform 


mac_os = platform.system() == 'Darwin'
windows = platform.system() == 'Windows'

def check_openmp_available():
  cmd = """echo "int main() {}" | clang -fopenmp -x c++ -"""
  with open(os.devnull, 'w') as devnull:
    p = subprocess.Popen(cmd, shell=True, stderr = devnull, stdout = devnull)
  p.wait()
  code = p.returncode
  return code == 0

openmp_available = check_openmp_available()