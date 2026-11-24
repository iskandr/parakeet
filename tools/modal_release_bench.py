"""Run with 'modal run tools/modal_release_bench.py --output comparison.json'.

All GPU providers use the same physical device. Each runs in a fresh process,
and JAX has a separate Python environment for its CUDA library dependencies.
"""

import json
from pathlib import Path

import modal

app = modal.App("parakeet-release-comparison")
revision = "0345559f4b2d3c59a07ce8c1a71bfb619f690063"
checkout = (
  "git init -q /opt/parakeet && cd /opt/parakeet && "
  f"git fetch -q --depth=1 https://github.com/iskandr/parakeet.git {revision} && git checkout -q FETCH_HEAD"
)
source = Path(__file__).resolve().parents[1] / "benchmarks/release_comparison.py"
base = (
  modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
  .apt_install("g++", "git")
  .pip_install("numpy==2.2.6", "cupy-cuda12x==13.6.0", "threadpoolctl==3.6.0")
)
gpu_image = (
  base.pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
  .pip_install("faiss-gpu-cu12==1.12.0")
  .pip_install("pykeops==2.3")
  .run_commands(
    "python -m venv /opt/jax && /opt/jax/bin/pip install numpy==2.2.6 'jax[cuda12]==0.6.2'",
    checkout,
  )
  .env({"PYTHONPATH": "/opt/parakeet", "OPENBLAS_NUM_THREADS": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
  .add_local_file(source, "/opt/comparison.py")
)
cpu_image = (
  base.run_commands(checkout)
  .env({"PYTHONPATH": "/opt/parakeet"})
  .pip_install("pykeops==2.3", "faiss-cpu==1.12.0", "scipy==1.15.3", "scikit-learn==1.7.2")
  .env({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
  .add_local_file(source, "/opt/comparison.py")
)


@app.function(image=gpu_image, gpu=["L4", "A10G", "A100-40GB"], memory=8192, cpu=2, timeout=1200, scaledown_window=2)
def gpu_run():
  import subprocess
  import sys
  import tempfile

  results = {}
  for provider in ("array", "faiss", "keops", "torch", "jax"):
    py = "/opt/jax/bin/python" if provider == "jax" else sys.executable
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "result.json"
      proc = subprocess.run([py, "/opt/comparison.py", provider, str(path)])
      results[provider] = json.loads(path.read_text()) if proc.returncode == 0 else {"error": f"exit {proc.returncode}"}
  return results


@app.function(image=cpu_image, cpu=(4, 4), memory=6144, timeout=1200, scaledown_window=2)
def cpu_run():
  import runpy

  return runpy.run_path("/opt/comparison.py")["run"]("cpu")


@app.local_entrypoint()
def main(output: str = "comparison.json", cpu: bool = True, gpu: bool = True):
  results = {"revision": revision, "providers": {}}
  remote_gpu = gpu_run.spawn() if gpu else None
  if cpu:
    results["providers"]["cpu"] = cpu_run.remote()
    Path(output).write_text(json.dumps(results, indent=2) + "\n")
  if remote_gpu:
    results["providers"].update(remote_gpu.get())
    Path(output).write_text(json.dumps(results, indent=2) + "\n")
  if any(p.get("error") or any("error" in row for row in p["rows"]) for p in results["providers"].values()):
    raise RuntimeError(f"Some measurements failed. See {output}.")
