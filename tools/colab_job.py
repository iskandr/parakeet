"""Run Parakeet's tests, examples and benchmarks on Google Colab hardware.

This is an on-demand hardware check, not part of the normal test suite or CI.
It needs `runplz <https://github.com/pirl-unc/runplz>`_ and a Colab account::

    runplz colab tools/colab_job.py                  # NVIDIA T4, real CUDA
    PK_COLAB_GPU=A100 runplz colab tools/colab_job.py
    PK_COLAB_TPU=v5e1 runplz colab tools/colab_job.py   # TPU host (CPU backends + probe)

Logs and a ``summary.json`` land in ``out/``.
"""

import os

from runplz import App, ColabConfig, Image

TPU = os.environ.get("PK_COLAB_TPU") or None
GPU = None if TPU else os.environ.get("PK_COLAB_GPU", "T4")

app = App("parakeet-hw", colab_config=ColabConfig(tpu=TPU))
image = Image.from_registry("python:3.12")

CUDA_CORE = """
import warnings
import numpy as np
import parakeet
from parakeet import jit
from parakeet.backends.cuda import CudaDevice
import tests.test_backends as tb

print(parakeet.available_backends())
assert parakeet.available_backends()["cuda"] == "ok"
dev = CudaDevice.get()
assert not dev.simulated
print("device:", dev.device_name)
with warnings.catch_warnings():
    warnings.simplefilter("error", parakeet.FallbackWarning)
    tb.run_everywhere("cuda", np.float64, 1e-10)
    tb.run_everywhere("cuda", np.float32, 2e-6)
    tb.aliased_arguments("cuda", np.float32)
    tb.aliased_arguments("cuda", np.float64)
f = jit(lambda x, idx: x[idx] * 2, backend="cuda")
x = np.zeros(50_000)
spec = f.specialize(x, np.arange(50_000))
assert "__global__ void pk_kernel" in spec.kernels[0]
try:
    f(x, np.arange(50_000) * 2)
except IndexError as e:
    assert "out of bounds" in str(e), e
else:
    raise SystemExit("expected IndexError")
assert spec.module.kernels[0].device.compiled is not None
assert dev.launches > 0
print("real-cuda-ok, loops run on the GPU:", dev.launches)
"""

TPU_PROBE = """
import os
print("cpus:", os.cpu_count())
try:
    import jax, jax.numpy as jnp
    print("jax", jax.__version__, "devices:", jax.devices())
    x = jnp.arange(1_000_000, dtype=jnp.float32)
    print("jit'd sum on", list(x.devices()), float(jax.jit(lambda v: (v * v).sum())(x)))
except Exception as e:
    print("jax probe failed:", type(e).__name__, e)
"""


@app.function(image=image, gpu=GPU, timeout=90 * 60)
def hardware_check(tpu: bool = False):
  import json
  import pathlib
  import subprocess
  import sys
  import time

  repo = pathlib.Path(__file__).resolve().parents[1]
  out = pathlib.Path(os.environ.get("RUNPLZ_OUT", "/content/out"))
  out.mkdir(parents=True, exist_ok=True)
  summary = {}

  def stage(name, cmd, env=None, timeout=3600):
    print(f"\n===== {name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    full_env = dict(os.environ, PYTHONPATH=str(repo), **(env or {}))
    try:
      p = subprocess.run(cmd, cwd=repo, env=full_env, capture_output=True, text=True, timeout=timeout)
      rc, text = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
      rc, text = 124, f"TIMEOUT after {timeout}s\n{e.stdout or ''}"
    (out / f"{name}.log").write_text(text)
    tail = "\n".join(text.splitlines()[-25:])
    print(tail, flush=True)
    summary[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "last_line": text.strip().splitlines()[-1:]}
    return rc

  py = sys.executable
  stage("system", ["bash", "-c", "nvidia-smi; nproc; free -g | head -2; cc --version | head -1; python --version"])
  stage("install", [py, "-m", "pip", "install", "-q", "-e", ".", "pytest", "pytest-xdist"])
  stage(
    "versions",
    [
      py,
      "-c",
      "import numpy, parakeet; print(numpy.__version__, parakeet.__version__, parakeet.available_backends())",
    ],
  )

  if tpu:
    (repo / "_tpu_probe.py").write_text(TPU_PROBE)
    stage("tpu_probe", [py, "_tpu_probe.py"])
    stage("cpu_suite", [py, "-m", "pytest", "tests", "-q", "-n", "8", "-p", "no:cacheprovider"])
    stage("benchmarks", [py, "benchmarks/run.py"])
    stage("crossover", [py, "benchmarks/crossover.py", "--json", str(out / "crossover.json")], timeout=3000)
  else:
    stage("cupy", [py, "-c", "import cupy; cupy.show_config()"])
    (repo / "_cuda_core.py").write_text(CUDA_CORE)
    stage("cuda_core", [py, "_cuda_core.py"])
    stage("crossover", [py, "benchmarks/crossover.py", "--json", str(out / "crossover.json")], timeout=3000)
    stats = out / "device_stats.txt"
    gpu_env = {
            "PARAKEET_BACKEND": "cuda", "PARAKEET_TEST_BACKENDS": "cuda", "PARAKEET_DEVICE_THRESHOLD": "1",
            "PARAKEET_DEVICE_FALLBACK": "silent", "PARAKEET_DEVICE_STATS": str(stats),
        }  # fmt: skip
    stage("cuda_suite", [py, "-m", "pytest", "tests", "-q", "-n", "2", "-p", "tools.device_stats",
                             "-p", "no:cacheprovider"], env=gpu_env, timeout=4200)  # fmt: skip
    stage("cuda_examples", [py, "-m", "pytest", "tests/test_examples.py", "-q", "-p", "tools.device_stats",
                                "-p", "no:cacheprovider"],
              env={"PARAKEET_BACKEND": "cuda", "PARAKEET_DEVICE_STATS": str(out / "device_stats_examples.txt")})  # fmt: skip
    stage("cpu_suite", [py, "-m", "pytest", "tests", "-q", "-n", "2", "-p", "no:cacheprovider"])
    stage("benchmarks_gpu", [py, "benchmarks/run.py", "--gpu"])
  (out / "summary.json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2), flush=True)


@app.local_entrypoint()
def main():
  hardware_check.remote(tpu=bool(TPU))
