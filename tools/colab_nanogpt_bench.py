"""Launch ``benchmarks/nanogpt_bench.py`` on a Colab GPU via ``runplz``.

Usage::

    PK_COLAB_GPU=A100 runplz colab tools/colab_nanogpt_bench.py
    PK_COLAB_GPU=L4   runplz colab tools/colab_nanogpt_bench.py
"""

import os
import pathlib
import subprocess
import sys

from runplz import App, Image

GPU = os.environ.get("PK_COLAB_GPU", "L4")
app = App("parakeet-nanogpt-bench")
image = Image.from_registry("python:3.12")


@app.function(image=image, gpu=GPU, timeout=30 * 60)
def run_bench():
  repo = pathlib.Path(__file__).resolve().parents[1]
  py = sys.executable
  subprocess.run(["bash", "-c", "nvidia-smi; nproc; python --version"], check=False)
  subprocess.run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=repo, check=True)
  env = dict(os.environ, PYTHONPATH=str(repo))
  subprocess.run([py, "-m", "benchmarks.nanogpt_bench", "--device", "cuda"], cwd=repo, env=env, check=True)


@app.local_entrypoint()
def main():
  run_bench.remote()
