"""Comprehensive Colab benchmark launcher for NVIDIA A100 GPU and high-core CPU.

Usage::

    # NVIDIA A100 GPU suite (release + large + nanoGPT + PTX/NVRTC/nvcc + gallery + pairwise fold)
    PK_BENCH_MODE=gpu PK_COLAB_GPU=A100 runplz colab tools/colab_full_bench.py

    # High-core CPU suite on Colab TPU v5e host (up to 32 threads: release_comparison + bio + trees)
    PK_BENCH_MODE=cpu PK_COLAB_TPU=v5e1 runplz colab tools/colab_full_bench.py
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from runplz import App, ColabConfig, Image

MODE = os.environ.get("PK_BENCH_MODE", "gpu")
TPU = os.environ.get("PK_COLAB_TPU") or None
GPU = None if (TPU or MODE == "cpu") else os.environ.get("PK_COLAB_GPU", "A100")
THREADS = int(os.environ.get("PK_CPU_THREADS", "32"))
STAGES = os.environ.get("PK_BENCH_STAGES", "")

app = App("parakeet-full-bench", colab_config=ColabConfig(tpu=TPU))
image = Image.from_registry("python:3.12")


def _run(cmd, *, cwd, env, label, allow_fail=False):
  print(f"\n===== {label}: {' '.join(cmd)}", flush=True)
  p = subprocess.run(cmd, cwd=cwd, env=env, check=False)
  if p.returncode != 0:
    print(f"WARNING: {label} exited with code {p.returncode}", flush=True)
    if not allow_fail:
      raise subprocess.CalledProcessError(p.returncode, cmd)


@app.function(image=image, gpu=GPU, timeout=60 * 60)
def run_suite(mode: str = "gpu", threads: int = 32, stages: str = ""):
  repo = pathlib.Path(__file__).resolve().parents[1]
  out = pathlib.Path(os.environ.get("RUNPLZ_OUT", "/content/out"))
  out.mkdir(parents=True, exist_ok=True)
  py = sys.executable
  env = dict(os.environ, PYTHONPATH=str(repo))
  only = {s.strip() for s in stages.split(",") if s.strip()}

  def want(name: str) -> bool:
    return not only or name in only

  _run(
    [
      "bash",
      "-c",
      "nvidia-smi || true; lscpu | grep -E 'Model name|CPU\\(s\\):' || nproc; free -g | head -2; nvcc --version || true; python --version",
    ],
    cwd=repo,
    env=env,
    label="system",
    allow_fail=True,
  )
  _run([py, "-m", "pip", "install", "-q", "-e", "."], cwd=repo, env=env, label="install_parakeet")

  if mode == "gpu":
    _run(
      [py, "-m", "pip", "install", "-q", "faiss-gpu-cu12", "pykeops"],
      cwd=repo,
      env=env,
      label="install_gpu_baselines",
      allow_fail=True,
    )

    if want("release_fp32"):
      for prov in ("array", "torch", "jax", "faiss", "keops"):
        _run(
          [
            py,
            "-m",
            "benchmarks.release_comparison",
            prov,
            str(out / f"gpu_release_fp32_{prov}.json"),
            "--scale",
            "release",
          ],
          cwd=repo,
          env=env,
          label=f"release_fp32_{prov}",
          allow_fail=True,
        )

    if want("release_tf32"):
      for prov in ("array", "torch", "jax", "faiss", "keops"):
        _run(
          [
            py,
            "-m",
            "benchmarks.release_comparison",
            prov,
            str(out / f"gpu_release_tf32_{prov}.json"),
            "--scale",
            "release",
            "--tensor-cores",
          ],
          cwd=repo,
          env=env,
          label=f"release_tf32_{prov}",
          allow_fail=True,
        )

    if want("large"):
      for prov in ("array", "torch", "jax", "faiss", "keops"):
        _run(
          [
            py,
            "-m",
            "benchmarks.release_comparison",
            prov,
            str(out / f"gpu_large_tf32_{prov}.json"),
            "--scale",
            "large",
            "--tensor-cores",
          ],
          cwd=repo,
          env=env,
          label=f"large_tf32_{prov}",
          allow_fail=True,
        )

    if want("compilers"):
      _run(
        [
          py,
          "-m",
          "benchmarks.nanogpt_bench",
          "--device",
          "cuda",
          "--compare-cuda-compilers",
          "--steps",
          "20",
          "--output",
          str(out / "gpu_nanogpt_compilers.json"),
        ],
        cwd=repo,
        env=env,
        label="nanogpt_cuda_compilers",
        allow_fail=True,
      )
      for comp in ("nvrtc", "ptx", "nvcc"):
        _run(
          [
            py,
            "-m",
            "benchmarks.release_comparison",
            "array",
            str(out / f"gpu_release_tf32_{comp}.json"),
            "--scale",
            "release",
            "--tensor-cores",
            "--cuda-compiler",
            comp,
          ],
          cwd=repo,
          env=env,
          label=f"release_tf32_{comp}",
          allow_fail=True,
        )

    if want("nanogpt"):
      _run(
        [
          py,
          "-m",
          "benchmarks.nanogpt_bench",
          "--device",
          "cuda",
          "--no-mapped",
          "--steps",
          "20",
          "--output",
          str(out / "gpu_nanogpt.json"),
        ],
        cwd=repo,
        env=env,
        label="nanogpt_all",
        allow_fail=True,
      )

    if want("gallery"):
      _run(
        [
          py,
          "-m",
          "benchmarks.gallery_bench",
          "--output",
          str(out / "gpu_gallery.json"),
        ],
        cwd=repo,
        env=env,
        label="gallery_gpu",
        allow_fail=True,
      )

    if want("pairwise"):
      _run(
        [py, "-m", "benchmarks.gpu_pairwise_fold", "--output", str(out / "gpu_pairwise_fold.json")],
        cwd=repo,
        env=env,
        label="gpu_pairwise_fold",
        allow_fail=True,
      )

  else:
    eff_threads = min(threads, os.cpu_count() or threads)
    print(f"CPU host detected {os.cpu_count()} cores; running with {eff_threads} threads", flush=True)
    _run(
      [
        py,
        "-m",
        "pip",
        "install",
        "-q",
        "numba",
        "faiss-cpu",
        "pykeops",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "edlib",
        "biopython",
        "biotite",
        "rdkit",
      ],
      cwd=repo,
      env=env,
      label="install_cpu_baselines",
      allow_fail=True,
    )

    if want("release_cpu"):
      _run(
        [
          py,
          "-m",
          "benchmarks.release_comparison",
          "cpu",
          str(out / f"cpu_{eff_threads}t_release.json"),
          "--threads",
          str(eff_threads),
          "--all-tasks",
        ],
        cwd=repo,
        env=env,
        label=f"cpu_{eff_threads}t_release",
        allow_fail=True,
      )

    if want("bio"):
      bio_env = dict(env, PARAKEET_NUM_THREADS=str(eff_threads))
      _run(
        [
          py,
          "-m",
          "benchmarks.bio_head_to_head",
          "--json",
          str(out / f"cpu_{eff_threads}t_bio.json"),
        ],
        cwd=repo,
        env=bio_env,
        label=f"cpu_{eff_threads}t_bio",
        allow_fail=True,
      )

    if want("trees"):
      tree_env = dict(env, PARAKEET_NUM_THREADS=str(eff_threads), OMP_NUM_THREADS=str(eff_threads))
      _run([py, "-m", "benchmarks.tree_models"], cwd=repo, env=tree_env, label="tree_models", allow_fail=True)

  print("\n===== Output files in", out, flush=True)
  for p in sorted(out.glob("*.json")):
    print(" ", p.name, p.stat().st_size, "bytes", flush=True)


@app.local_entrypoint()
def main():
  run_suite.remote(mode=MODE, threads=THREADS, stages=STAGES)
