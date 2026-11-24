"""Run the current checkout on one physical NVIDIA A10: modal run tools/modal_pairwise_fold.py."""

import hashlib
import json
import subprocess
from pathlib import Path

import modal

baseline = "ff7ebd7be1cc3ea2247364e88072237caa548a39"
root = Path(__file__).resolve().parents[1]
image = (
  modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
  .apt_install("g++", "git")
  .pip_install("numpy==2.2.6", "cupy-cuda12x==13.6.0", "threadpoolctl==3.6.0")
  .run_commands(
    f"git init -q /opt/baseline && git -C /opt/baseline fetch -q --depth=1 "
    f"https://github.com/iskandr/parakeet.git {baseline} && git -C /opt/baseline checkout -q FETCH_HEAD"
  )
  .env({"PYTHONPATH": "/opt/parakeet", "OPENBLAS_NUM_THREADS": "1"})
)
for folder in ("parakeet", "benchmarks", "examples"):
  image = image.add_local_dir(root / folder, f"/opt/parakeet/{folder}", ignore=["**/__pycache__/**", "**/*.pyc"])

app = modal.App("parakeet-ordered-pairwise-fold")


@app.function(image=image, gpu="A10G", cpu=2, memory=4096, timeout=600, scaledown_window=2)
def measure():
  import os
  import sys
  import tempfile

  # Running each checkout in a separate process isolates Python imports,
  # JIT specialization caches and CUDA native state on the same physical GPU.
  results = {}
  with tempfile.TemporaryDirectory() as tmp:
    for name, checkout in (("release_baseline", "/opt/baseline"), ("candidate", "/opt/parakeet")):
      target = Path(tmp) / f"{name}.json"
      cmd = [sys.executable, "/opt/parakeet/benchmarks/gpu_pairwise_fold.py", "--output", str(target)]
      if name == "release_baseline":
        cmd.append("--baseline")
      subprocess.run(
        cmd,
        cwd=tmp,
        env={**os.environ, "PYTHONPATH": checkout},
        check=True,
      )
      results[name] = json.loads(target.read_text())
  digest = hashlib.sha256()
  project = Path("/opt/parakeet")
  for path in sorted((project / "parakeet").rglob("*")):
    if path.suffix in (".py", ".h", ".c"):
      digest.update(path.relative_to(project).as_posix().encode() + b"\0" + path.read_bytes())
  results["compiler_source_sha256"] = digest.hexdigest()
  return results


@app.local_entrypoint()
def main(output: str = "pairwise.json"):
  revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=root).strip()
  result = measure.remote()
  result.update(release_revision=baseline, revision=revision)
  Path(output).write_text(json.dumps(result, indent=2) + "\n")
