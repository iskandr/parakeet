"""Every script in examples/ must run and pass its own checks."""

import importlib.util
import pathlib

import pytest

EXAMPLES = sorted((pathlib.Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example(path):
  spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  if path.stem == "gpu_distances":
    module.main(n=3000)
  else:
    module.main()
