

import parakeet

fn = parakeet.compile_string("def f(x):\n  return x + 1")
assert parakeet.run_compiled(fn, 1) == 1
