


def f(x):
  return x + 1


def test_interp():
  untyped = ast_conversion.translate_function_value(f)
  result = parakeet.interp(untyped, 1)
  assert result == 2

