import args
import core_types
import syntax
import syntax_helpers
import testing_helpers
import tile_adverbs

fn = syntax.TypedFn(
  args = args.Args(positional=["x"],),
  body = [syntax.Return(syntax_helpers.const(1))],
  return_type = core_types.Int32,
  type_env = {})

def test_tiling():
  tiling_transform = tile_adverbs.TileAdverbs(fn)
  new_fn = tiling_transform.apply(copy=True)
  assert isinstance(new_fn, syntax.TypedFn)

def test_lowering():
  lower_tiling = tile_adverbs.LowerTiledAdverbs(fn)
  new_fn = lower_tiling.apply(copy=True)
  assert isinstance(new_fn, syntax.TypedFn)

if __name__ == '__main__':
  testing_helpers.run_local_tests()
