# Supported functions

This page is generated from Parakeet's library registry by `tools/gen_supported.py`, so it always
matches the installed version.  Anything not listed here raises an `UnsupportedError` that names the
function (and suggests similarly named supported ones).

## NumPy element-wise functions (ufuncs)

All follow NumPy's broadcasting and type promotion rules exactly.

|  |  |  |  |
|---|---|---|---|
| `np.absolute` | `np.add` | `np.arccos` | `np.arccosh` |
| `np.arcsin` | `np.arcsinh` | `np.arctan` | `np.arctan2` |
| `np.arctanh` | `np.bitwise_and` | `np.bitwise_count` | `np.bitwise_or` |
| `np.bitwise_xor` | `np.cbrt` | `np.ceil` | `np.copysign` |
| `np.cos` | `np.cosh` | `np.deg2rad` | `np.degrees` |
| `np.divide` | `np.equal` | `np.exp` | `np.exp2` |
| `np.expm1` | `np.fabs` | `np.float_power` | `np.floor` |
| `np.floor_divide` | `np.fmax` | `np.fmin` | `np.fmod` |
| `np.greater` | `np.greater_equal` | `np.heaviside` | `np.hypot` |
| `np.invert` | `np.isfinite` | `np.isinf` | `np.isnan` |
| `np.left_shift` | `np.less` | `np.less_equal` | `np.log` |
| `np.log10` | `np.log1p` | `np.log2` | `np.logaddexp` |
| `np.logaddexp2` | `np.logical_and` | `np.logical_not` | `np.logical_or` |
| `np.logical_xor` | `np.maximum` | `np.minimum` | `np.multiply` |
| `np.negative` | `np.not_equal` | `np.positive` | `np.power` |
| `np.rad2deg` | `np.radians` | `np.reciprocal` | `np.remainder` |
| `np.right_shift` | `np.rint` | `np.sign` | `np.signbit` |
| `np.sin` | `np.sinh` | `np.sqrt` | `np.square` |
| `np.subtract` | `np.tan` | `np.tanh` | `np.trunc` |

## NumPy functions

|  |  |  |  |
|---|---|---|---|
| `np.abs` | `np.acos` | `np.acosh` | `np.all` |
| `np.allclose` | `np.amax` | `np.amin` | `np.angle` |
| `np.any` | `np.arange` | `np.argmax` | `np.argmin` |
| `np.argsort` | `np.around` | `np.array` | `np.array_equal` |
| `np.asanyarray` | `np.asarray` | `np.ascontiguousarray` | `np.asin` |
| `np.asinh` | `np.atan` | `np.atan2` | `np.atanh` |
| `np.atleast_1d` | `np.atleast_2d` | `np.bincount` | `np.bitwise_invert` |
| `np.bitwise_left_shift` | `np.bitwise_not` | `np.bitwise_right_shift` | `np.broadcast_to` |
| `np.cdouble` | `np.clip` | `np.column_stack` | `np.complex128` |
| `np.complex64` | `np.concat` | `np.concatenate` | `np.conj` |
| `np.conjugate` | `np.convolve` | `np.copy` | `np.count_nonzero` |
| `np.cross` | `np.csingle` | `np.cumprod` | `np.cumsum` |
| `np.diag` | `np.diagonal` | `np.diff` | `np.dot` |
| `np.dtype` | `np.einsum` | `np.empty` | `np.empty_like` |
| `np.expand_dims` | `np.eye` | `np.flatnonzero` | `np.flip` |
| `np.fliplr` | `np.flipud` | `np.full` | `np.full_like` |
| `np.histogram` | `np.hstack` | `np.identity` | `np.imag` |
| `np.inner` | `np.isclose` | `np.isscalar` | `np.linspace` |
| `np.matmul` | `np.max` | `np.mean` | `np.median` |
| `np.min` | `np.mod` | `np.moveaxis` | `np.ndim` |
| `np.nonzero` | `np.ones` | `np.ones_like` | `np.outer` |
| `np.permute_dims` | `np.pow` | `np.prod` | `np.promote_types` |
| `np.ravel` | `np.real` | `np.repeat` | `np.reshape` |
| `np.result_type` | `np.round` | `np.searchsorted` | `np.shape` |
| `np.size` | `np.sort` | `np.squeeze` | `np.stack` |
| `np.std` | `np.sum` | `np.swapaxes` | `np.tile` |
| `np.trace` | `np.transpose` | `np.tril` | `np.triu` |
| `np.true_divide` | `np.unique` | `np.var` | `np.vdot` |
| `np.vstack` | `np.where` | `np.zeros` | `np.zeros_like` |

## `np.linalg` and `np.random`

|  |  |  |  |
|---|---|---|---|
| `np.linalg.norm` | `np.random.normal` | `np.random.permutation` | `np.random.rand` |
| `np.random.randint` | `np.random.randn` | `np.random.random` | `np.random.random_sample` |
| `np.random.ranf` | `np.random.sample` | `np.random.seed` | `np.random.shuffle` |
| `np.random.standard_normal` | `np.random.uniform` |  |  |

## Array attributes and methods

|  |  |  |  |
|---|---|---|---|
| `.T` | `.dtype` | `.flat` | `.imag` |
| `.itemsize` | `.nbytes` | `.ndim` | `.real` |
| `.shape` | `.size` | `._replace()` | `.all()` |
| `.any()` | `.append()` | `.argmax()` | `.argmin()` |
| `.argsort()` | `.astype()` | `.bit_count()` | `.clear()` |
| `.clip()` | `.conj()` | `.conjugate()` | `.copy()` |
| `.cumprod()` | `.cumsum()` | `.diagonal()` | `.dot()` |
| `.extend()` | `.fill()` | `.flatten()` | `.get()` |
| `.item()` | `.max()` | `.mean()` | `.min()` |
| `.nonzero()` | `.pop()` | `.prod()` | `.range()` |
| `.ravel()` | `.reshape()` | `.round()` | `.squeeze()` |
| `.std()` | `.sum()` | `.swapaxes()` | `.tolist()` |
| `.trace()` | `.transpose()` | `.var()` |  |

## Python builtins

|  |  |  |  |
|---|---|---|---|
| `abs` | `all` | `any` | `complex` |
| `divmod` | `enumerate` | `filter` | `getattr` |
| `hasattr` | `isinstance` | `len` | `list` |
| `map` | `max` | `min` | `ord` |
| `pow` | `print` | `range` | `reversed` |
| `round` | `slice` | `sorted` | `sum` |
| `tuple` | `zip` | `bool` | `float` |
| `int` |  |  |  |

## `math`

|  |  |  |  |
|---|---|---|---|
| `math.acos` | `math.acosh` | `math.asin` | `math.asinh` |
| `math.atan` | `math.atan2` | `math.atanh` | `math.cbrt` |
| `math.ceil` | `math.copysign` | `math.cos` | `math.cosh` |
| `math.degrees` | `math.erf` | `math.erfc` | `math.exp` |
| `math.exp2` | `math.expm1` | `math.fabs` | `math.floor` |
| `math.fmod` | `math.gamma` | `math.hypot` | `math.isfinite` |
| `math.isinf` | `math.isnan` | `math.lgamma` | `math.log` |
| `math.log10` | `math.log1p` | `math.log2` | `math.pow` |
| `math.radians` | `math.sin` | `math.sinh` | `math.sqrt` |
| `math.tan` | `math.tanh` | `math.trunc` | `math.e` |
| `math.inf` | `math.nan` | `math.pi` | `math.tau` |

## Parakeet operators

|  |  |  |  |
|---|---|---|---|
| `parakeet.allpairs` | `parakeet.each` | `parakeet.filter` | `parakeet.imap` |
| `parakeet.ireduce` | `parakeet.map` | `parakeet.outer_map` | `parakeet.parfor` |
| `parakeet.pmap1` | `parakeet.pmap2` | `parakeet.pmap2_trim` | `parakeet.reduce` |
| `parakeet.scan` |  |  |  |
