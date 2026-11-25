"""Tests for #48: zero-copy bytes/mmap/memoryview, bitops, bfloat16, IntEnum,
match/case, in-kernel frozen dataclasses & replace(), and structured NumPy views."""

from __future__ import annotations

import dataclasses
import enum
import mmap
import os
import tempfile
from typing import NamedTuple

import numpy as np
import pytest

import parakeet as pk
from parakeet import dsl


class Base(enum.IntEnum):
  A = 0
  C = 1
  G = 2
  T = 3
  N = 4


class CigarFlag(enum.IntFlag):
  MATCH = 1
  INS = 2
  DEL = 4
  SOFT_CLIP = 8


@dataclasses.dataclass(frozen=True)
class Hit:
  pos: int
  score: float
  strand: int = 1


class PairRec(NamedTuple):
  index: int
  dist: float


# ---------------------------------------------------------------------------
# 1. Zero-copy bytes / bytearray / memoryview / mmap
# ---------------------------------------------------------------------------


def test_bytes_bytearray_memoryview_mmap_zero_copy():
  @pk.jit
  def gc_count(seq):
    total = 0
    for i in range(len(seq)):
      c = seq[i]
      if c == ord("G") or c == ord("C") or c == ord("g") or c == ord("c"):
        total += 1
    return total

  dna = b"ACGTacgtGGCCnnnn"
  expected = sum(ch in b"GCgc" for ch in dna)
  assert gc_count(dna) == expected
  assert gc_count(bytearray(dna)) == expected
  assert gc_count(memoryview(dna)) == expected
  assert gc_count(memoryview(dna)[2:12]) == sum(ch in b"GCgc" for ch in dna[2:12])

  with tempfile.TemporaryFile() as f:
    f.write(dna * 16)
    f.flush()
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
      assert gc_count(mm) == expected * 16
    finally:
      mm.close()


def test_bytearray_and_mmap_inplace_mutation_and_readonly_guard():
  @pk.jit
  def to_upper(buf):
    for i in range(len(buf)):
      b = buf[i]
      if b >= ord("a") and b <= ord("z"):
        buf[i] = b - np.uint8(32)

  ba = bytearray(b"acgtN123")
  to_upper(ba)
  assert ba == bytearray(b"ACGTN123")

  mv = memoryview(ba)
  to_upper(mv)
  assert ba == bytearray(b"ACGTN123")

  with pytest.raises(ValueError, match="read-only"):
    to_upper(b"acgt")

  with tempfile.TemporaryFile() as f:
    f.write(b"acgt")
    f.flush()
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    try:
      to_upper(mm)
      assert mm[:] == b"ACGT"
    finally:
      mm.close()


def test_bytes_literal_and_ord_inside_jit():
  @pk.jit
  def complement(seq):
    lut = b"ACGT"
    out = np.empty(len(seq), dtype=np.uint8)
    for i in range(len(seq)):
      c = seq[i]
      if c == lut[0]:
        out[i] = ord("T")
      elif c == lut[1]:
        out[i] = ord("G")
      elif c == lut[2]:
        out[i] = ord("C")
      elif c == lut[3]:
        out[i] = ord("A")
      else:
        out[i] = ord("N")
    return out

  res = complement(b"ACGNT")
  assert bytes(res) == b"TGCNA"


# ---------------------------------------------------------------------------
# 2. Bitwise intrinsics & bfloat16 conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64])
def test_bitops_all_widths_and_zero(dtype):
  bits = np.dtype(dtype).itemsize * 8
  mask = (1 << bits) - 1
  if np.issubdtype(dtype, np.unsignedinteger):
    vals = np.array([0, 1, 2, 7, 1 << (bits - 1), mask, 0x5555555555555555 & mask], dtype=dtype)
  else:
    vals = np.array([0, 1, -1, 2, -2, -(1 << (bits - 1)), (1 << (bits - 1)) - 1], dtype=dtype)

  @pk.jit
  def run_ops(x):
    return (
      pk.popcount(x),
      pk.clz(x),
      pk.ctz(x),
      pk.rotl(x, 3),
      pk.rotr(x, 3),
      pk.rotl(x, 0),
      pk.rotr(x, bits + 5),
      pk.bswap(x),
    )

  pc, lz, tz, rl, rr, rl0, rr_wrap, bs = run_ops(vals)
  np.testing.assert_array_equal(pc, pk.popcount(vals))
  np.testing.assert_array_equal(lz, pk.clz(vals))
  np.testing.assert_array_equal(tz, pk.ctz(vals))
  np.testing.assert_array_equal(rl, pk.rotl(vals, 3))
  np.testing.assert_array_equal(rr, pk.rotr(vals, 3))
  np.testing.assert_array_equal(rl0, vals)
  np.testing.assert_array_equal(rr_wrap, pk.rotr(vals, 5))
  np.testing.assert_array_equal(bs, vals.byteswap())
  # Explicit zero-safety check
  assert lz[0] == bits
  assert tz[0] == bits
  assert pc[0] == 0


def test_numpy_bitwise_count_and_int_bit_count():
  if not hasattr(np, "bitwise_count"):
    pytest.skip("NumPy < 2.0 has no bitwise_count")

  @pk.jit
  def np_bc(x):
    return np.bitwise_count(x)

  for dt in (np.uint8, np.uint64, np.int8, np.int32):
    arr = np.array([0, 1, 5, 127], dtype=dt)
    if np.issubdtype(dt, np.signedinteger):
      arr = np.concatenate([arr, np.array([-1, -128], dtype=dt)])
    res = np_bc(arr)
    ref = np.bitwise_count(arr)
    assert res.dtype == ref.dtype
    np.testing.assert_array_equal(res, ref)

  @pk.jit
  def scalar_bc(a, b):
    return a.bit_count() + b.bit_count()

  assert scalar_bc(0b101101, -7) == (0b101101).bit_count() + (-7).bit_count()


def test_bitops_hamming_tanimoto_and_bf16():
  @pk.jit
  def hamming(a, b):
    total = 0
    for i in range(len(a)):
      total += pk.popcount(a[i] ^ b[i])
    return total

  rng = np.random.default_rng(0)
  a = rng.integers(0, 1 << 63, size=32, dtype=np.uint64)
  b = rng.integers(0, 1 << 63, size=32, dtype=np.uint64)
  expected = sum(int(x ^ y).bit_count() for x, y in zip(a, b))
  assert hamming(a, b) == expected

  @pk.jit
  def roundtrip_bf16(x):
    return dsl.bf16_to_f32(dsl.f32_to_bf16(x))

  f = np.array([0.0, -0.0, 1.0, -3.5, 1.00390625, 1.005859375, np.inf, -np.inf, np.nan], dtype=np.float32)
  out = roundtrip_bf16(f)
  ref = pk.bf16_to_f32(pk.f32_to_bf16(f))
  np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
  finite = ~np.isnan(f)
  np.testing.assert_array_equal(out[finite], ref[finite])
  # Round-to-nearest-even check on midpoint: 1.0 + 2^-8 = 1.00390625 -> rounds to even 1.0
  assert out[4] == np.float32(1.0)


# ---------------------------------------------------------------------------
# 3. IntEnum / IntFlag + match/case statements
# ---------------------------------------------------------------------------


def test_intenum_and_match_case_dna_encoding():
  @pk.jit
  def encode_dna(seq, unknown_val):
    out = np.empty(len(seq), dtype=np.uint8)
    for i in range(len(seq)):
      match seq[i]:
        case 65 | 97:
          code = Base.A
        case 67 | 99:
          code = Base.C
        case 71 | 103:
          code = Base.G
        case 84 | 116:
          code = Base.T
        case _:
          code = unknown_val
      out[i] = code
    return out

  res = encode_dna(b"AaCcGgTtNx", Base.N)
  np.testing.assert_array_equal(res, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.uint8))


def test_match_case_guards_sequences_classes_and_control_flow():
  @pk.jit
  def classify_hit(hit, flag):
    match hit:
      case Hit(pos=p, score=s, strand=1) if s > 50.0:
        return p * 10 + 1
      case Hit(p, s, -1) if s > 50.0:
        return p * 10 + 2
      case Hit(score=s) if (flag & CigarFlag.MATCH) != 0 and s > 0.0:
        return 100
      case _:
        return -1

  assert classify_hit(Hit(7, 80.0, 1), CigarFlag.MATCH) == 71
  assert classify_hit(Hit(7, 80.0, -1), CigarFlag.MATCH) == 72
  assert classify_hit(Hit(7, 10.0, 1), CigarFlag.MATCH) == 100
  assert classify_hit(Hit(7, 10.0, 1), CigarFlag.DEL) == -1

  @pk.jit
  def match_or_bind_and_loop(xs):
    acc = 0
    for i in range(len(xs)):
      pair = (i % 3, xs[i])
      match pair:
        case (0, v) | (2, v) if v > 0:
          acc += v
        case (1, v) if v < 0:
          break
        case _:
          continue
    return acc

  arr = np.array([5, 10, 7, 3, -1, 100], dtype=np.int64)
  assert match_or_bind_and_loop(arr) == 5 + 7 + 3


# ---------------------------------------------------------------------------
# 4. In-kernel frozen @dataclass construction & dataclasses.replace / _replace
# ---------------------------------------------------------------------------


def test_in_kernel_dataclass_construct_replace_and_compare():
  @pk.jit
  def best_hit(scores):
    best = Hit(pos=-1, score=-1e9)
    for i in range(len(scores)):
      if scores[i] > best.score:
        best = dataclasses.replace(best, pos=i, score=scores[i])
    return best

  res = best_hit(np.array([1.5, 9.25, 4.0, 9.25], dtype=np.float64))
  assert isinstance(res, Hit)
  assert res == Hit(pos=1, score=9.25, strand=1)

  @pk.jit
  def update_namedtuple(p, new_dist):
    q = p._replace(dist=new_dist)
    return q, (q == p)

  q, same = update_namedtuple(PairRec(3, 5.0), 2.5)
  assert isinstance(q, PairRec)
  assert q == PairRec(3, 2.5)
  assert not same

  @pk.jit
  def cmp_hits(a, b):
    return a == b, a != b

  eq_res, ne_res = cmp_hits(Hit(1, 2.0), Hit(1, 2.0))
  assert bool(eq_res) is True
  assert bool(ne_res) is False


# ---------------------------------------------------------------------------
# 5. Structured NumPy array boundary views
# ---------------------------------------------------------------------------


def test_structured_numpy_array_views_and_mutation():
  atom_dt = np.dtype([("xyz", np.float32, (3,)), ("charge", np.float32), ("elem", np.int32)])
  atoms = np.zeros(8, dtype=atom_dt)
  atoms["xyz"] = np.arange(24, dtype=np.float32).reshape(8, 3)
  atoms["charge"] = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
  atoms["elem"] = np.array([6, 7, 8, 6, 1, 6, 8, 7], dtype=np.int32)

  @pk.jit
  def center_of_charge(a):
    total = np.float32(0.0)
    cx = np.float32(0.0)
    for i in range(len(a)):
      if a["elem"][i] == 6:
        total += a["charge"][i]
        cx += a["xyz"][i, 0] * a["charge"][i]
    return cx, total

  cx, total = center_of_charge(atoms)
  mask = atoms["elem"] == 6
  np.testing.assert_allclose(total, atoms["charge"][mask].sum(), rtol=1e-6)
  np.testing.assert_allclose(cx, (atoms["xyz"][mask, 0] * atoms["charge"][mask]).sum(), rtol=1e-6)

  # Iteration and row access + slicing + returning a structured array
  @pk.jit
  def heavy_charge_sum(a):
    s = np.float32(0.0)
    for row in a:
      if row.elem > 1:
        s += row["charge"]
    return s

  np.testing.assert_allclose(heavy_charge_sum(atoms), atoms["charge"][atoms["elem"] > 1].sum(), rtol=1e-6)

  @pk.jit
  def slice_atoms(a):
    return a[1:5]

  sliced = slice_atoms(atoms)
  assert sliced.dtype == atom_dt
  np.testing.assert_array_equal(sliced, atoms[1:5])

  # Packed unaligned struct mutation with automatic writeback
  packed_dt = np.dtype([("id", np.int64), ("flag", np.uint8)], align=False)
  recs = np.zeros(5, dtype=packed_dt)
  recs["id"] = np.arange(5, dtype=np.int64)

  @pk.jit
  def mark_even(r):
    for i in range(len(r)):
      if r["id"][i] % 2 == 0:
        r["flag"][i] = np.uint8(1)

  mark_even(recs)
  np.testing.assert_array_equal(recs["flag"], np.array([1, 0, 1, 0, 1], dtype=np.uint8))


# ---------------------------------------------------------------------------
# 6. GPU verification (CUDA simulator + native Metal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_bitops_and_bytes_on_gpu(backend):
  if backend == "cuda" and not os.environ.get("PARAKEET_CUDA_SIMULATE"):
    try:
      pk.available_backends()
    except Exception:
      pass
    if pk.available_backends().get("cuda") != "ok":
      pytest.skip("CUDA not available")
  if backend == "metal" and pk.available_backends().get("metal") != "ok":
    pytest.skip("Metal not available")

  @pk.jit(backend=backend)
  def gpu_bit_kernel(a, b):
    return pk.popcount(a ^ b) + pk.clz(a) + pk.ctz(b)

  @pk.jit(backend=backend)
  def gpu_bf16_kernel(x):
    return pk.bf16_to_f32(pk.f32_to_bf16(x))

  @pk.jit(backend=backend)
  def gpu_gc_mask(seq):
    return (seq == np.uint8(ord("G"))) | (seq == np.uint8(ord("C")))

  n = 2048
  rng = np.random.default_rng(7)
  a = rng.integers(0, 1 << 32, size=n, dtype=np.uint32)
  b = rng.integers(0, 1 << 32, size=n, dtype=np.uint32)
  a[0] = 0
  b[0] = 0
  f = rng.standard_normal(n).astype(np.float32)
  dna = b"ACGT" * (n // 4)

  with pk.options(device_fallback="error", device_threshold=1):
    res_bit = gpu_bit_kernel(a, b)
    res_bf16 = gpu_bf16_kernel(f)
    res_gc = gpu_gc_mask(dna)

  np.testing.assert_array_equal(res_bit, pk.popcount(a ^ b) + pk.clz(a) + pk.ctz(b))
  np.testing.assert_array_equal(res_bf16, pk.bf16_to_f32(pk.f32_to_bf16(f)))
  np.testing.assert_array_equal(
    res_gc,
    np.frombuffer(dna, dtype=np.uint8) % 2 != 0
    if False
    else (np.frombuffer(dna, dtype=np.uint8) == ord("G")) | (np.frombuffer(dna, dtype=np.uint8) == ord("C")),
  )
