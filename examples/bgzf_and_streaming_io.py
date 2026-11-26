"""Compiled block-parallel BGZF (RFC-1951 DEFLATE) inflate + overlapped CPU I/O pipeline.

Demonstrates both sides of the I/O and decompression boundary in Parakeet:

1. **Fits inside ``@par.jit``: block-parallel BGZF DEFLATE inflate.**
   Bioinformatics formats (``.bam``, ``.vcf.gz``, ``.fastq.bgz``) use BGZF
   (Blocked GNU Zip Format), which concatenates independent ``<= 64 KiB``
   RFC-1951 DEFLATE blocks whose compressed bounds (``BSIZE``) and exact
   uncompressed sizes (``ISIZE``) live in each block's header and trailer.
   After an ``O(#blocks)`` header scan, ``inflate_bgzf_blocks`` decodes all
   blocks in parallel with a compiled RFC-1951 decoder (stored, fixed Huffman,
   and dynamic Huffman with a 9-bit fast LUT + canonical fallback + LZ77).

2. **Overlapped CPU I/O + GIL-free Parakeet compute (``stream_fastq_overlapped``).**
   For general streaming files (including single-stream gzip/zstd that escape
   Parakeet's data-parallel hull), compiled Parakeet functions release the GIL,
   so a background producer thread decompressing chunks via ``zlib`` into a
   2-slot ring overlaps cleanly with Parakeet multicore FASTQ kernels.

Run:
    python examples/bgzf_and_streaming_io.py
"""

from __future__ import annotations

import gzip
import queue
import struct
import threading
import time
import zlib

import numpy as np

import parakeet as par

# ---------------------------------------------------------------------------
# 1. Standard BGZF writer & O(#blocks) header scanner (SAM/htslib format)
# ---------------------------------------------------------------------------

_BGZF_MAX_BLOCK = 65280  # <= 64 KiB uncompressed per BGZF block (htslib standard)


def make_bgzf_bytes(raw: bytes, block_size: int = _BGZF_MAX_BLOCK, level: int = 6) -> bytes:
  """Compress ``raw`` into standard BGZF bytes (readable by ``gzip.decompress`` / ``bgzip``)."""
  parts: list[bytes] = []
  n = len(raw)
  pos = 0
  while pos < n or (n == 0 and pos == 0):
    chunk = raw[pos : pos + block_size]
    pos += len(chunk)
    co = zlib.compressobj(level, zlib.DEFLATED, -15)
    cdata = co.compress(chunk) + co.flush()
    bsize = 18 + len(cdata) + 8 - 1  # total BGZF block size minus 1
    # RFC-1952 gzip header with FEXTRA (0x04), XLEN=6, subfield 'BC' (SLEN=2, BSIZE)
    header = struct.pack("<BBBBIBBHBBHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, bsize)
    trailer = struct.pack("<II", zlib.crc32(chunk) & 0xFFFFFFFF, len(chunk) & 0xFFFFFFFF)
    parts.append(header + cdata + trailer)
    if n == 0:
      break
  # Standard empty 28-byte BGZF EOF marker
  eof_header = struct.pack("<BBBBIBBHBBHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, 27)
  parts.append(eof_header + b"\x03\x00" + struct.pack("<II", 0, 0))
  return b"".join(parts)


def index_bgzf_blocks(bgzf_data: bytes):
  """Scan BGZF block headers in O(#blocks) to locate raw DEFLATE payloads and output slices."""
  buf = memoryview(bgzf_data)
  n = len(buf)
  comp_starts: list[int] = []
  comp_ends: list[int] = []
  uncomp_lens: list[int] = []
  pos = 0
  while pos + 18 <= n:
    if buf[pos] != 31 or buf[pos + 1] != 139 or buf[pos + 2] != 8:
      raise ValueError(f"invalid BGZF/gzip magic at byte {pos}")
    flg = buf[pos + 3]
    if not (flg & 0x04):
      raise ValueError(f"gzip member at byte {pos} is missing BGZF FEXTRA header")
    xlen = struct.unpack_from("<H", buf, pos + 10)[0]
    bsize = -1
    ex_pos = pos + 12
    ex_end = ex_pos + xlen
    while ex_pos + 4 <= ex_end:
      si1, si2, slen = struct.unpack_from("<BBH", buf, ex_pos)
      if si1 == 66 and si2 == 67 and slen == 2:
        bsize = struct.unpack_from("<H", buf, ex_pos + 4)[0] + 1
        break
      ex_pos += 4 + slen
    if bsize < 26 or pos + bsize > n:
      raise ValueError(f"invalid BGZF BSIZE={bsize} at byte {pos}")
    c_start = ex_end
    c_end = pos + bsize - 8
    isize = struct.unpack_from("<I", buf, c_end + 4)[0]
    if isize > 0:
      comp_starts.append(c_start)
      comp_ends.append(c_end)
      uncomp_lens.append(isize)
    pos += bsize

  uncomp_offsets = np.zeros(len(uncomp_lens) + 1, dtype=np.int64)
  if uncomp_lens:
    np.cumsum(np.array(uncomp_lens, dtype=np.int64), out=uncomp_offsets[1:])
  return (
    np.frombuffer(bgzf_data, dtype=np.uint8),
    np.array(comp_starts, dtype=np.int64),
    np.array(comp_ends, dtype=np.int64),
    uncomp_offsets,
  )


# ---------------------------------------------------------------------------
# 2. Compiled RFC-1951 DEFLATE inflater (parallel across BGZF blocks)
# ---------------------------------------------------------------------------

# RFC-1951 static lookup tables (lifted once into constant C arrays)
_CL_ORDER = np.array([16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15], dtype=np.int64)
_LEN_BASE = np.array(
  [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258],
  dtype=np.int64,
)
_LEN_EXTRA = np.array(
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0],
  dtype=np.int64,
)
_DIST_BASE = np.array(
  [
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    13,
    17,
    25,
    33,
    49,
    65,
    97,
    129,
    193,
    257,
    385,
    513,
    769,
    1025,
    1537,
    2049,
    3073,
    4097,
    6145,
    8193,
    12289,
    16385,
    24577,
  ],
  dtype=np.int64,
)
_DIST_EXTRA = np.array(
  [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13],
  dtype=np.int64,
)

# Precomputed bit-reverse table for 9-bit fast Huffman lookup
_REV9 = np.array([int(f"{i:09b}"[::-1], 2) for i in range(512)], dtype=np.int64)


def _build_huffman(lengths, n_syms, lut, count_tbl, first_code, first_sym, sorted_syms):
  """Build canonical Huffman tables + 9-bit LSB-first fast lookup table."""
  for i in range(16):
    count_tbl[i] = 0
  for s in range(n_syms):
    ln = lengths[s]
    if ln > 0:
      count_tbl[ln] += 1

  # Compute canonical first code and sorted symbol offsets per bit length
  code = 0
  sym_off = 0
  next_code = np.zeros(16, dtype=np.int64)
  next_off = np.zeros(16, dtype=np.int64)
  for bits in range(1, 16):
    code = (code + count_tbl[bits - 1]) << 1
    first_code[bits] = code
    next_code[bits] = code
    first_sym[bits] = sym_off
    next_off[bits] = sym_off
    sym_off += count_tbl[bits]

  for i in range(512):
    lut[i] = 0

  for s in range(n_syms):
    ln = lengths[s]
    if ln > 0:
      idx = next_off[ln]
      sorted_syms[idx] = s
      next_off[ln] = idx + 1
      c = next_code[ln]
      next_code[ln] = c + 1
      if ln <= 9:
        # Fill all 9-bit LSB suffixes whose low `ln` bits equal bit-reversed `c`
        rev = _REV9[c << (9 - ln)]
        step = 1 << ln
        packed = (s << 4) | ln
        slot = rev
        while slot < 512:
          lut[slot] = packed
          slot += step


def _decode_symbol(data, c_end, pos, bitbuf, nbits, lut, count_tbl, first_code, first_sym, sorted_syms):
  """Decode one Huffman symbol using 9-bit fast LUT + canonical fallback for 10..15 bits."""
  while nbits < 15 and pos < c_end:
    bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
    pos += 1
    nbits += 8

  entry = lut[int(bitbuf & np.uint64(511))]
  if entry > 0:
    sym = entry >> 4
    ln = entry & 15
    return sym, pos, bitbuf >> np.uint64(ln), nbits - ln

  # Rare 10..15-bit symbol: walk canonical code lengths MSB-first
  code = 0
  for bits in range(1, 16):
    code = (code << 1) | int(bitbuf & np.uint64(1))
    bitbuf >>= np.uint64(1)
    nbits -= 1
    cnt = count_tbl[bits]
    fc = first_code[bits]
    if cnt > 0 and code >= fc and (code - fc) < cnt:
      return sorted_syms[first_sym[bits] + (code - fc)], pos, bitbuf, nbits
  return -1, pos, bitbuf, nbits


def _inflate_one_block(data, c_start, c_end, out, u_start):
  """Inflate one raw RFC-1951 DEFLATE stream ``data[c_start:c_end]`` into ``out[u_start:]``."""
  pos = c_start
  out_pos = u_start
  bitbuf = np.uint64(0)
  nbits = 0

  # Per-block reusable Huffman tables (small fixed sizes, L1-resident)
  ll_lens = np.zeros(288, dtype=np.int64)
  d_lens = np.zeros(32, dtype=np.int64)
  cl_lens = np.zeros(19, dtype=np.int64)
  all_lens = np.zeros(320, dtype=np.int64)

  ll_lut = np.zeros(512, dtype=np.int64)
  ll_cnt = np.zeros(16, dtype=np.int64)
  ll_fc = np.zeros(16, dtype=np.int64)
  ll_fs = np.zeros(16, dtype=np.int64)
  ll_syms = np.zeros(288, dtype=np.int64)

  d_lut = np.zeros(512, dtype=np.int64)
  d_cnt = np.zeros(16, dtype=np.int64)
  d_fc = np.zeros(16, dtype=np.int64)
  d_fs = np.zeros(16, dtype=np.int64)
  d_syms = np.zeros(32, dtype=np.int64)

  cl_lut = np.zeros(512, dtype=np.int64)
  cl_cnt = np.zeros(16, dtype=np.int64)
  cl_fc = np.zeros(16, dtype=np.int64)
  cl_fs = np.zeros(16, dtype=np.int64)
  cl_syms = np.zeros(19, dtype=np.int64)

  bfinal = 0
  while bfinal == 0:
    while nbits < 3 and pos < c_end:
      bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
      pos += 1
      nbits += 8
    bfinal = int(bitbuf & np.uint64(1))
    btype = int((bitbuf >> np.uint64(1)) & np.uint64(3))
    bitbuf >>= np.uint64(3)
    nbits -= 3

    if btype == 0:
      # Stored uncompressed block: discard partial byte, read LEN / NLEN, copy raw bytes
      drop = nbits & 7
      bitbuf >>= np.uint64(drop)
      nbits -= drop
      while nbits < 32 and pos < c_end:
        bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
        pos += 1
        nbits += 8
      blk_len = int(bitbuf & np.uint64(0xFFFF))
      bitbuf >>= np.uint64(32)
      nbits -= 32
      # Drain any remaining buffered whole bytes first, then copy directly from data
      while blk_len > 0 and nbits >= 8:
        out[out_pos] = np.uint8(bitbuf & np.uint64(0xFF))
        out_pos += 1
        bitbuf >>= np.uint64(8)
        nbits -= 8
        blk_len -= 1
      for _ in range(blk_len):
        out[out_pos] = data[pos]
        out_pos += 1
        pos += 1
      continue

    if btype == 1:
      # Fixed RFC-1951 Huffman tables
      for i in range(144):
        ll_lens[i] = 8
      for i in range(144, 256):
        ll_lens[i] = 9
      for i in range(256, 280):
        ll_lens[i] = 7
      for i in range(280, 288):
        ll_lens[i] = 8
      for i in range(32):
        d_lens[i] = 5
      _build_huffman(ll_lens, 288, ll_lut, ll_cnt, ll_fc, ll_fs, ll_syms)
      _build_huffman(d_lens, 32, d_lut, d_cnt, d_fc, d_fs, d_syms)
    else:
      # Dynamic RFC-1951 Huffman tables
      while nbits < 14 and pos < c_end:
        bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
        pos += 1
        nbits += 8
      hlit = int(bitbuf & np.uint64(31)) + 257
      hdist = int((bitbuf >> np.uint64(5)) & np.uint64(31)) + 1
      hclen = int((bitbuf >> np.uint64(10)) & np.uint64(15)) + 4
      bitbuf >>= np.uint64(14)
      nbits -= 14

      for i in range(19):
        cl_lens[i] = 0
      for i in range(hclen):
        while nbits < 3 and pos < c_end:
          bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
          pos += 1
          nbits += 8
        cl_lens[_CL_ORDER[i]] = int(bitbuf & np.uint64(7))
        bitbuf >>= np.uint64(3)
        nbits -= 3
      _build_huffman(cl_lens, 19, cl_lut, cl_cnt, cl_fc, cl_fs, cl_syms)

      total_codes = hlit + hdist
      idx = 0
      while idx < total_codes:
        sym, pos, bitbuf, nbits = _decode_symbol(data, c_end, pos, bitbuf, nbits, cl_lut, cl_cnt, cl_fc, cl_fs, cl_syms)
        if sym < 16:
          all_lens[idx] = sym
          idx += 1
        elif sym == 16:
          while nbits < 2 and pos < c_end:
            bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
            pos += 1
            nbits += 8
          rep = int(bitbuf & np.uint64(3)) + 3
          bitbuf >>= np.uint64(2)
          nbits -= 2
          prev_l = all_lens[idx - 1]
          for _ in range(rep):
            all_lens[idx] = prev_l
            idx += 1
        elif sym == 17:
          while nbits < 3 and pos < c_end:
            bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
            pos += 1
            nbits += 8
          rep = int(bitbuf & np.uint64(7)) + 3
          bitbuf >>= np.uint64(3)
          nbits -= 3
          for _ in range(rep):
            all_lens[idx] = 0
            idx += 1
        else:
          while nbits < 7 and pos < c_end:
            bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
            pos += 1
            nbits += 8
          rep = int(bitbuf & np.uint64(127)) + 11
          bitbuf >>= np.uint64(7)
          nbits -= 7
          for _ in range(rep):
            all_lens[idx] = 0
            idx += 1

      for i in range(hlit):
        ll_lens[i] = all_lens[i]
      for i in range(hdist):
        d_lens[i] = all_lens[hlit + i]
      _build_huffman(ll_lens, hlit, ll_lut, ll_cnt, ll_fc, ll_fs, ll_syms)
      _build_huffman(d_lens, hdist, d_lut, d_cnt, d_fc, d_fs, d_syms)

    # Hot Huffman + LZ77 decode loop
    while True:
      sym, pos, bitbuf, nbits = _decode_symbol(data, c_end, pos, bitbuf, nbits, ll_lut, ll_cnt, ll_fc, ll_fs, ll_syms)
      if sym < 256:
        out[out_pos] = np.uint8(sym)
        out_pos += 1
      elif sym == 256:
        break
      else:
        li = sym - 257
        length = _LEN_BASE[li]
        lex = _LEN_EXTRA[li]
        if lex > 0:
          while nbits < lex and pos < c_end:
            bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
            pos += 1
            nbits += 8
          length += int(bitbuf & ((np.uint64(1) << np.uint64(lex)) - np.uint64(1)))
          bitbuf >>= np.uint64(lex)
          nbits -= lex

        dsym, pos, bitbuf, nbits = _decode_symbol(data, c_end, pos, bitbuf, nbits, d_lut, d_cnt, d_fc, d_fs, d_syms)
        dist = _DIST_BASE[dsym]
        dex = _DIST_EXTRA[dsym]
        if dex > 0:
          while nbits < dex and pos < c_end:
            bitbuf |= np.uint64(data[pos]) << np.uint64(nbits)
            pos += 1
            nbits += 8
          dist += int(bitbuf & ((np.uint64(1) << np.uint64(dex)) - np.uint64(1)))
          bitbuf >>= np.uint64(dex)
          nbits -= dex

        src = out_pos - dist
        for k in range(length):
          out[out_pos + k] = out[src + k]
        out_pos += length


@par.jit
def inflate_bgzf_blocks(data, comp_starts, comp_ends, uncomp_offsets):
  """Decompress all BGZF blocks in parallel into one flat ``uint8`` array."""
  n_blocks = len(comp_starts)
  total_bytes = uncomp_offsets[n_blocks]
  out = np.empty(total_bytes, dtype=np.uint8)
  par.parfor(
    lambda b: _inflate_one_block(data, comp_starts[b], comp_ends[b], out, uncomp_offsets[b]),
    n_blocks,
  )
  return out


# ---------------------------------------------------------------------------
# 3. Compiled FASTQ slab QC + Overlapped double-buffered CPU I/O orchestrator
# ---------------------------------------------------------------------------

GC_BASE = {"G": 1, "C": 1, "g": 1, "c": 1}


@par.jit
def fastq_slab_qc(fastq_bytes):
  """Parse a raw FASTQ byte slab in parallel, returning ``(n_reads, total_bases, gc_bases, phred_sum)``."""
  lines = par.split_bytes(fastq_bytes, ord("\n"), skip_empty=True)
  n_reads = len(lines) // 4

  def read_stats(r):
    seq = lines[4 * r + 1]
    qual = lines[4 * r + 3]
    gc = np.sum(GC_BASE[seq].astype(np.int64))
    qsum = np.sum(qual.astype(np.int64) - 33)
    return len(seq), gc, qsum

  lens, gcs, qsums = par.map(read_stats, np.arange(n_reads))
  return n_reads, np.sum(lens), np.sum(gcs), np.sum(qsums)


def stream_fastq_overlapped(bgzf_data: bytes, blocks_per_slab: int = 16):
  """Double-buffered streaming pipeline: background zlib reader overlaps with GIL-free Parakeet QC."""
  _, c_starts, c_ends, _ = index_bgzf_blocks(bgzf_data)
  n_blocks = len(c_starts)
  raw_mv = memoryview(bgzf_data)
  slabs: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)

  def producer():
    carry = b""
    for lo in range(0, n_blocks, blocks_per_slab):
      hi = min(lo + blocks_per_slab, n_blocks)
      chunk = carry + b"".join(zlib.decompress(raw_mv[int(c_starts[b]) : int(c_ends[b])], -15) for b in range(lo, hi))
      if hi < n_blocks:
        # Keep only complete 4-line FASTQ records in each slab; carry the partial tail forward
        n_nl = chunk.count(b"\n")
        keep_nl = (n_nl // 4) * 4
        if keep_nl == 0:
          carry = chunk
          continue
        cut = len(chunk)
        drop = n_nl - keep_nl
        # Strip trailing partial line after the last newline
        last_nl = chunk.rfind(b"\n")
        cut = last_nl + 1
        while drop > 0:
          last_nl = chunk.rfind(b"\n", 0, cut - 1)
          cut = last_nl + 1
          drop -= 1
        carry = chunk[cut:]
        chunk = chunk[:cut]
      else:
        carry = b""
      if chunk:
        slabs.put(np.frombuffer(chunk, dtype=np.uint8))
    slabs.put(None)

  t = threading.Thread(target=producer, daemon=True)
  t.start()

  total_reads = total_bases = total_gc = total_q = 0
  while True:
    slab = slabs.get()
    if slab is None:
      break
    nr, nb, ngc, nq = fastq_slab_qc(slab, _backend="multicore")
    total_reads += int(nr)
    total_bases += int(nb)
    total_gc += int(ngc)
    total_q += int(nq)
  t.join()
  return total_reads, total_bases, total_gc, total_q


def make_synthetic_fastq(n_reads: int = 20_000, read_len: int = 150, seed: int = 0) -> bytes:
  rng = np.random.default_rng(seed)
  alph = np.frombuffer(b"ACGT", dtype=np.uint8)
  seqs = alph[rng.integers(0, 4, size=(n_reads, read_len))]
  quals = rng.integers(33 + 20, 33 + 41, size=(n_reads, read_len), dtype=np.uint8)
  lines: list[bytes] = []
  for i in range(n_reads):
    lines.append(f"@read_{i}\n".encode() + seqs[i].tobytes() + b"\n+\n" + quals[i].tobytes() + b"\n")
  return b"".join(lines)


def main(n_reads: int = 20_000):
  raw_fastq = make_synthetic_fastq(n_reads=n_reads, read_len=150)
  bgzf = make_bgzf_bytes(raw_fastq, level=6)
  assert gzip.decompress(bgzf) == raw_fastq

  data, cs, ce, uo = index_bgzf_blocks(bgzf)
  raw_mb = len(raw_fastq) / 1e6
  comp_mb = len(bgzf) / 1e6

  # 1. Verify bit-for-bit exactness of compiled parallel BGZF inflate
  got_raw = inflate_bgzf_blocks(data, cs, ce, uo, _backend="multicore")
  assert got_raw.tobytes() == raw_fastq, "BGZF inflate mismatch!"

  # Warmup both kernels
  ref_qc = fastq_slab_qc(got_raw, _backend="multicore")
  ov_qc = stream_fastq_overlapped(bgzf, blocks_per_slab=16)
  assert tuple(int(x) for x in ref_qc) == ov_qc, (ref_qc, ov_qc)

  # Benchmark raw decompression throughput
  t0 = time.perf_counter()
  gzip.decompress(bgzf)
  t_gzip = (time.perf_counter() - t0) * 1e3

  inflate_bgzf_blocks(data, cs, ce, uo, _backend="c")
  t0 = time.perf_counter()
  inflate_bgzf_blocks(data, cs, ce, uo, _backend="c")
  t_pk_1t = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  inflate_bgzf_blocks(data, cs, ce, uo, _backend="multicore")
  t_pk_mc = (time.perf_counter() - t0) * 1e3

  # Benchmark end-to-end decompress + FASTQ QC (serial gzip vs overlapped vs Parakeet BGZF)
  t0 = time.perf_counter()
  buf_seq = np.frombuffer(gzip.decompress(bgzf), dtype=np.uint8)
  fastq_slab_qc(buf_seq, _backend="multicore")
  t_e2e_seq = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  stream_fastq_overlapped(bgzf, blocks_per_slab=16)
  t_e2e_ov = (time.perf_counter() - t0) * 1e3

  t0 = time.perf_counter()
  buf_pk = inflate_bgzf_blocks(data, cs, ce, uo, _backend="multicore")
  fastq_slab_qc(buf_pk, _backend="multicore")
  t_e2e_pk = (time.perf_counter() - t0) * 1e3

  print(f"FASTQ BGZF ({n_reads:,} reads, {raw_mb:.2f} MB raw / {comp_mb:.2f} MB BGZF, {len(cs)} blocks):")
  print(f"  Raw inflate  | Python gzip (C zlib 1T)   : {t_gzip:6.2f} ms ({raw_mb / (t_gzip / 1e3):6.1f} MB/s)")
  print(f"  Raw inflate  | Parakeet BGZF (c, 1T)     : {t_pk_1t:6.2f} ms ({raw_mb / (t_pk_1t / 1e3):6.1f} MB/s)")
  print(f"  Raw inflate  | Parakeet BGZF (multicore) : {t_pk_mc:6.2f} ms ({raw_mb / (t_pk_mc / 1e3):6.1f} MB/s)")
  print(f"  End-to-end   | Sequential gzip + QC      : {t_e2e_seq:6.2f} ms ({raw_mb / (t_e2e_seq / 1e3):6.1f} MB/s)")
  print(f"  End-to-end   | Overlapped zlib + QC      : {t_e2e_ov:6.2f} ms ({raw_mb / (t_e2e_ov / 1e3):6.1f} MB/s)")
  print(f"  End-to-end   | Parakeet BGZF + QC        : {t_e2e_pk:6.2f} ms ({raw_mb / (t_e2e_pk / 1e3):6.1f} MB/s)")


if __name__ == "__main__":
  main()
