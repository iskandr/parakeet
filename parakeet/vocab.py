"""Compact, Pythonic categorical encoders for sequences and text.

Provides two closely related abstractions backed by 256-entry lookup tables:

- :class:`Alphabet`: encodes ASCII character sequences (DNA, RNA, proteins,
  bitmasks) into compact ``uint8`` code arrays.  Works identically in Python
  and inside ``@par.jit`` via ``DNA.encode(seq)`` or ``DNA[seq]`` (compiling to
  a single L1-resident LUT gather with zero ``IntEnum`` or dict boilerplate).
- :class:`Vocab`: character-level vocabulary for language models, built from a
  corpus with ``Vocab.from_text(text)`` and supporting ``encode`` / ``decode``.
"""

from __future__ import annotations

import numpy as np

from .ragged import RaggedArray


def _to_byte_array(seq):
  if isinstance(seq, str):
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
  if isinstance(seq, (bytes, bytearray, memoryview)):
    return np.frombuffer(seq, dtype=np.uint8)
  if isinstance(seq, list):
    raw = [s.encode("ascii") if isinstance(s, str) else bytes(s) for s in seq]
    lens = {len(b) for b in raw}
    if len(lens) == 1:
      n = next(iter(lens))
      return np.frombuffer(b"".join(raw), dtype=np.uint8).reshape(len(raw), n).copy()
    return RaggedArray.from_sequences([np.frombuffer(b, dtype=np.uint8).copy() for b in raw])
  return np.asarray(seq, dtype=np.uint8)


class Alphabet:
  """Fixed ASCII categorical mapping compiled as an L1-resident 256-byte LUT.

  Examples::

      DNA = par.Alphabet("ACGT")
      codes = DNA.encode("ACGTACGT")        # array([0, 1, 2, 3, 0, 1, 2, 3], dtype=uint8)
      codes = DNA[read_bytes]               # also works inside @par.jit!
      GC = par.Alphabet.mask("GC", ignore_case=True)
      gc_frac = np.mean(GC[read_bytes])
  """

  __parakeet_record__ = True

  def __init__(
    self,
    symbols: str,
    *,
    ignore_case: bool = True,
    unknown: int | None = None,
    mapping: dict[str, int] | None = None,
  ):
    if unknown is None:
      unknown = len(symbols) if mapping is None else 0
    lut = np.full(256, int(unknown), dtype=np.uint8)
    if mapping is not None:
      for ch, code in mapping.items():
        if len(ch) != 1 or ord(ch) >= 128:
          raise ValueError(f"Alphabet keys must be single ASCII characters, got {ch!r}")
        lut[ord(ch)] = int(code)
        if ignore_case:
          lut[ord(ch.lower())] = int(code)
          lut[ord(ch.upper())] = int(code)
    else:
      for idx, ch in enumerate(symbols):
        if ord(ch) >= 128:
          raise ValueError(f"Alphabet symbols must be ASCII, got {ch!r}")
        lut[ord(ch)] = idx
        if ignore_case:
          lut[ord(ch.lower())] = idx
          lut[ord(ch.upper())] = idx
    self.lut = lut
    self.symbols = symbols
    self.unknown = int(unknown)

  @classmethod
  def mask(cls, symbols: str, *, ignore_case: bool = True) -> Alphabet:
    """Create a 0/1 indicator alphabet (1 for characters in ``symbols``, 0 elsewhere)."""
    return cls(symbols, ignore_case=ignore_case, unknown=0, mapping={ch: 1 for ch in symbols})

  def __len__(self) -> int:
    return len(self.symbols)

  def __repr__(self) -> str:
    return f"Alphabet({self.symbols!r}, unknown={self.unknown})"

  def code(self, ch: str) -> int:
    """Return the integer code for a single character."""
    return int(self.lut[ord(ch)])

  def __getitem__(self, seq):
    """Map a ``uint8`` byte or array through the 256-byte LUT (compiles inside ``@par.jit``)."""
    return self.lut[seq]

  def encode(self, seq, *, dtype=None):
    """Encode ``str``, ``bytes``, ``list[str]``, ``RaggedArray``, or ``uint8`` array to codes."""
    if isinstance(seq, RaggedArray):
      vals = self.lut[seq.values]
      if dtype is not None:
        vals = vals.astype(dtype)
      return RaggedArray(vals, seq.offsets)
    arr = _to_byte_array(seq)
    if isinstance(arr, RaggedArray):
      return self.encode(arr, dtype=dtype)
    out = self.lut[arr]
    return out.astype(dtype) if dtype is not None else out

  def decode(self, codes, *, unknown_char: str = "?") -> str:
    """Decode a 1D integer code array back into a Python string."""
    syms = self.symbols
    return "".join(syms[int(c)] if 0 <= int(c) < len(syms) else unknown_char for c in np.asarray(codes).ravel())


class Vocab:
  """Character vocabulary for tokenizing text into compact integer arrays.

  Example::

      vocab = par.Vocab.from_text(text)
      ids = vocab.encode(text)   # int64 token array
      vocab.decode(ids[:32])
  """

  __parakeet_record__ = True

  def __init__(self, chars: str | list[str] | tuple[str, ...]):
    self.chars = "".join(chars) if not isinstance(chars, str) else chars
    self._char_to_id = {ch: i for i, ch in enumerate(self.chars)}
    # Fast 256-entry LUT when all vocabulary characters are ASCII
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(self.chars):
      if ord(ch) < 256:
        lut[ord(ch)] = i
    self.lut = lut

  @classmethod
  def from_text(cls, text: str) -> Vocab:
    """Build a sorted vocabulary from the distinct characters in ``text``."""
    return cls(sorted(set(text)))

  def __len__(self) -> int:
    return len(self.chars)

  def __repr__(self) -> str:
    return f"Vocab(size={len(self.chars)})"

  def __getitem__(self, seq):
    """Map ASCII ``uint8`` bytes/arrays to ``int64`` token IDs (compiles inside ``@par.jit``)."""
    return self.lut[seq]

  def encode(self, text: str | bytes, *, dtype=np.int64) -> np.ndarray:
    """Encode a Python string or ASCII bytes into a 1D token ID array."""
    if isinstance(text, (bytes, bytearray)):
      arr = np.frombuffer(text, dtype=np.uint8)
      return self.lut[arr].astype(dtype, copy=False)
    if text.isascii():
      arr = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
      return self.lut[arr].astype(dtype, copy=False)
    return np.array([self._char_to_id[ch] for ch in text], dtype=dtype)

  def decode(self, ids) -> str:
    """Decode a 1D sequence of token IDs back into a Python string."""
    chars = self.chars
    return "".join(chars[int(i)] for i in np.asarray(ids).ravel())
