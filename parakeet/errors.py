"""Exception hierarchy.

Compile-time problems are reported as :class:`CompileError` subclasses that
carry the location of the offending Python source, so the message points at the
user's code rather than at Parakeet internals.  Runtime problems detected by
generated code (out-of-bounds indices, shape mismatches, ...) are re-raised as
the same built-in exception NumPy/Python would have raised.
"""

from __future__ import annotations

import linecache
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceLocation:
  """A position in the Python source of a function being compiled."""

  filename: str
  lineno: int
  col: int = 0
  end_col: int | None = None
  function: str = "?"

  def line(self) -> str | None:
    text = linecache.getline(self.filename, self.lineno)
    return text.rstrip("\n") if text else None

  def frame(self) -> str:
    """A small code frame: ``file:line`` plus the source line and a caret."""
    header = f'  File "{self.filename}", line {self.lineno}, in {self.function}'
    text = self.line()
    if text is None:
      return header
    stripped = text.lstrip()
    indent = len(text) - len(stripped)
    out = [header, "    " + stripped]
    if self.col >= indent:
      width = 1
      if self.end_col is not None and self.end_col > self.col:
        width = min(self.end_col, len(text)) - self.col
      out.append("    " + " " * (self.col - indent) + "^" * max(width, 1))
    return "\n".join(out)

  def short(self) -> str:
    return f"{self.filename}:{self.lineno} (in {self.function})"


class ParakeetError(Exception):
  """Base class of every error raised by Parakeet itself."""


class CompileError(ParakeetError):
  """The function could not be compiled.

  ``location`` is the innermost source position; ``call_chain`` lists the
  call sites (outermost first) through which the failing code was inlined.
  """

  def __init__(
    self,
    message: str,
    location: SourceLocation | None = None,
    call_chain: tuple[SourceLocation, ...] = (),
    hint: str | None = None,
  ):
    self.message = message
    self.location = location
    self.call_chain = call_chain
    self.hint = hint
    super().__init__(message)

  def __str__(self) -> str:
    parts = [self.message]
    if self.hint:
      parts.append(f"hint: {self.hint}")
    if self.location is not None:
      parts.append(self.location.frame())
    for site in reversed(self.call_chain):
      parts.append("called from\n" + site.frame())
    return "\n".join(parts)


class UnsupportedError(CompileError):
  """A Python/NumPy feature outside the subset Parakeet compiles."""


class TypeInferenceError(CompileError, TypeError):
  """Types could not be reconciled (also catchable as ``TypeError``)."""


class StaticValueError(CompileError, ValueError):
  """A value that must be known at compile time was not, or was invalid."""


class BackendError(ParakeetError):
  """A backend failed to build or run generated code."""


class BackendUnavailable(BackendError):
  """The requested backend cannot run on this machine.

  The message always says what is missing and how to get it.
  """


class CCompilerError(BackendError):
  """The system C compiler rejected generated code (a Parakeet bug)."""

  def __init__(self, message: str, source_path: str | None = None, stderr: str = ""):
    self.source_path = source_path
    self.stderr = stderr
    super().__init__(message)


class InternalError(ParakeetError):
  """An invariant inside Parakeet was violated - please report it."""


class FallbackWarning(UserWarning):
  """A parallel loop could not run on the requested device and ran on the CPU."""
