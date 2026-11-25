"""Locate and parse the Python source of functions to be compiled."""

from __future__ import annotations

import ast
import inspect
import linecache
import os
import types as pytypes
from dataclasses import dataclass

from ..errors import SourceLocation, UnsupportedError

FuncNode = ast.FunctionDef | ast.Lambda


@dataclass(eq=False)
class FunctionSource:
  node: FuncNode
  filename: str
  name: str
  _outside_loads: dict[int, frozenset[str]] | None = None
  _mutated_lists: frozenset[str] | None = None

  def mutated_list_names(self) -> frozenset[str]:
    """Local variable names on which list-mutating methods are called."""
    if self._mutated_lists is None:
      names: set[str] = set()
      for n in ast.walk(self.node):
        if (
          isinstance(n, ast.Call)
          and isinstance(n.func, ast.Attribute)
          and n.func.attr in ("append", "extend", "pop", "clear")
          and isinstance(n.func.value, ast.Name)
        ):
          names.add(n.func.value.id)
      self._mutated_lists = frozenset(names)
    return self._mutated_lists

  def location(self, node: ast.AST | None = None) -> SourceLocation:
    node = node if node is not None else self.node
    lineno = getattr(node, "lineno", getattr(self.node, "lineno", 0))
    col = getattr(node, "col_offset", 0)
    end_col = getattr(node, "end_col_offset", None)
    if getattr(node, "end_lineno", lineno) != lineno:
      end_col = None
    return SourceLocation(self.filename, lineno, col, end_col, self.name)

  def loads_outside(self, loop: ast.AST) -> frozenset[str]:
    """Names read anywhere in this function *outside* ``loop``."""
    if self._outside_loads is None:
      self._outside_loads = {}
    cached = self._outside_loads.get(id(loop))
    if cached is None:
      inside = {id(n) for n in ast.walk(loop)}
      cached = frozenset(
        n.id
        for n in ast.walk(self.node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in inside
      )
      self._outside_loads[id(loop)] = cached
    return cached


_module_cache: dict[str, tuple[tuple, ast.Module]] = {}
_source_cache: dict[pytypes.CodeType, FunctionSource] = {}


def _parse_file(filename: str) -> ast.Module | None:
  lines = linecache.getlines(filename)
  if not lines:
    return None
  try:
    stamp: tuple = (os.stat(filename).st_mtime_ns, len(lines))
  except OSError:
    stamp = (len(lines), hash(lines[0]) if lines else 0)
  hit = _module_cache.get(filename)
  if hit is not None and hit[0] == stamp:
    return hit[1]
  try:
    tree = ast.parse("".join(lines), filename=filename)
  except SyntaxError:
    return None
  _module_cache[filename] = (stamp, tree)
  return tree


def _first_line(node: ast.AST) -> int:
  decorators = getattr(node, "decorator_list", [])
  return min([node.lineno] + [d.lineno for d in decorators])  # type: ignore[attr-defined]


def _arg_names(args: ast.arguments) -> tuple[str, ...]:
  names = [a.arg for a in args.posonlyargs + args.args]
  if args.vararg:
    names.append(args.vararg.arg)
  names += [a.arg for a in args.kwonlyargs]
  if args.kwarg:
    names.append(args.kwarg.arg)
  return tuple(names)


def _code_arg_names(code: pytypes.CodeType) -> tuple[str, ...]:
  n = code.co_argcount + code.co_kwonlyargcount
  n += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
  names = list(code.co_varnames[:n])
  # co_varnames order: positional, kwonly, *args, **kwargs -> normalise to source order
  pos = names[: code.co_argcount]
  kwonly = names[code.co_argcount : code.co_argcount + code.co_kwonlyargcount]
  rest = names[code.co_argcount + code.co_kwonlyargcount :]
  out = pos
  if code.co_flags & inspect.CO_VARARGS:
    out = out + rest[:1]
    rest = rest[1:]
  return tuple(out + kwonly + rest)


def _find_node(tree: ast.AST, code: pytypes.CodeType) -> FuncNode | None:
  is_lambda = code.co_name == "<lambda>"
  want_args = _code_arg_names(code)
  candidates: list[FuncNode] = []
  for node in ast.walk(tree):
    if is_lambda:
      if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno:
        if _arg_names(node.args) == want_args:
          candidates.append(node)
    elif isinstance(node, ast.FunctionDef):
      if node.name == code.co_name and _first_line(node) == code.co_firstlineno:
        candidates.append(node)
  if not candidates:
    return None
  if len(candidates) == 1:
    return candidates[0]
  # several lambdas with the same signature on one line: compare bytecode
  for node in candidates:
    try:
      compiled = compile(ast.Expression(node), "<parakeet>", "eval")  # type: ignore[arg-type]
    except (SyntaxError, ValueError):
      continue
    for const in compiled.co_consts:
      if isinstance(const, pytypes.CodeType) and const.co_code == code.co_code:
        if const.co_consts == code.co_consts and const.co_names == code.co_names:
          return node
  # fall back on the columns recorded in the code object: the instructions of
  # a lambda's body lie inside the lambda
  try:
    cols = [p[2] for p in code.co_positions() if p[0] == code.co_firstlineno and p[2]]
  except (AttributeError, TypeError):
    cols = []
  if cols:
    col = min(cols)
    inside = [n for n in candidates if n.col_offset <= col < (getattr(n, "end_col_offset", None) or col + 1)]
    if inside:
      return max(inside, key=lambda n: n.col_offset)
  raise UnsupportedError(
    f"cannot tell apart the lambdas on line {code.co_firstlineno} of {code.co_filename}; "
    "put them on separate lines (or use def)"
  )


def get_source(fn: pytypes.FunctionType) -> FunctionSource:
  """Return the parsed definition of ``fn`` (cached per code object)."""
  code = fn.__code__
  hit = _source_cache.get(code)
  if hit is not None:
    return hit
  name = getattr(fn, "__qualname__", code.co_name)
  if code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
    raise UnsupportedError(f"{name} is a generator or coroutine; only plain functions can be compiled")
  filename = code.co_filename
  node: FuncNode | None = None
  tree = _parse_file(filename)
  if tree is not None:
    node = _find_node(tree, code)
  if node is None:
    # e.g. the file changed on disk, or the function was built by exec()
    try:
      lines, first = inspect.getsourcelines(fn)
      import textwrap

      snippet = ast.parse(textwrap.dedent("".join(lines)))
      ast.increment_lineno(snippet, first - 1)
      fake = pytypes.CodeType.replace(code, co_firstlineno=first)
      node = _find_node(snippet, fake)
    except (OSError, TypeError, SyntaxError, IndentationError):
      node = None
  if node is None:
    raise UnsupportedError(
      f"cannot find the source code of {name} (defined in {filename!r}); "
      "Parakeet compiles functions from their Python source, so functions built with "
      "exec/eval or defined in an interactive prompt without source cannot be compiled"
    )
  src = FunctionSource(node, filename, fn.__name__)
  _source_cache[code] = src
  return src


class _AssignedNames(ast.NodeVisitor):
  def __init__(self) -> None:
    self.names: set[str] = set()

  def visit_Name(self, node: ast.Name) -> None:
    if isinstance(node.ctx, (ast.Store, ast.Del)):
      self.names.add(node.id)

  def visit_Attribute(self, node: ast.Attribute) -> None:
    if isinstance(node.ctx, (ast.Store, ast.Del)):
      cur = node.value
      while isinstance(cur, ast.Attribute):
        cur = cur.value
      if isinstance(cur, ast.Name):
        self.names.add(cur.id)
    self.generic_visit(node)

  def visit_MatchAs(self, node: ast.MatchAs) -> None:
    if node.name is not None:
      self.names.add(node.name)
    self.generic_visit(node)

  def visit_MatchStar(self, node: ast.MatchStar) -> None:
    if node.name is not None:
      self.names.add(node.name)

  def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
    if node.rest is not None:
      self.names.add(node.rest)
    self.generic_visit(node)

  def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
    self.names.add(node.name)  # do not descend: inner scopes bind their own names

  def visit_Lambda(self, node: ast.Lambda) -> None:
    pass

  def _comprehension(self, node) -> None:
    # comprehension targets are local to the comprehension, but walrus targets leak
    for child in ast.walk(node):
      if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
        self.names.add(child.target.id)

  visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension


_assigned_cache: dict[int, tuple[ast.AST, frozenset[str]]] = {}


def assigned_names(nodes: list[ast.stmt] | ast.AST, cache_key: ast.AST | None = None) -> frozenset[str]:
  """Names (re)bound by the given statements, excluding nested function bodies."""
  if cache_key is not None:
    hit = _assigned_cache.get(id(cache_key))
    if hit is not None and hit[0] is cache_key:
      return hit[1]
  visitor = _AssignedNames()
  for n in nodes if isinstance(nodes, list) else [nodes]:
    visitor.visit(n)
  result = frozenset(visitor.names)
  if cache_key is not None:
    _assigned_cache[id(cache_key)] = (cache_key, result)
  return result
