"""Fixed Python record structure carried alongside flattened numerical leaves."""

from dataclasses import dataclass, fields, is_dataclass


@dataclass(frozen=True)
class Record:
  cls: type
  names: tuple[str, ...]

  def values(self, obj):
    if self.cls is dict:
      return tuple(obj[name] for name in self.names)
    return tuple(getattr(obj, name) for name in self.names)

  def build(self, values):
    if self.cls is dict:
      return dict(zip(self.names, values, strict=True))
    if issubclass(self.cls, tuple):
      return tuple.__new__(self.cls, values)
    obj = object.__new__(self.cls)
    for name, value in zip(self.names, values, strict=True):
      object.__setattr__(obj, name, value)
    return obj


def record_type(obj):
  cls = type(obj)
  if cls is dict and all(type(key) is str for key in obj):
    return Record(dict, tuple(obj))
  if isinstance(obj, tuple) and isinstance(names := getattr(cls, "_fields", None), tuple):
    return Record(cls, names)
  if (
    not isinstance(obj, type)
    and is_dataclass(obj)
    and cls.__dataclass_params__.frozen
    and not cls.__module__.startswith("parakeet.")
  ):
    return Record(cls, tuple(f.name for f in fields(obj)))
  return None
