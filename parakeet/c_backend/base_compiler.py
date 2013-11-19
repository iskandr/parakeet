from .. import names 
from ..ndtypes import BoolT, IntT 
import type_mappings 
from reserved_names import is_reserved


class BaseCompiler(object):
  
  def __init__(self, extra_link_flags = None, extra_compile_flags = None):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}
    self.extra_link_flags = extra_link_flags if extra_link_flags else []
    self.extra_compile_flags = extra_compile_flags if extra_compile_flags else []
    
  def add_compile_flag(self, flag):
    if flag not in self.extra_compile_flags:
      self.extra_compile_flags.append(flag)
  
  def add_link_flag(self, flag):
    if flag not in self.extra_link_flags:
      self.extra_link_flags.append(flag)
        
  
  def visit_expr(self, expr):
    expr_class_name = expr.__class__.__name__
    method_name = "visit_" + expr_class_name
    assert hasattr(self, method_name), "Unsupported expression %s" % expr_class_name  
    result = getattr(self, method_name)(expr)
    assert result is not None, \
      "Compilation method for expression %s returned None, expected code string" % expr_class_name
    return result 
      
  def visit_expr_list(self, exprs):
    return [self.visit_expr(e) for e in exprs]
  
  def breakpoint(self):
    self.append("raise(SIGINT);")
    
  def visit_stmt(self, stmt):
    stmt_class_name = stmt.__class__.__name__
    method_name = "visit_" + stmt_class_name
    assert hasattr(self, method_name), \
      "Statement %s not supported by %s" % (stmt_class_name, self.__class__.__name__)  
    result = getattr(self, method_name)(stmt)
    assert result is not None, "Compilation method for statement %s return None" % stmt_class_name
    return result 
  
  def push(self):
    self.blocks.append([])
  
  def pop(self):
    stmts = self.blocks.pop()
    return "  " + self.indent("\n".join("  " + stmt for stmt in stmts))
  
  def indent(self, block_str):
    return block_str.replace("\n", "\n  ")
  
  def append(self, stmt):
    stripped = stmt.strip()
    
    assert len(stripped) == 0 or \
      ";" in stripped or \
      stripped.startswith("//") or \
      stripped.startswith("/*"), "Invalid statement: %s" % stmt
    self.blocks[-1].append(stmt)
  
  def newline(self):
    self.append("\n")
    
  def comment(self, text):
    self.append("// %s" % text)
  
  def printf(self, fmt, *args):
    
    result = 'printf("%s\\n"' % fmt
    if len(args) > 0:
      result = result + ", " + ", ".join(str(arg) for arg in args)
    self.append( result + ");" )
  
  def fresh_name(self, prefix):
    prefix = names.original(prefix)
    
    prefix = prefix.replace(".", "")
  
    version = self.name_versions.get(prefix, 1)
    self.name_versions[prefix] = version + 1
    
    # not valid chars!
    if not any(c.isalpha() for c in prefix):
      prefix = self.fresh_name("temp" + prefix) 
      
    if version == 1 and not is_reserved(prefix):
      return prefix 
    elif prefix[-1] != "_":
      return "%s_%d" % (prefix, version)
    else:
      return prefix + str(version)
    
  def to_ctype(self, t):
    """
    Convert Parakeet type to string representing its C type.
    The base class implementation only handles scalars, 
    support for Tuples, Slices, and Arrays is in the overload FlatFnCompiler.to_ctype
    """
    return type_mappings.to_ctype(t)
  
  def fresh_var(self, t, prefix = None, init = None):
    if prefix is None:
      prefix = "temp"
    name = self.fresh_name(prefix)
    if isinstance(t, str):
      t_str = t
    else:
      t_str = self.to_ctype(t)
    if init is None:
      self.append("%s %s;" % (t_str, name))
    else:
      self.append("%s %s = %s;" % (t_str, name, init))
    return name
  
  def fresh_array_var(self, t, n, prefix = None):
    if prefix is None:
      prefix = "temp"
    name = self.fresh_name(prefix)
    if isinstance(t, str):
      t_str = t
    else:
      t_str = self.to_ctype(t)
    self.append("%s %s[%d];" % (t_str, name, n))
    return name
  
  def assign(self, name, rhs):
    self.append("%s = %s;" % (name, rhs))
  
  def name(self, ssa_name, overwrite = False):
    """
    Convert from ssa names, which might have large version numbers and contain 
    syntactically invalid characters to valid local C names
    """
    if ssa_name in self.name_mappings and not overwrite:
      return self.name_mappings[ssa_name]
    prefix = names.original(ssa_name)
    prefix = prefix.replace(".", "")

    name = self.fresh_name(prefix)
    self.name_mappings[ssa_name] = name 
    return name
   
  def return_if_null(self, obj):
    self.append("if (!%s) { return NULL; }" % obj)
    
  def not_(self, x):
    if x == "1":
      return "0"
    elif x == "0":
      return "1"
    return "!%s" % x
  
  def and_(self, x, y):
    if x == "0" or y == "0":
      return "0"
    elif x == "1" and y == "1":
      return "1"
    elif x == "1":
      return y 
    elif y == "1":
      return x
    return "%s && %s" % (x,y) 
  
  def or_(self, x, y):
    if x == "1" or y == "1":
      return "1"
    elif x == "0":
      return y
    elif y == "0":
      return x 
    return "%s || %s" % (x,y) 
  
  def gt(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s > %s" % (x, y)
  
  def gte(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s >= %s" % (x,y) 
  
  def lt(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s < %s" % (x,y)
  
  def lte(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s <= %s" % (x, y) 
  
  def neq(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "0"
    return "%s != %s" % (x, y) 
  
  def eq(self, x, y, t):
    if isinstance(t, (BoolT, IntT)) and x == y:
      return "1"
    return "%s == %s" % (x, y)
  
  
  def add(self, x, y):
    if x == "0":
      return y
    elif y == "0":
      return x
    return "%s + %s" % (x,y)
  
  def sub(self, x, y):
    if x == "0":
      return "-(%s)" % y 
    elif y == "0":
      return x 
    return "%s - %s" % (x,y)
  
  def mul(self, x, y):
    if x == "1":
      return y 
    elif y == "1":
      return x 
    elif x == 0 or y == 0:
      return "0"
    return "%s * %s" % (x,y)
  
  def div(self, x, y):
    if x == y:
      return "1"
    elif x == "0":
      return "0"
    elif y == "1":
      return x
    else:
      return "%s / %s" % (x,y)
    