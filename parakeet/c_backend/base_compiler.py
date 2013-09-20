from .. import names 

from c_types import to_ctype
from reserved_names import is_reserved


class BaseCompiler(object):
  
  def __init__(self):
    self.blocks = []
    self.name_versions = {}
    self.name_mappings = {}
  
  
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
    assert hasattr(self, method_name), "Unsupported statemet %s" % stmt_class_name  
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
      prefix = "temp" + prefix 
      
    if version == 1 and not is_reserved(prefix):
      return prefix 
    elif prefix[-1] != "_":
      return "%s_%d" % (prefix, version)
    else:
      return prefix + str(version)
    
  def fresh_var(self, t, prefix = None, init = None):
    if prefix is None:
      prefix = "temp"
    name = self.fresh_name(prefix)
    if isinstance(t, str):
      t_str = t
    else:
      t_str = to_ctype(t)
    if init is None:
      self.append("%s %s;" % (t_str, name))
    else:
      self.append("%s %s = %s;" % (t_str, name, init))
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