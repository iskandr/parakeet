
from dsltools import ScopedSet 

from .. analysis.collect_vars import collect_var_names, collect_binding_names
from .. analysis.escape_analysis import may_alias
from .. analysis.syntax_visitor import SyntaxVisitor

from .. ndtypes import ImmutableT, ArrayT, PtrT 
from .. syntax import Var, Assign, Return, While, If, Index, Alloc, AllocArray  
from .. syntax import Array, ArrayView, Slice, Struct 

from transform import Transform

class Find_LICM_Candidates(SyntaxVisitor):
  def __init__(self):
    SyntaxVisitor.__init__(self)
    self.mutable_types = None
    self.volatile_vars = ScopedSet()
    self.depends_on = {}
    self.safe_to_move = set([])
    self.curr_block_id = None
    self.block_contains_return = set([])
    self.may_alias = None

  def visit_fn(self, fn):
    self.volatile_vars.push(fn.arg_names)
    self.may_alias = may_alias(fn)
    SyntaxVisitor.visit_fn(self, fn)
    return self.safe_to_move

  def mark_safe_assignments(self, block, volatile_set):
    for stmt in block:
      klass = stmt.__class__
      if klass is Assign and \
         stmt.lhs.__class__ is Var:
        name = stmt.lhs.name
        dependencies = self.depends_on.get(name, set([]))
        volatile = name in volatile_set or \
                   any(d in volatile_set for d in dependencies)
        if not volatile:
          self.safe_to_move.add(name)
      # just in case there are Returns in nested control flow
      # we should probably avoid changing the performance characteristics
      # by pulling out statements which will never run
      elif klass is If:
        if id(stmt.true) in self.block_contains_return or \
           id(stmt.false) in self.block_contains_return:
          break
      elif klass is While:
        if id(stmt.body) in self.block_contains_return:
          break
      elif klass is Return:
        break

  def mark_curr_block_returns(self):
    self.block_contains_return.add(self.curr_block_id)

  def does_block_return(self, block):
    return id(block) in self.block_contains_return

  def visit_Return(self, stmt):
    self.mark_curr_block_returns()

  def visit_block(self, stmts):
    self.curr_block_id = id(stmts)
    SyntaxVisitor.visit_block(self, stmts)

  def visit_merge(self, merge, both_branches = True):
    pass

  def visit_ForLoop(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    self.volatile_vars.add(stmt.var.name)
    SyntaxVisitor.visit_ForLoop(self, stmt)
    if self.does_block_return(stmt.body):
      self.block_contains_return()
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.body, volatile_in_scope)

  def visit_While(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    SyntaxVisitor.visit_While(self, stmt)
    if self.does_block_return(stmt.body):
      self.block_contains_return()
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.body, volatile_in_scope)

  def visit_Var(self, expr):
    self.volatile_vars.add(expr.name)

  def visit_If(self, stmt):
    self.volatile_vars.push(stmt.merge.keys())
    self.visit_expr(stmt.cond)
    SyntaxVisitor.visit_If(self, stmt) 
    if self.does_block_return(stmt.true) or self.does_block_return(stmt.false):
      self.mark_curr_block_returns()
    volatile_in_scope = self.volatile_vars.pop()
    self.mark_safe_assignments(stmt.true, volatile_in_scope)
    self.mark_safe_assignments(stmt.false, volatile_in_scope)

  def is_mutable_alloc(self, expr):
    return isinstance(expr.type, (PtrT, ArrayT)) 
    #c = expr.__class__
    #return c in (Alloc, AllocArray, )
    #return c is Alloc or \
    #       c is AllocArray or \
    #       c is Array or \
    #       c is ArrayView or \
    #       c is 
    #       (c is Struct and not isinstance(expr.type, ImmutableT))

  def visit_Assign(self, stmt):
     
    lhs_names = collect_binding_names(stmt.lhs)
    rhs_names = collect_var_names(stmt.rhs)
   
    for x in lhs_names:
      dependencies = self.depends_on.get(x, set([]))
      dependencies.update(rhs_names)
      self.depends_on[x] = dependencies

    if any(x in self.volatile_vars for x in rhs_names):
      self.volatile_vars.update(lhs_names)
    elif self.is_mutable_alloc(stmt.rhs):
      if len(lhs_names) == 1 and \
         len(self.may_alias.get(lhs_names[0], [])) <= 1:
        pass
      else:
        self.volatile_vars.update(lhs_names)
    # mark any array writes as volatile 
    if stmt.lhs.__class__ is Index:
      assert stmt.lhs.value.__class__ is Var, \
        "Expected LHS array to be variable but instead got %s" % stmt  
      self.volatile_vars.add(stmt.lhs.value.name)
      #print 
      #print "STMT", stmt
      #print "lhs names", lhs_names 
      #print "rhs names", rhs_names 
      #print "volatile vars", self.volatile_vars
      #print 
      

class LoopInvariantCodeMotion(Transform):
  def __init__(self):
    Transform.__init__(self)

  def pre_apply(self, fn):
    self.analysis = Find_LICM_Candidates()
    self.safe_to_move = self.analysis.visit_fn(fn)
    self.binding_depth = {}
    self.mark_binding_depths(fn.arg_names, 0)

  def mark_binding_depths(self, names, depth_offset = 0):
    curr_depth = len(self.blocks._blocks) - 1 + depth_offset
    for name in names:
      self.binding_depth[name] = curr_depth

  def transform_ForLoop(self, stmt):
    self.mark_binding_depths(stmt.merge.iterkeys(), 1)
    self.mark_binding_depths([stmt.var.name], 1)
    return Transform.transform_ForLoop(self, stmt)

  def transform_While(self, stmt):
    self.mark_binding_depths(stmt.merge.iterkeys(), 1)
    return Transform.transform_While(self, stmt)

  def transform_If(self, stmt):
    self.mark_binding_depths(stmt.merge.iterkeys(), 1)
    return Transform.transform_If(self, stmt)

  def transform_Assign(self, stmt):
    # TODO: Allow possibility of indexing into variables that are read-only
    if stmt.lhs.__class__ is Var and stmt.rhs.__class__ is not Index:
      name = stmt.lhs.name
      if name in self.safe_to_move:
        deps = self.analysis.depends_on[name]

        if all(d in self.binding_depth for d in deps):
          if len(deps) > 0:
            target_level = max(self.binding_depth[d] for d in deps)
          else:
            target_level = 0
 
          if target_level >= 0 and target_level < self.blocks.depth():
            self.blocks._blocks[target_level].append(stmt)
            self.binding_depth[name] = target_level
            return None
    self.mark_binding_depths(collect_binding_names(stmt.lhs))
    return stmt
