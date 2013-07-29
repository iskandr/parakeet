from .. import syntax
from .. syntax import Const, ForLoop, Var
from .. syntax.helpers import const_int
from ..transforms import CloneStmt 

from loop_transform import LoopTransform


def safediv(m,n):
  return (m+n-1)/n

class LoopUnrolling(LoopTransform):
  def __init__(self, unroll_factor = 4,
                      max_static_unrolling = 8,
                      max_block_size = 50):
    LoopTransform.__init__(self)
    self.unroll_factor = unroll_factor
    if max_static_unrolling is not None:
    # should we unroll static loops more than ones with unknown iters?
      self.max_static_unrolling = max_static_unrolling
    else:
      self.max_static_unrolling = unroll_factor

    self.max_block_size = max_block_size

  def pre_apply(self, fn):
    # skip the alias analysis that's default for LoopTransform
    return fn

  def copy_loop_body(self, stmt, outer_loop_var, iter_num, phi_values = None):
    """Assume the current codegen block is the unrolled loop"""

    cloner = CloneStmt(self.type_env)
    # make a fresh copy of the loop
    loop = cloner.transform_ForLoop(stmt)
    i = const_int(iter_num, loop.var.type)
    loop_var_value = self.add(outer_loop_var, self.mul(stmt.step, i))
    self.assign(loop.var, loop_var_value)

    # if this isn't the first iteration of unrolling
    # then propagate old versions of phi-bound values
    # into this block
    if phi_values is not None:
      for (old_name, input_value) in phi_values.iteritems():
        new_var = cloner.rename_dict[old_name]
        self.assign(new_var, input_value)

    self.blocks.top().extend(loop.body)

    output_values = {}
    for old_name in stmt.merge.iterkeys():
        new_var = cloner.rename_dict[old_name]
        output_values[old_name] = loop.merge[new_var.name][1]

    return output_values, cloner.rename_dict

  def transform_ForLoop(self, stmt):
    assert self.unroll_factor > 0
    if self.unroll_factor == 1:
      return stmt

    if stmt.step.__class__ is Const:
      assert stmt.step.value > 0, "Downward loops not yet supported"

    stmt = LoopTransform.transform_ForLoop(self, stmt)

    if not self.is_simple_block(stmt.body) or \
       len(stmt.body) > self.max_block_size:
      return stmt

    start, stop, step = stmt.start, stmt.stop, stmt.step

    # if loop has static bounds, fully unroll unless it's too big
    unroll_factor = self.unroll_factor

    # number of iterations of loop iterations is not generally known
    if start.__class__ is Const and \
       stop.__class__ is Const and \
       step.__class__ is Const:
      niters = safediv(stop.value - start.value, step.value)
      if niters <= self.max_static_unrolling:
        unroll_factor = niters

    # push the unrolled body onto the stack
    self.blocks.push()

    phi_values = None
    loop_var = self.fresh_var(stmt.var.type,  "loop_counter")
    name_mappings = None
    for iter_num in xrange(unroll_factor):
      #self.comment("Unrolling iteration %d" % iter_num)
      phi_values, curr_names = \
          self.copy_loop_body(stmt, loop_var, iter_num, phi_values)
      if name_mappings is None:
        name_mappings = curr_names

    unrolled_body = self.blocks.pop()
    unroll_value = syntax.helpers.const_int(unroll_factor, stmt.var.type)
    unrolled_step = self.mul(unroll_value, stmt.step, "unrolled_step")
    n_total_iters = self.sub(stop, start, name = "niters")
    n_unrolled_iters = self.div(n_total_iters, unrolled_step, name = "unrolled_iters")
    # Python's division doesn't behave like C, so that small_negative/big_positive = -1
    # ..which is crappy when expecting truncation! 
    n_unrolled_iters = self.max(n_unrolled_iters, self.int(0), name = "unrolled_iters")
    trunc = self.mul(n_unrolled_iters, unrolled_step, "trunc")
    unrolled_stop = self.add(stmt.start, trunc, "unrolled_stop")
    final_merge = {}
    for (old_name, (input_value, _)) in stmt.merge.iteritems():
      first_name_in_loop = name_mappings[old_name].name
      output_value = phi_values[old_name]
      final_merge[first_name_in_loop] = (input_value, output_value)

    unrolled_loop = ForLoop(var = loop_var,
                            start = stmt.start,
                            stop = unrolled_stop,
                            step = unrolled_step,
                            body = unrolled_body,
                            merge = final_merge)

    if unrolled_loop.start.__class__ is Const and \
       unrolled_loop.stop.__class__ is Const and \
       unrolled_loop.step.__class__ is Const:
      start_value = unrolled_loop.start.value
      stop_value = unrolled_stop.value
      step_value = unrolled_loop.step.value
      if start_value + step_value == stop_value:
        self.assign(unrolled_loop.var, unrolled_loop.start)
        # assign all loop-carried variables to their initial values
        if len(final_merge) > 0:
          self.comment("Initialize loop-carried values")
        for (acc_name, (input_value, _)) in final_merge.iteritems():
          var = Var(acc_name, type = input_value.type)
          self.assign(var, input_value)
        # inline loop body
        self.blocks.top().extend(unrolled_body)
        # since we're not going to have a cleanup loop,
        # need to assign all the original phi-carried variables
        if len(stmt.merge) > 0:
          self.comment("Finalize loop-carried values")
        for old_acc_name in stmt.merge.iterkeys():
          last_value = phi_values[old_acc_name]
          var = Var(old_acc_name, last_value.type)
          self.assign(var, last_value)
        return None

    self.blocks.append(unrolled_loop)

    if unrolled_loop.stop.__class__ is not Const or \
       stop.__class__ is not Const or \
       unrolled_loop.stop.value != stop.value:
      cleanup_merge = {}
      for (old_name, (_, output_value)) in stmt.merge.iteritems():
        input_var = name_mappings[old_name]
        cleanup_merge[old_name] = (input_var, output_value)
      stmt.merge = cleanup_merge
      stmt.start = unrolled_loop.stop
    return stmt
