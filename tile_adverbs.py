import adverb_helpers
import adverbs
import array_type
import closure_type
import copy
import names
import syntax
import tuple_type

from core_types import Int32, Int64
from transform import Transform

def free_vars_list(expr_list):
  rslt = set()
  for expr in expr_list:
    rslt.update(free_vars(expr))
  return rslt

def free_vars(expr):
  if isinstance(expr, syntax.Var):
    return set([expr])
  elif isinstance(expr, (syntax.PrimCall,syntax.Call)):
    return free_vars_list(expr.args)
  elif isinstance(expr, syntax.Index):
    return free_vars(expr.value).union(free_vars(expr.index))
  elif isinstance(expr, syntax.Tuple):
    return free_vars_list(expr.elts)
  else:
    assert isinstance(expr, syntax.Const), "%s is not a Const" % expr
    return set()

class FindAdverbs(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.has_adverbs = False

  def transform_Map(self, expr):
    self.has_adverbs = True
    return expr

  def transform_Reduce(self, expr):
    self.has_adverbs = True
    return expr

  def transform_Scan(self, expr):
    self.has_adverbs = True
    return expr

class TileAdverbs(Transform):
  def __init__(self, fn):
    Transform.__init__(self, fn)
    self.adverbs_visited = []
    self.adverb_args = []
    self.expansions = {}
    self.exp_stack = []

  def push_exp(self, adv):
    self.exp_stack.append(self.expansions)
    self.expansions = copy.deepcopy(self.expansions)
    self.adverbs_visited.append(adv)

  def pop_exp(self):
    self.expansions = self.exp_stack.pop()
    self.adverbs_visited.pop()

  def get_expansions(self, arg):
    if arg in self.expansions:
      return self.expansions[arg]
    else:
      return []

  def get_num_expansions_at_depth(self, arg, depth):
    exps = self.get_expansions(arg)

    for i,v in enumerate(exps):
      if v >= depth:
        return len(exps) - i

    return 0

  def gen_unpack_tree(self, adverb_tree, depths, v_names, block, type_env):
    def order_args(depth):
      cur_depth_args = []
      other_args = []
      for arg in v_names:
        arg_exps = self.get_expansions(arg)
        if depth in arg_exps:
          cur_depth_args.append(arg)
        else:
          other_args.append(arg)
      return (cur_depth_args, other_args)

    def gen_unpack_fn(depth_idx):
      if depth_idx >= len(depths):
        # Create type env for innermost fn - just the original types
        inner_type_env = {}
        for arg in v_names:
          inner_type_env[arg] = type_env[arg]

        # For each stmt in body, add its lhs free vars to the type env
        return_t = Int32 # Dummy type
        for s in block:
          if isinstance(s, syntax.Assign):
            lhs_vars = free_vars(s.lhs)
            lhs_names = [var.name for var in lhs_vars]
            lhs_types = [type_env[name] for name in lhs_names]
            for name, t in zip(lhs_names, lhs_types):
              inner_type_env[name] = t
          elif isinstance(s, syntax.Return):
            if isinstance(s.value, str):
              return_t = type_env[s.value.name]
            else:
              return_t = s.value.type

        # The innermost function always uses all the variables
        arg_types = [array_type.increase_rank(type_env[arg], 1)
                     for arg in v_names]
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=v_names,
                            body=block,
                            input_types=arg_types,
                            return_type=return_t,
                            type_env=inner_type_env)
        return (v_names, arg_types, [], [], fn)
      else:
        # Get the current depth
        depth = depths[depth_idx]

        # Order the arguments for the current depth, i.e. for the nested fn
        cur_arg_names, fixed_arg_names = order_args(depth)

        # Make a type env for this function based on the number of expansions
        # left for each arg
        new_type_env = {}
        new_adverb = adverb_tree[depth_idx]
        print "fixed_arg_names:", fixed_arg_names
        print "cur_arg_names:", cur_arg_names
        if isinstance(new_adverb, adverbs.Reduce):
          arg = fixed_arg_names[0]
          print "expansions:", self.get_num_expansions_at_depth(arg, depth)
          print "type:", type_env[arg]
          new_type_env[arg] = type_env[arg]
          for arg in cur_arg_names + fixed_arg_names[1:]:
            rank_increase = self.get_num_expansions_at_depth(arg, depth)
            new_type_env[arg] = \
                array_type.increase_rank(type_env[arg], rank_increase)
        else:
          for arg in cur_arg_names + fixed_arg_names:
            rank_increase = self.get_num_expansions_at_depth(arg, depth)
            new_type_env[arg] = \
                array_type.increase_rank(type_env[arg], rank_increase)

        cur_arg_types = []
        for arg in cur_arg_names:
          cur_arg_types.append(array_type.increase_rank(new_type_env[arg], 1))
        fixed_arg_types = [type_env[arg] for arg in fixed_arg_names]

        # Generate the nested fn and its fixed and normal args
        nested_arg_names, nested_arg_types, \
            nested_fixed_names, nested_fixed_types, nested_fn = \
            gen_unpack_fn(depth_idx+1)
        nested_args = [syntax.Var(name, type=t)
                       for name, t in zip(nested_arg_names, nested_arg_types)]
        nested_fixed_args = \
            [syntax.Var(name, type=t)
             for name, t in zip(nested_fixed_names, nested_fixed_types)]
        closure_t = closure_type.make_closure_type(nested_fn,
                                                   nested_fixed_types)
        nested_closure = syntax.Closure(nested_fn, nested_fixed_args,
                                        type=closure_t)

        # Make an adverb that wraps the nested fn
        axis = 0 # When unpacking a non-adverb assignment, all axes are 0
        return_t = array_type.increase_rank(nested_fn.return_type, 1)
        if isinstance(new_adverb, adverbs.Reduce):
          new_adverb.combine = nested_closure
          return_t = nested_fn.return_type
          new_adverb.init = syntax.Var(fixed_arg_names[0], type=return_t)
          new_type_env[nested_args[0].name] = return_t
          new_adverb.args = nested_args[1:]
        else:
          new_adverb.args = nested_args
          new_adverb.fn = nested_closure
        new_adverb.type=return_t
        new_adverb.axis = axis

        # Add the adverb to the body of the current fn and return the fn
        fn = syntax.TypedFn(name=names.fresh("expanded_assign"),
                            arg_names=fixed_arg_names + cur_arg_names,
                            body=[syntax.Return(new_adverb)],
                            input_types=fixed_arg_types + cur_arg_types,
                            return_type=return_t,
                            type_env=new_type_env)
        return (cur_arg_names, cur_arg_types,
                fixed_arg_names, fixed_arg_types, fn)

    return gen_unpack_fn(0)

  def get_depths_list(self, v_names):
    depths = set()
    for name in v_names:
      for e in self.get_expansions(name):
        depths.add(e)
    depths = list(depths)
    depths.sort()
    return depths

  def transform_Assign(self, stmt):
    if isinstance(stmt.rhs, adverbs.Adverb):
      new_rhs = self.transform_expr(stmt.rhs)
      return syntax.Assign(stmt.lhs, new_rhs)
    elif len(self.adverbs_visited) > 0:
      fv = free_vars(stmt.rhs)
      fv_names = [v.name for v in fv]
      depths = self.get_depths_list(fv_names)
      map_tree = [adverbs.Map for _ in depths]
      inner_body = [stmt, syntax.Return(stmt.lhs)]
      nested_args, unpack_fn = \
          self.gen_unpack_tree(map_tree, depths, fv_names, inner_body,
                               self.fn.type_env)
      new_rhs = syntax.Call(unpack_fn, nested_args)
      return syntax.Assign(stmt.lhs, new_rhs)
    else:
      # Do nothing if we're not inside a nesting of tiled adverbs
      return stmt

  def transform_Return(self, stmt):
    if isinstance(stmt.value, adverbs.Adverb):
      return syntax.Return(self.transform_expr(stmt.value))

    return stmt

  def transform_Map(self, expr):
    # TODO: Have to handle naming collisions in the expansions dict
    depth = len(self.adverbs_visited)
    self.push_exp(adverbs.Map(expr.fn, expr.args, expr.axis))
    for fn_arg, adverb_arg in zip(expr.fn.arg_names, expr.args):
      new_expansions = copy.deepcopy(self.get_expansions(adverb_arg.name))
      new_expansions.append(depth)
      self.expansions[fn_arg] = new_expansions
      print self.expansions

    new_fn = syntax.TypedFn
    arg_names = fixed_arg_names = []
    depths = self.get_depths_list(expr.fn.arg_names)
    find_adverbs = FindAdverbs(expr.fn)
    find_adverbs.apply(copy=False)

    if find_adverbs.has_adverbs:
      arg_names = expr.fn.arg_names
      input_types = []
      new_type_env = copy.copy(expr.fn.type_env)
      for arg, t in zip(arg_names, expr.fn.input_types):
        new_type = array_type.increase_rank(t, 1)
        input_types.append(new_type)
        new_type_env[arg] = new_type
      return_t = array_type.increase_rank(expr.fn.return_type, 1)
      new_fn = syntax.TypedFn(name=names.fresh("expanded_adverb_fn"),
                              arg_names=arg_names,
                              body=self.transform_block(expr.fn.body),
                              input_types=input_types,
                              return_type=return_t,
                              type_env=new_type_env)
    else:
      arg_names, _, fixed_arg_names, _, new_fn = \
          self.gen_unpack_tree(self.adverbs_visited, depths, expr.fn.arg_names,
                               expr.fn.body, expr.fn.type_env)

    #TODO: below is for when we have multiple axes
    #axis = [len(self.get_expansions(arg)) + a
    #        for arg, a in zip(expr.args, expr.axis)]
    # print new_fn
    arg_idxs = [expr.fn.arg_names.index(arg)
                for arg in fixed_arg_names + arg_names]
    print "arg_idxs:", arg_idxs
    print "expr.args:", expr.args
    args = [expr.args[idx] for idx in arg_idxs]
    for arg in args:
      rank_increase = len(self.get_expansions(arg.name))
      if depth in self.get_expansions(arg.name):
        rank_increase -= 1
      arg.type = array_type.increase_rank(arg.type, rank_increase)
    axis = len(self.get_expansions(expr.fn.arg_names[0])) + expr.axis - 1
    #axis = expr.axis
    self.pop_exp()
    return adverbs.TiledMap(new_fn, args, axis, type=new_fn.return_type)

  def transform_Reduce(self, expr):
    depth = len(self.adverbs_visited)
    new_adverb = adverbs.Reduce(expr.combine, expr.init, expr.fn, expr.args,
                                expr.axis)
    self.push_exp(new_adverb)
    # TODO: assuming all inputs to a multi-input reduce have the same number
    #       of expansions.  Is this true?
    exps = self.get_expansions(expr.args[0])
    for fn_arg in expr.combine.arg_names[1:]:
      new_expansions = copy.deepcopy(exps)
      new_expansions.append(depth)
      self.expansions[fn_arg] = new_expansions
      print self.expansions

    new_fn = syntax.TypedFn
    arg_names = fixed_arg_names = []
    depths = self.get_depths_list(expr.combine.arg_names)
    find_adverbs = FindAdverbs(expr.combine)
    find_adverbs.apply(copy=False)

    if find_adverbs.has_adverbs:
      arg_names = expr.combine.arg_names
      input_types = []
      new_type_env = copy.copy(expr.combine.type_env)
      for arg, t in zip(arg_names, expr.combine.input_types):
        new_type = array_type.increase_rank(t, 1)
        input_types.append(new_type)
        new_type_env[arg] = new_type
      return_t = array_type.increase_rank(expr.combine.return_type, 1)
      new_fn = syntax.TypedFn(name=names.fresh("expanded_adverb_fn"),
                              arg_names=arg_names,
                              body=self.transform_block(expr.combine.body),
                              input_types=input_types,
                              return_type=return_t,
                              type_env=new_type_env)
    else:
      arg_names, _, fixed_arg_names, _, new_fn = \
          self.gen_unpack_tree(self.adverbs_visited, depths,
                               expr.combine.arg_names,
                               expr.combine.body,
                               expr.combine.type_env)

    #TODO: below is for when we have multiple axes
    #axis = [len(self.get_expansions(arg)) + a
    #        for arg, a in zip(expr.args, expr.axis)]
    print new_fn
    arg_idxs = [expr.combine.arg_names.index(arg)
                for arg in fixed_arg_names + arg_names]
    print "arg_idxs:", arg_idxs
    print "expr.args:", expr.args
    args = []
    for i, _ in enumerate(expr.args):
      args.append(expr.args[arg_idxs[i]])
    for arg in args:
      rank_increase = len(self.get_expansions(arg.name))
      if depth in self.get_expansions(arg.name):
        rank_increase -= 1
      arg.type = array_type.increase_rank(arg.type, rank_increase)
      print "increasing rank of", arg, "by", rank_increase
    axis = len(self.get_expansions(expr.combine.arg_names[1])) + expr.axis - 1
    init = expr.init # TODO: lift init
    self.pop_exp()
    print "return type:", new_fn.return_type
    return adverbs.TiledReduce(new_fn, init, expr.fn, args, axis,
                               type=new_fn.return_type)

  def transform_Scan(self, expr):
    self.adverb_args.append((expr.combine, expr.init, expr.emit))
    return self.tile_adverb(expr, adverbs.Scan, adverbs.TiledScan)

  def post_apply(self, fn):
    print fn
    return fn
