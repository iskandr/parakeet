
import syntax

def subst(node, rename_dict):
  if isinstance(node, syntax.Var):
    return syntax.Var(rename_dict.get(node.name, node.name), type = node.type)
  if isinstance(node, (syntax.Expr, syntax.Stmt)):
    new_values = {}
    for member_name in node.members:
      old_v = getattr(node, member_name)
      if member_name == 'merge':
        new_v = subst_phi_nodes(old_v, rename_dict)
      else:
        new_v = subst(old_v, rename_dict)
      new_values[member_name] = new_v
    new_node = node.__class__(**new_values)
    return new_node

  elif isinstance(node, list):
    return subst_list(node, rename_dict)
  elif isinstance(node, tuple):
    return subst_tuple(node, rename_dict)
  elif isinstance(node, dict):
    return subst_dict(node, rename_dict)
  else:
    return node 

def subst_dict(old_dict, rename_dict):
  new_dict = {}
  for (k,v) in old_dict.iteritems():
    new_dict[subst(k, rename_dict)] = subst(v, rename_dict)
  return new_dict 

def subst_list(nodes, rename_dict):
  return [subst(node, rename_dict) for node in nodes]

def subst_tuple(elts, rename_dict):
  return tuple(subst_list(elts, rename_dict))

def subst_phi_nodes(old_merge, d):
  new_merge = {}
  for (k,(l,r)) in old_merge.iteritems():
    new_merge[d.get(k,k)] = subst(l, d), subst(r, d)
  return new_merge

