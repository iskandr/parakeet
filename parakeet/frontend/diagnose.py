
from dsltools.testing_helpers import eq
from run_function import specialize 
from ..transforms import Phase, Transform, CloneFunction
from ..transforms.pipeline import loopify

def transform_name(t):
  assert not isinstance(t, Phase)
  if hasattr(t, 'name') and t.name is not None:
    return t.name 
  elif hasattr(t, '__name__'):
    return t.__name__
  else:
    assert hasattr(t, '__class__')
    return t.__class__.__name__ 
  
def linearize_phase(typed_fn, phase):
  if isinstance(phase, (list, tuple)):
    return linearize_phases(typed_fn, phase)
  elif not isinstance(phase, Phase):
    name = transform_name(phase)
    return [ (name, phase) ]
  else:
    group_name = str(phase)
    previous = linearize_phase(typed_fn, phase.depends_on)
    if phase.run_if is not None and not phase.run_if(typed_fn):
      return previous  
    interleave = []
    for (name, t) in linearize_phases(typed_fn, phase.cleanup):
      interleave.append( ("  %s" % ( name, ), t) )
    combined = previous 
    for (name, t) in linearize_phases(typed_fn, phase.transforms):
      new_name = "%s:%s" % (group_name, name)
      combined.append((new_name, t))
      for (cleanup_name, t2) in interleave:
        cleanup_name = cleanup_name + " (after %s)" % new_name
        combined.append( (cleanup_name, t) )
    return combined 
    
def linearize_phases(typed_fn, phases):
  combined = []
  for phase in phases:
    combined.extend(linearize_phase(typed_fn, phase))
  return combined 
         
def get_transform_list(typed_fn, last_phase = loopify):
  return linearize_phase(typed_fn, last_phase)  

def find_broken_transform(fn, inputs, expected, 
                             print_transforms = True, 
                             print_functions = True, 
                             last_phase = loopify):
  from .. import interp 
  

  fn, args = specialize(fn, inputs, optimize=False)  

  try:  
    interp_result = interp.eval(fn, args)
  except:
    print "[Diagnose] Runtime before any optimizations while trying to run", fn
    raise  
    
  
   
  if not eq(interp_result, expected):
    print "[Diagnose] Expected:", expected 
    print "[Diagnose] Result:", interp_result
    print "[Diagnose] This function was busted before we optimized anything!" 
    return None 
  transforms = get_transform_list(fn, last_phase = last_phase)
  print "[Diagnose] Full list of transforms:"
  for (name, t) in transforms:
    print "  -> ", name 
  old_fn = fn 

  for (name, t) in transforms:
    print "[Diagnose] Running %s..." % (name, )
    # in case t is just a class, instantiate it 
    if not isinstance(t, Transform):
      t = t()
    assert isinstance(t, Transform)
    cloner = CloneFunction(parent_transform = t)
    new_fn = t.apply(cloner.apply(old_fn))
    if print_functions:
      print new_fn 
      
    result = interp.eval(new_fn, args)
    if not eq(result, expected):
      print "[Diagnose] Expected %s but got %s " % (expected, result)
      print "[Diagnose] After running ", t
      print "[Diagnose] Old function ", old_fn 
      print "[Diagnose] Transformed function ", new_fn 
      print "[Diagnose] The culprit was:", name 
      return t 
    old_fn = new_fn 
  print "[Diagnose] No problems found!"
  