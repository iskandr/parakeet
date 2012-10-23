import sys
import traceback 

def run_local_functions(prefix, locals_dict = None):   
  if locals_dict is None:
    last_frame = sys._getframe() 
    locals_dict = last_frame.f_back.f_locals
  
  good = set([])
  bad = set([])
  for k, test in locals_dict.iteritems():
    
    if k.startswith(prefix):
      print "Running %s..." % k
      try:
        test()
        print "\n --- %s passed\n" % k
        good.add(k)
      
      except:
        #traceback.print_tb(sys.exc_info()[2])
        #print sys.exc_info()
        raise 
        #traceback.print_tb()
        #print sys.exc_info()[1]
        print "\n --- %s failed\n" % k
        bad.add(k)
  print "\n%d tests passed: %s\n" % (len(good), ", ".join(good))
  print "%d failed: %s" % (len(bad),", ".join(bad))

def run_local_tests(locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals
  return run_local_functions("test_", locals_dict)
  