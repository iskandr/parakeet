
"""
When the adverb_api creates python functions which can be called from 
outside of Parakeet to run an adverb, they should be registered 
through this module to prevent the ast_translator from trying to 
parse their source
  """
registered_functions = {}
  
def is_registered(python_fn):
  return python_fn in registered_functions
  
def register(python_fn, wrapper):
  registered_functions[python_fn] = wrapper 
  
def get_wrapper(python_fn):
  return registered_functions[python_fn]
  