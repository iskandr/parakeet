import numpy as np 


def _find_all_dtypes():
  all_dtypes = []
  # collect all the numpy scalar types 
  for (category, dtypes) in np.sctypes.iteritems():
    if category != 'others':
      all_dtypes.extend(dtypes)
  return all_dtypes 

dtypes = _find_all_dtypes()

def _generate_byte_sizes():
  byte_sizes = {}
  for dtype in dtypes:
    x = dtype(0)
    # look up the buffer storing this numpy scalar instance
    # and get its length in bytes
    if hasattr(x, 'data'):
      byte_sizes[dtype] = len(x.data)
  return byte_sizes

byte_sizes = _generate_byte_sizes()


def _generate_conversion_table():
  conversions = {}
  # for all pairs of dtypes, 
  # create instances of that type
  # and add them together
  # the type of result type should be what we
  # always convert that combination of types to
  for t1 in dtypes:
    x = t1(1)
    for t2 in dtypes:
      result = x + t2(1)
      try:
        conversions[ (t1, t2) ] = result.dtype 
      except:
        pass    
  return conversions

conversion_table = _generate_conversion_table()
  
  
   
def is_int(dtype):
  return dtype in np.sctypes['int']

def is_uint(dtype):
  return dtype in np.sctypes['uint']

def is_floating(dtype):
  return dtype in np.sctypes['float']

def nbytes(dtype):
  return byte_sizes[dtype]

def combine(t1, t2):
  return conversion_table.get( (t1, t2) )
  

