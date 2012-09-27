import numpy as np 


def _find_all_dtypes():
  all_dtypes = []
  # collect all the numpy scalar types 
  for (category, numpy_types) in np.sctypes.iteritems():
    # skip currently unsupported types like 'unicode' 
    if category == 'others':
      numpy_types = [np.bool8]
    dtypes = map(np.dtype, numpy_types)
    all_dtypes.extend(dtypes)
  return all_dtypes 

dtypes = _find_all_dtypes()
float_dtypes = map(np.dtype, np.sctypes['float'])
int_dtypes = map(np.dtype, np.sctypes['int'])
uint_dtypes = map(np.dtype, np.sctypes['uint'])

 
def is_int(dt):
  return dt in int_dtypes

def is_uint(dt):
  return dt in uint_dtypes

def is_float(dt):
  return dt in float_dtypes 


def _generate_conversion_table():
  conversions  = {}
  
  # for all pairs of dtypes, 
  # create instances of that type
  # and add them together
  # the type of result type should be what we
  # always convert that combination of types to
  for dtype1 in dtypes:
    
    x = dtype1.type(1)  
    for dtype2 in dtypes:
      result = x + dtype2.type(1)
      try:
        conversions[ (dtype1, dtype2) ] = result.dtype 
      except:
        pass    
  return conversions 

# map pairs of numpy type objects to result of operations between them 
conversion_table = _generate_conversion_table()
  
  
def combine(dt1, dt2):
  return conversion_table.get( (dt1, dt2) )
  