import parakeet 

def nocalls(x):
  return x + 1

def call1(x):
  return nocalls(x)

def call2(x):
  return call1(x) 

def test_inline():
  