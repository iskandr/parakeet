
from parakeet import jit 

@jit
def sum_loop(xs):
    total = 0
    for x in xs:
        total += x
    return total 

xs = [1,2,3,4,5]
print "Python result", sum(xs)
print "Parakeet result", sum_loop(xs)
