#
# GF2 Power and Multiplication functions 
# Copied from Jason Sachs's libgf2: https://bitbucket.org/jason_s/libgf2/
#

def gf2mulmod(x,y,m):
    z = 0
    while x > 0:
        if (x & 1) != 0:
            z ^= y
        y <<= 1
        y2 = y ^ m
        if y2 < y:
            y = y2
        x >>= 1
    return z

def gf2powmod(x,k,m):
    z = 1
    while k > 0:
        if (k & 1) != 0:
            z = gf2mulmod(z,x,m)
        x = gf2mulmod(x,x,m)
        k >>= 1
    return z

x = 0b101010101010
y = 3
power = 1024


from compare_perf import compare_perf 
compare_perf(gf2powmod, [x, power, y])
