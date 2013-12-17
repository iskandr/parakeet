#
# Longest hailstone sequence from http://www.mit.edu/~mtikekar/posts/stream-fusion.html
#
import sys

def collatzLen(a0):
    a = a0
    length = 0
    while a != 1:
        a = (a if a%2 == 0 else 3*a+1) / 2
        length += 1
    return length

def maxLen(max_a0):
    max_length = 0
    longest = 0
    for a0 in xrange(1, max_a0 + 1):
        length = collatzLen(a0)
        if length > max_length:
            max_length = length
            longest = a0
    return max_length, longest

from compare_perf import compare_perf

compare_perf(maxLen, [1000000])

