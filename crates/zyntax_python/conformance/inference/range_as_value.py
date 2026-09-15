import sys

if sys.version_info[0] > 2:
    xrange = range

def numbers(n):
    total = 0
    for i in xrange(n):
        total += i
    return total

print(numbers(5))
print(list(xrange(4)))
print(list(xrange(2, 6)))
print(list(xrange(7, 0, -2)))
print(list(xrange(0)))

def apply(f):
    return list(f(1, 5, 2))

print(apply(range))
functions = [range]
print(list(functions[0](3)))

try:
    xrange()
except TypeError:
    print("missing bound")
try:
    xrange(1, 2, 3, 4)
except TypeError:
    print("too many bounds")
try:
    xrange(1.5)
except TypeError:
    print("noninteger bound")
try:
    xrange(0, 2, 0)
except ValueError:
    print("zero step")

def local_alias():
    seq = range
    print(list(seq(2)))
    seq = lambda n: [n]
    print(seq(2))

local_alias()
