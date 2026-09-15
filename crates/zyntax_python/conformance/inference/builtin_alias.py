# A name bound once to a builtin is called as that builtin.

import sys
if sys.version_info[0] > 2:
    xrange = range

total = 0
for i in xrange(5):
    total += i
print(total)
print(list(xrange(2, 8, 3)), len(list(xrange(3))))


def count(n):
    c = 0
    for j in xrange(n):
        c += j * j
    return c


print(count(10))
