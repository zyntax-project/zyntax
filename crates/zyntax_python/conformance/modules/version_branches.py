# a branch on the interpreter's version is decided when the program is compiled;
# an import nothing reads is dropped, whatever module it names
import sys
import re
import string as text_tools
if sys.version_info[0] < 3:
    from itertools import izip
    range = xrange
else:
    izip = zip
if sys.version_info >= (3, 0):
    kind = "three"
elif sys.version_info.major == 2:
    kind = "two"
else:
    kind = "other"
total = 0
for a, b in izip([1, 2, 3], [10, 20, 30]):
    total += a * b
print(kind, total, sys.version_info[1] > 5)
def pairs(xs, ys):
    return list(izip(xs, ys))
print(pairs("ab", "cd"))
