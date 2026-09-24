# test_builtin: a builtin bound to another name is called through it,
# in loops, comprehensions and as an argument.
import sys
if sys.version_info[0] > 2:
    xrange = range
    unicode = str
    long = int
    izip = zip

total = 0
for i in xrange(5):
    total += i
print(total)
print([i * i for i in xrange(1, 4)])
print(list(xrange(3)), list(xrange(2, 8, 3)), list(xrange(5, 0, -2)))
print(sum(xrange(10)), max(xrange(4)), len(list(xrange(7))))
print(unicode(3), long("12") + 1, list(izip("ab", xrange(2))))


def total_below(n):
    count = 0
    for j in xrange(n):
        for k in xrange(j):
            count += 1
    return count


print(total_below(6))


def local_alias(n):
    rng = range
    return [x for x in rng(n)]


print(local_alias(4))


def shadowed(xrange):
    return xrange * 2


print(shadowed(21))
