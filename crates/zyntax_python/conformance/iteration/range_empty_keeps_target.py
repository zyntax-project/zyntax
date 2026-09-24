# test_range / test_grammar: a loop over an empty range leaves a bound
# target untouched; a loop that runs leaves the last value.
import sys
if sys.version_info[0] > 2:
    xrange = range


def up(n):
    i = 7
    for i in range(n):
        pass
    return i


def by_three(n):
    i = 7
    for i in range(2, n, 3):
        pass
    return i


def down(n):
    i = 7
    for i in range(n, -5, -2):
        pass
    return i


def alias(n):
    i = 9
    for i in xrange(n):
        pass
    return i


def alias_down(n):
    i = 9
    for i in xrange(n, 0, -1):
        pass
    return i


def broken(n):
    i = 7
    for i in range(n):
        if i == 2:
            break
    return i


print(up(0), up(3), up(-2))
print(by_three(0), by_three(10), by_three(2), by_three(3))
print(down(-10), down(4), down(-5), down(-4))
print(alias(0), alias(4))
print(alias_down(0), alias_down(3))
print(broken(0), broken(2), broken(5))

k = 42
for k in range(0):
    pass
print(k)
for k in range(5, 5):
    pass
print(k)
for k in range(5, 1):
    pass
print(k)
for k in range(1, 5, -1):
    pass
print(k)
for k in range(3):
    pass
print(k)
for k in xrange(10, 0, -3):
    pass
print(k)
