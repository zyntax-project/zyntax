# A typed set against a value only the runtime knows: a set of its own
# kind is used as it is, a set of another kind is compared value by
# value, anything else is the TypeError.
def f(free, fp):
    n = free - fp
    m = free & fp
    return sorted(n), [int(x) for x in sorted(m)], fp <= free, free >= fp, fp < free, free > fp, free <= fp, min(n), len(free & fp), len(free - fp)


def g(x: object):
    return x


a = frozenset(range(10))
print(f(a, g(frozenset([2, 3]))))
print(f(a, g({2, 5})))
print(f(a, g({1.0, 9.0})))
try:
    print(a - g(3))
except TypeError as e:
    print("TypeError", e)
b = {1, 2, 3}
print(sorted(b - g({1})), b <= g(frozenset([1, 2, 3, 4])), b < g(b), g(b) <= b)
try:
    print(b & g([1]))
except TypeError as e:
    print("TypeError", e)
