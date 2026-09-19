# Tuples of a settled shape: built, passed, returned, indexed, unpacked,
# nested, compared, boxed and read back, without a list in sight.
def pair(x, y):
    return (x, y * 2.0)


def swap(p):
    a, b = p
    return (b, a)


def total(points):
    s = 0.0
    for (x, y) in points:
        s += x * y
    return s


p = pair(3, 4.0)
print(p, p[0], p[1], p[-1], len(p))
q = swap(p)
print(q, q[0] + 1.0, q[1] * 2)
nested = (p, q, 7)
(a, b), (c, d), e = nested
print(a, b, c, d, e)
print(nested[0][1], nested[1], nested[2])
print(nested[1:], nested[:1], nested[::-1], nested[::2])
print(p + q, p + (1, "s"))
print(p == (3, 8.0), p != q, (1, 2) < (1, 3), (2, 1) > (1, 9))
print(4.0 in q, 5 in q, 3 in p)
print(p * 2, 2 * q)
print(tuple(p), list(p), sorted((3, 1, 2)), max((3, 1, 2)))
print(str(p), repr(q), "%d-%s" % (p[0], q))
d = {}
d[p] = "p"
d[(3, 8.0)] = "same"
print(d, len(d))
s = set()
s.add(p)
s.add((3, 8.0))
s.add(q)
print(len(s), p in s, (8.0, 3) in s, (1, 1) in s)
boxed = [p, q, nested]
print(boxed[0], boxed[2][0])
x, y = boxed[1]
print(x, y)
for t in (p, q):
    print(t)
print(divmod(17, 5), divmod(7.5, 2.0))
print(bool(p), (0,) and 1, not (0, 0))
one = (5,)
print(one, one[0], len(one))
(only,) = one
print(only)
mixed = (1, "two", 3.0, None, [4], {"five": 5})
print(mixed)
print(isinstance(p, tuple), type(p) == tuple, isinstance(p, list))
f = (1.5, 2.5, 3.5)
print(f[0] + f[1] + f[2], sum(f), min(f), f.count(2.5), f.index(3.5))
