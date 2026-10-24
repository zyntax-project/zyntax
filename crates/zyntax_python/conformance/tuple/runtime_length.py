# Tuples whose length only the run time knows, made from typed sources,
# behave as tuples wherever they go.
t = tuple(x * 2 for x in range(5))
print(t, len(t), t[0], t[-1], t[1:3], t[::-1], t[3:], t[:0])
print(3 in t, 4 in t, sum(t), max(t), min(t), sorted(t, reverse=True))
print(list(t), tuple(t) == t, t == (0, 2, 4, 6, 8), t != (0, 2, 4, 6, 8))
print(isinstance(t, tuple), type(t) is tuple)
d = {t: "evens"}
print(d[(0, 2, 4, 6, 8)], (0, 2, 4, 6, 8) in {t})
a, b, c, e, f = t
print(a, b, c, e, f)
print(bool(t), bool(tuple(x for x in range(0))), tuple(x for x in range(0)))
print(tuple(x for x in range(1)), tuple(range(3)), tuple([1, 2, 3]))
print(tuple(s.upper() for s in ["a", "bb"]), tuple(v / 2 for v in [1, 3]))
print(t + t, t * 2, t < (0, 3), t > (0, 2, 4))
try:
    t[10]
except IndexError as exc:
    print("IndexError:", exc)
print(t is t, [t, t], str(t), repr(t))


class Holder:
    def __init__(self, n):
        self.items = tuple(i * i for i in range(n))

    def total(self):
        s = 0
        for v in self.items:
            s += v
        return s


h = Holder(4)
print(h.items, h.total(), h.items[2])


def first_two(xs):
    return xs[:2]


def squares(n):
    return tuple(i * i for i in range(n))


print(first_two(t), squares(3), squares(0), len(squares(6)))

acc = 0
for k in range(100):
    u = tuple(k + j for j in range(3))
    acc += u[0] + u[2] * 2
print(acc)
